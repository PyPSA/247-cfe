
import pypsa, numpy as np, pandas as pd
from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints
from vresutils.costdata import annuity

import logging
logger = logging.getLogger(__name__)

# Suppress logging of the slack bus choices
pypsa.pf.logger.setLevel(logging.WARNING)

from vresutils.benchmark import memory_logger

# from https://github.com/PyPSA/pypsa-eur-sec/blob/93eb86eec87d34832ebc061697e289eabb38c105/scripts/solve_network.py
override_component_attrs = pypsa.descriptors.Dict({k : v.copy() for k,v in pypsa.components.component_attrs.items()})
override_component_attrs["Link"].loc["bus2"] = ["string",np.nan,np.nan,"2nd bus","Input (optional)"]
override_component_attrs["Link"].loc["bus3"] = ["string",np.nan,np.nan,"3rd bus","Input (optional)"]
override_component_attrs["Link"].loc["bus4"] = ["string",np.nan,np.nan,"4th bus","Input (optional)"]
override_component_attrs["Link"].loc["efficiency2"] = ["static or series","per unit",1.,"2nd bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["efficiency3"] = ["static or series","per unit",1.,"3rd bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["efficiency4"] = ["static or series","per unit",1.,"4th bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["p2"] = ["series","MW",0.,"2nd bus output","Output"]
override_component_attrs["Link"].loc["p3"] = ["series","MW",0.,"3rd bus output","Output"]
override_component_attrs["Link"].loc["p4"] = ["series","MW",0.,"4th bus output","Output"]




# TODO checkout PyPSA-Eur script
def prepare_costs(cost_file, USD_to_EUR, discount_rate, Nyears, lifetime):

    #set all asset costs and other parameters
    costs = pd.read_csv(cost_file, index_col=[0,1]).sort_index()

    #correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
    costs.loc[costs.unit.str.contains("USD"), "value"] *= USD_to_EUR

    #min_count=1 is important to generate NaNs which are then filled by fillna
    costs = costs.loc[:, "value"].unstack(level=1).groupby("technology").sum(min_count=1)
    costs = costs.fillna({"CO2 intensity" : 0,
                          "FOM" : 0,
                          "VOM" : 0,
                          "discount rate" : discount_rate,
                          "efficiency" : 1,
                          "fuel" : 0,
                          "investment" : 0,
                          "lifetime" : lifetime
    })

    annuity_factor = lambda v: annuity(v["lifetime"], v["discount rate"]) + v["FOM"] / 100
    costs["fixed"] = [annuity_factor(v) * v["investment"] * Nyears for i, v in costs.iterrows()]

    return costs


def strip_network(n):
    nodes_to_keep = snakemake.config['basenodes_to_keep'][:]
    for b in snakemake.config['basenodes_to_keep']:
        for s in snakemake.config['node_suffixes_to_keep']:
            nodes_to_keep.append(b + " " + s)

    nodes_to_keep.extend(snakemake.config['additional_nodes'])

    print("keeping nodes:",nodes_to_keep)

    n.mremove('Bus', n.buses.index.symmetric_difference(nodes_to_keep))
    
    #make sure lines are kept
    n.lines.carrier = "AC"

    carrier_to_keep = snakemake.config['carrier_to_keep']

    for c in n.iterate_components(["Generator","Link","Line","Store","StorageUnit","Load"]):
        if c.name in ["Link","Line"]:
            location_boolean = c.df.bus0.isin(nodes_to_keep) & c.df.bus1.isin(nodes_to_keep)
        else:
            location_boolean = c.df.bus.isin(nodes_to_keep)
        to_keep = c.df.index[location_boolean & c.df.carrier.isin(carrier_to_keep)]
        to_drop = c.df.index.symmetric_difference(to_keep)
        n.mremove(c.name, to_drop)


def add_ci(n):
    """Add C&I at its own node"""

    #first deal with global policy environment
    gl_policy = snakemake.config['global']
    if gl_policy['policy_type'] == "co2 cap":
        co2_cap = gl_policy['co2_share']*gl_policy['co2_baseline']
        print("Setting global CO2 cap to ",co2_cap)
        n.global_constraints.at["CO2Limit","constant"] = co2_cap
    elif gl_policy['policy_type'] == "co2 price":
        n.global_constraints.drop("CO2Limit",
                                  inplace=True)
        print("Setting CO2 price to",gl_policy['co2_price'])
        for carrier in ["coal", "oil", "gas"]:
            n.generators.at[f"EU {carrier}","marginal_cost"] += gl_policy['co2_price']*costs.at[carrier, 'CO2 intensity']

    #local C&I properties
    ci = snakemake.config['ci']
    name = ci['name']
    node = ci['node']
    load = ci['load']

    n.add("Bus",
          name)

    n.add("Link",
          name + " export",
          bus0=name,
          bus1=node,
          marginal_cost=1, #to stop link being used to destroy energy
          p_nom=1e6)

    n.add("Link",
          name + " import",
          bus0=node,
          bus1=name,
          marginal_cost=1, #to stop link being used to destroy energy
          p_nom=1e6)

    #baseload clean energy generator
    if "green hydrogen OCGT" in ci["clean_techs"]:
        n.add("Generator",
              name + " green hydrogen OCGT",
              carrier="green hydrogen OCGT",
              bus=name,
              p_nom_extendable=True,
              capital_cost=costs.at['OCGT', 'fixed'],
              marginal_cost=costs.at['OCGT', 'VOM']  + snakemake.config['costs']['price_green_hydrogen']/0.033/costs.at['OCGT', 'efficiency']) #hydrogen cost in EUR/kg, 0.033 MWhLHV/kg

    #RES generator
    for carrier in ["onwind","solar"]:
        if carrier not in ci["clean_techs"]:
            continue
        gen_template = node + " " + carrier
        n.add("Generator",
              f"{name} {carrier}",
              carrier=carrier,
              bus=name,
              p_nom_extendable=True,
              p_max_pu=n.generators_t.p_max_pu[gen_template],
              capital_cost=n.generators.at[gen_template,"capital_cost"],
              marginal_cost=n.generators.at[gen_template,"marginal_cost"])

    n.add("Load",
          name + " load",
          carrier=name,
          bus=name,
          p_set=pd.Series(load,index=n.snapshots))

    if "battery" in ci["storage_techs"]:
        n.add("Bus",
              f"{name} battery",
              carrier="battery"
              )

        n.add("Store",
              f"{name} battery",
              bus=f"{name} battery",
              e_cyclic=True,
              e_nom_extendable=True,
              carrier="battery",
              capital_cost=n.stores.at[f"{node} battery", "capital_cost"],
              lifetime=n.stores.at[f"{node} battery", "lifetime"]
              )

        n.add("Link",
              f"{name} battery charger",
              bus0=name,
              bus1=f"{name} battery",
              carrier="battery charger",
              efficiency=n.links.at[f"{node} battery charger", "efficiency"],
              capital_cost=n.links.at[f"{node} battery charger", "capital_cost"],
              p_nom_extendable=True,
              lifetime=n.links.at[f"{node} battery charger", "lifetime"]
              )

        n.add("Link",
              f"{name} battery discharger",
              bus0=f"{name} battery",
              bus1=name,
              carrier="battery discharger",
              efficiency=n.links.at[f"{node} battery discharger", "efficiency"],
              marginal_cost=n.links.at[f"{node} battery discharger", "marginal_cost"],
              p_nom_extendable=True,
              lifetime=n.links.at[f"{node} battery discharger", "lifetime"]
              )

    if "hydrogen" in ci["storage_techs"]:
        n.add("Bus",
              f"{name} H2",
              carrier="H2"
              )

        n.add("Store",
              f"{name} H2 Store",
              bus=f"{name} H2",
              e_cyclic=True,
              e_nom_extendable=True,
              carrier="H2 Store",
              capital_cost=n.stores.at[f"{node} H2 Store", "capital_cost"],
              lifetime=n.stores.at[f"{node} H2 Store", "lifetime"]
              )

        n.add("Link",
              f"{name} H2 Electrolysis",
              bus0=name,
              bus1=f"{name} H2",
              carrier="H2 Electrolysis",
              efficiency=n.links.at[f"{node} H2 Electrolysis", "efficiency"],
              capital_cost=n.links.at[f"{node} H2 Electrolysis", "capital_cost"],
              p_nom_extendable=True,
              lifetime=n.links.at[f"{node} H2 Electrolysis", "lifetime"]
              )

        n.add("Link",
              f"{name} H2 Fuel Cell",
              bus0=f"{name} H2",
              bus1=name,
              carrier="H2 Fuel Cell",
              efficiency=n.links.at[f"{node} H2 Fuel Cell", "efficiency"],
              capital_cost=n.links.at[f"{node} H2 Fuel Cell", "capital_cost"],
              p_nom_extendable=True,
              lifetime=n.links.at[f"{node} H2 Fuel Cell", "lifetime"]
              )


def calculate_grid_cfe(n):

    name = snakemake.config['ci']['name']

    grid_buses = n.buses.index[~n.buses.index.str.contains(name)]

    grid_generators = n.generators.index[n.generators.bus.isin(grid_buses)]

    grid_clean_techs = snakemake.config['global']['grid_clean_techs']

    grid_loads = n.loads.index[n.loads.bus.isin(grid_buses)]

    grid_cfe = n.generators_t.p[grid_generators].groupby(n.generators.carrier,axis=1).sum()[grid_clean_techs].sum(axis=1)/n.loads_t.p[grid_loads].sum(axis=1)

    print("Grid CFE has following stats:")
    print(grid_cfe.describe())

    return grid_cfe


def solve_network(n, policy, penetration):

    ci = snakemake.config['ci']
    name = ci['name']

    if policy == "res":
        n_iterations = 1
        res_gens = [name + " " + g for g in ci['res_techs']]
    elif policy == "cfe":
        n_iterations = snakemake.config['solving']['options']['n_iterations']
        clean_gens = [name + " " + g for g in ci['clean_techs']]
        storage_dischargers = [name + " " + g for g in ci['storage_dischargers']]
        storage_chargers = [name + " " + g for g in ci['storage_chargers']]


    def cfe_constraints(n):

        weightings = pd.DataFrame(np.outer(n.snapshot_weightings["generators"],[1.]*len(clean_gens)),
                                  index = n.snapshots,
                                  columns = clean_gens)
        gen_sum = join_exprs(linexpr((weightings,get_var(n, "Generator", "p")[clean_gens]))) # single line sum

        weightings = pd.DataFrame(np.outer(n.snapshot_weightings["generators"],n.links.loc[storage_dischargers,"efficiency"]),
                                  index = n.snapshots,
                                  columns = storage_dischargers)
        discharge_sum = join_exprs(linexpr((weightings, get_var(n, "Link", "p")[storage_dischargers])))

        weightings = pd.DataFrame(np.outer(n.snapshot_weightings["generators"],[1.]*len(storage_chargers)),
                                  index = n.snapshots,
                                  columns = storage_chargers)
        charge_sum = join_exprs(linexpr((-weightings, get_var(n, "Link", "p")[storage_chargers])))

        gexport = get_var(n, "Link", "p")[name + " export"] # a series
        gimport = get_var(n, "Link", "p")[name + " import"] # a series
        grid_sum = join_exprs(linexpr((-n.snapshot_weightings["generators"],gexport),
                                      (n.links.at[name + " import","efficiency"]*grid_cfe*n.snapshot_weightings["generators"],gimport))) # single line sum

        lhs = gen_sum + '\n' + discharge_sum  + '\n' + charge_sum + '\n' + grid_sum
        total_load = (n.loads_t.p_set[name + " load"]*n.snapshot_weightings["generators"]).sum() # number
        con = define_constraints(n, lhs, '>=', penetration*total_load, 'CFEconstraints','CFEtarget')


    def res_constraints(n):
        
        weightings = pd.DataFrame(np.outer(n.snapshot_weightings["generators"],[1.]*len(res_gens)),
                                  index = n.snapshots,
                                  columns = res_gens)
        lhs = join_exprs(linexpr((weightings,get_var(n, "Generator", "p")[res_gens]))) # single line sum
        total_load = (n.loads_t.p_set[name + " load"]*n.snapshot_weightings["generators"]).sum() # number
        con = define_constraints(n, lhs, '>=', penetration*total_load, 'RESconstraints','REStarget')

    def add_battery_constraints(n):

        chargers_b = n.links.carrier.str.contains("battery charger")
        chargers = n.links.index[chargers_b & n.links.p_nom_extendable]
        dischargers = chargers.str.replace("charger", "discharger")

        if chargers.empty or ('Link', 'p_nom') not in n.variables.index:
            return

        link_p_nom = get_var(n, "Link", "p_nom")

        lhs = linexpr((1,link_p_nom[chargers]),
                      (-n.links.loc[dischargers, "efficiency"].values,
                       link_p_nom[dischargers].values))

        define_constraints(n, lhs, "=", 0, 'Link', 'charger_ratio')


    def extra_functionality(n, snapshots):

        add_battery_constraints(n)

        if policy == "cfe":
            print("setting CFE target of",penetration)
            cfe_constraints(n)
        elif policy == "res":
            print("setting annual RES target of",penetration)
            res_constraints(n)
        else:
            print("no target set")
            sys.exit()

    n.consistency_check()

    formulation = snakemake.config['solving']['options']['formulation']
    solver_options = snakemake.config['solving']['solver']
    solver_name = solver_options.pop('name')

    grid_cfe_df = pd.DataFrame(0.,index=n.snapshots,columns=[f"iteration {i}" for i in range(n_iterations+1)])

    for i in range(n_iterations):

        grid_cfe = grid_cfe_df[f"iteration {i}"]

        n.lopf(pyomo=False,
               extra_functionality=extra_functionality,
               formulation=formulation,
               solver_name=solver_name,
               solver_options=solver_options,
               solver_logfile=snakemake.log.solver)

        grid_cfe_df[f"iteration {i+1}"] = calculate_grid_cfe(n)

    grid_cfe_df.to_csv(snakemake.output.grid_cfe)


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('solve_network', policy="cfe80")

    # When running via snakemake
    n = pypsa.Network(snakemake.input.network,
                      override_component_attrs=override_component_attrs)

    policy = snakemake.wildcards.policy[:3]
    penetration = float(snakemake.wildcards.policy[3:])/100

    print(f"solving network for policy {policy} and penetration {penetration}")


    Nyears = 1 # years in simulation
    costs = prepare_costs(snakemake.input.costs,
                          snakemake.config['costs']['USD2013_to_EUR2013'],
                          snakemake.config['costs']['discountrate'],
                          Nyears,
                          snakemake.config['costs']['lifetime'])


    with memory_logger(filename=getattr(snakemake.log, 'memory', None), interval=30.) as mem:

        strip_network(n)

        add_ci(n)

        solve_network(n, policy, penetration)

        n.export_to_netcdf(snakemake.output.network)

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
