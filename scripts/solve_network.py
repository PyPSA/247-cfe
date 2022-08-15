
import pypsa, numpy as np, pandas as pd
from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints
from vresutils.costdata import annuity

import logging
logger = logging.getLogger(__name__)
import sys

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


def palette(tech_palette):
    '''
    Define technology palette at CI node based on wildcard value
    '''

    if tech_palette == 'p1':
        clean_techs = ["onwind", "solar"]
        storage_techs = ["battery"]
        storage_chargers = ["battery charger"]
        storage_dischargers = ["battery discharger"]
    elif tech_palette == 'p2':
        clean_techs = ["onwind", "solar"]
        storage_techs = ["battery", "hydrogen"]
        storage_chargers = ["battery charger", "H2 Electrolysis"]
        storage_dischargers = ["battery discharger", "H2 Fuel Cell"]
    elif tech_palette == 'p3':
        clean_techs = ["onwind", "solar", "adv_nuclear"]
        storage_techs = ["battery", "hydrogen"]
        storage_chargers = ["battery charger", "H2 Electrolysis"]
        storage_dischargers = ["battery discharger", "H2 Fuel Cell"]
    else: 
        print(f"'palette' wildcard must be one of 'p1', 'p2' or 'p3'. Now is {tech_palette}.")
        sys.exit()

    return clean_techs, storage_techs, storage_chargers, storage_dischargers


def geoscope(zone):
    '''
    basenodes_to_keep -> geographical scope of the model
    country_nodes -> scope for national RES policy constraint
    country_res_target -> value for national RES policy constraint
    node -> zone where C&I load is located
    '''
    d = dict(); 

    if zone == 'Ireland':
        d['basenodes_to_keep'] = ["IE5 0", "GB0 0", "GB5 0"]
        d['country_nodes'] = ["IE5 0"]
        d['country_res_target'] = 0.8
        d['node'] = "IE5 0" 
    elif zone == 'Denmark':
        d['basenodes_to_keep'] = ["DK1 0", "DK2 0", "SE2 0", "NO2 0", "NL1 0", "DE1 0"]
        d['country_nodes:'] = ["DK1 0", "DK2 0"]
        d['country_res_target'] = 1.2
        d['node'] = "DK1 0"
    else: 
        print(f"'zone' wildcard must be one of 'Ireland', 'Denmark'. Now is {zone}.")
        sys.exit()
    
    return d


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

    # Advanced nuclear
    new_row = pd.Series({'CO2 intensity': 0, 
            'FOM': costs.loc['nuclear']['FOM'], 
            'VOM': costs.loc['nuclear']['VOM'], 
            'discount rate': costs.loc['nuclear']['discount rate'],
            'efficiency': 0.37,             #higher than nuclear, see data
            'fuel': costs.loc['nuclear']['fuel'],
            'investment': snakemake.config['costs']['adv_nuclear_overnight'] * 1e3 * USD_to_EUR,
            'lifetime': 40.0
            }, name="adv_nuclear")
            
    costs = costs.append(new_row, ignore_index=False)

    annuity_factor = lambda v: annuity(v["lifetime"], v["discount rate"]) + v["FOM"] / 100
    costs["fixed"] = [annuity_factor(v) * v["investment"] * Nyears for i, v in costs.iterrows()]

    return costs


def strip_network(n):
    nodes_to_keep = geoscope(zone)['basenodes_to_keep']
    for b in geoscope(zone)['basenodes_to_keep']:
        for s in snakemake.config['node_suffixes_to_keep']:
            nodes_to_keep.append(b + " " + s)

    nodes_to_keep.extend(snakemake.config['additional_nodes'])

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


def shutdown_lineexp(n):
    '''
    remove line expansion option
    '''
    n.lines.s_nom_extendable = False
    n.links.loc[n.links.carrier=='DC', 'p_nom_extendable'] = False


def nuclear_policy(n):
    '''
    remove demand for solid biomass from industrial processes from overall biomass potential
    '''
    for node in snakemake.config['nodes_with_nucsban']:
        if node in geoscope(zone)['basenodes_to_keep']:
            nn.links.loc[(n.links['bus1'] == f'{node}') & (n.links.index.str.contains('nuclear')), 'p_nom'] = 0


def biomass_potential(n):
    '''
    remove solid biomass demand for industrial processes from overall biomass potential
    '''
    n.stores.loc[n.stores.index=='EU solid biomass', 'e_nom'] *= 0.45
    n.stores.loc[n.stores.index=='EU solid biomass', 'e_initial'] *= 0.45


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
    name = snakemake.config['ci']['name']
    load = snakemake.config['ci']['load']
    node = geoscope(zone)['node']

    #tech_palette options
    clean_techs = palette(tech_palette)[0]
    storage_techs = palette(tech_palette)[1]


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
    if "green hydrogen OCGT" in clean_techs:
        n.add("Generator",
              name + " green hydrogen OCGT",
              carrier="green hydrogen OCGT",
              bus=name,
              p_nom_extendable=True,
              capital_cost=costs.at['OCGT', 'fixed'],
              marginal_cost=costs.at['OCGT', 'VOM']  + snakemake.config['costs']['price_green_hydrogen']/0.033/costs.at['OCGT', 'efficiency']) #hydrogen cost in EUR/kg, 0.033 MWhLHV/kg

    #baseload clean energy generator
    if "adv_nuclear" in clean_techs:      
        n.add("Generator",
              f"{name} adv_nuclear",
              bus = name,
              carrier = 'nuclear',
              capital_cost = costs.loc['adv_nuclear']['fixed'],
              marginal_cost= costs.loc['adv_nuclear']['VOM']  + costs.loc['adv_nuclear']['fuel']/costs.loc['adv_nuclear']['efficiency'],
              p_nom_extendable = True,
              lifetime = costs.loc['adv_nuclear']['lifetime']
              )

    #RES generator
    for carrier in ["onwind","solar"]:
        if carrier not in clean_techs:
            continue
        gen_template = node + " " + carrier+"-2030" #"-2030" fix for a pypsa-eur-sec brownfield 2030 network
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

    if "battery" in storage_techs and policy == "cfe":
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
              capital_cost=n.stores.at[f"{node} battery"+"-2030", "capital_cost"],
              lifetime=n.stores.at[f"{node} battery"+"-2030", "lifetime"]
              )

        n.add("Link",
              f"{name} battery charger",
              bus0=name,
              bus1=f"{name} battery",
              carrier="battery charger",
              efficiency=n.links.at[f"{node} battery charger"+"-2030", "efficiency"],
              capital_cost=n.links.at[f"{node} battery charger"+"-2030", "capital_cost"],
              p_nom_extendable=True, 
              lifetime=n.links.at[f"{node} battery charger"+"-2030", "lifetime"] 
              )

        n.add("Link",
              f"{name} battery discharger",
              bus0=f"{name} battery",
              bus1=name,
              carrier="battery discharger",
              efficiency=n.links.at[f"{node} battery discharger"+"-2030", "efficiency"],
              marginal_cost=n.links.at[f"{node} battery discharger"+"-2030", "marginal_cost"],
              p_nom_extendable=True,
              lifetime=n.links.at[f"{node} battery discharger"+"-2030", "lifetime"]
              )

    if "hydrogen" in storage_techs and policy == "cfe":
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
              capital_cost=costs.at["hydrogen storage underground","fixed"],
              lifetime=costs.at["hydrogen storage underground","lifetime"],
              #capital_cost=n.stores.filter(like=(f'{node}'+' '+'H2 Store'), axis=0)["capital_cost"],
              #lifetime=n.stores.filter(like=(f'{node}'+' '+'H2 Store'), axis=0)["lifetime"]
              )        

        n.add("Link",
              f"{name} H2 Electrolysis",
              bus0=name,
              bus1=f"{name} H2",
              carrier="H2 Electrolysis",
              efficiency=n.links.at[f"{node} H2 Electrolysis"+"-2030", "efficiency"],
              capital_cost=n.links.at[f"{node} H2 Electrolysis"+"-2030", "capital_cost"],
              p_nom_extendable=True,
              lifetime=n.links.at[f"{node} H2 Electrolysis"+"-2030", "lifetime"] 
              )

        n.add("Link",
              f"{name} H2 Fuel Cell",
              bus0=f"{name} H2",
              bus1=name,
              carrier="H2 Fuel Cell",
              efficiency=n.links.at[f"{node} H2 Fuel Cell"+"-2030", "efficiency"],
              capital_cost=n.links.at[f"{node} H2 Fuel Cell"+"-2030", "capital_cost"],
              p_nom_extendable=True,
              lifetime=n.links.at[f"{node} H2 Fuel Cell"+"-2030", "lifetime"]
              )


def calculate_grid_cfe(n):

    name = snakemake.config['ci']['name']
    country = geoscope(zone)['node'] 
    grid_buses = n.buses.index[~n.buses.index.str.contains(name) & ~n.buses.index.str.contains(country)]
    country_buses = n.buses.index[n.buses.index.str.contains(country)]

    #grid_buses = n.buses.index[n.buses.location.isin(snakemake.config['nodes_for_cfe'])]

    clean_techs = pd.Index(snakemake.config['global']['grid_clean_techs'])
    emitters = pd.Index(snakemake.config['global']['emitters'])

    clean_grid_generators = n.generators.index[n.generators.bus.isin(grid_buses) & n.generators.carrier.isin(clean_techs)]
    clean_grid_links = n.links.index[n.links.bus1.isin(grid_buses) & n.links.carrier.isin(clean_techs)]
    clean_grid_storage_units = n.storage_units.index[n.storage_units.bus.isin(grid_buses) & n.storage_units.carrier.isin(clean_techs)]
    dirty_grid_links = n.links.index[n.links.bus1.isin(grid_buses) & n.links.carrier.isin(emitters)]

    logger.info(f"clean grid generators are {clean_grid_generators}")
    logger.info(f"clean grid links are {clean_grid_links}")
    logger.info(f"clean grid storage units are {clean_grid_storage_units}")
    logger.info(f"dirty grid links are {dirty_grid_links}")

    clean_country_generators = n.generators.index[n.generators.bus.isin(country_buses) & n.generators.carrier.isin(clean_techs)]
    clean_country_links = n.links.index[n.links.bus1.isin(country_buses) & n.links.carrier.isin(clean_techs)]
    clean_country_storage_units = n.storage_units.index[n.storage_units.bus.isin(country_buses) & n.storage_units.carrier.isin(clean_techs)]
    dirty_country_links = n.links.index[n.links.bus1.isin(country_buses) & n.links.carrier.isin(emitters)]

    logger.info(f"clean country generators are {clean_country_generators}")
    logger.info(f"clean country links are {clean_country_links}")
    logger.info(f"clean country storage units are {clean_country_storage_units}")
    logger.info(f"dirty country links are {dirty_country_links}")

    #grid_loads = n.loads.index[n.loads.bus.isin(grid_buses)]
    country_loads = n.loads.index[n.loads.bus.isin(country_buses)]

    clean_grid_gens = n.generators_t.p[clean_grid_generators].sum(axis=1)
    clean_grid_ls = (- n.links_t.p1[clean_grid_links].sum(axis=1))
    clean_grid_sus = n.storage_units_t.p[clean_grid_storage_units].sum(axis=1)
    clean_grid_resources = clean_grid_gens + clean_grid_ls + clean_grid_sus
    
    dirty_grid_resources = (- n.links_t.p1[dirty_grid_links].sum(axis=1))

    #grid_cfe =  clean_grid_resources / n.loads_t.p[grid_loads].sum(axis=1)
    #grid_cfe[grid_cfe > 1] = 1.

    import_cfe =  clean_grid_resources / (clean_grid_resources + dirty_grid_resources)

    clean_country_gens = n.generators_t.p[clean_country_generators].sum(axis=1)
    clean_country_ls = (- n.links_t.p1[clean_country_links].sum(axis=1))
    clean_country_sus = n.storage_units_t.p[clean_country_storage_units].sum(axis=1)
    clean_country_resources = clean_country_gens + clean_country_ls + clean_country_sus

    dirty_country_resources = (- n.links_t.p1[dirty_country_links].sum(axis=1))

    ##################
    # Country imports | 
    # NB lines and links are bidirectional, thus we track imports for both subsets 
    # of interconnectors: where [country] node is bus0 and bus1. Subsets are exclusive.
    
    line_imp_subsetA = n.lines_t.p1.loc[:,n.lines.bus0.str.contains(country)].sum(axis=1)
    line_imp_subsetB = n.lines_t.p0.loc[:,n.lines.bus1.str.contains(country)].sum(axis=1)
    line_imp_subsetA[line_imp_subsetA < 0] = 0.
    line_imp_subsetB[line_imp_subsetB < 0] = 0.

    links_imp_subsetA = n.links_t.p1.loc[:,n.links.bus0.str.contains(country) & 
                        (n.links.carrier == "DC") & ~(n.links.index.str.contains(name))].sum(axis=1)
    links_imp_subsetB = n.links_t.p0.loc[:,n.links.bus1.str.contains(country) & 
                        (n.links.carrier == "DC") & ~(n.links.index.str.contains(name))].sum(axis=1)
    links_imp_subsetA[links_imp_subsetA < 0] = 0.
    links_imp_subsetB[links_imp_subsetB < 0] = 0.

    country_import =   line_imp_subsetA + line_imp_subsetB + links_imp_subsetA + links_imp_subsetB

    grid_supply_cfe = (clean_country_resources + country_import * import_cfe) / \
                    (clean_country_resources + dirty_country_resources + country_import)


    print("Grid_supply_CFE has following stats:")
    print(grid_supply_cfe.describe())

    return grid_supply_cfe


def solve_network(n, policy, penetration, tech_palette):
    
    ci = snakemake.config['ci']
    name = ci['name']

    clean_techs = palette(tech_palette)[0]
    storage_techs = palette(tech_palette)[1]
    storage_chargers = palette(tech_palette)[2]
    storage_dischargers = palette(tech_palette)[3]

    if policy == "res":
        n_iterations = 2
        res_gens = [name + " " + g for g in ci['res_techs']]
    elif policy == "cfe":
        n_iterations = snakemake.config['solving']['options']['n_iterations']
        clean_gens = [name + " " + g for g in clean_techs]
        storage_dischargers = [name + " " + g for g in storage_dischargers]
        storage_chargers = [name + " " + g for g in storage_chargers]

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
                                      (n.links.at[name + " import","efficiency"]*grid_supply_cfe*n.snapshot_weightings["generators"],gimport))) # single line sum

        lhs = gen_sum + '\n' + discharge_sum  + '\n' + charge_sum + '\n' + grid_sum
        total_load = (n.loads_t.p_set[name + " load"]*n.snapshot_weightings["generators"]).sum() # number
        con = define_constraints(n, lhs, '>=', penetration*total_load, 'CFEconstraints','CFEtarget')


    def excess_constraints(n):
        
        gexport = get_var(n, "Link", "p")[name + " export"] # a series
        
        excess = linexpr((n.snapshot_weightings["generators"], gexport)).sum(axis=0)
        lhs = excess

        total_load = (n.loads_t.p_set[name + " load"]*n.snapshot_weightings["generators"]).sum()
        rhs = (penetration - 0.8) * total_load

        con = define_constraints(n, lhs, '<=', rhs, 'Excess_constraint')


    def res_constraints(n):
        
        weightings = pd.DataFrame(np.outer(n.snapshot_weightings["generators"],[1.]*len(res_gens)),
                                  index = n.snapshots,
                                  columns = res_gens)
        lhs = join_exprs(linexpr((weightings,get_var(n, "Generator", "p")[res_gens]))) # single line sum

        total_load = (n.loads_t.p_set[name + " load"]*n.snapshot_weightings["generators"]).sum() # number

        # (lhs '>=' penetration*total_load) ??
        con = define_constraints(n, lhs, '=', penetration*total_load, 'RESconstraints','REStarget')


    def country_res_constraints(n):

        grid_buses = n.buses.index[n.buses.location.isin(geoscope(zone)['country_nodes'])]

        grid_res_techs = snakemake.config['global']['grid_res_techs']

        grid_loads = n.loads.index[n.loads.bus.isin(grid_buses)]

        country_res_gens = n.generators.index[n.generators.bus.isin(grid_buses) & n.generators.carrier.isin(grid_res_techs)]
        country_res_links = n.links.index[n.links.bus1.isin(grid_buses) & n.links.carrier.isin(grid_res_techs)]
        country_res_storage_units = n.storage_units.index[n.storage_units.bus.isin(grid_buses) & n.storage_units.carrier.isin(grid_res_techs)]

        #res_gens = n.generators_t.p[country_res_gens].sum(axis=1)
        #res_links = (- n.links_t.p1[country_res_links].sum(axis=1))
        #res_sus = n.storage_units_t.p[country_res_storage_units].sum(axis=1)
        
        weigt_gens = pd.DataFrame(np.outer(n.snapshot_weightings["generators"],[1.]*len(country_res_gens)),
                                  index = n.snapshots,
                                  columns = country_res_gens)
        weigt_links = pd.DataFrame(np.outer(n.snapshot_weightings["generators"],[1.]*len(country_res_links)),
                                  index = n.snapshots,
                                  columns = country_res_links)
        weigt_sus= pd.DataFrame(np.outer(n.snapshot_weightings["generators"],[1.]*len(country_res_storage_units)),
                                  index = n.snapshots,
                                  columns = country_res_storage_units)

        gens = linexpr((weigt_gens, get_var(n, "Generator", "p")[country_res_gens]))
        links = linexpr((weigt_links*n.links.loc[country_res_links, "efficiency"].values, get_var(n, "Link", "p")[country_res_links]))
        sus = linexpr((weigt_sus, get_var(n, "StorageUnit", "p_dispatch")[country_res_storage_units]))
        lhs_temp = pd.concat([gens, links, sus], axis=1)

        lhs = join_exprs(lhs_temp)
        target = geoscope(zone)["country_res_target"]
        total_load = (n.loads_t.p_set[grid_loads].sum(axis=1)*n.snapshot_weightings["generators"]).sum() # number

        logger.info(f"country RES constraints for {country_res_gens} and total load {total_load}")

        con = define_constraints(n, lhs, '=', target*total_load, 'countryRESconstraints','countryREStarget')


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

        country_res_constraints(n)

        if policy == "cfe":
            print("setting CFE target of",penetration)
            cfe_constraints(n)
            excess_constraints(n)
        elif policy == "res":
            print("setting annual RES target of",penetration)
            res_constraints(n)
        else:
            print("no target set")
            sys.exit()

    n.consistency_check()

    formulation = snakemake.config['solving']['options']['formulation']
    solver_options = snakemake.config['solving']['solver']
    solver_name = solver_options['algo']

    grid_cfe_df = pd.DataFrame(0.,index=n.snapshots,columns=[f"iteration {i}" for i in range(n_iterations+1)])

    for i in range(n_iterations):

        grid_supply_cfe = grid_cfe_df[f"iteration {i}"]

        n.lopf(pyomo=False,
               extra_functionality=extra_functionality,
               formulation=formulation,
               solver_name=solver_name,
               solver_options=solver_options,
               solver_logfile=snakemake.log.solver)

        grid_cfe_df[f"iteration {i+1}"] = calculate_grid_cfe(n)
        #print(grid_cfe_df)

    grid_cfe_df.to_csv(snakemake.output.grid_cfe)


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('solve_network', policy="cfe80", palette='p1')

    logging.basicConfig(filename=snakemake.log.python,
                    level=snakemake.config['logging_level'])

    # When running via snakemake
    n = pypsa.Network(snakemake.input.network,
                      override_component_attrs=override_component_attrs)

    #Wildcards & Settings
    policy = snakemake.wildcards.policy[:3]
    penetration = float(snakemake.wildcards.policy[3:])/100
    print(f"solving network for policy {policy} and penetration {penetration}")

    tech_palette = snakemake.wildcards.palette
    print(f"solving network for palette {tech_palette}")

    zone = snakemake.config['scenario']['zone']
    print(f"solving network for bidding zone {zone}")

    # Compute technology costs
    Nyears = 1 # years in simulation
    costs = prepare_costs(snakemake.input.costs,
                          snakemake.config['costs']['USD2013_to_EUR2013'],
                          snakemake.config['costs']['discountrate'],
                          Nyears,
                          snakemake.config['costs']['lifetime'])


    with memory_logger(filename=getattr(snakemake.log, 'memory', None), interval=30.) as mem:

        strip_network(n)
        shutdown_lineexp(n)
        nuclear_policy(n)
        biomass_potential(n)
        add_ci(n)

        solve_network(n, policy, penetration, tech_palette)

        n.export_to_netcdf(snakemake.output.network)

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))

