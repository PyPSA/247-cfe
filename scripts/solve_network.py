
import pypsa, numpy as np, pandas as pd
from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints
from vresutils.costdata import annuity

import logging
logger = logging.getLogger(__name__)
import sys

# Suppress logging of the slack bus choices
pypsa.pf.logger.setLevel(logging.WARNING)

from vresutils.benchmark import memory_logger
from _helpers import override_component_attrs


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
        clean_techs = ["onwind", "solar", "allam_ccs", "adv_geothermal"]  #"adv_nuclear", "adv_geothermal"
        storage_techs = ["battery", "hydrogen"]
        storage_chargers = ["battery charger", "H2 Electrolysis"]
        storage_dischargers = ["battery discharger", "H2 Fuel Cell"]

    else:
        print(f"'palette' wildcard must be one of 'p1', 'p2' or 'p3'. Now is {tech_palette}.")
        sys.exit()

    return clean_techs, storage_techs, storage_chargers, storage_dischargers


def geoscope(zone, area):
    '''
    basenodes_to_keep -> geographical scope of the model
    country_nodes -> scope for national RES policy constraint
    node -> zone where C&I load is located
    '''
    d = dict();

    if zone == 'IE':
        d['basenodes_to_keep'] = ["IE5 0", "GB0 0", "GB5 0"]
        d['country_nodes'] = ["IE5 0"]
        d['node'] = "IE5 0"

    elif zone == 'DK':
        d['basenodes_to_keep'] = ["DK1 0", "DK2 0", "SE2 0", "NO2 0", "NL1 0", "DE1 0"]
        d['country_nodes'] = ["DK1 0", "DK2 0"]
        d['node'] = "DK1 0"

    elif zone == 'DE':
        d['basenodes_to_keep'] = ['DE1 0', 'BE1 0', 'NO2 0', 'DK1 0', 'DK2 0', 'SE2 0', 'GB0 0',
                                  'FR1 0', 'LU1 0', 'NL1 0', 'PL1 0', 'AT1 0', 'CH1 0', 'CZ1 0']
        d['country_nodes'] = ['DE1 0']
        d['node'] = 'DE1 0'

    elif zone == 'NL':
        d['basenodes_to_keep'] = ['NL1 0', 'GB0 0', 'DK1 0', 'NO2 0', 'BE1 0', 'DE1 0']
        d['country_nodes'] = ['NL1 0']
        d['node'] = 'NL1 0'

    else:
        print(f"'zone' wildcard must be one of 'IE', 'DK', 'DE', 'NL'. Now is {zone}.")
        sys.exit()

    if area == 'EU':
        d['basenodes_to_keep'] = ['AL1 0', 'AT1 0',  'BA1 0',  'BE1 0',  'BG1 0',
            'CH1 0', 'CZ1 0',  'DE1 0',  'DK1 0',  'DK2 0',
            'EE6 0', 'ES1 0',  'ES4 0',  'FI2 0',  'FR1 0',
            'GB0 0', 'GB5 0',  'GR1 0',  'HR1 0',  'HU1 0',
            'IE5 0', 'IT1 0',  'IT3 0',  'LT6 0',  'LU1 0',
            'LV6 0', 'ME1 0',  'MK1 0',  'NL1 0',  'NO2 0',
            'PL1 0', 'PT1 0',  'RO1 0',  'RS1 0',  'SE2 0',
            'SI1 0', 'SK1 0']

    return d


def timescope(zone, year):
    '''
    country_res_target -> value of national RES policy constraint for {year} and {zone}
    coal_phaseout -> countries that implement coal phase-out policy until {year}
    network_file -> input file with pypsa-eur-sec brownfield network for {year}
    costs_projection -> input file with technology costs for {year}
    '''

    d = dict();

    d['country_res_target'] = snakemake.config[f'res_target_{year}'][f'{zone}']
    d['coal_phaseout'] = snakemake.config[f'policy_{year}']

    if year == '2030':
        d['network_file']  = snakemake.input.network2030
        d['costs_projection'] = snakemake.input.costs2030
    elif year == '2025':
        d['network_file']  = snakemake.input.network2025
        d['costs_projection'] = snakemake.input.costs2025
    else:
        print(f"'year' wildcard must be one of '2025', '2030'. Now is {year}.")
        sys.exit()

    return d


def cost_parametrization(n):
    '''
    overwrite default price assumptions for primary energy carriers
    only for virtual generators located in 'EU {carrier}' buses
    '''

    for carrier in ['lignite', 'coal', 'gas']:
        n.generators.loc[n.generators.index.str.contains(f'EU {carrier}'), 'marginal_cost'] = snakemake.config['costs'][f'price_{carrier}']
    #n.generators[n.generators.index.str.contains('EU')].T


def prepare_costs(cost_file, USD_to_EUR, discount_rate, Nyears, lifetime, year):

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
    data_nuc = pd.Series({'CO2 intensity': 0,
            'FOM': costs.loc['nuclear']['FOM'],
            'VOM': costs.loc['nuclear']['VOM'],
            'discount rate': discount_rate,
            'efficiency': 0.36,
            'fuel': costs.loc['nuclear']['fuel'],
            'investment': snakemake.config['costs']['adv_nuclear_overnight'] * 1e3 * snakemake.config['costs']['USD2021_to_EUR2021'],
            'lifetime': 40.0
            }, name="adv_nuclear")

    if year == '2025':
        adv_geo_overnight = snakemake.config['costs']['adv_geo_overnight_2025']
        allam_ccs_overnight = snakemake.config['costs']['allam_ccs_overnight_2025']
    elif year == '2030':
        adv_geo_overnight = snakemake.config['costs']['adv_geo_overnight_2030']
        allam_ccs_overnight = snakemake.config['costs']['allam_ccs_overnight_2030']

    # Advanced geothermal
    data_geo = pd.Series({'CO2 intensity': 0,
            'FOM': 0,
            'VOM': 0,
            'discount rate': discount_rate,
            'efficiency': 1,
            'fuel': 0,
            'investment':  adv_geo_overnight * 1e3 * 1,
            'lifetime': 30.0
            }, name="adv_geothermal")

    # Allam cycle ccs
    data_allam = pd.Series({'CO2 intensity': 0,
            'FOM': 0, #%/year
            'FOM-abs' : 33000, #$/MW-yr
            'VOM': 3.2, #EUR/MWh
            'co2_seq': 40, #$/ton
            'discount rate': discount_rate,
            'efficiency': 0.54,
            'fuel': snakemake.config['costs']['price_gas'],
            'investment':  allam_ccs_overnight * 1e3 * 1,
            'lifetime': 30.0
            }, name="allam_ccs")

    costs = costs.append(data_nuc, ignore_index=False)
    costs = costs.append(data_geo, ignore_index=False)
    costs = costs.append(data_allam, ignore_index=False)

    annuity_factor = lambda v: annuity(v["lifetime"], v["discount rate"]) + v["FOM"] / 100
    costs["fixed"] = [annuity_factor(v) * v["investment"] * Nyears for i, v in costs.iterrows()]

    return costs


def strip_network(n):
    nodes_to_keep = geoscope(zone, area)['basenodes_to_keep']
    new_nodes = []

    for b in nodes_to_keep:
        for s in snakemake.config['node_suffixes_to_keep']:
            new_nodes.append(b + " " + s)

    nodes_to_keep.extend(new_nodes)
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


def limit_resexp(n, year):
    '''
    limit expansion of renewable technologies per zone and carrier type
    as a ratio of max increase to 2021 capacity fleet
    (additional to zonal place availability constraint)
    '''
    name = snakemake.config['ci']['name']
    ratio = snakemake.config['global'][f'limit_res_exp_{year}']
    node = geoscope(zone, area)['node']

    res = n.generators[(~n.generators.index.str.contains('EU')) & (~n.generators.index.str.contains(name)) & (~n.generators.index.str.contains(f'{node}'))]
    fleet = res[res.p_nom_extendable==False]

    #fleet.groupby([fleet.carrier, fleet.bus]).p_nom.sum()
    off_c = fleet[fleet.index.str.contains('offwind')].carrier + '-' + \
            fleet[fleet.index.str.contains('offwind')].index.str.extract('(ac)|(dc)').fillna('').sum(axis=1).values

    fleet["carrier_s"] = off_c.reindex(fleet.index).fillna(fleet.carrier)

    for bus in fleet.bus.unique():
        for carrier in ['solar', 'onwind', 'offwind-ac', 'offwind-dc']:
            p_nom_fleet = 0
            p_nom_fleet = fleet.loc[(fleet.bus == bus) & (fleet.carrier_s == carrier), "p_nom"].sum()
            #print(f'bus: {bus}, carrier: {carrier}' ,p_nom_fleet)
            n.generators.loc[(n.generators.p_nom_extendable==True) & (n.generators.bus == bus) & \
                             (n.generators.carrier == carrier), "p_nom_max"] = ratio * p_nom_fleet


def nuclear_policy(n):
    '''
    remove nuclear PPs fleet for countries with nuclear ban policy
    '''
    for node in snakemake.config['nodes_with_nucsban']:
            n.links.loc[n.links['bus1'].str.contains(f'{node}') & (n.links.index.str.contains('nuclear')), 'p_nom'] = 0


def coal_policy(n):
    '''
    remove coal PPs fleet for countries with coal phase-out policy for {year}
    '''

    countries = timescope(zone, year)['coal_phaseout']

    for country in countries:
        n.links.loc[n.links['bus1'].str.contains(f'{country}') & (n.links.index.str.contains('coal')), 'p_nom'] = 0
        n.links.loc[n.links['bus1'].str.contains(f'{country}') & (n.links.index.str.contains('lignite')), 'p_nom'] = 0


def biomass_potential(n):
    '''
    remove solid biomass demand for industrial processes from overall biomass potential
    '''
    n.stores.loc[n.stores.index=='EU solid biomass', 'e_nom'] *= 0.45
    n.stores.loc[n.stores.index=='EU solid biomass', 'e_initial'] *= 0.45


def add_ci(n, policy, participation, year):
    """Add C&I at its own node"""

    #first deal with global policy environment
    gl_policy = snakemake.config['global']

    if gl_policy['policy_type'] == "co2 cap":
        co2_cap = gl_policy['co2_share']*gl_policy['co2_baseline']
        print(f"Setting global CO2 cap to {co2_cap}")
        n.global_constraints.at["CO2Limit","constant"] = co2_cap

    elif gl_policy['policy_type'] == "co2 price":
        n.global_constraints.drop("CO2Limit", inplace=True)
        co2_price = gl_policy[f'co2_price_{year}']
        print(f"Setting CO2 price to {co2_price}")
        for carrier in ["coal", "oil", "gas", "lignite"]:
            n.generators.at[f"EU {carrier}","marginal_cost"] += co2_price*costs.at[carrier, 'CO2 intensity']


    #local C&I properties
    name = snakemake.config['ci']['name']
    #load = snakemake.config['ci']['load']
    ci_load = snakemake.config['ci_load'][f'{zone}']
    load = ci_load * float(participation)/100  #C&I baseload MW
    node = geoscope(zone, area)['node']


    if policy == "ref":
        return None

    #tech_palette options
    clean_techs, storage_techs, storage_chargers, storage_dischargers = palette(tech_palette)

    n.add("Bus",
          name)

    n.add("Bus",
          f"{name} H2",
          carrier="H2"
          )

    n.add("Link",
          f"{name} H2 Electrolysis",
          bus0=name,
          bus1=f"{name} H2",
          carrier="H2 Electrolysis",
          efficiency=n.links.at[f"{node} H2 Electrolysis"+"-{}".format(year), "efficiency"],
          capital_cost=n.links.at[f"{node} H2 Electrolysis"+"-{}".format(year), "capital_cost"],
          p_nom_extendable=True,
          lifetime=n.links.at[f"{node} H2 Electrolysis"+"-{}".format(year), "lifetime"]
          )

    n.add("Load",
          f"{name} H2",
          carrier=f"{name} H2",
          bus=f"{name} H2",
          p_set=pd.Series(load,index=n.snapshots))

    #cost-less storage to indivcate flexible demand
    n.add("Store",
          f"{name} H2 Store",
          bus=f"{name} H2",
          e_cyclic=True,
          e_nom_extendable=True,
          carrier="H2 Store",
          capital_cost=0.001,#costs.at["hydrogen storage underground","fixed"],
          lifetime=costs.at["hydrogen storage underground","lifetime"],
          )

    if policy in ["res","cfe","exl","grf"]:
        n.add("Link",
              name + " export",
              bus0=name,
              bus1=node,
              marginal_cost=0.1, #large enough to avoid optimization artifacts, small enough not to influence PPA portfolio
              p_nom=1e6)

    if policy in ["grd","res","grf"]:
        n.add("Link",
              name + " import",
              bus0=node,
              bus1=name,
              marginal_cost=0.001, #large enough to avoid optimization artifacts, small enough not to influence PPA portfolio
              p_nom=1e6)

    if policy == "grd":
        return None


    #baseload clean energy generator
    if "green hydrogen OCGT" in clean_techs:
        n.add("Generator",
              name + " green hydrogen OCGT",
              carrier="green hydrogen OCGT",
              bus=name,
              p_nom_extendable = True,
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

    #baseload clean energy generator
    if "allam_ccs" in clean_techs:
        n.add("Generator",
              f"{name} allam_ccs",
              bus = name,
              carrier = 'gas',
              capital_cost = costs.loc['allam_ccs']['fixed'] + costs.loc['allam_ccs']['FOM-abs'],
              marginal_cost = costs.loc['allam_ccs']['VOM'] + \
                              costs.loc['allam_ccs']['fuel']/costs.loc['allam_ccs']['efficiency'] + \
                              costs.loc['allam_ccs']['co2_seq']*costs.at['gas', 'CO2 intensity']/costs.loc['allam_ccs']['efficiency'],
              p_nom_extendable = True,
              lifetime = costs.loc['allam_ccs']['lifetime'],
              efficiency = costs.loc['allam_ccs']['efficiency'],
              )

    #baseload clean energy generator
    if "adv_geothermal" in clean_techs:
        n.add("Generator",
              f"{name} adv_geothermal",
              bus = name,
              #carrier = '',
              capital_cost = costs.loc['adv_geothermal']['fixed'],
              marginal_cost= costs.loc['adv_geothermal']['VOM'],
              p_nom_extendable = True,
              lifetime = costs.loc['adv_geothermal']['lifetime']
              )

    #RES generator
    for carrier in ["onwind","solar"]:
        if carrier not in clean_techs:
            continue
        gen_template = node+" "+carrier+"-{}".format(year)
        n.add("Generator",
              f"{name} {carrier}",
              carrier=carrier,
              bus=name,
              p_nom_extendable=True,
              p_max_pu=n.generators_t.p_max_pu[gen_template],
              capital_cost=n.generators.at[gen_template,"capital_cost"],
              marginal_cost=n.generators.at[gen_template,"marginal_cost"])


    if "battery" in storage_techs:
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
              capital_cost=n.stores.at[f"{node} battery"+"-{}".format(year), "capital_cost"],
              lifetime=n.stores.at[f"{node} battery"+"-{}".format(year), "lifetime"]
              )

        n.add("Link",
              f"{name} battery charger",
              bus0=name,
              bus1=f"{name} battery",
              carrier="battery charger",
              efficiency=n.links.at[f"{node} battery charger"+"-{}".format(year), "efficiency"],
              capital_cost=n.links.at[f"{node} battery charger"+"-{}".format(year), "capital_cost"],
              p_nom_extendable=True,
              lifetime=n.links.at[f"{node} battery charger"+"-{}".format(year), "lifetime"]
              )

        n.add("Link",
              f"{name} battery discharger",
              bus0=f"{name} battery",
              bus1=name,
              carrier="battery discharger",
              efficiency=n.links.at[f"{node} battery discharger"+"-{}".format(year), "efficiency"],
              marginal_cost=n.links.at[f"{node} battery discharger"+"-{}".format(year), "marginal_cost"],
              p_nom_extendable=True,
              lifetime=n.links.at[f"{node} battery discharger"+"-{}".format(year), "lifetime"]
              )


def solve_network(n, policy, penetration, tech_palette):

    ci = snakemake.config['ci']
    name = ci['name']

    clean_techs, storage_techs, storage_chargers, storage_dischargers = palette(tech_palette)

    def res_constraints(n):

        res_gens = [name + " " + g for g in ci['res_techs']]

        weightings = pd.DataFrame(np.outer(n.snapshot_weightings["generators"],[1.]*len(res_gens)),
                                  index = n.snapshots,
                                  columns = res_gens)
        res = join_exprs(linexpr((weightings,get_var(n, "Generator", "p")[res_gens]))) # single line sum

        electrolysis = get_var(n, "Link", "p")[f"{name} H2 Electrolysis"]

        load = join_exprs(linexpr((-n.snapshot_weightings["generators"],electrolysis)))

        lhs = res + "\n" + load

        con = define_constraints(n, lhs, '=', 0., 'RESconstraints','REStarget')


    def excess_constraints(n):

        res_gens = [name + " " + g for g in ci['res_techs']]

        weightings = pd.DataFrame(np.outer(n.snapshot_weightings["generators"],[1.]*len(res_gens)),
                                  index = n.snapshots,
                                  columns = res_gens)
        res = join_exprs(linexpr((weightings,get_var(n, "Generator", "p")[res_gens]))) # single line sum

        electrolysis = get_var(n, "Link", "p")[f"{name} H2 Electrolysis"]

        allowed_excess = 1.2

        load = join_exprs(linexpr((-allowed_excess*n.snapshot_weightings["generators"],electrolysis)))

        lhs = res + "\n" + load

        con = define_constraints(n, lhs, '<=', 0., 'RESconstraints','REStarget')


    def country_res_constraints(n):

        grid_buses = n.buses.index[n.buses.location.isin(geoscope(zone, area)['country_nodes'])]

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
        target = timescope(zone, year)["country_res_target"]
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

        if policy == "res":
            print("setting annual RES target")
            res_constraints(n)
        elif policy == "exl":
            print("setting excess limit on hourly matching")
            excess_constraints(n)

    n.consistency_check()

    formulation = snakemake.config['solving']['options']['formulation']
    solver_options = snakemake.config['solving']['solver']
    solver_name = solver_options['name']

    n.lopf(pyomo=False,
           extra_functionality=extra_functionality,
           formulation=formulation,
           solver_name=solver_name,
           solver_options=solver_options,
           solver_logfile=snakemake.log.solver)


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('solve_network',
                                policy="cfe80", palette='p3', zone='DE', year='2030', participation='25')

    logging.basicConfig(filename=snakemake.log.python,
                    level=snakemake.config['logging_level'])

    #Wildcards & Settings
    policy = snakemake.wildcards.policy[:3]
    print(f"solving network for policy: {policy}")

    penetration = snakemake.wildcards.policy[3:]
    if penetration != "":
        print(f"warning, penetration {penetration} not supported, only 100%")

    tech_palette = snakemake.wildcards.palette
    print(f"solving network for palette: {tech_palette}")

    zone = snakemake.wildcards.zone
    print(f"solving network for bidding zone: {zone}")

    year = snakemake.wildcards.year
    print(f"solving network year: {year}")

    area = snakemake.config['area']
    print(f"solving with geoscope: {area}")

    participation = snakemake.wildcards.participation
    print(f"solving with participation: {participation}")

    # When running via snakemake
    n = pypsa.Network(timescope(zone, year)['network_file'],
                      override_component_attrs=override_component_attrs())

    Nyears = 1 # years in simulation
    costs = prepare_costs(timescope(zone, year)['costs_projection'],
                          snakemake.config['costs']['USD2013_to_EUR2013'],
                          snakemake.config['costs']['discountrate'],
                          Nyears,
                          snakemake.config['costs']['lifetime'],
                          year)


    with memory_logger(filename=getattr(snakemake.log, 'memory', None), interval=30.) as mem:

        strip_network(n)

        shutdown_lineexp(n)
        #limit_resexp(n,year)
        nuclear_policy(n)
        coal_policy(n)
        biomass_potential(n)
        cost_parametrization(n)

        add_ci(n, policy, participation, year)

        solve_network(n, policy, penetration, tech_palette)

        n.export_to_netcdf(snakemake.output.network)

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
