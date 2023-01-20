
import pypsa, numpy as np, pandas as pd

import logging
logger = logging.getLogger(__name__)
import sys
import os

# Suppress logging of the slack bus choices
pypsa.pf.logger.setLevel(logging.WARNING)
from vresutils.costdata import annuity
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


def load_profile(n, zone, profile_shape):
    '''
    create daily load profile for 24/7 CFE buyers based on config setting
    '''
    shape_base = [1/24]*24

    shape_it =  [0.034,0.034,0.034,0.034,0.034,0.034,
                 0.038,0.042,0.044,0.046,0.047,0.048,
                 0.049,0.049,0.049,0.049,0.049,0.048,
                 0.047,0.043,0.038,0.037,0.036,0.035]

    shape_ind = [0.009,0.009,0.009,0.009,0.009,0.009,
                 0.016,0.031,0.070,0.072,0.073,0.072,
                 0.070,0.052,0.054,0.066,0.070,0.068,
                 0.063,0.035,0.037,0.045,0.045,0.009]

    if profile_shape == 'baseload':
        shape = np.array(shape_base).reshape(-1, 3).mean(axis=1)
    elif profile_shape == 'datacenter':
        shape = np.array(shape_it).reshape(-1, 3).mean(axis=1)
    elif profile_shape == 'industry':
        shape = np.array(shape_ind).reshape(-1, 3).mean(axis=1)   
    else: 
        print(f"'profile_shape' option must be one of 'baseload', 'datacenter' or 'industry'. Now is {profile_shape}.")
        sys.exit()

    ci_load = snakemake.config['ci_load'][f'{zone}']
    load = ci_load * float(participation)/100  #C&I baseload MW

    load_day = load*8 #24h with 3h sampling. to avoid use of hard-coded value
    load_profile_day = pd.Series(shape*load_day*3)
    load_profile_year = pd.concat([load_profile_day]*int(len(n.snapshots)/8))
    
    profile = load_profile_year.set_axis(n.snapshots)
    #baseload = pd.Series(load,index=n.snapshots)

    return profile


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


def add_ci(n, participation, year):
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
    node = geoscope(zone, area)['node']

    #tech_palette options
    clean_techs = palette(tech_palette)[0]
    storage_techs = palette(tech_palette)[1]


    n.add("Bus",
          name)

    n.add("Link",
          name + " export",
          bus0=name,
          bus1=node,
          marginal_cost=0.1, #large enough to avoid optimization artifacts, small enough not to influence PPA portfolio
          p_nom=1e6)

    n.add("Link",
          name + " import",
          bus0=node,
          bus1=name,
          marginal_cost=0.001, #large enough to avoid optimization artifacts, small enough not to influence PPA portfolio
          p_nom=1e6)


    #Add C&I load
    n.add("Load",
          name + " load",
          carrier=name,
          bus=name,
          p_set=load_profile(n, zone, profile_shape))

    #C&I following 24/7 approach is a share of all C&I load -> thus substract it from node's profile
    n.loads_t.p_set[f'{node}'] -= n.loads_t.p_set[f'{name}'+' load']


    #Add generators 
    #baseload clean energy generator
    if "green hydrogen OCGT" in clean_techs:
        n.add("Generator",
              name + " green hydrogen OCGT",
              carrier="green hydrogen OCGT",
              bus=name,
              p_nom_extendable = True if policy == "cfe" else False,
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
              p_nom_extendable = True if policy == "cfe" else False,
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
              p_nom_extendable = True if policy == "cfe" else False,
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
              p_nom_extendable = True if policy == "cfe" else False,
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
              p_nom_extendable=False if policy == "ref" else True,
              p_max_pu=n.generators_t.p_max_pu[gen_template],
              capital_cost=n.generators.at[gen_template,"capital_cost"],
              marginal_cost=n.generators.at[gen_template,"marginal_cost"])

    #Add storage tech
    if "battery" in storage_techs:
        n.add("Bus",
              f"{name} battery",
              carrier="battery"
              )

        n.add("Store",
              f"{name} battery",
              bus=f"{name} battery",
              e_cyclic=True,
              e_nom_extendable=True if policy == "cfe" else False,
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
              p_nom_extendable=True if policy == "cfe" else False,
              lifetime=n.links.at[f"{node} battery charger"+"-{}".format(year), "lifetime"] 
              )

        n.add("Link",
              f"{name} battery discharger",
              bus0=f"{name} battery",
              bus1=name,
              carrier="battery discharger",
              efficiency=n.links.at[f"{node} battery discharger"+"-{}".format(year), "efficiency"],
              marginal_cost=n.links.at[f"{node} battery discharger"+"-{}".format(year), "marginal_cost"],
              p_nom_extendable=True if policy == "cfe" else False,
              lifetime=n.links.at[f"{node} battery discharger"+"-{}".format(year), "lifetime"]
              )

    if "hydrogen" in storage_techs:
        n.add("Bus",
              f"{name} H2",
              carrier="H2"
              )

        n.add("Store",
              f"{name} H2 Store",
              bus=f"{name} H2",
              e_cyclic=True,
              e_nom_extendable=True if policy == "cfe" else False,
              carrier="H2 Store",
              capital_cost=costs.at["hydrogen storage underground","fixed"],
              lifetime=costs.at["hydrogen storage underground","lifetime"],
              )        

        n.add("Link",
              f"{name} H2 Electrolysis",
              bus0=name,
              bus1=f"{name} H2",
              carrier="H2 Electrolysis",
              efficiency=n.links.at[f"{node} H2 Electrolysis"+"-{}".format(year), "efficiency"],
              capital_cost=n.links.at[f"{node} H2 Electrolysis"+"-{}".format(year), "capital_cost"],
              p_nom_extendable=True if policy == "cfe" else False,
              lifetime=n.links.at[f"{node} H2 Electrolysis"+"-{}".format(year), "lifetime"] 
              )

        n.add("Link",
              f"{name} H2 Fuel Cell",
              bus0=f"{name} H2",
              bus1=name,
              carrier="H2 Fuel Cell",
              efficiency=n.links.at[f"{node} H2 Fuel Cell"+"-{}".format(year), "efficiency"],
              capital_cost=n.links.at[f"{node} H2 Fuel Cell"+"-{}".format(year), "capital_cost"],
              p_nom_extendable=True if policy == "cfe" else False,
              lifetime=n.links.at[f"{node} H2 Fuel Cell"+"-{}".format(year), "lifetime"]
              )


def calculate_grid_cfe(n):

    name = snakemake.config['ci']['name']
    country = geoscope(zone, area)['node'] 
    grid_buses = n.buses.index[~n.buses.index.str.contains(name) & ~n.buses.index.str.contains(country)]
    country_buses = n.buses.index[n.buses.index.str.contains(country)]

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

    n_iterations = snakemake.config['solving']['options']['n_iterations']

    if policy == "res":
        res_gens = [name + " " + g for g in ci['res_techs']]
    elif policy == "cfe":
        clean_gens = [name + " " + g for g in clean_techs]
        storage_dischargers = [name + " " + g for g in storage_dischargers]
        storage_chargers = [name + " " + g for g in storage_chargers]


    def cfe_constraints(n): #done

        weights = n.snapshot_weightings["generators"]

        gen_sum = (n.model['Generator-p'].loc[:,clean_gens] * weights).sum()
        discharge_sum = (n.model['Link-p'].loc[:,storage_dischargers] * 
                         n.links.loc[storage_dischargers, "efficiency"] * weights).sum()
        charge_sum = -1*(n.model['Link-p'].loc[:,storage_chargers] * weights).sum()
        n.links.loc[storage_dischargers,"efficiency"]
        ci_export = n.model['Link-p'].loc[:,[name + " export"]]
        ci_import = n.model['Link-p'].loc[:,[name + " import"]] 
        grid_sum = (
            (-1*ci_export*weights) + 
            (ci_import*n.links.at[name + " import","efficiency"]*grid_supply_cfe*weights)
            ).sum() # linear expr

        lhs = gen_sum + discharge_sum + charge_sum  + grid_sum
        total_load = (n.loads_t.p_set[name + " load"]*weights).sum() # number
        
        n.model.add_constraints(lhs >= penetration*total_load, name="CFE_constraint")


    def excess_constraints(n): #done
        
        weights = n.snapshot_weightings["generators"]

        ci_export = n.model['Link-p'].loc[:,[name + " export"]]
        excess =  (ci_export * weights).sum()

        total_load = (n.loads_t.p_set[name + " load"] * weights).sum()
        share = 0.2 # max(0., penetration - 0.8) -> no sliding share
        
        n.model.add_constraints(excess <= share*total_load, name="Excess_constraint")


    def res_constraints(n): #done
        
        weights = n.snapshot_weightings["generators"]

        lhs = (n.model['Generator-p'].loc[:,res_gens] * weights).sum()
        total_load = (n.loads_t.p_set[name + " load"] * weights).sum()

        # Note equality sign
        n.model.add_constraints(lhs == penetration*total_load, name="100RES_annual_constraint")


    def country_res_constraints(n): #done

        grid_buses = n.buses.index[n.buses.location.isin(geoscope(zone, area)['country_nodes'])]
        grid_res_techs = snakemake.config['global']['grid_res_techs']
        grid_loads = n.loads.index[n.loads.bus.isin(grid_buses)]

        country_res_gens = n.generators.index[n.generators.bus.isin(grid_buses) & n.generators.carrier.isin(grid_res_techs)]
        country_res_links = n.links.index[n.links.bus1.isin(grid_buses) & n.links.carrier.isin(grid_res_techs)]
        country_res_storage_units = n.storage_units.index[n.storage_units.bus.isin(grid_buses) & n.storage_units.carrier.isin(grid_res_techs)]

        weights = n.snapshot_weightings["generators"]
        gens = n.model['Generator-p'].loc[:,country_res_gens] * weights
        links = n.model['Link-p'].loc[:,country_res_links] * n.links.loc[country_res_links, "efficiency"] * weights
        sus = n.model['StorageUnit-p_dispatch'].loc[:,country_res_storage_units] * weights
        lhs = gens.sum() + sus.sum() + links.sum()

        target = timescope(zone, year)["country_res_target"]
        total_load = (n.loads_t.p_set[grid_loads].sum(axis=1)*weights).sum() # number

        n.model.add_constraints(lhs == target*total_load, name="country_res_constraints")


    def add_battery_constraints(n):
        """
        Add constraint ensuring that charger = discharger:
         1 * charger_size - efficiency * discharger_size = 0
        """
        discharger_bool = n.links.index.str.contains("battery discharger")
        charger_bool = n.links.index.str.contains("battery charger")

        dischargers_ext= n.links[discharger_bool].query("p_nom_extendable").index
        chargers_ext= n.links[charger_bool].query("p_nom_extendable").index

        eff = n.links.efficiency[dischargers_ext].values
        lhs = n.model["Link-p_nom"].loc[chargers_ext] - n.model["Link-p_nom"].loc[dischargers_ext] * eff
        
        n.model.add_constraints(lhs == 0, name="Link-charger_ratio")


    def extra_functionality(n, snapshots):

        add_battery_constraints(n)
        country_res_constraints(n)

        if policy == "ref":
            print("no target set")
        elif policy == "cfe":
            print("setting CFE target of",penetration)
            cfe_constraints(n)
            excess_constraints(n)
        elif policy == "res":
            print("setting annual RES target of",penetration)
            res_constraints(n)
            excess_constraints(n)
        else:
            print(f"'policy' wildcard must be one of 'ref', 'res__' or 'cfe__'. Now is {policy}.")
            sys.exit()

    n.consistency_check()

    formulation = snakemake.config['solving']['options']['formulation']
    solver_options = snakemake.config['solving']['solver']
    solver_name = solver_options['name']

    grid_cfe_df = pd.DataFrame(0.,index=n.snapshots,columns=[f"iteration {i}" for i in range(n_iterations+1)])

    for i in range(n_iterations):

        grid_supply_cfe = grid_cfe_df[f"iteration {i}"]

        n.optimize.create_model()
        
        extra_functionality(n, n.snapshots)

        n.optimize.solve_model(
               solver_name=solver_name,
               log_fn=snakemake.log.solver,
               **solver_options)

        grid_cfe_df[f"iteration {i+1}"] = calculate_grid_cfe(n)
        #print(grid_cfe_df)

    grid_cfe_df.to_csv(snakemake.output.grid_cfe)


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        snakemake = mock_snakemake('solve_network', 
                    policy="cfe100", palette='p1', zone='IE', year='2025', participation='10')

    logging.basicConfig(filename=snakemake.log.python, level=snakemake.config['logging_level'])

    #Wildcards & Settings
    policy = snakemake.wildcards.policy[:3]
    penetration = float(snakemake.wildcards.policy[3:])/100 if policy != "ref" else 0
    tech_palette = snakemake.wildcards.palette
    zone = snakemake.wildcards.zone
    year = snakemake.wildcards.year
    area = snakemake.config['area']
    participation = snakemake.wildcards.participation
    profile_shape = snakemake.config['ci']['profile_shape']

    print(f"solving network for policy {policy} and penetration {penetration}")
    print(f"solving network for palette: {tech_palette}")
    print(f"solving network for bidding zone: {zone}")
    print(f"solving network year: {year}")
    print(f"solving with geoscope: {area}")
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

    #Temp model reduction
    # nhours = 1000
    # n.set_snapshots(n.snapshots[:nhours])
    # n.snapshot_weightings[:] = 8760.0 / nhours

    with memory_logger(filename=getattr(snakemake.log, 'memory', None), interval=30.) as mem:

        strip_network(n)

        shutdown_lineexp(n)
        nuclear_policy(n)
        coal_policy(n)
        biomass_potential(n)
        #limit_resexp(n,year)

        cost_parametrization(n)
        load_profile(n, zone, profile_shape)
        add_ci(n, participation, year)

        solve_network(n, policy, penetration, tech_palette)

        n.export_to_netcdf(snakemake.output.network)

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))

