

import pypsa, numpy as np, pandas as pd
import yaml
from solve_network import palette, geoscope, timescope
from _helpers import override_component_attrs


def summarise_network(n, policy, tech_palette):

    #tech_palette options
    clean_techs = palette(tech_palette)[0]
    storage_techs = palette(tech_palette)[1]
    storage_chargers = palette(tech_palette)[2]
    storage_dischargers = palette(tech_palette)[3]

    if policy == "ref":
        n_iterations = 2
    elif policy == "res":
        n_iterations = 2
    elif policy == "cfe":
        n_iterations = snakemake.config['solving']['options']['n_iterations']

    name = snakemake.config['ci']['name']
    node = geoscope(zone, area)['node']

    clean_gens = [name + " " + g for g in clean_techs]
    clean_dischargers = [name + " " + g for g in storage_dischargers]
    clean_chargers = [name + " " + g for g in storage_chargers]


    # for calculation of system expansion beyond CI node in pypsa-eur-sec brownfield network for {year}
    exp_generators = ['offwind-ac-%s' % year, 
                    'offwind-dc-%s' % year, 
                    'onwind-%s' % year, 
                    'solar-%s' % year, 
                    'solar rooftop-%s' % year]
    exp_links = ['OCGT-%s' % year]
    exp_chargers = ['battery charger-%s' % year, 'H2 Electrolysis-%s' % year]
    exp_dischargers = ['battery discharger-%s' % year, 'H2 Fuel Cell-%s' % year]


    grid_buses = n.buses.index[~n.buses.index.str.contains(name)]
    grid_loads = n.loads.index[n.loads.bus.isin(grid_buses)]
    grid_cfe = grid_cfe_df[f"iteration {n_iterations-1}"]
 
    results = {}
    temp = {}


    # Processing, calculating and storing model results
    
    results['objective'] = n.objective
    
    # 1: Generation & imports at C&I node

    p_clean = n.generators_t.p[clean_gens].multiply(n.snapshot_weightings["generators"],axis=0).sum(axis=1)
    p_storage = - n.links_t.p1[clean_dischargers].multiply(n.snapshot_weightings["generators"],axis=0).sum(axis=1) \
                - n.links_t.p0[clean_chargers].multiply(n.snapshot_weightings["generators"],axis=0).sum(axis=1) 
    p_demand = n.loads_t.p["google load"].multiply(n.snapshot_weightings["generators"],axis=0)

    p_diff = p_clean + p_storage - p_demand
    excess = p_diff.copy()
    excess[excess < 0] = 0.

    used_local = p_clean + p_storage - excess

    used_grid = (-n.links_t.p1[name + " import"].multiply(n.snapshot_weightings["generators"],axis=0))*grid_cfe

    results['ci_clean_total']  = (p_clean + p_storage).sum()
    results['ci_clean_used_local_total']  = used_local.sum()
    results['ci_clean_used_grid_total']  = used_grid.sum()
    results['ci_clean_used_total']  = used_grid.sum() + used_local.sum()
    results['ci_clean_excess_total']  = excess.sum()
    results['ci_demand_total']  = p_demand.sum()

    results['ci_fraction_clean'] = results['ci_clean_total']/results['ci_demand_total']
    results['ci_fraction_clean_used_local'] = results['ci_clean_used_local_total']/results['ci_demand_total']
    results['ci_fraction_clean_used_grid'] = results['ci_clean_used_grid_total']/results['ci_demand_total']
    results['ci_fraction_clean_used'] = results['ci_clean_used_total']/results['ci_demand_total']
    results['ci_fraction_clean_excess'] = results['ci_clean_excess_total']/results['ci_demand_total']

    # 2: compute grid average emission rate

    emitters = snakemake.config['global']['emitters']

    fossil_links = n.links.index[n.links.carrier.isin(emitters)]
    hourly_emissions = n.links_t.p0[fossil_links].multiply(n.links.efficiency2[fossil_links],axis=1).sum(axis=1)
    load = n.loads_t.p[grid_loads].sum(axis=1)

    results['system_emissions'] = hourly_emissions.sum() 
    results['system_emission_rate'] = hourly_emissions.sum() / load.sum()

    # 3: compute emissions & emission rates
    
    country = geoscope(zone, area)['node']
    grid_clean_techs = snakemake.config['global']['grid_clean_techs']

    #Careful: clean_techs (at C&I node) != grid_clean_techs (rest of system)
    #3.1: ci_emission_rate based on Princeton (considering network congestions)

    rest_system_buses = n.buses.index[~n.buses.index.str.contains(name) & ~n.buses.index.str.contains(country)]
    country_buses = n.buses.index[n.buses.index.str.contains(country)]

    clean_grid_generators = n.generators.index[n.generators.bus.isin(rest_system_buses) & n.generators.carrier.isin(grid_clean_techs)]
    clean_grid_links = n.links.index[n.links.bus1.isin(rest_system_buses) & n.links.carrier.isin(grid_clean_techs)]
    clean_grid_storage_units = n.storage_units.index[n.storage_units.bus.isin(rest_system_buses) & n.storage_units.carrier.isin(grid_clean_techs)]
    dirty_grid_links = n.links.index[n.links.bus1.isin(rest_system_buses) & n.links.carrier.isin(emitters)]

    clean_country_generators = n.generators.index[n.generators.bus.isin(country_buses) & n.generators.carrier.isin(grid_clean_techs)]
    clean_country_links = n.links.index[n.links.bus1.isin(country_buses) & n.links.carrier.isin(grid_clean_techs)]
    clean_country_storage_units = n.storage_units.index[n.storage_units.bus.isin(country_buses) & n.storage_units.carrier.isin(grid_clean_techs)]
    dirty_country_links = n.links.index[n.links.bus1.isin(country_buses) & n.links.carrier.isin(emitters)]

    country_loads = n.loads.index[n.loads.bus.isin(country_buses)]

    clean_grid_gens = n.generators_t.p[clean_grid_generators].sum(axis=1)
    clean_grid_ls = (- n.links_t.p1[clean_grid_links].sum(axis=1))
    clean_grid_sus = n.storage_units_t.p[clean_grid_storage_units].sum(axis=1)
    clean_grid_resources = clean_grid_gens + clean_grid_ls + clean_grid_sus
    dirty_grid_resources = (- n.links_t.p1[dirty_grid_links].sum(axis=1))
    
    clean_country_gens = n.generators_t.p[clean_country_generators].sum(axis=1)
    clean_country_ls = (- n.links_t.p1[clean_country_links].sum(axis=1))
    clean_country_sus = n.storage_units_t.p[clean_country_storage_units].sum(axis=1)
    clean_country_resources = clean_country_gens + clean_country_ls + clean_country_sus
    dirty_country_resources = (- n.links_t.p1[dirty_country_links].sum(axis=1))

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

    grid_hourly_emissions = n.links_t.p0[dirty_grid_links].multiply(n.links.efficiency2[dirty_grid_links],axis=1).sum(axis=1)
    
    grid_emission_rate =  grid_hourly_emissions / (clean_grid_resources + dirty_grid_resources)

    country_hourly_emissions = n.links_t.p0[dirty_country_links].multiply(n.links.efficiency2[dirty_country_links],axis=1).sum(axis=1)

    grid_supply_emission_rate = (country_hourly_emissions + country_import*grid_emission_rate) / \
                                (clean_country_resources + dirty_country_resources + country_import)

    ci_emissions_t = n.links_t.p0["google import"]*grid_supply_emission_rate
    
    results['ci_emission_rate_true'] = ci_emissions_t.sum() / n.loads_t.p["google load"].sum()

    #3.2: considering only country node(local bidding zone) 
    country_load = n.loads_t.p[country_loads].sum(axis=1)
    emissions_factor_local = country_hourly_emissions / country_load
    results['ci_emission_rate_local'] = (n.links_t.p0["google import"]*emissions_factor_local).sum()/n.loads_t.p["google load"].sum()

    #3.3: Our original ci_emission_rate (ignoring network congestions)
    emissions_factor = hourly_emissions / load #global average emissions
    results['ci_emission_rate_myopic'] = (n.links_t.p0["google import"]*emissions_factor).sum()/n.loads_t.p["google load"].sum()
    
    #3.3: Total CO2 emissions for 24/7 participating customers
    results['ci_emissions'] = (ci_emissions_t*n.snapshot_weightings["generators"]).sum()

    #3.4: Total CO2 emissions in zone where 24/7 participating customers are located
    results['emissions_zone'] = (country_hourly_emissions*n.snapshot_weightings["generators"]).sum()/1e6


    # 4: Storing invested capacities at CI node

    for tech in clean_techs:
        results['ci_cap_' + tech] = n.generators.at[name + " " + tech,"p_nom_opt"]

    for charger in storage_chargers:
        temp['eff_' + charger] = n.links.loc[n.links.index.str.contains(f'{charger}'), 'efficiency'][0]
        if name + " " + charger in n.links.index:
            results['ci_cap_' + charger.replace(' ', '_')] = n.links.at[name + " " + charger, "p_nom_opt"]*temp['eff_' + charger]
        else:
            results['ci_cap_' + charger.replace(' ', '_')] = 0.

    for discharger in storage_dischargers:
        temp['eff_' + discharger] = n.links.loc[n.links.index.str.contains(f'{discharger}'), 'efficiency'][0]
        if name + " " + discharger in n.links.index:
            results['ci_cap_' + discharger.replace(' ', '_')] = n.links.at[name + " " + discharger, "p_nom_opt"]*temp['eff_' + discharger]
        else:
            results['ci_cap_' + discharger.replace(' ', '_')] = 0.

    # Storing generation at CI node  
    
    for tech in clean_techs:
        results['ci_generation_' + tech] = n.generators_t.p[name + " " + tech].multiply(n.snapshot_weightings["generators"],axis=0).sum()

    for discharger in storage_dischargers:
        if name + " " + discharger in n.links.index:
            results['ci_generation_' + discharger.replace(' ', '_')] = - n.links_t.p1[name + " " + discharger].multiply(n.snapshot_weightings["generators"],axis=0).sum()
        else:
            results['ci_generation_' + discharger.replace(' ', '_')] = 0.

    # 5: Storing invested capacities in the rest of energy system

    # Generators
    gen_expandable = n.generators[n.generators.p_nom_extendable == True] #all gens that can be expanded
    
    for g in snakemake.config['additional_nodes']: # drop additional nodes
        if g in gen_expandable.index:
            gen_expandable = gen_expandable.drop(g)

    system_gens = gen_expandable[~gen_expandable.index.str.contains('google')] # drop google gens

    for gen in exp_generators:
        temp['system_optcap_' + gen] = system_gens.loc[system_gens.index.str.contains(f'{gen}'), 'p_nom_opt'].sum()
        results['system_inv_' + gen.replace(' ', '_')] = temp['system_optcap_' + gen] - system_gens.loc[system_gens.index.str.contains(f'{gen}'), 'p_nom'].sum()

    #Links
    for link in exp_links:
        system_links = n.links[n.links.index.str.contains(f'{link}')]

    for link in exp_links:
        temp['system_optcap_' + link] = system_links.loc[system_links.index.str.contains(f'{link}'), 'p_nom_opt'].sum()
        temp['eff_' + link] = system_links.loc[system_links.index.str.contains(f'{link}'), 'efficiency'][0]
        results['system_inv_' + link] = temp['system_optcap_' + link] - system_links.loc[system_links.index.str.contains(f'{link}'), 'p_nom'].sum()
        # capacity of PyPSA-eur-sec links are in MWth. Converting MWth -> MWel.
        results['system_inv_' + link] *= temp['eff_' + link]

    #Chargers & Dischargers
    HV_links = n.links
    for l in n.links[n.links.index.str.contains("home battery")].index:
        HV_links = HV_links.drop(l) #remove low voltage batteries 

    batteries = HV_links[HV_links.index.str.contains(f"battery charger"+"-{}".format(year))]
    electrolysis = HV_links[HV_links.index.str.contains(f"H2 Electrolysis"+"-{}".format(year))]
    system_chargers = pd.concat([batteries, electrolysis])

    inverters = HV_links[HV_links.index.str.contains(f"battery discharger"+"-{}".format(year))]
    fuelcells = HV_links[HV_links.index.str.contains(f"H2 Fuel Cell"+"-{}".format(year))]
    system_dischargers = pd.concat([inverters, fuelcells])

    for charger in exp_chargers:
        temp['system_optcap_' + charger] = system_chargers.loc[system_chargers.index.str.contains(f'{charger}'), 'p_nom_opt'].sum()
        temp['eff_' + charger] = system_chargers.loc[system_chargers.index.str.contains(f'{charger}'), 'efficiency'][0]
        results['system_inv_' + charger.replace(' ', '_')] = temp['system_optcap_' + charger] - system_chargers.loc[system_chargers.index.str.contains(f'{charger}'), 'p_nom'].sum()
        results['system_inv_' + charger.replace(' ', '_')] *= temp['eff_' + charger]

    for discharger in exp_dischargers:
        temp['system_optcap_' + discharger] = system_dischargers.loc[system_dischargers.index.str.contains(f'{discharger}'), 'p_nom_opt'].sum()
        temp['eff_' + discharger] = system_dischargers.loc[system_dischargers.index.str.contains(f'{discharger}'), 'efficiency'][0]
        results['system_inv_' + discharger.replace(' ', '_')] = temp['system_optcap_' + discharger] - system_dischargers.loc[system_dischargers.index.str.contains(f'{discharger}'), 'p_nom'].sum()
        results['system_inv_' + discharger.replace(' ', '_')] *= temp['eff_' + discharger]

    # 6: Storing costs at CI node

    total_cost = 0.
    for tech in clean_techs:
        results['ci_capital_cost_' + tech] = results['ci_cap_' + tech]*n.generators.at[name + " " + tech,"capital_cost"]
        results['ci_marginal_cost_' + tech] = results['ci_generation_' + tech]*n.generators.at[name + " " + tech,"marginal_cost"]
        cost =  results['ci_capital_cost_' + tech] + results['ci_marginal_cost_' + tech]
        results['ci_cost_' + tech] = cost
        total_cost += cost

    results['ci_capital_cost_grid'] = 0
    results['ci_marginal_cost_grid'] = (n.links_t.p0[name + " import"]*n.snapshot_weightings["generators"]*n.buses_t.marginal_price[node]).sum()
    cost =  results['ci_capital_cost_grid'] + results['ci_marginal_cost_grid']
    results['ci_cost_grid'] = cost
    total_cost += cost

    #check_average_cost_grid = results['ci_cost_grid'] / results['ci_demand_total']
    results['ci_revenue_grid'] = (n.links_t.p0[name + " export"]*n.snapshot_weightings["generators"]*n.buses_t.marginal_price[node]).sum()
    results['ci_average_revenue'] =  results['ci_revenue_grid'] / results['ci_demand_total']

    if "battery" in storage_techs:
        results['ci_capital_cost_battery_storage'] = n.stores.at[f"{name} battery","e_nom_opt"]*n.stores.at[f"{name} battery","capital_cost"]
        results['ci_cost_battery_storage'] = results['ci_capital_cost_battery_storage']
        total_cost += results['ci_cost_battery_storage']

        results['ci_capital_cost_battery_inverter'] = n.links.at[f"{name} battery charger","p_nom_opt"]*n.links.at[f"{name} battery charger","capital_cost"]
        results['ci_cost_battery_inverter'] = results['ci_capital_cost_battery_inverter']
        total_cost += results['ci_cost_battery_inverter']
    else:
        results['ci_capital_cost_battery_storage'] = 0.
        results['ci_cost_battery_storage'] = 0.
        results['ci_capital_cost_battery_inverter'] = 0.
        results['ci_cost_battery_inverter'] = 0.

    if "hydrogen" in storage_techs:
        results['ci_capital_cost_hydrogen_storage'] = n.stores.at[f"{name} H2 Store","e_nom_opt"]*n.stores.at[f"{name} H2 Store","capital_cost"]
        results['ci_cost_hydrogen_storage'] = results['ci_capital_cost_hydrogen_storage']
        total_cost += results['ci_cost_hydrogen_storage']

        results['ci_capital_cost_hydrogen_electrolysis'] = n.links.at[f"{name} H2 Electrolysis","p_nom_opt"]*n.links.at[f"{name} H2 Electrolysis","capital_cost"]
        results['ci_cost_hydrogen_electrolysis'] = results['ci_capital_cost_hydrogen_electrolysis']
        total_cost += results['ci_cost_hydrogen_electrolysis']

        results['ci_capital_cost_hydrogen_fuel_cell'] = n.links.at[f"{name} H2 Fuel Cell","p_nom_opt"]*n.links.at[f"{name} H2 Fuel Cell","capital_cost"]
        results['ci_cost_hydrogen_fuel_cell'] = results['ci_capital_cost_hydrogen_fuel_cell']
        total_cost += results['ci_cost_hydrogen_fuel_cell']
    else:
        results['ci_capital_cost_hydrogen_storage'] = 0.
        results['ci_cost_hydrogen_storage'] = 0.
        results['ci_capital_cost_hydrogen_electrolysis'] = 0.
        results['ci_cost_hydrogen_electrolysis'] = 0.
        results['ci_capital_cost_hydrogen_fuel_cell'] = 0.
        results['ci_cost_hydrogen_fuel_cell'] = 0.

    results["ci_total_cost"] = total_cost
    results["ci_average cost"] = results['ci_total_cost']/results['ci_demand_total']

    # 7: Other calculations

    results['zone_average_marginal_price'] = n.buses_t.marginal_price[node].sum() / len(n.snapshots)
    
    # Storing system emissions and co2 price
    results["emissions"] = n.stores_t.e["co2 atmosphere"][-1]

    #Compute weighted average grid CFE
    def weighted_avg(cfe, weights):
        weighted_sum = []
        for value, weight in zip(cfe, weights):
            weighted_sum.append(value * weight)
        return sum(weighted_sum) / sum(weights) 
    
    system_grid_cfe_wavg = weighted_avg(grid_cfe, n.loads_t.p[grid_loads].sum(axis=1))
    results['system_grid_cfe_wavg'] = system_grid_cfe_wavg
    #print(system_grid_cfe_wavg)
    #print(grid_cfe.mean())

    if snakemake.config['global']['policy_type'] == "co2 cap":
        results["co2_price"] = n.global_constraints.at["CO2Limit","mu"]
    elif snakemake.config['global']['policy_type'] == "co2 price":
        results["co2_price"] = snakemake.config['global'][f'co2_price_{year}']
    else:
        results["co2_price"] = 0

    # 8: Saving resutls as ../summaries/{}.yaml

    for k in results:
        results[k] = float(results[k])

    print(results)

    with open(snakemake.output.yaml, 'w') as outfile:
        yaml.dump(results, outfile)


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('summarise_network', policy="ref", palette='p3', zone='DE', year='2025', participation='10')

    #Wildcards
    policy = snakemake.wildcards.policy[:3]
    penetration = float(snakemake.wildcards.policy[3:])/100 if policy != "ref" else 0
    print(f"summarising network for policy {policy} and penetration {penetration}")

    tech_palette = snakemake.wildcards.palette
    print(f"summarising network for palette: {tech_palette}")

    zone = snakemake.wildcards.zone
    year = snakemake.wildcards.year
    print(f"summarising network for bidding zone: {zone} and year: {year}")

    area = snakemake.config['area']
    print(f"solving with geographcial scope: {area}")

    participation = snakemake.wildcards.participation
    print(f"solving with participation: {participation}")

    #Read data
    n = pypsa.Network(snakemake.input.network,
                      override_component_attrs=override_component_attrs())

    grid_cfe_df = pd.read_csv(snakemake.input.grid_cfe,
                              index_col=0,
                              parse_dates=True)
    print(grid_cfe_df)
    

    summarise_network(n, policy, tech_palette)

