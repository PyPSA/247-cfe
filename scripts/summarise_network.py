
import pypsa, numpy as np, pandas as pd

import yaml

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


def summarise_network(n):
    policy = snakemake.wildcards.policy[:3]
    ci = snakemake.config['ci']
    name = ci['name']
    node = ci['node']
    clean_techs = ci['clean_techs']
    clean_gens = [name + " " + g for g in clean_techs]
    clean_dischargers = [name + " " + g for g in ci['storage_dischargers']]
    clean_chargers = [name + " " + g for g in ci['storage_chargers']]
    exp_generators = snakemake.config['exp_generators']
    exp_links = snakemake.config['exp_links']
    exp_chargers = snakemake.config['exp_chargers']
    exp_dischargers = snakemake.config['exp_dischargers']

    if policy == "res":
        n_iterations = 1
    elif policy == "cfe":
        n_iterations = snakemake.config['solving']['options']['n_iterations']

    grid_cfe = grid_cfe_df[f"iteration {n_iterations}"]

    results = {}
    temp = {}

    results['objective'] = n.objective

    p_clean = n.generators_t.p[clean_gens].multiply(n.snapshot_weightings["generators"],axis=0).sum(axis=1)
    p_storage = - n.links_t.p1[clean_dischargers].multiply(n.snapshot_weightings["generators"],axis=0).sum(axis=1) - n.links_t.p0[clean_chargers].multiply(n.snapshot_weightings["generators"],axis=0).sum(axis=1)
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


    # Storing invested capacities at CI node
    for tech in clean_techs:
        results['ci_cap_' + tech] = n.generators.at[name + " " + tech,"p_nom_opt"]

    for charger in ci['storage_chargers']:
        temp['eff_' + charger] = n.links.loc[n.links.index.str.contains(f'{charger}'), 'efficiency'][0]
        results['ci_cap_' + charger.replace(' ', '_')] = n.links.at[name + " " + charger, "p_nom_opt"]*temp['eff_' + charger] 
       
    for discharger in ci['storage_dischargers']:
        temp['eff_' + discharger] = n.links.loc[n.links.index.str.contains(f'{discharger}'), 'efficiency'][0]
        results['ci_cap_' + discharger.replace(' ', '_')] = n.links.at[name + " " + discharger, "p_nom_opt"]*temp['eff_' + discharger] 


    # Storing generation at CI node    
    for tech in clean_techs:
        results['ci_generation_' + tech] = n.generators_t.p[name + " " + tech].multiply(n.snapshot_weightings["generators"],axis=0).sum()

    for discharger in ci['storage_dischargers']:
        results['ci_generation_' + discharger.replace(' ', '_')] = - n.links_t.p1[name + " " + discharger].multiply(n.snapshot_weightings["generators"],axis=0).sum()


    # Storing invested capacities in the rest of energy system
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

    if "battery charger-2030" in exp_chargers:
        batteries = HV_links[HV_links.index.str.contains(f"battery charger-2030")]
    if "H2 Electrolysis-2030" in exp_chargers:
        electrolysis = HV_links[HV_links.index.str.contains(f"H2 Electrolysis-2030")]
    system_chargers = pd.concat([batteries, electrolysis])

    if "battery discharger-2030" in exp_dischargers:
        inverters = HV_links[HV_links.index.str.contains(f"battery discharger-2030")]
    if "H2 Fuel Cell-2030" in exp_dischargers:
        fuelcells = HV_links[HV_links.index.str.contains(f"H2 Fuel Cell-2030")]
    system_dischargers = pd.concat([inverters, fuelcells])

    for charger in exp_chargers:
        temp['system_optcap_' + charger] = system_chargers.loc[system_chargers.index.str.contains(f'{charger}'), 'p_nom_opt'].sum()
        temp['eff_' + charger] = system_chargers.loc[system_chargers.index.str.contains(f'{charger}'), 'efficiency'][0]
        results['system_inv_' + charger.replace(' ', '_')] = temp['system_optcap_' + charger] - system_chargers.loc[system_chargers.index.str.contains(f'{charger}'), 'p_nom'].sum()
        #print(temp['eff_' + charger])
        results['system_inv_' + charger.replace(' ', '_')] *= temp['eff_' + charger]

    for discharger in exp_dischargers:
        temp['system_optcap_' + discharger] = system_dischargers.loc[system_dischargers.index.str.contains(f'{discharger}'), 'p_nom_opt'].sum()
        temp['eff_' + discharger] = system_dischargers.loc[system_dischargers.index.str.contains(f'{discharger}'), 'efficiency'][0]
        results['system_inv_' + discharger.replace(' ', '_')] = temp['system_optcap_' + discharger] - system_dischargers.loc[system_dischargers.index.str.contains(f'{discharger}'), 'p_nom'].sum()
        #print(temp['eff_' + discharger])
        results['system_inv_' + discharger.replace(' ', '_')] *= temp['eff_' + discharger]


    # Storing costs at CI node
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

    if "battery" in ci["storage_techs"]:
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

    if "hydrogen" in ci["storage_techs"]:
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

    # Storing system emissions and co2 price
    results["emissions"] = n.stores_t.e["co2 atmosphere"][-1]

    if snakemake.config['global']['policy_type'] == "co2 cap":
        results["co2_price"] = n.global_constraints.at["CO2Limit","mu"]
    elif snakemake.config['global']['policy_type'] == "co2 price":
        results["co2_price"] = snakemake.config['global']['co2_price']
    else:
        results["co2_price"] = 0

    # Saving resutls as ../summaries/{}.yaml
    for k in results:
        results[k] = float(results[k])

    print(results)

    with open(snakemake.output.yaml, 'w') as outfile:
        yaml.dump(results, outfile)

if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('summarise_network', policy="cfe100")

    # When running via snakemake
    policy = snakemake.wildcards.policy[:3]
    penetration = float(snakemake.wildcards.policy[3:])/100

    print(f"summarising network for policy {policy} and penetration {penetration}")

    n = pypsa.Network(snakemake.input.network,
                      override_component_attrs=override_component_attrs)

    grid_cfe_df = pd.read_csv(snakemake.input.grid_cfe,
                              index_col=0,
                              parse_dates=True)
    print(grid_cfe_df)
    
    summarise_network(n)
