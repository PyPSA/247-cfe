
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

    name = snakemake.config['ci']['name']
    clean_techs = snakemake.config['ci']['clean_techs']
    clean_gens = [name + " " + g for g in clean_techs]

    results = {}

    results['objective'] = n.objective

    p_clean = n.generators_t.p[clean_gens].multiply(n.snapshot_weightings["generators"],axis=0).sum(axis=1)
    p_demand = n.loads_t.p["google load"].multiply(n.snapshot_weightings["generators"],axis=0)

    p_diff = p_clean - p_demand
    excess = p_diff.copy()
    excess[excess < 0] = 0.

    used = p_clean - excess

    results['ci_clean_total']  = p_clean.sum()
    results['ci_clean_used_total']  = used.sum()
    results['ci_clean_excess_total']  = excess.sum()
    results['ci_demand_total']  = p_demand.sum()

    results['ci_fraction_clean'] = p_clean.sum()/p_demand.sum()
    results['ci_fraction_clean_used'] = used.sum()/p_demand.sum()
    results['ci_fraction_clean_excess'] = excess.sum()/p_demand.sum()


    for tech in clean_techs:
        results['ci_cap_' + tech] = n.generators.at[name + " " + tech,"p_nom_opt"]

    for tech in clean_techs:
        results['ci_generation_' + tech] = n.generators_t.p[name + " " + tech].multiply(n.snapshot_weightings["generators"],axis=0).sum()

    total_cost = 0.
    for tech in clean_techs:
        results['ci_capital_cost_' + tech] = results['ci_cap_' + tech]*n.generators.at[name + " " + tech,"capital_cost"]
        results['ci_marginal_cost_' + tech] = results['ci_generation_' + tech]*n.generators.at[name + " " + tech,"marginal_cost"]
        cost =  results['ci_capital_cost_' + tech] + results['ci_marginal_cost_' + tech]
        results['ci_cost_' + tech] = cost
        total_cost += cost

    results["ci_total_cost"] = total_cost

    results["ci_average cost"] = results['ci_total_cost']/results['ci_demand_total']

    results["emissions"] = n.stores_t.e["co2 atmosphere"][-1]

    results["co2_price"] = n.global_constraints.at["CO2Limit","mu"]

    for k in results:
        results[k] = float(results[k])

    print(results)

    with open(snakemake.output.yaml, 'w') as outfile:
        yaml.dump(results, outfile)

if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils.snakemake import MockSnakemake, Dict
        snakemake = MockSnakemake(
            path='',
            wildcards=dict(policy='co2120-trans-storage-wind1040-sola510-nuclNone-lCCSNone',parameter="0"),
            output=dict(network="results/test/0after.nc"),
            log=dict(solver="results/test/log_0after.log")
        )
        import yaml
        with open('config.yaml') as f:
            snakemake.config = yaml.load(f)


    policy = snakemake.wildcards.policy[:3]
    penetration = float(snakemake.wildcards.policy[3:])/100

    print(f"summarising network for policy {policy} and penetration {penetration}")

    n = pypsa.Network(snakemake.input.network,
                      override_component_attrs=override_component_attrs)

    summarise_network(n)
