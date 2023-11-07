# SPDX-FileCopyrightText: 2023 Iegor Riepin, Tom Brown
#
# SPDX-License-Identifier: MIT

import pypsa, numpy as np, pandas as pd
import yaml
from solve_network import palette
from _helpers import override_component_attrs


def weighted_avg(cfe, weights):
    weighted_sum = []
    for value, weight in zip(cfe, weights):
        weighted_sum.append(value * weight)
    return sum(weighted_sum) / sum(weights)


# Helper functions


def _aggregate_resources(n, buses, grid_clean_techs, emitters):
    clean_gens = [
        gen
        for gen in n.generators.index
        if n.generators.loc[gen, "bus"] in buses
        and n.generators.loc[gen, "carrier"] in grid_clean_techs
    ]
    clean_ls = [
        link
        for link in n.links.index
        if n.links.loc[link, "bus1"] in buses
        and n.links.loc[link, "carrier"] in grid_clean_techs
    ]
    clean_sus = [
        su
        for su in n.storage_units.index
        if n.storage_units.loc[su, "bus"] in buses
        and n.storage_units.loc[su, "carrier"] in grid_clean_techs
    ]
    dirty_links = [
        link
        for link in n.links.index
        if n.links.loc[link, "bus1"] in buses
        and n.links.loc[link, "carrier"] in emitters
    ]

    clean_gens_t = n.generators_t.p[clean_gens].sum()
    clean_ls_t = -n.links_t.p1[clean_ls].sum()
    clean_sus_t = n.storage_units_t.p[clean_sus].sum()
    dirty_links_t = -n.links_t.p1[dirty_links].sum()

    return clean_gens_t, clean_ls_t, clean_sus_t, dirty_links_t


def _calculate_imports(n, location, name):
    line_imp_subsetA = n.lines_t.p1.loc[:, n.lines.bus0.str.contains(location)].sum(
        axis=1
    )
    line_imp_subsetB = n.lines_t.p0.loc[:, n.lines.bus1.str.contains(location)].sum(
        axis=1
    )
    line_imp_subsetA = line_imp_subsetA.clip(lower=0)
    line_imp_subsetB = line_imp_subsetB.clip(lower=0)

    links_imp_subsetA = n.links_t.p1.loc[
        :,
        n.links.bus0.str.contains(location)
        & (n.links.carrier == "DC")
        & ~n.links.index.str.contains(name),
    ].sum(axis=1)
    links_imp_subsetB = n.links_t.p0.loc[
        :,
        n.links.bus1.str.contains(location)
        & (n.links.carrier == "DC")
        & ~n.links.index.str.contains(name),
    ].sum(axis=1)
    links_imp_subsetA = links_imp_subsetA.clip(lower=0)
    links_imp_subsetB = links_imp_subsetB.clip(lower=0)

    return line_imp_subsetA + line_imp_subsetB + links_imp_subsetA + links_imp_subsetB


def _calculate_emissions(n, dirty_links):
    return (
        n.links_t.p0[dirty_links.index]
        .multiply(n.links.efficiency2[dirty_links.index], axis=1)
        .sum(axis=1)
    )


def _calculate_tech_cost(tech, location, name, network, results_dict):
    capital_cost = (
        results_dict[location][f"ci_cap_{tech}"]
        * network.generators.at[f"{name} {tech}", "capital_cost"]
    )
    marginal_cost = (
        results_dict[location][f"ci_generation_{tech}"]
        * network.generators.at[f"{name} {tech}", "marginal_cost"]
    )
    total_cost = capital_cost + marginal_cost

    return {
        f"ci_capital_cost_{tech}": capital_cost,
        f"ci_marginal_cost_{tech}": marginal_cost,
        f"ci_cost_{tech}": total_cost,
    }


def _calculate_storage_costs(storage_techs, name, network):
    battery_costs = {
        "ci_capital_cost_battery_storage": 0.0,
        "ci_cost_battery_storage": 0.0,
        "ci_capital_cost_battery_inverter": 0.0,
        "ci_cost_battery_inverter": 0.0,
    }

    hydrogen_costs = {
        "ci_capital_cost_hydrogen_storage": 0.0,
        "ci_cost_hydrogen_storage": 0.0,
        "ci_capital_cost_hydrogen_electrolysis": 0.0,
        "ci_cost_hydrogen_electrolysis": 0.0,
        "ci_capital_cost_hydrogen_fuel_cell": 0.0,
        "ci_cost_hydrogen_fuel_cell": 0.0,
    }

    if "battery" in storage_techs:
        battery_costs["ci_capital_cost_battery_storage"] = (
            network.stores.at[f"{name} battery", "e_nom_opt"]
            * network.stores.at[f"{name} battery", "capital_cost"]
        )
        battery_costs["ci_cost_battery_storage"] = battery_costs[
            "ci_capital_cost_battery_storage"
        ]

        battery_costs["ci_capital_cost_battery_inverter"] = (
            network.links.at[f"{name} battery charger", "p_nom_opt"]
            * network.links.at[f"{name} battery charger", "capital_cost"]
        )
        battery_costs["ci_cost_battery_inverter"] = battery_costs[
            "ci_capital_cost_battery_inverter"
        ]

    if "hydrogen" in storage_techs:
        hydrogen_costs["ci_capital_cost_hydrogen_storage"] = (
            network.stores.at[f"{name} H2 Store", "e_nom_opt"]
            * network.stores.at[f"{name} H2 Store", "capital_cost"]
        )
        hydrogen_costs["ci_cost_hydrogen_storage"] = hydrogen_costs[
            "ci_capital_cost_hydrogen_storage"
        ]

        hydrogen_costs["ci_capital_cost_hydrogen_electrolysis"] = (
            network.links.at[f"{name} H2 Electrolysis", "p_nom_opt"]
            * network.links.at[f"{name} H2 Electrolysis", "capital_cost"]
        )
        hydrogen_costs["ci_cost_hydrogen_electrolysis"] = hydrogen_costs[
            "ci_capital_cost_hydrogen_electrolysis"
        ]

        hydrogen_costs["ci_capital_cost_hydrogen_fuel_cell"] = (
            network.links.at[f"{name} H2 Fuel Cell", "p_nom_opt"]
            * network.links.at[f"{name} H2 Fuel Cell", "capital_cost"]
        )
        hydrogen_costs["ci_cost_hydrogen_fuel_cell"] = hydrogen_costs[
            "ci_capital_cost_hydrogen_fuel_cell"
        ]

    return battery_costs, hydrogen_costs


def _calculate_curtailment(network, tech, buses, weights):
    """
    Calculate the curtailment for a given technology and bus.

    :param network: The PyPSA network object
    :param tech: Technology type (string)
    :param buses: Buses to consider for the calculation (iterable of strings)
    :param weights: Weights for the calculation (pandas Series or similar)
    :return: Total curtailment for the given technology and buses
    """
    gens = network.generators.query("carrier == @tech and bus in @buses").index
    curtailment = (
        (
            network.generators_t.p_max_pu[gens] * network.generators.p_nom_opt[gens]
            - network.generators_t.p[gens]
        )
        .clip(lower=0)
        .multiply(weights, axis=0)
        .sum()
        .sum()
    )
    return curtailment


# Main function


def summarise_network(n, policy, tech_palette):
    """
    This function refactors summarise_network_old() to improve readability and maintainability.
        List Comprehensions and Generators are used to simplify loops and list creation.
        Redundant Code Removal: Helper functions (build_name_list, sum_technology_power, compute_emission_rate) are used to avoid repetition.
        Tuple Unpacking: Used for extracting technology types from tech_palette.
        Dictionary Comprehensions: Applied in constructing location_results.
        Destructuring for Multiple Assignments: Implemented in various places.
        Simplified Conditional Statements: Used in calculations like excess.
    """

    clean_techs, storage_techs, storage_charge_techs, storage_discharge_techs = palette(
        tech_palette
    )

    n_iterations = snakemake.config["solving"]["options"]["n_iterations"]
    results = {}

    for location, name in datacenters.items():
        clean_gens = [f"{name} {g}" for g in clean_techs]
        clean_dischargers = [f"{name} {g}" for g in storage_discharge_techs]
        clean_chargers = [f"{name} {g}" for g in storage_charge_techs]
        grid_buses = [bus for bus in n.buses.index if not name in bus]
        grid_loads = n.loads.index[n.loads.bus.isin(grid_buses)]

        exp_generators = [
            f"{tech}-{year}"
            for tech in ["offwind-ac", "offwind-dc", "onwind", "solar", "solar rooftop"]
        ]
        exp_links = [f"OCGT-{year}"]
        exp_chargers = [
            f"{tech}-{year}" for tech in ["battery charger", "H2 Electrolysis"]
        ]
        exp_dischargers = [
            f"{tech}-{year}" for tech in ["battery discharger", "H2 Fuel Cell"]
        ]

        grid_cfe = grid_cfe_df.loc[:, (location, f"iteration {n_iterations-1}")]

        results[location] = {}
        weights = n.snapshot_weightings["generators"]

        results[f"{location}"].update({"objective": n.objective})

        # 1: Generation & imports at C&I node

        p_clean = n.generators_t.p[clean_gens].multiply(weights, axis=0).sum(axis=1)
        p_storage = -n.links_t.p1[clean_dischargers].multiply(weights, axis=0).sum(
            axis=1
        ) - n.links_t.p0[clean_chargers].multiply(weights, axis=0).sum(axis=1)
        p_demand = n.loads_t.p[f"{name} load"].multiply(weights, axis=0)

        p_diff = p_clean + p_storage - p_demand
        excess = p_diff.clip(lower=0)
        used_local = p_clean + p_storage - excess
        used_grid = -n.links_t.p1[f"{name} import"].multiply(weights, axis=0) * grid_cfe

        results[location].update(
            {
                "ci_clean_total": (p_clean + p_storage).sum(),
                "ci_clean_used_local_total": used_local.sum(),
                "ci_clean_used_grid_total": used_grid.sum(),
                "ci_clean_used_total": used_grid.sum() + used_local.sum(),
                "ci_clean_excess_total": excess.sum(),
                "ci_demand_total": p_demand.sum(),
                "ci_fraction_clean": (p_clean + p_storage).sum() / p_demand.sum(),
                "ci_fraction_clean_used_local": used_local.sum() / p_demand.sum(),
                "ci_fraction_clean_used_grid": used_grid.sum() / p_demand.sum(),
                "ci_fraction_clean_used": (used_grid.sum() + used_local.sum())
                / p_demand.sum(),
                "ci_fraction_clean_excess": excess.sum() / p_demand.sum(),
            }
        )

    # 2: Compute grid average emission rate
    emitters = snakemake.config["global"]["emitters"]

    fossil_links = n.links.index[n.links.carrier.isin(emitters)]
    hourly_emissions = (
        n.links_t.p0[fossil_links]
        .multiply(n.links.efficiency2[fossil_links], axis=1)
        .sum(axis=1)
    )
    load = n.loads_t.p[grid_loads].sum(axis=1)

    results[location].update(
        {
            "system_emissions": hourly_emissions.sum(),
            "system_emission_rate": hourly_emissions.sum() / load.sum(),
        }
    )

    # 3: Compute emissions & emission rates
    grid_clean_techs = snakemake.config["global"]["grid_clean_techs"]

    # clean_techs (at C&I node) != grid_clean_techs (rest of system)
    # 3.1: ci_emission_rate based on Princeton (considering network congestions)

    rest_system_buses = n.buses.index[
        ~n.buses.index.str.contains(name) & ~n.buses.index.str.contains(location)
    ]
    country_buses = n.buses.index[n.buses.index.str.contains(location)]

    (
        clean_grid_generators,
        clean_grid_links,
        clean_grid_storage_units,
        dirty_grid_links,
    ) = _aggregate_resources(n, rest_system_buses, grid_clean_techs, emitters)

    (
        clean_country_generators,
        clean_country_links,
        clean_country_storage_units,
        dirty_country_links,
    ) = _aggregate_resources(n, country_buses, grid_clean_techs, emitters)

    country_loads = n.loads.index[n.loads.bus.isin(country_buses)]

    clean_grid_gens = n.generators_t.p[clean_grid_generators.index].sum(axis=1)
    clean_grid_ls = -n.links_t.p1[clean_grid_links.index].sum(axis=1)
    clean_grid_sus = n.storage_units_t.p[clean_grid_storage_units.index].sum(axis=1)
    clean_grid_resources = clean_grid_gens + clean_grid_ls + clean_grid_sus
    dirty_grid_resources = -n.links_t.p1[dirty_grid_links.index].sum(axis=1)

    clean_country_gens = n.generators_t.p[clean_country_generators.index].sum(axis=1)
    clean_country_ls = -n.links_t.p1[clean_country_links.index].sum(axis=1)
    clean_country_sus = n.storage_units_t.p[clean_country_storage_units.index].sum(
        axis=1
    )
    clean_country_resources = clean_country_gens + clean_country_ls + clean_country_sus
    dirty_country_resources = -n.links_t.p1[dirty_country_links.index].sum(axis=1)

    country_import = _calculate_imports(n, location, name)

    grid_hourly_emissions = _calculate_emissions(n, dirty_grid_links)

    grid_emission_rate = grid_hourly_emissions / (
        clean_grid_resources + dirty_grid_resources
    )

    country_hourly_emissions = _calculate_emissions(n, dirty_country_links)
    grid_supply_emission_rate = (
        country_hourly_emissions + country_import * grid_emission_rate
    ) / (clean_country_resources + dirty_country_resources + country_import)

    ci_emissions_t = n.links_t.p0[f"{name} import"] * grid_supply_emission_rate

    # Compute CI emission rates: true, local (local bidding zone), myopic (ignoring network congestions)
    ci_load = n.loads_t.p[f"{name} load"].sum()
    country_load = n.loads_t.p[country_loads].sum(axis=1)

    ci_emission_rate_true = ci_emissions_t.sum() / ci_load
    ci_emission_rate_local = (
        n.links_t.p0[f"{name} import"] * (country_hourly_emissions / country_load)
    ).sum() / ci_load
    ci_emission_rate_myopic = (
        n.links_t.p0[f"{name} import"] * (hourly_emissions / load)
    ).sum() / ci_load

    # Compute total CO2 emissions
    ci_emissions = (ci_emissions_t * weights).sum()
    emissions_zone = (country_hourly_emissions * weights).sum() / 1e6

    # Update results dictionary
    results[location].update(
        {
            "ci_emission_rate_true": ci_emission_rate_true,
            "ci_emission_rate_local": ci_emission_rate_local,
            "ci_emission_rate_myopic": ci_emission_rate_myopic,
            "ci_emissions": ci_emissions,
            "emissions_zone": emissions_zone,
        }
    )

    # 4: Storing invested capacities at CI node
    results[location].update(
        {
            f"ci_cap_{tech}": n.generators.at[name + " " + tech, "p_nom_opt"]
            for tech in clean_techs
        }
    )

    results[location].update(
        {
            f"ci_cap_{charger.replace(' ', '_')}": (
                n.links.at[name + " " + charger, "p_nom_opt"]
                * n.links.loc[n.links.index.str.contains(charger), "efficiency"].iloc[0]
                if name + " " + charger in n.links.index
                else 0.0
            )
            for charger in storage_charge_techs
        }
    )

    results[location].update(
        {
            f"ci_cap_{discharger.replace(' ', '_')}": (
                n.links.at[name + " " + discharger, "p_nom_opt"]
                * n.links.loc[
                    n.links.index.str.contains(discharger), "efficiency"
                ].iloc[0]
                if name + " " + discharger in n.links.index
                else 0.0
            )
            for discharger in storage_discharge_techs
        }
    )

    # Storing generation at CI node
    results[location].update(
        {
            f"ci_generation_{tech}": n.generators_t.p[name + " " + tech]
            .multiply(weights, axis=0)
            .sum()
            for tech in clean_techs
        }
    )

    results[location].update(
        {
            f"ci_generation_{discharger.replace(' ', '_')}": (
                -n.links_t.p1[name + " " + discharger].multiply(weights, axis=0).sum()
                if name + " " + discharger in n.links.index
                else 0.0
            )
            for discharger in storage_discharge_techs
        }
    )

    # 5: Storing invested capacities in the rest of energy system
    gen_expandable = n.generators[n.generators.p_nom_extendable == True]
    gen_expandable = gen_expandable.drop(
        labels=snakemake.config["additional_nodes"], errors="ignore"
    )
    system_gens = gen_expandable[~gen_expandable.index.str.contains(name)]

    results[location].update(
        {
            f"system_inv_{gen.replace(' ', '_')}": (
                system_gens.loc[system_gens.index.str.contains(gen), "p_nom_opt"].sum()
                - system_gens.loc[system_gens.index.str.contains(gen), "p_nom"].sum()
            )
            for gen in exp_generators
        }
    )

    system_links = n.links[n.links.index.str.contains("|".join(exp_links))]
    results[location].update(
        {
            f"system_inv_{link}": (
                system_links.loc[
                    system_links.index.str.contains(link), "p_nom_opt"
                ].sum()
                - system_links.loc[system_links.index.str.contains(link), "p_nom"].sum()
            )
            * system_links.loc[
                system_links.index.str.contains(link), "efficiency"
            ].iloc[0]
            for link in exp_links
        }
    )

    # Chargers & Dischargers
    HV_links = n.links.drop(n.links[n.links.index.str.contains("home battery")].index)
    system_chargers = HV_links[
        HV_links.index.str.contains(f"battery charger-{year}|H2 Electrolysis-{year}")
    ]
    system_dischargers = HV_links[
        HV_links.index.str.contains(f"battery discharger-{year}|H2 Fuel Cell-{year}")
    ]

    results[location].update(
        {
            f"system_inv_{charger.replace(' ', '_')}": (
                system_chargers.loc[
                    system_chargers.index.str.contains(charger), "p_nom_opt"
                ].sum()
                - system_chargers.loc[
                    system_chargers.index.str.contains(charger), "p_nom"
                ].sum()
            )
            * system_chargers.loc[
                system_chargers.index.str.contains(charger), "efficiency"
            ].iloc[0]
            for charger in exp_chargers
        }
    )

    results[location].update(
        {
            f"system_inv_{discharger.replace(' ', '_')}": (
                system_dischargers.loc[
                    system_dischargers.index.str.contains(discharger), "p_nom_opt"
                ].sum()
                - system_dischargers.loc[
                    system_dischargers.index.str.contains(discharger), "p_nom"
                ].sum()
            )
            * system_dischargers.loc[
                system_dischargers.index.str.contains(discharger), "efficiency"
            ].iloc[0]
            for discharger in exp_dischargers
        }
    )

    # 6: Storing costs at CI node
    total_cost = 0.0
    for tech in clean_techs:
        tech_costs = _calculate_tech_cost(tech, location, name, n, results)
        results[location].update(tech_costs)
        total_cost += tech_costs[f"ci_cost_{tech}"]

    results[location].update(
        {
            "ci_capital_cost_grid": 0,
            "ci_marginal_cost_grid": (
                n.links_t.p0[name + " import"]
                * weights
                * n.buses_t.marginal_price[location]
            ).sum(),
        }
    )

    results[location].update(
        {
            "ci_cost_grid": (
                results[location]["ci_capital_cost_grid"]
                + results[location]["ci_marginal_cost_grid"]
            )
        }
    )

    total_cost += results[location]["ci_cost_grid"]

    ci_revenue_grid = (
        n.links_t.p0[name + " export"] * weights * n.buses_t.marginal_price[location]
    ).sum()

    results[location].update(
        {
            "ci_revenue_grid": ci_revenue_grid,
            "ci_average_revenue": ci_revenue_grid
            / results[location]["ci_demand_total"],
        }
    )

    battery_costs, hydrogen_costs = _calculate_storage_costs(storage_techs, name, n)
    results[location].update(battery_costs)
    results[location].update(hydrogen_costs)
    # Only add capital costs (or operational costs, as per your model's requirement) to the total_cost
    total_cost += (
        battery_costs["ci_cost_battery_storage"]
        + battery_costs["ci_cost_battery_inverter"]
    )
    total_cost += (
        hydrogen_costs["ci_cost_hydrogen_storage"]
        + hydrogen_costs["ci_cost_hydrogen_electrolysis"]
        + hydrogen_costs["ci_cost_hydrogen_fuel_cell"]
    )

    results[location]["ci_total_cost"] = total_cost
    results[location]["ci_average_cost"] = (
        total_cost / results[location]["ci_demand_total"]
    )

    # 7: Other calculations
    results[location].update(
        {
            "zone_average_marginal_price": n.buses_t.marginal_price[location].sum()
            / len(n.snapshots),
            "emissions": n.stores_t.e["co2 atmosphere"][-1],
            "system_grid_cfe_wavg": weighted_avg(
                grid_cfe, n.loads_t.p[grid_loads].sum(axis=1)
            ),
        }
    )

    # 8: Store RES curtailment
    system_res = n.generators[
        ~n.generators.index.str.contains("EU")
        & ~n.generators.carrier.str.contains("gas")
    ].carrier.unique()
    ci_res = snakemake.config["ci"]["res_techs"]

    pattern = "|".join(names)
    system_buses = n.buses.index[~n.buses.index.str.contains(pattern)]
    ci_buses = n.buses.index[n.buses.index.str.contains(pattern)]
    ci_bus = n.buses.index[n.buses.index.str.contains(name)]

    results[location].update(
        {
            f"system_curtailment_{tech}": _calculate_curtailment(
                n, tech, system_buses, weights
            )
            for tech in system_res
        }
    )

    results[location].update(
        {
            f"ci_curtailment_{tech}": _calculate_curtailment(n, tech, ci_bus, weights)
            for tech in ci_res
        }
    )

    policy_type = snakemake.config["global"]["policy_type"]
    results[location]["co2_price"] = (
        n.global_constraints.at["CO2Limit", "mu"]
        if policy_type == "co2 cap"
        else snakemake.config["global"][f"co2_price_{year}"]
    )

    # Convert all results to float and round to 2 decimal places
    results[location] = {k: round(float(v), 2) for k, v in results[location].items()}

    # Saving resutls as ../summaries/{}.yaml
    print(f"Summary for is completed! Saving to \n {snakemake.output.yaml}")
    with open(snakemake.output.yaml, "w") as outfile:
        yaml.dump(results, outfile)


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "summarise_network",
            year="2030",
            zone="IE",
            palette="p3",
            policy="cfe100",
            participation="10",
        )

    # Wildcards & Settings
    policy = snakemake.wildcards.policy[:3]
    penetration = float(snakemake.wildcards.policy[3:]) / 100 if policy != "ref" else 0
    tech_palette = snakemake.wildcards.palette
    zone = snakemake.wildcards.zone
    year = snakemake.wildcards.year
    participation = snakemake.wildcards.participation

    datacenters = snakemake.config["ci"]["datacenters"]
    locations = list(datacenters.keys())
    names = list(datacenters.values())
    flexibility = snakemake.config["ci"]["flexibility"]

    print(f"Summary for policy {policy} and penetration {penetration}")
    print(f"Summary for palette: {tech_palette}")
    print(f"Summary for bidding zone: {zone}")
    print(f"Summary for year: {year}")
    print(f"Summary for datacenters: {datacenters}")
    print(f"Summary for flexibility: {flexibility}")

    # Read data
    n = pypsa.Network(
        snakemake.input.network, override_component_attrs=override_component_attrs()
    )

    grid_cfe_df = pd.read_csv(
        snakemake.input.grid_cfe, index_col=0, parse_dates=True, header=[0, 1]
    )

    print(grid_cfe_df)

    summarise_network(n, policy, tech_palette)
