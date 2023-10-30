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


def summarise_network(n, policy, tech_palette):
    # techs for CFE hourly matching
    clean_techs = palette(tech_palette)[0]
    storage_techs = palette(tech_palette)[1]
    storage_charge_techs = palette(tech_palette)[2]
    storage_discharge_techs = palette(tech_palette)[3]

    n_iterations = snakemake.config["solving"]["options"]["n_iterations"]
    results = {}

    for location, name in datacenters.items():
        clean_gens = [name + " " + g for g in clean_techs]
        clean_dischargers = [name + " " + g for g in storage_discharge_techs]
        clean_chargers = [name + " " + g for g in storage_charge_techs]
        grid_buses = n.buses.index[~n.buses.index.str.contains(name)]
        grid_loads = n.loads.index[n.loads.bus.isin(grid_buses)]

        # for calculation of system expansion beyond CI node in pypsa-eur-sec myopic network for {year}
        exp_generators = [
            "offwind-ac-%s" % year,
            "offwind-dc-%s" % year,
            "onwind-%s" % year,
            "solar-%s" % year,
            "solar rooftop-%s" % year,
        ]
        exp_links = ["OCGT-%s" % year]
        exp_chargers = ["battery charger-%s" % year, "H2 Electrolysis-%s" % year]
        exp_dischargers = ["battery discharger-%s" % year, "H2 Fuel Cell-%s" % year]

        # Grid CFE hourly score [0,1]
        # grid_cfe_df.xs("iteration 2", axis=1, level=1)
        grid_cfe = grid_cfe_df.loc[:, (location, f"iteration {n_iterations-1}")]

        #######################################################################
        results[f"{location}"] = {}
        temp = {}
        #######################################################################

        # Processing, calculating and storing model results

        results[f"{location}"]["objective"] = n.objective
        weights = n.snapshot_weightings["generators"]

        # 1: Generation & imports at C&I node

        p_clean = n.generators_t.p[clean_gens].multiply(weights, axis=0).sum(axis=1)
        p_storage = -n.links_t.p1[clean_dischargers].multiply(weights, axis=0).sum(
            axis=1
        ) - n.links_t.p0[clean_chargers].multiply(weights, axis=0).sum(axis=1)
        p_demand = n.loads_t.p[f"{name} load"].multiply(weights, axis=0)

        p_diff = p_clean + p_storage - p_demand
        excess = p_diff.copy()
        excess[excess < 0] = 0.0

        used_local = p_clean + p_storage - excess

        used_grid = (
            -n.links_t.p1[name + " import"].multiply(weights, axis=0)
        ) * grid_cfe

        results[f"{location}"]["ci_clean_total"] = (p_clean + p_storage).sum()
        results[f"{location}"]["ci_clean_used_local_total"] = used_local.sum()
        results[f"{location}"]["ci_clean_used_grid_total"] = used_grid.sum()
        results[f"{location}"]["ci_clean_used_total"] = (
            used_grid.sum() + used_local.sum()
        )
        results[f"{location}"]["ci_clean_excess_total"] = excess.sum()
        results[f"{location}"]["ci_demand_total"] = p_demand.sum()

        results[f"{location}"]["ci_fraction_clean"] = (
            results[f"{location}"]["ci_clean_total"]
            / results[f"{location}"]["ci_demand_total"]
        )
        results[f"{location}"]["ci_fraction_clean_used_local"] = (
            results[f"{location}"]["ci_clean_used_local_total"]
            / results[f"{location}"]["ci_demand_total"]
        )
        results[f"{location}"]["ci_fraction_clean_used_grid"] = (
            results[f"{location}"]["ci_clean_used_grid_total"]
            / results[f"{location}"]["ci_demand_total"]
        )
        results[f"{location}"]["ci_fraction_clean_used"] = (
            results[f"{location}"]["ci_clean_used_total"]
            / results[f"{location}"]["ci_demand_total"]
        )
        results[f"{location}"]["ci_fraction_clean_excess"] = (
            results[f"{location}"]["ci_clean_excess_total"]
            / results[f"{location}"]["ci_demand_total"]
        )

        # 2: compute grid average emission rate

        emitters = snakemake.config["global"]["emitters"]

        fossil_links = n.links.index[n.links.carrier.isin(emitters)]
        hourly_emissions = (
            n.links_t.p0[fossil_links]
            .multiply(n.links.efficiency2[fossil_links], axis=1)
            .sum(axis=1)
        )
        load = n.loads_t.p[grid_loads].sum(axis=1)

        results[f"{location}"]["system_emissions"] = hourly_emissions.sum()
        results[f"{location}"]["system_emission_rate"] = (
            hourly_emissions.sum() / load.sum()
        )

        # 3: compute emissions & emission rates

        grid_clean_techs = snakemake.config["global"]["grid_clean_techs"]

        # NB: clean_techs (at C&I node) != grid_clean_techs (rest of system)
        # 3.1: ci_emission_rate based on Princeton (considering network congestions)

        rest_system_buses = n.buses.index[
            ~n.buses.index.str.contains(name) & ~n.buses.index.str.contains(location)
        ]
        country_buses = n.buses.index[n.buses.index.str.contains(location)]

        clean_grid_generators = n.generators.index[
            n.generators.bus.isin(rest_system_buses)
            & n.generators.carrier.isin(grid_clean_techs)
        ]
        clean_grid_links = n.links.index[
            n.links.bus1.isin(rest_system_buses)
            & n.links.carrier.isin(grid_clean_techs)
        ]
        clean_grid_storage_units = n.storage_units.index[
            n.storage_units.bus.isin(rest_system_buses)
            & n.storage_units.carrier.isin(grid_clean_techs)
        ]
        dirty_grid_links = n.links.index[
            n.links.bus1.isin(rest_system_buses) & n.links.carrier.isin(emitters)
        ]

        clean_country_generators = n.generators.index[
            n.generators.bus.isin(country_buses)
            & n.generators.carrier.isin(grid_clean_techs)
        ]
        clean_country_links = n.links.index[
            n.links.bus1.isin(country_buses) & n.links.carrier.isin(grid_clean_techs)
        ]
        clean_country_storage_units = n.storage_units.index[
            n.storage_units.bus.isin(country_buses)
            & n.storage_units.carrier.isin(grid_clean_techs)
        ]
        dirty_country_links = n.links.index[
            n.links.bus1.isin(country_buses) & n.links.carrier.isin(emitters)
        ]

        country_loads = n.loads.index[n.loads.bus.isin(country_buses)]

        clean_grid_gens = n.generators_t.p[clean_grid_generators].sum(axis=1)
        clean_grid_ls = -n.links_t.p1[clean_grid_links].sum(axis=1)
        clean_grid_sus = n.storage_units_t.p[clean_grid_storage_units].sum(axis=1)
        clean_grid_resources = clean_grid_gens + clean_grid_ls + clean_grid_sus
        dirty_grid_resources = -n.links_t.p1[dirty_grid_links].sum(axis=1)

        clean_country_gens = n.generators_t.p[clean_country_generators].sum(axis=1)
        clean_country_ls = -n.links_t.p1[clean_country_links].sum(axis=1)
        clean_country_sus = n.storage_units_t.p[clean_country_storage_units].sum(axis=1)
        clean_country_resources = (
            clean_country_gens + clean_country_ls + clean_country_sus
        )
        dirty_country_resources = -n.links_t.p1[dirty_country_links].sum(axis=1)

        line_imp_subsetA = n.lines_t.p1.loc[:, n.lines.bus0.str.contains(location)].sum(
            axis=1
        )
        line_imp_subsetB = n.lines_t.p0.loc[:, n.lines.bus1.str.contains(location)].sum(
            axis=1
        )
        line_imp_subsetA[line_imp_subsetA < 0] = 0.0
        line_imp_subsetB[line_imp_subsetB < 0] = 0.0

        links_imp_subsetA = n.links_t.p1.loc[
            :,
            n.links.bus0.str.contains(location)
            & (n.links.carrier == "DC")
            & ~(n.links.index.str.contains(name)),
        ].sum(axis=1)
        links_imp_subsetB = n.links_t.p0.loc[
            :,
            n.links.bus1.str.contains(location)
            & (n.links.carrier == "DC")
            & ~(n.links.index.str.contains(name)),
        ].sum(axis=1)
        links_imp_subsetA[links_imp_subsetA < 0] = 0.0
        links_imp_subsetB[links_imp_subsetB < 0] = 0.0

        country_import = (
            line_imp_subsetA + line_imp_subsetB + links_imp_subsetA + links_imp_subsetB
        )

        grid_hourly_emissions = (
            n.links_t.p0[dirty_grid_links]
            .multiply(n.links.efficiency2[dirty_grid_links], axis=1)
            .sum(axis=1)
        )

        grid_emission_rate = grid_hourly_emissions / (
            clean_grid_resources + dirty_grid_resources
        )

        country_hourly_emissions = (
            n.links_t.p0[dirty_country_links]
            .multiply(n.links.efficiency2[dirty_country_links], axis=1)
            .sum(axis=1)
        )

        grid_supply_emission_rate = (
            country_hourly_emissions + country_import * grid_emission_rate
        ) / (clean_country_resources + dirty_country_resources + country_import)

        ci_emissions_t = n.links_t.p0[f"{name} import"] * grid_supply_emission_rate

        results[f"{location}"]["ci_emission_rate_true"] = (
            ci_emissions_t.sum() / n.loads_t.p[f"{name} load"].sum()
        )

        # 3.2: considering only country node(local bidding zone)
        country_load = n.loads_t.p[country_loads].sum(axis=1)
        emissions_factor_local = country_hourly_emissions / country_load
        results[f"{location}"]["ci_emission_rate_local"] = (
            n.links_t.p0[f"{name} import"] * emissions_factor_local
        ).sum() / n.loads_t.p[f"{name} load"].sum()

        # 3.3: Our original ci_emission_rate (ignoring network congestions)
        emissions_factor = hourly_emissions / load  # global average emissions
        results[f"{location}"]["ci_emission_rate_myopic"] = (
            n.links_t.p0[f"{name} import"] * emissions_factor
        ).sum() / n.loads_t.p[f"{name} load"].sum()

        # 3.3: Total CO2 emissions for 24/7 participating customers
        results[f"{location}"]["ci_emissions"] = (ci_emissions_t * weights).sum()

        # 3.4: Total CO2 emissions in zone where 24/7 participating customers are located
        results[f"{location}"]["emissions_zone"] = (
            country_hourly_emissions * weights
        ).sum() / 1e6

        # 4: Storing invested capacities at CI node

        for tech in clean_techs:
            results[f"{location}"]["ci_cap_" + tech] = n.generators.at[
                name + " " + tech, "p_nom_opt"
            ]

        for charger in storage_charge_techs:
            temp["eff_" + charger] = n.links.loc[
                n.links.index.str.contains(f"{charger}"), "efficiency"
            ][0]
            if name + " " + charger in n.links.index:
                results[f"{location}"]["ci_cap_" + charger.replace(" ", "_")] = (
                    n.links.at[name + " " + charger, "p_nom_opt"]
                    * temp["eff_" + charger]
                )
            else:
                results[f"{location}"]["ci_cap_" + charger.replace(" ", "_")] = 0.0

        for discharger in storage_discharge_techs:
            temp["eff_" + discharger] = n.links.loc[
                n.links.index.str.contains(f"{discharger}"), "efficiency"
            ][0]
            if name + " " + discharger in n.links.index:
                results[f"{location}"]["ci_cap_" + discharger.replace(" ", "_")] = (
                    n.links.at[name + " " + discharger, "p_nom_opt"]
                    * temp["eff_" + discharger]
                )
            else:
                results[f"{location}"]["ci_cap_" + discharger.replace(" ", "_")] = 0.0

        # Storing generation at CI node

        for tech in clean_techs:
            results[f"{location}"]["ci_generation_" + tech] = (
                n.generators_t.p[name + " " + tech].multiply(weights, axis=0).sum()
            )

        for discharger in storage_discharge_techs:
            if name + " " + discharger in n.links.index:
                results[f"{location}"][
                    "ci_generation_" + discharger.replace(" ", "_")
                ] = (
                    -n.links_t.p1[name + " " + discharger]
                    .multiply(weights, axis=0)
                    .sum()
                )
            else:
                results[f"{location}"][
                    "ci_generation_" + discharger.replace(" ", "_")
                ] = 0.0

        # 5: Storing invested capacities in the rest of energy system

        # Generators
        gen_expandable = n.generators[
            n.generators.p_nom_extendable == True
        ]  # all gens that can be expanded

        for g in snakemake.config["additional_nodes"]:  # drop additional nodes
            if g in gen_expandable.index:
                gen_expandable = gen_expandable.drop(g)

        system_gens = gen_expandable[
            ~gen_expandable.index.str.contains(f"{name}")
        ]  # drop google gens

        for gen in exp_generators:
            temp["system_optcap_" + gen] = system_gens.loc[
                system_gens.index.str.contains(f"{gen}"), "p_nom_opt"
            ].sum()
            results[f"{location}"]["system_inv_" + gen.replace(" ", "_")] = (
                temp["system_optcap_" + gen]
                - system_gens.loc[
                    system_gens.index.str.contains(f"{gen}"), "p_nom"
                ].sum()
            )

        # Links
        for link in exp_links:
            system_links = n.links[n.links.index.str.contains(f"{link}")]

        for link in exp_links:
            temp["system_optcap_" + link] = system_links.loc[
                system_links.index.str.contains(f"{link}"), "p_nom_opt"
            ].sum()
            temp["eff_" + link] = system_links.loc[
                system_links.index.str.contains(f"{link}"), "efficiency"
            ][0]
            results[f"{location}"]["system_inv_" + link] = (
                temp["system_optcap_" + link]
                - system_links.loc[
                    system_links.index.str.contains(f"{link}"), "p_nom"
                ].sum()
            )
            # capacity of PyPSA-eur-sec links are in MWth. Converting MWth -> MWel.
            results[f"{location}"]["system_inv_" + link] *= temp["eff_" + link]

        # Chargers & Dischargers
        HV_links = n.links
        for l in n.links[n.links.index.str.contains("home battery")].index:
            HV_links = HV_links.drop(l)  # remove low voltage batteries

        batteries = HV_links[
            HV_links.index.str.contains(f"battery charger" + "-{}".format(year))
        ]
        electrolysis = HV_links[
            HV_links.index.str.contains(f"H2 Electrolysis" + "-{}".format(year))
        ]
        system_chargers = pd.concat([batteries, electrolysis])

        inverters = HV_links[
            HV_links.index.str.contains(f"battery discharger" + "-{}".format(year))
        ]
        fuelcells = HV_links[
            HV_links.index.str.contains(f"H2 Fuel Cell" + "-{}".format(year))
        ]
        system_dischargers = pd.concat([inverters, fuelcells])

        for charger in exp_chargers:
            temp["system_optcap_" + charger] = system_chargers.loc[
                system_chargers.index.str.contains(f"{charger}"), "p_nom_opt"
            ].sum()
            temp["eff_" + charger] = system_chargers.loc[
                system_chargers.index.str.contains(f"{charger}"), "efficiency"
            ][0]
            results[f"{location}"]["system_inv_" + charger.replace(" ", "_")] = (
                temp["system_optcap_" + charger]
                - system_chargers.loc[
                    system_chargers.index.str.contains(f"{charger}"), "p_nom"
                ].sum()
            )
            results[f"{location}"]["system_inv_" + charger.replace(" ", "_")] *= temp[
                "eff_" + charger
            ]

        for discharger in exp_dischargers:
            temp["system_optcap_" + discharger] = system_dischargers.loc[
                system_dischargers.index.str.contains(f"{discharger}"), "p_nom_opt"
            ].sum()
            temp["eff_" + discharger] = system_dischargers.loc[
                system_dischargers.index.str.contains(f"{discharger}"), "efficiency"
            ][0]
            results[f"{location}"]["system_inv_" + discharger.replace(" ", "_")] = (
                temp["system_optcap_" + discharger]
                - system_dischargers.loc[
                    system_dischargers.index.str.contains(f"{discharger}"), "p_nom"
                ].sum()
            )
            results[f"{location}"][
                "system_inv_" + discharger.replace(" ", "_")
            ] *= temp["eff_" + discharger]

        # 6: Storing costs at CI node

        total_cost = 0.0
        for tech in clean_techs:
            results[f"{location}"]["ci_capital_cost_" + tech] = (
                results[f"{location}"]["ci_cap_" + tech]
                * n.generators.at[name + " " + tech, "capital_cost"]
            )
            results[f"{location}"]["ci_marginal_cost_" + tech] = (
                results[f"{location}"]["ci_generation_" + tech]
                * n.generators.at[name + " " + tech, "marginal_cost"]
            )
            cost = (
                results[f"{location}"]["ci_capital_cost_" + tech]
                + results[f"{location}"]["ci_marginal_cost_" + tech]
            )
            results[f"{location}"]["ci_cost_" + tech] = cost
            total_cost += cost

        results[f"{location}"]["ci_capital_cost_grid"] = 0
        results[f"{location}"]["ci_marginal_cost_grid"] = (
            n.links_t.p0[name + " import"]
            * weights
            * n.buses_t.marginal_price[location]
        ).sum()
        cost = (
            results[f"{location}"]["ci_capital_cost_grid"]
            + results[f"{location}"]["ci_marginal_cost_grid"]
        )
        results[f"{location}"]["ci_cost_grid"] = cost
        total_cost += cost

        # check_average_cost_grid = results[f'{location}']['ci_cost_grid'] / results[f'{location}']['ci_demand_total']
        results[f"{location}"]["ci_revenue_grid"] = (
            n.links_t.p0[name + " export"]
            * weights
            * n.buses_t.marginal_price[location]
        ).sum()
        results[f"{location}"]["ci_average_revenue"] = (
            results[f"{location}"]["ci_revenue_grid"]
            / results[f"{location}"]["ci_demand_total"]
        )

        if "battery" in storage_techs:
            results[f"{location}"]["ci_capital_cost_battery_storage"] = (
                n.stores.at[f"{name} battery", "e_nom_opt"]
                * n.stores.at[f"{name} battery", "capital_cost"]
            )
            results[f"{location}"]["ci_cost_battery_storage"] = results[f"{location}"][
                "ci_capital_cost_battery_storage"
            ]
            total_cost += results[f"{location}"]["ci_cost_battery_storage"]

            results[f"{location}"]["ci_capital_cost_battery_inverter"] = (
                n.links.at[f"{name} battery charger", "p_nom_opt"]
                * n.links.at[f"{name} battery charger", "capital_cost"]
            )
            results[f"{location}"]["ci_cost_battery_inverter"] = results[f"{location}"][
                "ci_capital_cost_battery_inverter"
            ]
            total_cost += results[f"{location}"]["ci_cost_battery_inverter"]
        else:
            results[f"{location}"]["ci_capital_cost_battery_storage"] = 0.0
            results[f"{location}"]["ci_cost_battery_storage"] = 0.0
            results[f"{location}"]["ci_capital_cost_battery_inverter"] = 0.0
            results[f"{location}"]["ci_cost_battery_inverter"] = 0.0

        if "hydrogen" in storage_techs:
            results[f"{location}"]["ci_capital_cost_hydrogen_storage"] = (
                n.stores.at[f"{name} H2 Store", "e_nom_opt"]
                * n.stores.at[f"{name} H2 Store", "capital_cost"]
            )
            results[f"{location}"]["ci_cost_hydrogen_storage"] = results[f"{location}"][
                "ci_capital_cost_hydrogen_storage"
            ]
            total_cost += results[f"{location}"]["ci_cost_hydrogen_storage"]

            results[f"{location}"]["ci_capital_cost_hydrogen_electrolysis"] = (
                n.links.at[f"{name} H2 Electrolysis", "p_nom_opt"]
                * n.links.at[f"{name} H2 Electrolysis", "capital_cost"]
            )
            results[f"{location}"]["ci_cost_hydrogen_electrolysis"] = results[
                f"{location}"
            ]["ci_capital_cost_hydrogen_electrolysis"]
            total_cost += results[f"{location}"]["ci_cost_hydrogen_electrolysis"]

            results[f"{location}"]["ci_capital_cost_hydrogen_fuel_cell"] = (
                n.links.at[f"{name} H2 Fuel Cell", "p_nom_opt"]
                * n.links.at[f"{name} H2 Fuel Cell", "capital_cost"]
            )
            results[f"{location}"]["ci_cost_hydrogen_fuel_cell"] = results[
                f"{location}"
            ]["ci_capital_cost_hydrogen_fuel_cell"]
            total_cost += results[f"{location}"]["ci_cost_hydrogen_fuel_cell"]
        else:
            results[f"{location}"]["ci_capital_cost_hydrogen_storage"] = 0.0
            results[f"{location}"]["ci_cost_hydrogen_storage"] = 0.0
            results[f"{location}"]["ci_capital_cost_hydrogen_electrolysis"] = 0.0
            results[f"{location}"]["ci_cost_hydrogen_electrolysis"] = 0.0
            results[f"{location}"]["ci_capital_cost_hydrogen_fuel_cell"] = 0.0
            results[f"{location}"]["ci_cost_hydrogen_fuel_cell"] = 0.0

        results[f"{location}"]["ci_total_cost"] = total_cost
        results[f"{location}"]["ci_average cost"] = (
            results[f"{location}"]["ci_total_cost"]
            / results[f"{location}"]["ci_demand_total"]
        )

        # 7: Other calculations

        results[f"{location}"][
            "zone_average_marginal_price"
        ] = n.buses_t.marginal_price[location].sum() / len(n.snapshots)

        # Storing system emissions and co2 price
        results[f"{location}"]["emissions"] = n.stores_t.e["co2 atmosphere"][-1]

        # Compute weighted average grid CFE
        system_grid_cfe_wavg = weighted_avg(
            grid_cfe, n.loads_t.p[grid_loads].sum(axis=1)
        )
        results[f"{location}"]["system_grid_cfe_wavg"] = system_grid_cfe_wavg
        # print(system_grid_cfe_wavg) #print(grid_cfe.mean())

        # 8: Store RES curtailment
        system_res = n.generators[
            ~n.generators.index.str.contains("EU")
        ].carrier.unique()
        ci_res = snakemake.config["ci"]["res_techs"]

        # Combine DC names using a regular expression pattern
        pattern = "|".join(names)
        system_buses = n.buses.index[~n.buses.index.str.contains(pattern)]
        ci_buses = n.buses.index[n.buses.index.str.contains(pattern)]
        ci_bus = n.buses.index[n.buses.index.str.contains(name)]

        for tech in system_res:
            gens = n.generators.query("carrier == @tech and bus in @system_buses").index
            results[f"{location}"]["system_curtailment_" + tech] = (
                (
                    n.generators_t.p_max_pu[gens] * n.generators.p_nom_opt[gens]
                    - n.generators_t.p[gens]
                )
                .clip(lower=0)
                .multiply(weights, axis=0)
                .sum()
                .sum()
            )

        for tech in ci_res:
            gens = n.generators.query("carrier == @tech and bus in @ci_bus").index
            results[f"{location}"]["ci_curtailment_" + tech] = (
                (
                    n.generators_t.p_max_pu[gens] * n.generators.p_nom_opt[gens]
                    - n.generators_t.p[gens]
                )
                .clip(lower=0)
                .multiply(weights, axis=0)
                .sum()
                .sum()
            )

        if snakemake.config["global"]["policy_type"] == "co2 cap":
            results[f"{location}"]["co2_price"] = n.global_constraints.at[
                "CO2Limit", "mu"
            ]
        elif snakemake.config["global"]["policy_type"] == "co2 price":
            results[f"{location}"]["co2_price"] = snakemake.config["global"][
                f"co2_price_{year}"
            ]

        for k in results[f"{location}"]:
            results[f"{location}"][k] = float(results[f"{location}"][k])

    # 8: Saving resutls as ../summaries/{}.yaml

    print(f"Summary for is completed! Saving to \n {snakemake.output.yaml}")
    with open(snakemake.output.yaml, "w") as outfile:
        yaml.dump(results, outfile)


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "summarise_network",
            year="2025",
            zone="IEDK",
            palette="p1",
            policy="cfe100",
            flexibility="0",
        )

    # Wildcards & Settings
    policy = snakemake.wildcards.policy[:3]
    penetration = float(snakemake.wildcards.policy[3:]) / 100 if policy != "ref" else 0
    tech_palette = snakemake.wildcards.palette
    zone = snakemake.wildcards.zone
    year = snakemake.wildcards.year

    datacenters = snakemake.config["ci"]["datacenters"]
    locations = list(datacenters.keys())
    names = list(datacenters.values())
    flexibility = snakemake.wildcards.flexibility

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
