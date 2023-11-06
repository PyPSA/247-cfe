# SPDX-FileCopyrightText: 2023 Iegor Riepin, Tom Brown
#
# SPDX-License-Identifier: MIT

import pypsa
import numpy as np, pandas as pd
import sys
import os

import logging

logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)

from vresutils.costdata import annuity
from vresutils.benchmark import memory_logger
from _helpers import override_component_attrs

from typing import Dict, List, Tuple, Any


def palette(tech_palette: str) -> tuple:
    """
    Define technology palette available for C&I clean energy buyers

    Args:
    tech_palette (str): The technology palette to use based on config setting.

    Returns:
    tuple: A tuple containing the available clean technologies, storage technologies, storage chargers, and storage dischargers.
    """

    palettes = {
        "p1": {
            "clean_techs": ["onwind", "solar"],
            "storage_techs": ["battery"],
            "storage_chargers": ["battery charger"],
            "storage_dischargers": ["battery discharger"],
        },
        "p2": {
            "clean_techs": ["onwind", "solar"],
            "storage_techs": ["battery", "hydrogen"],
            "storage_chargers": ["battery charger", "H2 Electrolysis"],
            "storage_dischargers": ["battery discharger", "H2 Fuel Cell"],
        },
        "p3": {
            "clean_techs": ["onwind", "solar", "allam_ccs"],
            "storage_techs": ["battery"],
            "storage_chargers": ["battery charger"],
            "storage_dischargers": ["battery discharger"],
        },
    }

    if tech_palette not in palettes:
        print(
            f"'palette' wildcard must be one of 'p1', 'p2' or 'p3'. Now is {tech_palette}."
        )
        sys.exit()

    return tuple(
        palettes[tech_palette][key]
        for key in [
            "clean_techs",
            "storage_techs",
            "storage_chargers",
            "storage_dischargers",
        ]
    )


def geoscope(zone: str):
    """
    Returns a dictionary containing the geographical scope of the model based on the given zone.

    Args:
    - zone (str): controls basenodes_to_keep list -> sets geographical scope of the model

    Returns:
    - A dictionary containing the following keys:
        - basenodes_to_keep (List[str]): list of bus IDs representing the geographical scope of the model
        - country_nodes (Dict[str, List[str]]): dictionary containing bus IDs of countries subject to national RES policy constraints

    NB zone is used as a wildcard, while area as a switcher option.
    """
    # A few toy regional networks for test & play purposes
    IRELAND = ["IE5 0", "GB0 0", "GB5 0"]
    GERMANY = [
        "DE1 0",
        "BE1 0",
        "NO2 0",
        "DK1 0",
        "DK2 0",
        "SE2 0",
        "GB0 0",
        "FR1 0",
        "LU1 0",
        "NL1 0",
        "PL1 0",
        "AT1 0",
        "CH1 0",
        "CZ1 0",
    ]
    DKDE = ["DE1 0", "DK1 0", "DK2 0", "PL1 0"]
    IEDK = (
        IRELAND
        + ["DK1 0", "DK2 0"]
        + ["FR1 0", "LU1 0", "DE1 0", "BE1 0", "NL1 0", "NO2 0", "SE2 0"]
    )

    # Full geographical scope
    EU = n.buses[n.buses["carrier"] == "AC"].index.tolist()

    basenodes_to_keep = {
        "IE": IRELAND,
        "DE": GERMANY,
        "IEDK": IEDK,
        "DKDE": DKDE,
        "EU": EU,
    }.get(zone)

    if not basenodes_to_keep:
        print(f"'zone' wildcard cannot be {zone}.")
        sys.exit()

    country_nodes = {
        "IE5 0": ["IE5 0"],
        "DK1 0": ["DK1 0", "DK2 0"],
        "DE1 0": ["DE1 0"],
        "NL1 0": ["NL1 0"],
        "GB0 0": ["GB0 0", "GB5 0"],
        "FR1 0": ["FR1 0"],
    }
    country_nodes = {k: v for k, v in country_nodes.items() if k in basenodes_to_keep}

    return {"basenodes_to_keep": basenodes_to_keep, "country_nodes": country_nodes}


def timescope(year: str) -> Dict[str, str]:
    """
    Args:
    - year (str): the year of optimisation based on config setting

    Returns:
    - A dictionary with the following keys:
        - "coal_phaseout": the list of countries that implement coal phase-out policy for the given year.
        - "network_file": the path to the input file with the pypsa-eur brownfield network.
        - "costs_projection": the path to the input file with technology costs for the given year.
    """
    return {
        "network_file": snakemake.input.network,
        "costs_projection": snakemake.input.costs,
    }


def cost_parametrization(n, config) -> None:
    """
    Overwrites default price assumptions for primary energy carriers only for virtual generators located in 'EU {carrier}' buses.

    Args:
    - n: a PyPSA network object.
    - config: config.yaml settings

    Returns:
    - None
    """

    carriers = ["lignite", "coal", "gas"]
    prices = [config["costs"][f"price_{carrier}"] for carrier in carriers]

    n.generators.loc[
        n.generators.index.str.contains(f"EU {'|'.join(carriers)}"), "marginal_cost"
    ] = prices
    n.generators.loc[n.generators.carrier == "onwind", "marginal_cost"] = 0.015


def load_profile(
    n: pypsa.Network,
    profile_shape: str,
    config: Dict[str, Any],
) -> pd.Series:
    """
    Create daily load profile for C&I buyers based on config setting.

    Args:
    - n (object): object
    - profile_shape (str): shape of the load profile, must be one of 'baseload' or 'industry'
    - config (dict): config settings

    Returns:
    - pd.Series: annual load profile for C&I buyers
    """

    scaling = int(config["time_sampling"][0])  # 3/1 for 3H/1H

    shapes = {
        "baseload": [1 / 24] * 24,
        "industry": [0.009] * 5
        + [0.016, 0.031, 0.07, 0.072, 0.073, 0.072, 0.07]
        + [0.052, 0.054, 0.066, 0.07, 0.068, 0.063]
        + [0.035] * 2
        + [0.045] * 2
        + [0.009],
    }

    try:
        shape = shapes[profile_shape]
    except KeyError:
        print(
            f"'profile_shape' option must be one of 'baseload' or 'industry'. Now is {profile_shape}."
        )
        sys.exit()

    # CI consumer nominal load in MW
    load = config["ci_load"][zone] * float(participation) / 100

    load_day = load * 24  # 24h
    load_profile_day = pd.Series(shape) * load_day

    if scaling == 3:
        load_profile_day = load_profile_day.groupby(
            np.arange(len(load_profile_day)) // 3
        ).mean()  # 3H sampling

    load_profile_year = pd.concat([load_profile_day] * 365)
    profile = load_profile_year.set_axis(n.snapshots)

    return profile


def prepare_costs(
    cost_file: str,
    USD_to_EUR: float,
    discount_rate: float,
    lifetime: int,
    year: str,
    config: Dict[str, Any],
    Nyears: int = 1,
) -> pd.DataFrame:
    """
    Reads in a cost file and prepares the costs for use in the model.

    Args:
    - cost_file (str): path to the cost file
    - USD_to_EUR (float): conversion rate from USD to EUR
    - discount_rate (float): discount rate to use for calculating annuity factor
    - Nyears (int): number of years to run the model
    - lifetime (float): lifetime of the asset in years
    - year (int): year of the model run

    Returns:
    - costs (pd.DataFrame): a DataFrame containing the prepared costs
    """

    # set all asset costs and other parameters
    costs = pd.read_csv(cost_file, index_col=[0, 1]).sort_index()

    # correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
    costs.loc[costs.unit.str.contains("USD"), "value"] *= USD_to_EUR

    # min_count=1 is important to generate NaNs which are then filled by fillna
    costs = (
        costs.loc[:, "value"]
        .unstack(level=1)
        .groupby("technology")
        .sum(min_count=1)
        .fillna(
            {
                "CO2 intensity": 0,
                "FOM": 0,
                "VOM": 0,
                "discount rate": discount_rate,
                "efficiency": 1,
                "fuel": 0,
                "investment": 0,
                "lifetime": lifetime,
            }
        )
    )

    # Advanced nuclear
    data_nuc = pd.Series(
        {
            "CO2 intensity": 0,
            "FOM": costs.loc["nuclear"]["FOM"],
            "VOM": costs.loc["nuclear"]["VOM"],
            "discount rate": discount_rate,
            "efficiency": 0.36,
            "fuel": costs.loc["nuclear"]["fuel"],
            "investment": config["costs"]["adv_nuclear_overnight"]
            * 1e3
            * config["costs"]["USD2021_to_EUR2021"],
            "lifetime": 40.0,
        },
        name="adv_nuclear",
    )

    # Advanced geothermal
    adv_geo_overnight = config["costs"][f"adv_geo_overnight_{year}"]
    data_geo = pd.Series(
        {
            "CO2 intensity": 0,
            "FOM": 0,
            "VOM": 0,
            "discount rate": discount_rate,
            "efficiency": 1,
            "fuel": 0,
            "investment": adv_geo_overnight * 1e3 * 1,
            "lifetime": 30.0,
        },
        name="adv_geothermal",
    )

    # Allam cycle ccs
    allam_ccs_overnight = config["costs"][f"allam_ccs_overnight_{year}"]
    data_allam = pd.Series(
        {
            "CO2 intensity": 0,
            "FOM": 0,  # %/year
            "FOM-abs": 33000,  # $/MW-yr
            "VOM": 3.2,  # EUR/MWh
            "co2_seq": 40,  # $/ton
            "discount rate": discount_rate,
            "efficiency": 0.54,
            "fuel": config["costs"]["price_gas"],
            "investment": allam_ccs_overnight * 1e3 * 1,
            "lifetime": 30.0,
        },
        name="allam_ccs",
    )

    tech_list = [data_nuc, data_geo, data_allam]
    for tech in tech_list:
        costs = pd.concat([costs, tech.to_frame().transpose()], ignore_index=False)

    annuity_factor = (
        lambda v: annuity(v["lifetime"], v["discount rate"]) + v["FOM"] / 100
    )
    costs["fixed"] = [
        annuity_factor(v) * v["investment"] * Nyears for _, v in costs.iterrows()
    ]

    return costs


def strip_network(n, config) -> None:
    """
    Removes unnecessary components from a pypsa network.

    Args:
    - n (pypsa.Network): The network object to be stripped.

    Returns:
    - None
    """
    nodes_to_keep = geoscope(zone)["basenodes_to_keep"]

    new_nodes = [
        f"{b} {s}" for b in nodes_to_keep for s in config["node_suffixes_to_keep"]
    ]

    nodes_to_keep.extend(new_nodes)
    nodes_to_keep.extend(config["additional_nodes"])

    n.mremove("Bus", n.buses.index.symmetric_difference(nodes_to_keep))

    # make sure lines are kept
    n.lines.carrier = "AC"

    carrier_to_keep = config["carrier_to_keep"]

    for c in n.iterate_components(
        ["Generator", "Link", "Line", "Store", "StorageUnit", "Load"]
    ):
        if c.name in ["Link", "Line"]:
            location_boolean = c.df.bus0.isin(nodes_to_keep) & c.df.bus1.isin(
                nodes_to_keep
            )
        else:
            location_boolean = c.df.bus.isin(nodes_to_keep)
        to_keep = c.df.index[location_boolean & c.df.carrier.isin(carrier_to_keep)]
        to_drop = c.df.index.symmetric_difference(to_keep)
        n.mremove(c.name, to_drop)


def shutdown_lineexp(n: pypsa.Network) -> None:
    """
    Removes the option to expand lines and DC links.

    Args:
    - n (pypsa.Network): The network object to be modified.

    Returns:
    - None
    """
    n.lines.s_nom_extendable = False
    n.links.loc[n.links.carrier == "DC", "p_nom_extendable"] = False


def limit_resexp(n: pypsa.Network, year: str, config: Dict[str, Any]) -> None:
    """
    Limit expansion of renewable technologies per zone and carrier type
    as a ratio of max increase to 2021 capacity fleet (additional to zonal place availability constraint)

    Args:
        n: The network object to be modified.
        year: The year of optimisation based on config setting.
        config: config.yaml settings

    Returns:
        None
    """
    ratio = config["global"][f"limit_res_exp_{year}"]

    datacenters = config["ci"]["datacenters"]
    list_datacenters = list(datacenters.values())
    mask_datacenters = n.generators.index.str.contains(
        "|".join(list_datacenters), case=False
    )
    system_gens = n.generators[~mask_datacenters]

    fleet = system_gens.groupby(
        [system_gens.bus.str[:2], system_gens.carrier]
    ).p_nom.sum()
    fleet = fleet.rename(lambda x: x.split("-")[0], level=1).groupby(level=[0, 1]).sum()
    ct_national_target = list(config[f"res_target_{year}"].keys()) + ["EU"]

    fleet.drop(ct_national_target, errors="ignore", level=0, inplace=True)

    for ct, carrier in fleet.index:
        gen_i = (
            (n.generators.p_nom_extendable)
            & (n.generators.bus.str[:2] == ct)
            & (n.generators.carrier.str.contains(carrier))
        )
        n.generators.loc[gen_i, "p_nom_max"] = ratio * fleet.loc[ct, carrier]


def nuclear_policy(n: pypsa.Network, config: Dict[str, Any]) -> None:
    """
    Remove nuclear power plant fleet for countries with nuclear ban policy.

    Args:
    - n: The network object to be modified.
    - config: config.yaml settings
    -> nuclear_phaseout: List of countries with nuclear ban policy.

    Returns:
    - None
    """
    countries = config["nuclear_phaseout"]
    for country in countries:
        n.links.loc[
            (n.links["bus1"].str.contains(country))
            & (n.links.index.str.contains("nuclear")),
            "p_nom",
        ] = 0


def coal_policy(n: pypsa.Network, year: str, config: Dict[str, Any]) -> None:
    """
    Remove coal power plant fleet for countries with coal phase-out policy for {year}.

    Args:
    - n: The network object to be modified.
    - year: The year of optimisation (i.e. year for which the coal phase-out policy is in effect).
    - config: config.yaml settings

    Returns:
    - None
    """
    countries = config[f"coal_phaseout_{year}"]
    coal_links = n.links.index.str.contains("coal")
    lignite_links = n.links.index.str.contains("lignite")

    for country in countries:
        n.links.loc[n.links["bus1"].str.contains(country) & coal_links, "p_nom"] = 0
        n.links.loc[n.links["bus1"].str.contains(country) & lignite_links, "p_nom"] = 0


def biomass_potential(n: pypsa.Network) -> None:
    """
    Remove solid biomass demand for industrial processes from overall biomass potential.

    Args:
    - n: pypsa network to modify.

    Returns:
    - None
    """
    n.stores.loc[n.stores.index == "EU solid biomass", "e_nom"] *= 0.45
    n.stores.loc[n.stores.index == "EU solid biomass", "e_initial"] *= 0.45


def co2_policy(n, year: str, config: Dict[str, Any]) -> None:
    """
    Set EU carbon emissions policy as cap or price, update costs.

    Args:
    - n: a pypsa network object
    - year: a year of optimisation based on config setting
    - config: config.yaml settings

    Returns:
    - None
    """
    gl_policy = config["global"]["policy_type"]

    if gl_policy == "co2 cap":
        co2_cap = config["global"]["co2_share"] * config["global"]["co2_baseline"]
        n.global_constraints.at["CO2Limit", "constant"] = co2_cap
        print(f"Setting global CO2 cap to {co2_cap}")

    elif gl_policy == "co2 price":
        n.global_constraints.drop("CO2Limit", inplace=True)
        co2_price = config["global"][f"co2_price_{year}"]
        print(f"Setting CO2 price to {co2_price}")
        for carrier in ["coal", "oil", "gas", "lignite"]:
            n.generators.at[f"EU {carrier}", "marginal_cost"] += (
                co2_price * costs.at[carrier, "CO2 intensity"]
            )


def add_ci(n: pypsa.Network, year: str) -> None:
    """
    Add C&I buyer(s) to the network.

    Args:
    - n: pypsa.Network to which the C&I buyer(s) will be added.
    year: the year of optimisation based on config setting.

    Returns:
    - None
    """
    # tech_palette options
    clean_techs = palette(tech_palette)[0]
    storage_techs = palette(tech_palette)[1]

    for location, name in datacenters.items():
        n.add("Bus", name)

        n.add(
            "Link",
            f"{name}" + " export",
            bus0=name,
            bus1=location,
            marginal_cost=0.1,  # large enough to avoid optimization artifacts, small enough not to influence PPA portfolio
            p_nom=1e6,
        )

        n.add(
            "Link",
            f"{name}" + " import",
            bus0=location,
            bus1=name,
            marginal_cost=0.001,  # large enough to avoid optimization artifacts, small enough not to influence PPA portfolio
            p_nom=1e6,
        )

        n.add(
            "Load",
            f"{name}" + " load",
            carrier="electricity",
            bus=name,
            p_set=load_profile(n, profile_shape, config),
        )

        # C&I following voluntary clean energy procurement is a share of C&I load -> subtract it from node's profile
        n.loads_t.p_set[location] -= n.loads_t.p_set[f"{name}" + " load"]

        # Add clean firm advanced generators
        if "green hydrogen OCGT" in clean_techs:
            n.add(
                "Generator",
                f"{name} green hydrogen OCGT",
                carrier="green hydrogen OCGT",
                bus=name,
                p_nom_extendable=True if policy == "cfe" else False,
                capital_cost=costs.at["OCGT", "fixed"],
                marginal_cost=costs.at["OCGT", "VOM"]
                + snakemake.config["costs"]["price_green_hydrogen"]
                / 0.033
                / costs.at["OCGT", "efficiency"],
            )
            # hydrogen cost in EUR/kg, 0.033 MWhLHV/kg

        if "adv_nuclear" in clean_techs:
            n.add(
                "Generator",
                f"{name} adv_nuclear",
                bus=name,
                carrier="nuclear",
                capital_cost=costs.loc["adv_nuclear"]["fixed"],
                marginal_cost=costs.loc["adv_nuclear"]["VOM"]
                + costs.loc["adv_nuclear"]["fuel"]
                / costs.loc["adv_nuclear"]["efficiency"],
                p_nom_extendable=True if policy == "cfe" else False,
                lifetime=costs.loc["adv_nuclear"]["lifetime"],
            )

        if "allam_ccs" in clean_techs:
            n.add(
                "Generator",
                f"{name} allam_ccs",
                bus=name,
                carrier="gas",
                capital_cost=costs.loc["allam_ccs"]["fixed"]
                + costs.loc["allam_ccs"]["FOM-abs"],
                marginal_cost=costs.loc["allam_ccs"]["VOM"]
                + costs.loc["allam_ccs"]["fuel"] / costs.loc["allam_ccs"]["efficiency"]
                + costs.loc["allam_ccs"]["co2_seq"]
                * costs.at["gas", "CO2 intensity"]
                / costs.loc["allam_ccs"]["efficiency"],
                p_nom_extendable=True if policy == "cfe" else False,
                lifetime=costs.loc["allam_ccs"]["lifetime"],
                efficiency=costs.loc["allam_ccs"]["efficiency"],
            )

        if "adv_geothermal" in clean_techs:
            n.add(
                "Generator",
                f"{name} adv_geothermal",
                bus=name,
                # carrier = '',
                capital_cost=costs.loc["adv_geothermal"]["fixed"],
                marginal_cost=costs.loc["adv_geothermal"]["VOM"],
                p_nom_extendable=True if policy == "cfe" else False,
                lifetime=costs.loc["adv_geothermal"]["lifetime"],
            )

        # Add RES generators
        for carrier in ["onwind", "solar"]:
            if carrier not in clean_techs:
                continue
            gen_template = location + " " + carrier + f"-{year}"

            n.add(
                "Generator",
                f"{name} {carrier}",
                carrier=carrier,
                bus=name,
                p_nom_extendable=False if policy == "ref" else True,
                p_max_pu=n.generators_t.p_max_pu[gen_template],
                capital_cost=n.generators.at[gen_template, "capital_cost"],
                marginal_cost=n.generators.at[gen_template, "marginal_cost"],
            )

        # Add storage techs
        if "battery" in storage_techs:
            n.add("Bus", f"{name} battery", carrier="battery")

            n.add(
                "Store",
                f"{name} battery",
                bus=f"{name} battery",
                e_cyclic=True,
                e_nom_extendable=True if policy == "cfe" else False,
                carrier="battery",
                capital_cost=n.stores.at[
                    f"{location} battery" + "-{}".format(year), "capital_cost"
                ],
                lifetime=n.stores.at[
                    f"{location} battery" + "-{}".format(year), "lifetime"
                ],
            )

            n.add(
                "Link",
                f"{name} battery charger",
                bus0=name,
                bus1=f"{name} battery",
                carrier="battery charger",
                efficiency=n.links.at[
                    f"{location} battery charger" + "-{}".format(year), "efficiency"
                ],
                capital_cost=n.links.at[
                    f"{location} battery charger" + "-{}".format(year), "capital_cost"
                ],
                p_nom_extendable=True if policy == "cfe" else False,
                lifetime=n.links.at[
                    f"{location} battery charger" + "-{}".format(year), "lifetime"
                ],
            )

            n.add(
                "Link",
                f"{name} battery discharger",
                bus0=f"{name} battery",
                bus1=name,
                carrier="battery discharger",
                efficiency=n.links.at[
                    f"{location} battery discharger" + "-{}".format(year), "efficiency"
                ],
                marginal_cost=n.links.at[
                    f"{location} battery discharger" + "-{}".format(year),
                    "marginal_cost",
                ],
                p_nom_extendable=True if policy == "cfe" else False,
                lifetime=n.links.at[
                    f"{location} battery discharger" + "-{}".format(year), "lifetime"
                ],
            )

        if "hydrogen" in storage_techs:
            n.add("Bus", f"{name} H2", carrier="H2")

            n.add(
                "Store",
                f"{name} H2 Store",
                bus=f"{name} H2",
                e_cyclic=True,
                e_nom_extendable=True if policy == "cfe" else False,
                carrier="H2 Store",
                capital_cost=costs.at["hydrogen storage underground", "fixed"],
                lifetime=costs.at["hydrogen storage underground", "lifetime"],
            )

            n.add(
                "Link",
                f"{name} H2 Electrolysis",
                bus0=name,
                bus1=f"{name} H2",
                carrier="H2 Electrolysis",
                efficiency=n.links.at[
                    f"{location} H2 Electrolysis" + "-{}".format(year), "efficiency"
                ],
                capital_cost=n.links.at[
                    f"{location} H2 Electrolysis" + "-{}".format(year), "capital_cost"
                ],
                p_nom_extendable=True if policy == "cfe" else False,
                lifetime=n.links.at[
                    f"{location} H2 Electrolysis" + "-{}".format(year), "lifetime"
                ],
            )

            n.add(
                "Link",
                f"{name} H2 Fuel Cell",
                bus0=f"{name} H2",
                bus1=name,
                carrier="H2 Fuel Cell",
                efficiency=n.links.at[
                    f"{location} H2 Fuel Cell" + "-{}".format(year), "efficiency"
                ],
                capital_cost=n.links.at[
                    f"{location} H2 Fuel Cell" + "-{}".format(year), "capital_cost"
                ],
                p_nom_extendable=True if policy == "cfe" else False,
                lifetime=n.links.at[
                    f"{location} H2 Fuel Cell" + "-{}".format(year), "lifetime"
                ],
            )


def add_vl(n, names: List[str]) -> None:
    """
    Add virtual links connecting data centers across physical network.

    Args:
    - n: The network to add virtual links to.
    - names: A list of data center names.

    Returns:
    - None
    """
    # Complete graph: n * (n - 1) / 2 edges
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:  # Exclude the case of the same datacenter
                link_name = f"vcc_{names[i]}_{names[j]}"
                n.add(
                    "Link",
                    link_name,
                    bus0=names[i],
                    bus1=names[j],
                    carrier="virtual_link",
                    marginal_cost=0.001,  # large enough to avoid optimization artifacts, small enough not to influence PPA portfolio
                    p_nom=1e6,
                )


def add_shifters(n) -> None:
    """
    Alternative form of virtual links connecting data centers across physical network

    Args:
    - n: The network to which the virtual links will be added

    Returns:
    - None
    """
    for i in range(len(names)):
        gen_name = f"vl_{names[i]}"
        n.add(
            "Generator",
            gen_name,
            bus=names[i],
            carrier="virtual_link",
            p_nom=40,
            marginal_cost=0.001,
            p_min_pu=-1,
            p_max_pu=1,
        )


def add_dsm(n) -> None:
    """
    Add option to shift loads over time, aka temporal DSM

    Args:
    - n: The network object to which the DSM components will be added

    Returns:
    - None
    """
    for location, name in datacenters.items():
        n.add("Bus", f"{name} DSM", carrier="dsm")

        n.add(
            "Store",
            f"{name} DSM",
            bus=f"{name} DSM",
            carrier="dsm",
            e_cyclic=True,
            e_nom=1e6,
            p_nom_extendable=False,
        )

        n.add(
            "Link",
            f"{name} DSM-delayin",
            bus0=name,
            bus1=f"{name} DSM",
            carrier="dsm",
            efficiency=1,
            p_nom=1e6,
            marginal_cost=0.1,
            p_nom_extendable=False,
        )

        n.add(
            "Link",
            f"{name} DSM-delayout",
            bus0=f"{name} DSM",
            bus1=name,
            carrier="dsm",
            efficiency=1,
            p_nom=1e6,
            marginal_cost=0.1,
            p_nom_extendable=False,
        )


def revert_links(n: pypsa.Network) -> None:
    """
    Modifies the sign attribute of links in the given PyPSA network object to -1 for virtual links and DSM mechanisms,
    and 1 for all other links. This reverts the sign in nodal balance constraint and aligns the extra_functionality code accordingly.

    Args:
    - n: The PyPSA network object to modify.

    Returns:
    - None
    """
    n.links.loc[
        (n.links.carrier == "virtual_link") | (n.links.carrier == "dsm"), "sign"
    ] = -1
    n.links.loc[
        (n.links.carrier != "virtual_link") & (n.links.carrier != "dsm"), "sign"
    ] = 1


def calculate_grid_cfe(n, name: str, node: str, config) -> pd.Series:
    """
    Calculates the time-series of grid supply CFE score for each C&I consumer.

    Args:
    - n: pypsa network.
    - name: name of a C&I consumer.
    - node: location (node) of a C&I consumer.
    - config: config.yaml settings

    Returns:
    - pd.Series: A pandas series containing the grid CFE supply score.
    """
    grid_buses = n.buses.index[
        ~n.buses.index.str.contains(name) & ~n.buses.index.str.contains(node)
    ]
    country_buses = n.buses.index[n.buses.index.str.contains(node)]

    clean_techs = pd.Index(config["global"]["grid_clean_techs"])
    emitters = pd.Index(config["global"]["emitters"])

    clean_grid_generators = n.generators.index[
        n.generators.bus.isin(grid_buses) & n.generators.carrier.isin(clean_techs)
    ]
    clean_grid_links = n.links.index[
        n.links.bus1.isin(grid_buses) & n.links.carrier.isin(clean_techs)
    ]
    clean_grid_storage_units = n.storage_units.index[
        n.storage_units.bus.isin(grid_buses) & n.storage_units.carrier.isin(clean_techs)
    ]
    dirty_grid_links = n.links.index[
        n.links.bus1.isin(grid_buses) & n.links.carrier.isin(emitters)
    ]

    clean_country_generators = n.generators.index[
        n.generators.bus.isin(country_buses) & n.generators.carrier.isin(clean_techs)
    ]
    clean_country_links = n.links.index[
        n.links.bus1.isin(country_buses) & n.links.carrier.isin(clean_techs)
    ]
    clean_country_storage_units = n.storage_units.index[
        n.storage_units.bus.isin(country_buses)
        & n.storage_units.carrier.isin(clean_techs)
    ]
    dirty_country_links = n.links.index[
        n.links.bus1.isin(country_buses) & n.links.carrier.isin(emitters)
    ]

    clean_grid_gens = n.generators_t.p[clean_grid_generators].sum(axis=1)
    clean_grid_ls = -n.links_t.p1[clean_grid_links].sum(axis=1)
    clean_grid_sus = n.storage_units_t.p[clean_grid_storage_units].sum(axis=1)
    clean_grid_resources = clean_grid_gens + clean_grid_ls + clean_grid_sus

    dirty_grid_resources = -n.links_t.p1[dirty_grid_links].sum(axis=1)

    # grid_cfe =  clean_grid_resources / n.loads_t.p[grid_loads].sum(axis=1)
    # grid_cfe[grid_cfe > 1] = 1.

    import_cfe = clean_grid_resources / (clean_grid_resources + dirty_grid_resources)

    clean_country_gens = n.generators_t.p[clean_country_generators].sum(axis=1)
    clean_country_ls = -n.links_t.p1[clean_country_links].sum(axis=1)
    clean_country_sus = n.storage_units_t.p[clean_country_storage_units].sum(axis=1)
    clean_country_resources = clean_country_gens + clean_country_ls + clean_country_sus

    dirty_country_resources = -n.links_t.p1[dirty_country_links].sum(axis=1)

    ##################
    # Country imports |
    # NB lines and links are bidirectional, thus we track imports for both subsets
    # of interconnectors: where [country] node is bus0 and bus1. Subsets are exclusive.

    line_imp_subsetA = n.lines_t.p1.loc[:, n.lines.bus0.str.contains(node)].sum(axis=1)
    line_imp_subsetB = n.lines_t.p0.loc[:, n.lines.bus1.str.contains(node)].sum(axis=1)
    line_imp_subsetA[line_imp_subsetA < 0] = 0.0
    line_imp_subsetB[line_imp_subsetB < 0] = 0.0

    links_imp_subsetA = (
        n.links_t.p1.loc[
            :,
            (
                n.links.bus0.str.contains(node)
                & (n.links.carrier == "DC")
                & ~(n.links.index.str.contains(name))
            ),
        ]
        .clip(lower=0)
        .sum(axis=1)
    )

    links_imp_subsetB = (
        n.links_t.p0.loc[
            :,
            (
                n.links.bus1.str.contains(node)
                & (n.links.carrier == "DC")
                & ~(n.links.index.str.contains(name))
            ),
        ]
        .clip(lower=0)
        .sum(axis=1)
    )

    country_import = (
        line_imp_subsetA + line_imp_subsetB + links_imp_subsetA + links_imp_subsetB
    )

    grid_supply_cfe = (clean_country_resources + country_import * import_cfe) / (
        clean_country_resources + dirty_country_resources + country_import
    )

    print(f"Grid_supply_CFE for {node} has following stats:")
    print(grid_supply_cfe.describe())

    return grid_supply_cfe


def solve_network(
    n: pypsa.Network,
    policy: str,
    penetration: float,
    tech_palette: str,
    n_iterations: int,
    config: Dict[str, Any],
) -> None:
    """
    Solves a network optimization problem given a network object, policy, penetration rate, technology palette, and number of iterations.

    Args:
    - n: The network object to be optimized.
    - policy: The voluntary procurement policy of C&I consumers.
    - penetration: Effectively the target CFE score.
    - tech_palette: The technology palette available for C&I consumers.
    - n_iterations: The number of iterations to be used for optimization (-> relaxed bilinear term in 24/7 CFE constraint).
    - config: config.yaml settings

    Returns:
    - None
    """

    # techs for RES annual matching
    res_techs = config["ci"]["res_techs"]

    # techs for CFE hourly matching
    clean_techs = palette(tech_palette)[0]
    storage_techs = palette(tech_palette)[1]
    storage_charge_techs = palette(tech_palette)[2]
    storage_discharge_techs = palette(tech_palette)[3]

    def vl_constraints(n):
        delta = float(flexibility) / 100
        weights = n.snapshot_weightings["generators"]
        vls = n.links[n.links.carrier == "virtual_link"]

        for location, name in datacenters.items():
            vls_snd = vls.query("bus0==@name").index
            vls_rec = vls.query("bus1==@name").index

            snd = n.model["Link-p"].loc[:, vls_snd].sum(dims=["Link"])
            rec = n.model["Link-p"].loc[:, vls_rec].sum(dims=["Link"])
            load = n.loads_t.p_set[name + " load"]
            # requested_load = load + rec - snd
            rhs_up = load * (1 + delta) - load
            rhs_lo = load * (1 - delta) - load

            n.model.add_constraints(rec - snd <= rhs_up, name=f"vl_limit-upper_{name}")
            n.model.add_constraints(rec - snd >= rhs_lo, name=f"vl_limit-lower_{name}")

    def shifts_conservation(n):
        vls = n.generators[n.generators.carrier == "virtual_link"]
        shifts = n.model["Generator-p"].loc[:, vls.index].sum(dims=["Generator"])
        # sum of loads shifts across all DC are equal 0 per time period
        n.model.add_constraints(shifts == 0, name=f"vl_limit-upper_{name}")

    def DSM_constraints(n):
        delta = float(flexibility) / 100
        weights = n.snapshot_weightings["generators"]
        dsm = n.links[n.links.carrier == "dsm"]

        for location, name in datacenters.items():
            dsm_delayin = dsm.query("bus0==@name").index
            dsm_delayout = dsm.query("bus1==@name").index

            delayin = n.model["Link-p"].loc[:, dsm_delayin].sum(dims=["Link"])
            delayout = n.model["Link-p"].loc[:, dsm_delayout].sum(dims=["Link"])

            load = n.loads_t.p_set[name + " load"]
            rhs_up = load * (1 + delta) - load
            rhs_lo = load * (1 - delta) - load

            n.model.add_constraints(
                delayout - delayin <= rhs_up, name=f"DSM-upper_{name}"
            )
            n.model.add_constraints(
                delayout - delayin >= rhs_lo, name=f"DSM-lower_{name}"
            )

    def DSM_conservation(n):
        dsm = n.links[n.links.carrier == "dsm"]

        for location, name in datacenters.items():
            dsm_link_delayout = dsm.query("bus0==@name").index
            dsm_link_delayin = dsm.query("bus1==@name").index

            delayout = n.model["Link-p"].loc[:, dsm_link_delayout].sum(dims=["Link"])
            delayin = n.model["Link-p"].loc[:, dsm_link_delayin].sum(dims=["Link"])

            daily_outs = delayout.groupby("snapshot.dayofyear").sum()
            daily_ins = delayin.groupby("snapshot.dayofyear").sum()

            n.model.add_constraints(
                daily_outs - daily_ins == 0, name=f"DSM-conservation_{name}"
            )

    def DC_constraints(n):
        "A general case when both spatial and temporal flexibility mechanisms are enabled"

        delta = float(flexibility) / 100
        weights = n.snapshot_weightings["generators"]
        vls = n.links[n.links.carrier == "virtual_link"]
        dsm = n.links[n.links.carrier == "dsm"]

        for location, name in datacenters.items():
            vls_snd = vls.query("bus0==@name").index
            vls_rec = vls.query("bus1==@name").index
            dsm_delayin = dsm.query("bus0==@name").index
            dsm_delayout = dsm.query("bus1==@name").index

            snd = n.model["Link-p"].loc[:, vls_snd].sum(dims=["Link"])
            rec = n.model["Link-p"].loc[:, vls_rec].sum(dims=["Link"])
            delayin = n.model["Link-p"].loc[:, dsm_delayin].sum(dims=["Link"])
            delayout = n.model["Link-p"].loc[:, dsm_delayout].sum(dims=["Link"])

            load = n.loads_t.p_set[name + " load"]
            # requested_load = load + rec - snd
            rhs_up = load * (1 + delta) - load
            rhs_lo = load * (1 - delta) - load

            n.model.add_constraints(
                rec - snd + delayout - delayin <= rhs_up, name=f"DC-upper_{name}"
            )
            n.model.add_constraints(
                rec - snd + delayout - delayin >= rhs_lo, name=f"DC-lower_{name}"
            )

    def cfe_constraints(n):
        weights = n.snapshot_weightings["generators"]
        delta = float(flexibility) / 100
        vls = n.links[n.links.carrier == "virtual_link"]
        dsm = n.links[n.links.carrier == "dsm"]
        # vls = n.generators[n.generators.carrier=='virtual_link']

        for location, name in datacenters.items():
            # LHS
            clean_gens = [name + " " + g for g in clean_techs]
            storage_dischargers = [name + " " + g for g in storage_discharge_techs]
            storage_chargers = [name + " " + g for g in storage_charge_techs]

            gen_sum = (n.model["Generator-p"].loc[:, clean_gens] * weights).sum()
            discharge_sum = (
                n.model["Link-p"].loc[:, storage_dischargers]
                * n.links.loc[storage_dischargers, "efficiency"]
                * weights
            ).sum()
            charge_sum = (
                -1 * (n.model["Link-p"].loc[:, storage_chargers] * weights).sum()
            )

            ci_export = n.model["Link-p"].loc[:, [name + " export"]]
            ci_import = n.model["Link-p"].loc[:, [name + " import"]]
            grid_sum = (
                (-1 * ci_export * weights)
                + (
                    ci_import
                    * n.links.at[name + " import", "efficiency"]
                    * grid_supply_cfe
                    * weights
                )
            ).sum()  # linear expr

            lhs = gen_sum + discharge_sum + charge_sum + grid_sum

            # RHS
            total_load = (n.loads_t.p_set[name + " load"] * weights).sum()

            vls_snd = vls.query("bus0==@name").index
            vls_rec = vls.query("bus1==@name").index
            total_snd = (
                n.model["Link-p"].loc[:, vls_snd] * weights
            ).sum()  # NB sum over both axes
            total_rec = (n.model["Link-p"].loc[:, vls_rec] * weights).sum()

            dsm_delayin = dsm.query("bus0==@name").index
            dsm_delayout = dsm.query("bus1==@name").index
            total_delayin = (
                n.model["Link-p"].loc[:, dsm_delayin] * weights
            ).sum()  # NB sum over both axes
            total_delayout = (n.model["Link-p"].loc[:, dsm_delayout] * weights).sum()

            flex = penetration * (
                total_rec - total_snd + total_delayout - total_delayin
            )

            n.model.add_constraints(
                lhs - flex >= penetration * (total_load), name=f"CFE_constraint_{name}"
            )

    def excess_constraints(n, config):
        weights = n.snapshot_weightings["generators"]

        for location, name in datacenters.items():
            ci_export = n.model["Link-p"].loc[:, [name + " export"]]
            excess = (ci_export * weights).sum()
            total_load = (n.loads_t.p_set[name + " load"] * weights).sum()
            share = config["ci"][
                "excess_share"
            ]  # 'sliding': max(0., penetration - 0.8)

            n.model.add_constraints(
                excess <= share * total_load, name=f"Excess_constraint_{name}"
            )

    def res_constraints(n):
        weights = n.snapshot_weightings["generators"]

        for location, name in datacenters.items():
            res_gens = [name + " " + g for g in res_techs]
            lhs = (n.model["Generator-p"].loc[:, res_gens] * weights).sum()
            total_load = (n.loads_t.p_set[name + " load"] * weights).sum()

            # Note equality sign
            n.model.add_constraints(
                lhs == penetration * total_load, name=f"100RES_annual_matching_{name}"
            )

    def country_res_constraints(n):
        weights = n.snapshot_weightings["generators"]

        for location, name in datacenters.items():
            zone = n.buses.loc[f"{location}", :].country

            grid_res_techs = config["global"]["grid_res_techs"]
            grid_buses = n.buses.index[
                n.buses.location.isin(geoscope(zone)["country_nodes"][location])
            ]
            grid_loads = n.loads.index[n.loads.bus.isin(grid_buses)]

            country_res_gens = n.generators.index[
                n.generators.bus.isin(grid_buses)
                & n.generators.carrier.isin(grid_res_techs)
            ]
            country_res_links = n.links.index[
                n.links.bus1.isin(grid_buses) & n.links.carrier.isin(grid_res_techs)
            ]
            country_res_storage_units = n.storage_units.index[
                n.storage_units.bus.isin(grid_buses)
                & n.storage_units.carrier.isin(grid_res_techs)
            ]

            gens = n.model["Generator-p"].loc[:, country_res_gens] * weights
            links = (
                n.model["Link-p"].loc[:, country_res_links]
                * n.links.loc[country_res_links, "efficiency"]
                * weights
            )
            sus = (
                n.model["StorageUnit-p_dispatch"].loc[:, country_res_storage_units]
                * weights
            )
            lhs = gens.sum() + sus.sum() + links.sum()

            target = config[f"res_target_{year}"][f"{zone}"]
            total_load = (
                n.loads_t.p_set[grid_loads].sum(axis=1) * weights
            ).sum()  # number

            n.model.add_constraints(
                lhs == target * total_load, name=f"country_res_constraints_{zone}"
            )

    def system_res_constraints(n, year, config) -> None:
        """
        Set a system-wide national RES constraints based on NECPs.

        Here CI load is not counted within country_load ->
        this avoids avoid big overshoot of national RES targets due to CI-procured portfolio. Note that EU RE directive counts corporate PPA within NECPs.
        """
        zones = [key[:2] for key in datacenters.keys()]
        country_targets = config[f"res_target_{year}"]

        grid_res_techs = config["global"]["grid_res_techs"]
        weights = n.snapshot_weightings["generators"]

        for ct in country_targets.keys():
            country_buses = n.buses.index[(n.buses.index.str[:2] == ct)]
            if country_buses.empty:
                continue

            country_loads = n.loads.index[n.loads.bus.isin(country_buses)]
            country_res_gens = n.generators.index[
                n.generators.bus.isin(country_buses)
                & n.generators.carrier.isin(grid_res_techs)
            ]
            country_res_links = n.links.index[
                n.links.bus1.isin(country_buses) & n.links.carrier.isin(grid_res_techs)
            ]
            country_res_storage_units = n.storage_units.index[
                n.storage_units.bus.isin(country_buses)
                & n.storage_units.carrier.isin(grid_res_techs)
            ]

            gens = n.model["Generator-p"].loc[:, country_res_gens] * weights
            links = (
                n.model["Link-p"].loc[:, country_res_links]
                * n.links.loc[country_res_links, "efficiency"]
                * weights
            )
            sus = (
                n.model["StorageUnit-p_dispatch"].loc[:, country_res_storage_units]
                * weights
            )
            lhs = gens.sum() + sus.sum() + links.sum()

            target = config[f"res_target_{year}"][f"{ct}"]
            total_load = (n.loads_t.p_set[country_loads].sum(axis=1) * weights).sum()

            print(
                f"country RES constraint for {ct} {target} and total load {round(total_load/1e6, 2)} TWh"
            )
            logger.info(
                f"country RES constraint for {ct} {target} and total load {round(total_load/1e6, 2)} TWh"
            )

            n.model.add_constraints(
                lhs == target * total_load, name=f"{ct}_res_constraint"
            )

    def add_battery_constraints(n):
        """
        Add constraint ensuring that charger = discharger:
         1 * charger_size - efficiency * discharger_size = 0
        """
        discharger_bool = n.links.index.str.contains("battery discharger")
        charger_bool = n.links.index.str.contains("battery charger")

        dischargers_ext = n.links[discharger_bool].query("p_nom_extendable").index
        chargers_ext = n.links[charger_bool].query("p_nom_extendable").index

        eff = n.links.efficiency[dischargers_ext].values
        lhs = (
            n.model["Link-p_nom"].loc[chargers_ext]
            - n.model["Link-p_nom"].loc[dischargers_ext] * eff
        )

        n.model.add_constraints(lhs == 0, name="Link-charger_ratio")

    def extra_functionality(n, snapshots):
        add_battery_constraints(n)
        # country_res_constraints(n)
        system_res_constraints(n, year, config)

        if policy == "ref":
            print("no target set")
        elif policy == "cfe":
            print("setting CFE target of", penetration)
            cfe_constraints(n)
            excess_constraints(n, config)
            # vl_constraints(n) # redundant with DSM_constraints(n)
            # DSM_constraints(n) # redundant with DC_constraints(n)
            DC_constraints(n)
            DSM_conservation(n) if config["ci"]["temporal_shifting"] else None
        elif policy == "res":
            print("setting annual RES target of", penetration)
            res_constraints(n)
            excess_constraints(n, config)
        else:
            print(
                f"'policy' wildcard must be one of 'ref', 'resXX' or 'cfeXX'. Now is {policy}."
            )
            sys.exit()

    n.consistency_check()

    set_of_options = config["solving"]["solver"]["options"]
    solver_options = (
        config["solving"]["solver_options"][set_of_options] if set_of_options else {}
    )
    solver_name = config["solving"]["solver"]["name"]

    # CFE dataframe
    values = [f"iteration {i}" for i in range(n_iterations + 1)]

    def create_tuples(locations, values):
        # Use a nested list comprehension to create the list of tuples
        tuples_list = [(location, value) for location in locations for value in values]
        return tuples_list

    cols = pd.MultiIndex.from_tuples(create_tuples(locations, values))
    grid_cfe_df = pd.DataFrame(0.0, index=n.snapshots, columns=cols)

    for i in range(n_iterations):
        for location, name in zip(locations, names):
            grid_supply_cfe = grid_cfe_df.loc[:, (location, f"iteration {i}")]
            print(grid_supply_cfe.describe())

        n.optimize.create_model()

        extra_functionality(n, n.snapshots)

        n.optimize.solve_model(
            solver_name=solver_name, log_fn=snakemake.log.solver, **solver_options
        )

        for location, name in zip(locations, names):
            grid_cfe_df.loc[
                :, (f"{location}", f"iteration {i+1}")
            ] = calculate_grid_cfe(n, name=name, node=location, config=config)

    grid_cfe_df.to_csv(snakemake.output.grid_cfe)


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        snakemake = mock_snakemake(
            "solve_network",
            year="2030",
            zone="IE",
            palette="p3",
            policy="cfe100",
            participation="10",
        )

    logging.basicConfig(
        filename=snakemake.log.python, level=snakemake.config["logging_level"]
    )

    config = snakemake.config

    # Wildcards & Settings
    policy = snakemake.wildcards.policy[:3]
    penetration = float(snakemake.wildcards.policy[3:]) / 100 if policy != "ref" else 0
    tech_palette = snakemake.wildcards.palette
    zone = snakemake.wildcards.zone
    year = snakemake.wildcards.year
    profile_shape = config["ci"]["profile_shape"]
    participation = snakemake.wildcards.participation

    datacenters = config["ci"]["datacenters"]
    locations = list(datacenters.keys())
    names = list(datacenters.values())
    flexibility = config["ci"]["flexibility"]

    print(f"solving network for policy {policy} and penetration {penetration}")
    print(f"solving network for palette: {tech_palette}")
    print(f"solving network for bidding zone: {zone}")
    print(f"solving network year: {year}")
    print(f"solving with datacenters: {datacenters}")
    print(f"solving with flexibility: {flexibility}")

    # When running via snakemake
    n = pypsa.Network(
        timescope(year)["network_file"],
        override_component_attrs=override_component_attrs(),
    )

    Nyears = 1  # years in simulation
    costs = prepare_costs(
        cost_file=timescope(year)["costs_projection"],
        USD_to_EUR=config["costs"]["USD2013_to_EUR2013"],
        discount_rate=config["costs"]["discountrate"],
        year=year,
        Nyears=Nyears,
        config=config,
        lifetime=config["costs"]["lifetime"],
    )

    # nhours = 240
    # n.set_snapshots(n.snapshots[:nhours])
    # n.snapshot_weightings[:] = 8760.0 / nhours

    with memory_logger(
        filename=getattr(snakemake.log, "memory", None), interval=30.0
    ) as mem:
        strip_network(n, config)
        cost_parametrization(n, config)

        shutdown_lineexp(n)
        limit_resexp(n, year, config)
        nuclear_policy(n, config)
        coal_policy(n, year, config)
        biomass_potential(n)
        co2_policy(n, year, config)

        add_ci(n, year)
        add_vl(n) if config["ci"]["spatial_shifting"] else None
        revert_links(n) if config["ci"]["spatial_shifting"] else None
        add_dsm(n) if config["ci"]["temporal_shifting"] else None

        solve_network(
            n=n,
            policy=policy,
            penetration=penetration,
            tech_palette=tech_palette,
            n_iterations=config["solving"]["options"]["n_iterations"],
            config=config,
        )

        n.export_to_netcdf(snakemake.output.network)

    logger.info(f"Maximum memory usage: {mem.mem_usage}")
