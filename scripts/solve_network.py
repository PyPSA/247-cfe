# SPDX-FileCopyrightText: 2023 Iegor Riepin, Tom Brown
#
# SPDX-License-Identifier: MIT

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
    """
    Define technology palette available for C&I clean energy buyers
    """

    if tech_palette == "p1":
        clean_techs = ["onwind", "solar"]
        storage_techs = ["battery"]
        storage_chargers = ["battery charger"]
        storage_dischargers = ["battery discharger"]

    elif tech_palette == "p2":
        clean_techs = ["onwind", "solar"]
        storage_techs = ["battery", "hydrogen"]
        storage_chargers = ["battery charger", "H2 Electrolysis"]
        storage_dischargers = ["battery discharger", "H2 Fuel Cell"]

    elif tech_palette == "p3":
        clean_techs = [
            "onwind",
            "solar",
            "allam_ccs",
            "adv_geothermal",
        ]  # "adv_nuclear", "adv_geothermal"
        storage_techs = ["battery", "hydrogen"]
        storage_chargers = ["battery charger", "H2 Electrolysis"]
        storage_dischargers = ["battery discharger", "H2 Fuel Cell"]

    else:
        print(
            f"'palette' wildcard must be one of 'p1', 'p2' or 'p3'. Now is {tech_palette}."
        )
        sys.exit()

    return clean_techs, storage_techs, storage_chargers, storage_dischargers


def geoscope(zone):
    """
    zone: controls basenodes_to_keep list -> sets geographical scope of the model
    country_nodes -> countries subject to national RES policy constraints
    """
    d = dict()

    # A few toy regional networks for test & play purposes and the whole network
    # NB zone is used as a wildcard, while area as a switcher option; thus these are not merged

    IRELAND = ["IE5 0", "GB0 0", "GB5 0"]
    DENMARK = ["DK1 0", "DK2 0", "SE2 0", "NO2 0", "NL1 0", "DE1 0"]
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
    NETHERLANDS = ["NL1 0", "GB0 0", "DK1 0", "NO2 0", "BE1 0", "DE1 0"]
    IEDK = (
        IRELAND
        + ["DK1 0", "DK2 0"]
        + ["FR1 0", "LU1 0", "DE1 0", "BE1 0", "NL1 0", "NO2 0", "SE2 0"]
    )
    DKDE = ["DE1 0", "DK1 0", "DK2 0", "PL1 0"]
    EU = [
        "AL1 0",
        "AT1 0",
        "BA1 0",
        "BE1 0",
        "BG1 0",
        "CH1 0",
        "CZ1 0",
        "DE1 0",
        "DK1 0",
        "DK2 0",
        "EE6 0",
        "ES1 0",
        "ES4 0",
        "FI2 0",
        "FR1 0",
        "GB0 0",
        "GB5 0",
        "GR1 0",
        "HR1 0",
        "HU1 0",
        "IE5 0",
        "IT1 0",
        "IT3 0",
        "LT6 0",
        "LU1 0",
        "LV6 0",
        "ME1 0",
        "MK1 0",
        "NL1 0",
        "NO2 0",
        "PL1 0",
        "PT1 0",
        "RO1 0",
        "RS1 0",
        "SE2 0",
        "SI1 0",
        "SK1 0",
    ]

    if zone == "IE":
        d["basenodes_to_keep"] = IRELAND
    elif zone == "DK":
        d["basenodes_to_keep"] = DENMARK
    elif zone == "DE":
        d["basenodes_to_keep"] = GERMANY
    elif zone == "NL":
        d["basenodes_to_keep"] = NETHERLANDS
    elif zone == "GB":
        d["basenodes_to_keep"] = IRELAND
    elif zone == "IEDK":
        d["basenodes_to_keep"] = IEDK
    elif zone == "FR":
        d["basenodes_to_keep"] = IEDK  # intentionally larger network
    elif zone == "DKDE":
        d["basenodes_to_keep"] = DKDE
    elif zone == "EU":
        d["basenodes_to_keep"] = EU
    else:
        print(f"'zone' wildcard cannot be {zone}.")
        sys.exit()

    temp = dict()
    if "IE5 0" in locations:
        temp["IE5 0"] = ["IE5 0"]
    if "DK1 0" in locations:
        temp["DK1 0"] = ["DK1 0", "DK2 0"]
    if "DE1 0" in locations:
        temp["DE1 0"] = ["DE1 0"]
    if "NL1 0" in locations:
        temp["NL1 0"] = ["NL1 0"]
    if "GB0 0" in locations:
        temp["GB0 0"] = ["GB0 0", "GB5 0"]
    if "FR1 0" in locations:
        temp["FR1 0"] = ["FR1 0"]
    d["country_nodes"] = temp

    return d


def timescope(year):
    """
    coal_phaseout -> countries that implement coal phase-out policy until {year}
    network_file -> input file with pypsa-eur-sec brownfield network for {year}
    costs_projection -> input file with technology costs for {year}
    """

    d = dict()

    d["coal_phaseout"] = snakemake.config[f"policy_{year}"]

    if year == "2030":
        d["network_file"] = snakemake.input.network2030
        d["costs_projection"] = snakemake.input.costs2030
    elif year == "2025":
        d["network_file"] = snakemake.input.network2025
        d["costs_projection"] = snakemake.input.costs2025
    else:
        print(f"'year' wildcard must be one of '2025', '2030'. Now is {year}.")
        sys.exit()

    return d


def cost_parametrization(n):
    """
    overwrite default price assumptions for primary energy carriers
    only for virtual generators located in 'EU {carrier}' buses
    """

    for carrier in ["lignite", "coal", "gas"]:
        n.generators.loc[
            n.generators.index.str.contains(f"EU {carrier}"), "marginal_cost"
        ] = snakemake.config["costs"][f"price_{carrier}"]
    # n.generators[n.generators.index.str.contains('EU')].T

    n.generators.loc[n.generators.carrier == "onwind", "marginal_cost"] = 0.015


def load_profile(n, profile_shape, config):
    """
    create daily load profile for 24/7 CFE buyers based on config setting
    """

    scaling = int(config["time_sampling"][0])  # 3/1 for 3H/1H

    shape_base = [1 / 24] * 24

    shape_it = [
        0.034,
        0.034,
        0.034,
        0.034,
        0.034,
        0.034,
        0.038,
        0.042,
        0.044,
        0.046,
        0.047,
        0.048,
        0.049,
        0.049,
        0.049,
        0.049,
        0.049,
        0.048,
        0.047,
        0.043,
        0.038,
        0.037,
        0.036,
        0.035,
    ]

    shape_ind = [
        0.009,
        0.009,
        0.009,
        0.009,
        0.009,
        0.009,
        0.016,
        0.031,
        0.070,
        0.072,
        0.073,
        0.072,
        0.070,
        0.052,
        0.054,
        0.066,
        0.070,
        0.068,
        0.063,
        0.035,
        0.037,
        0.045,
        0.045,
        0.009,
    ]

    if profile_shape == "baseload":
        shape = np.array(shape_base).reshape(-1, scaling).mean(axis=1)
    elif profile_shape == "datacenter":
        shape = np.array(shape_it).reshape(-1, scaling).mean(axis=1)
    elif profile_shape == "industry":
        shape = np.array(shape_ind).reshape(-1, scaling).mean(axis=1)
    else:
        print(
            f"'profile_shape' option must be one of 'baseload', 'datacenter' or 'industry'. Now is {profile_shape}."
        )
        sys.exit()

    load = snakemake.config["ci"]["load"]  # data center nominal load in MW

    load_day = load * 24  # 24h
    load_profile_day = pd.Series(shape * load_day)  # 3H (or any n-hour) sampling is in
    load_profile_year = pd.concat(
        [load_profile_day] * int(len(n.snapshots) / (24 / scaling))
    )

    profile = load_profile_year.set_axis(n.snapshots)
    # baseload = pd.Series(load,index=n.snapshots)

    return profile


def prepare_costs(cost_file, USD_to_EUR, discount_rate, Nyears, lifetime, year):
    # set all asset costs and other parameters
    costs = pd.read_csv(cost_file, index_col=[0, 1]).sort_index()

    # correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
    costs.loc[costs.unit.str.contains("USD"), "value"] *= USD_to_EUR

    # min_count=1 is important to generate NaNs which are then filled by fillna
    costs = (
        costs.loc[:, "value"].unstack(level=1).groupby("technology").sum(min_count=1)
    )
    costs = costs.fillna(
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

    # Advanced nuclear
    data_nuc = pd.Series(
        {
            "CO2 intensity": 0,
            "FOM": costs.loc["nuclear"]["FOM"],
            "VOM": costs.loc["nuclear"]["VOM"],
            "discount rate": discount_rate,
            "efficiency": 0.36,
            "fuel": costs.loc["nuclear"]["fuel"],
            "investment": snakemake.config["costs"]["adv_nuclear_overnight"]
            * 1e3
            * snakemake.config["costs"]["USD2021_to_EUR2021"],
            "lifetime": 40.0,
        },
        name="adv_nuclear",
    )

    if year == "2025":
        adv_geo_overnight = snakemake.config["costs"]["adv_geo_overnight_2025"]
        allam_ccs_overnight = snakemake.config["costs"]["allam_ccs_overnight_2025"]
    elif year == "2030":
        adv_geo_overnight = snakemake.config["costs"]["adv_geo_overnight_2030"]
        allam_ccs_overnight = snakemake.config["costs"]["allam_ccs_overnight_2030"]

    # Advanced geothermal
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
    data_allam = pd.Series(
        {
            "CO2 intensity": 0,
            "FOM": 0,  # %/year
            "FOM-abs": 33000,  # $/MW-yr
            "VOM": 3.2,  # EUR/MWh
            "co2_seq": 40,  # $/ton
            "discount rate": discount_rate,
            "efficiency": 0.54,
            "fuel": snakemake.config["costs"]["price_gas"],
            "investment": allam_ccs_overnight * 1e3 * 1,
            "lifetime": 30.0,
        },
        name="allam_ccs",
    )

    costs = costs.append(data_nuc, ignore_index=False)
    costs = costs.append(data_geo, ignore_index=False)
    costs = costs.append(data_allam, ignore_index=False)

    annuity_factor = (
        lambda v: annuity(v["lifetime"], v["discount rate"]) + v["FOM"] / 100
    )
    costs["fixed"] = [
        annuity_factor(v) * v["investment"] * Nyears for i, v in costs.iterrows()
    ]

    return costs


def strip_network(n):
    nodes_to_keep = geoscope(zone)["basenodes_to_keep"]
    new_nodes = []

    for b in nodes_to_keep:
        for s in snakemake.config["node_suffixes_to_keep"]:
            new_nodes.append(b + " " + s)

    nodes_to_keep.extend(new_nodes)
    nodes_to_keep.extend(snakemake.config["additional_nodes"])

    n.mremove("Bus", n.buses.index.symmetric_difference(nodes_to_keep))

    # make sure lines are kept
    n.lines.carrier = "AC"

    carrier_to_keep = snakemake.config["carrier_to_keep"]

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


def shutdown_lineexp(n):
    """
    remove line expansion option
    """
    n.lines.s_nom_extendable = False
    n.links.loc[n.links.carrier == "DC", "p_nom_extendable"] = False


def limit_resexp(n, year, snakemake):
    """
    limit expansion of renewable technologies per zone and carrier type
    as a ratio of max increase to 2021 capacity fleet
    (additional to zonal place availability constraint)
    """
    ratio = snakemake.config["global"][f"limit_res_exp_{year}"]

    list_datacenters = list(datacenters.values())
    mask_datacenters = n.generators.index.str.contains(
        "|".join(list_datacenters), case=False
    )
    system_gens = n.generators[~mask_datacenters]

    fleet = system_gens.groupby(
        [system_gens.bus.str[:2], system_gens.carrier]
    ).p_nom.sum()
    fleet = fleet.rename(lambda x: x.split("-")[0], level=1).groupby(level=[0, 1]).sum()
    ct_national_target = list(snakemake.config[f"res_target_{year}"].keys()) + ["EU"]

    fleet.drop(ct_national_target, errors="ignore", level=0, inplace=True)

    # option to allow build out of carriers which are not build yet
    # fleet[fleet==0] = 1
    for ct, carrier in fleet.index:
        gen_i = (
            (n.generators.p_nom_extendable)
            & (n.generators.bus.str[:2] == ct)
            & (n.generators.carrier.str.contains(carrier))
        )
        n.generators.loc[gen_i, "p_nom_max"] = ratio * fleet.loc[ct, carrier]


def groupby_assets(n):
    """
    groupby generators and links of same type/node over year vintage classes
    is supposed to yield same solution somewhat faster
    """

    # Groupby generators
    # mask
    list_datacenters = list(datacenters.values())
    mask_datacenters = n.generators.index.str.contains(
        "|".join(list_datacenters), case=False
    )
    df = n.generators[~mask_datacenters]
    system_gens = df[~df.index.str.contains("EU")]
    system_gens = df[~df.index.str.contains("ror")]
    fleet = system_gens[system_gens["p_nom_extendable"] == False]

    # group
    grouped = fleet.groupby(["bus", "carrier"]).agg({"p_nom": "sum"}).reset_index()
    grouped["bus_carrier"] = grouped["bus"] + " " + grouped["carrier"]
    grouped = grouped.set_index("bus_carrier")

    # Merge
    fleet["bus_carrier"] = fleet["bus"] + " " + fleet["carrier"]
    other_columns = fleet.set_index("bus_carrier").drop(
        columns=["bus", "carrier", "p_nom"]
    )
    # other_columns.drop(['build_year'], axis=1, inplace=True)
    other_columns = other_columns.groupby("bus_carrier").agg(
        lambda x: x.dropna().iloc[0] if not x.dropna().empty else None
    )
    result = grouped.merge(other_columns, left_index=True, right_index=True, how="left")
    result.index.name = "Generator"

    # Update generators
    rows_to_drop = n.generators.index.intersection(fleet.index)
    n.generators = n.generators.drop(rows_to_drop)
    n.generators = n.generators.append(result)

    # Give new generators a time series
    def find_timeseries(col_name):
        second_space = col_name.find(" ", col_name.find(" ") + 1)
        node = col_name[:second_space]
        carrier = col_name[second_space + 1 :]

        # If carrier is 'offwind', use 'offwind-dc' to match column names
        if carrier == "offwind":
            carrier = "offwind-ac"

        search_pattern = f"{node}  {carrier}"
        for col in n.generators_t.p_max_pu.columns:
            if search_pattern in col:
                return col
        return None

    for col_name in result.index:
        col = find_timeseries(col_name)
        if col is not None:
            n.generators_t.p_max_pu[col_name] = n.generators_t.p_max_pu[col]

    # Groupby links
    # mask
    gen_links = [
        "OCGT",
        "CCGT",
        "coal",
        "lignite",
        "nuclear",
        "oil",
        "urban central solid biomass CHP",
    ]
    df = n.links[n.links["carrier"].isin(gen_links)]
    lfleet = df[df["p_nom_extendable"] == False]

    # group
    grouped = lfleet.groupby(["bus1", "carrier"]).agg({"p_nom": "sum"}).reset_index()
    grouped["bus_carrier"] = grouped["bus1"] + " " + grouped["carrier"]
    grouped = grouped.set_index("bus_carrier")

    # Merge
    lfleet["bus_carrier"] = lfleet["bus1"] + " " + lfleet["carrier"]
    other_columns = lfleet.set_index("bus_carrier").drop(
        columns=["bus1", "carrier", "p_nom"]
    )
    # other_columns.drop(['build_year', 'lifetime'], axis=1, inplace=True)
    other_columns = other_columns.groupby("bus_carrier").agg(
        lambda x: x.dropna().iloc[0] if not x.dropna().empty else None
    )
    result = grouped.merge(other_columns, left_index=True, right_index=True, how="left")
    result.index.name = "Link"

    # Update links
    rows_to_drop = n.links.index.intersection(lfleet.index)
    n.links = n.links.drop(rows_to_drop)
    n.links = n.links.append(result)


def nuclear_policy(n):
    """
    remove nuclear PPs fleet for countries with nuclear ban policy
    """
    for node in snakemake.config["nodes_with_nucsban"]:
        n.links.loc[
            n.links["bus1"].str.contains(f"{node}")
            & (n.links.index.str.contains("nuclear")),
            "p_nom",
        ] = 0


def coal_policy(n):
    """
    remove coal PPs fleet for countries with coal phase-out policy for {year}
    """

    countries = timescope(year)["coal_phaseout"]

    for country in countries:
        n.links.loc[
            n.links["bus1"].str.contains(f"{country}")
            & (n.links.index.str.contains("coal")),
            "p_nom",
        ] = 0
        n.links.loc[
            n.links["bus1"].str.contains(f"{country}")
            & (n.links.index.str.contains("lignite")),
            "p_nom",
        ] = 0


def biomass_potential(n):
    """
    remove solid biomass demand for industrial processes from overall biomass potential
    """
    n.stores.loc[n.stores.index == "EU solid biomass", "e_nom"] *= 0.45
    n.stores.loc[n.stores.index == "EU solid biomass", "e_initial"] *= 0.45


def co2_policy(n, year):
    """
    set EU carbon emissions policy as cap or price, update costs
    """
    gl_policy = snakemake.config["global"]

    if gl_policy["policy_type"] == "co2 cap":
        co2_cap = gl_policy["co2_share"] * gl_policy["co2_baseline"]
        n.global_constraints.at["CO2Limit", "constant"] = co2_cap
        print(f"Setting global CO2 cap to {co2_cap}")

    elif gl_policy["policy_type"] == "co2 price":
        n.global_constraints.drop("CO2Limit", inplace=True)
        co2_price = gl_policy[f"co2_price_{year}"]
        print(f"Setting CO2 price to {co2_price}")
        for carrier in ["coal", "oil", "gas", "lignite"]:
            n.generators.at[f"EU {carrier}", "marginal_cost"] += (
                co2_price * costs.at[carrier, "CO2 intensity"]
            )


def add_ci(n, year):
    """
    Add C&I buyer(s)
    """

    # tech_palette options
    clean_techs = palette(tech_palette)[0]
    storage_techs = palette(tech_palette)[1]

    for location, name in datacenters.items():
        # Add C&I bus
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

        # Add C&I load
        n.add(
            "Load",
            f"{name}" + " load",
            carrier="electricity",
            bus=name,
            p_set=load_profile(n, profile_shape, config),
        )

        # C&I following 24/7 approach is a share of C&I load -> subtract it from node's profile
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


def add_vl(n):
    "Add virtual links connecting data centers across physical network"
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


def add_shifters(n):
    "Alternative form of virtual links connecting data centers across physical network"
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


def add_dsm(n):
    "Add option to shift loads over time, aka temporal DSM"

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


def hack_links(n):
    """
    Virtual links and DSM mechanism shift loads, while <link> object in pypsa architecture shifts energy
    Here we add additional attribute "sign" and fix it to -1. This reverts sign in nodal balance constraint.
    extra_functionality code is aligned accordingly.
    """
    n.links.loc[
        (n.links.carrier == "virtual_link") | (n.links.carrier == "dsm"), "sign"
    ] = -1
    n.links.loc[
        (n.links.carrier != "virtual_link") & (n.links.carrier != "dsm"), "sign"
    ] = 1


def calculate_grid_cfe(n, name, node):
    grid_buses = n.buses.index[
        ~n.buses.index.str.contains(name) & ~n.buses.index.str.contains(node)
    ]
    country_buses = n.buses.index[n.buses.index.str.contains(node)]

    clean_techs = pd.Index(snakemake.config["global"]["grid_clean_techs"])
    emitters = pd.Index(snakemake.config["global"]["emitters"])

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

    links_imp_subsetA = n.links_t.p1.loc[
        :,
        n.links.bus0.str.contains(node)
        & (n.links.carrier == "DC")
        & ~(n.links.index.str.contains(name)),
    ].sum(axis=1)
    links_imp_subsetB = n.links_t.p0.loc[
        :,
        n.links.bus1.str.contains(node)
        & (n.links.carrier == "DC")
        & ~(n.links.index.str.contains(name)),
    ].sum(axis=1)
    links_imp_subsetA[links_imp_subsetA < 0] = 0.0
    links_imp_subsetB[links_imp_subsetB < 0] = 0.0

    country_import = (
        line_imp_subsetA + line_imp_subsetB + links_imp_subsetA + links_imp_subsetB
    )

    grid_supply_cfe = (clean_country_resources + country_import * import_cfe) / (
        clean_country_resources + dirty_country_resources + country_import
    )

    print(f"Grid_supply_CFE for {node} has following stats:")
    print(grid_supply_cfe.describe())

    return grid_supply_cfe


def solve_network(n, policy, penetration, tech_palette):
    n_iterations = snakemake.config["solving"]["options"]["n_iterations"]

    # techs for RES annual matching
    res_techs = snakemake.config["ci"]["res_techs"]

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

            # vls_local = vls.query('bus==@name').index
            # shift_local = (n.model['Generator-p'].loc[:, vls_local]*weights).sum()

            n.model.add_constraints(
                lhs - flex >= penetration * (total_load), name=f"CFE_constraint_{name}"
            )
            # n.model.add_constraints(lhs + shift_local >= penetration*(total_load), name=f"CFE_constraint_{name}")

    def excess_constraints(n, snakemake):
        weights = n.snapshot_weightings["generators"]

        for location, name in datacenters.items():
            ci_export = n.model["Link-p"].loc[:, [name + " export"]]
            excess = (ci_export * weights).sum()
            total_load = (n.loads_t.p_set[name + " load"] * weights).sum()
            share = snakemake.config["ci"][
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

            grid_res_techs = snakemake.config["global"]["grid_res_techs"]
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

            target = snakemake.config[f"res_target_{year}"][f"{zone}"]
            total_load = (
                n.loads_t.p_set[grid_loads].sum(axis=1) * weights
            ).sum()  # number

            n.model.add_constraints(
                lhs == target * total_load, name=f"country_res_constraints_{zone}"
            )

    def system_res_constraints(n, snakemake):
        """
        An alternative implementation of system-wide level RES constraint.
        NB CI load is not counted within country_load ->
        - to avoid big overshoot of national RES targets due to CI-procured portfolio
        - also EU RE directive counts corporate PPA within NECPs.
        """
        zones = [key[:2] for key in datacenters.keys()]
        year = snakemake.wildcards.year
        country_targets = snakemake.config[f"res_target_{year}"]

        grid_res_techs = snakemake.config["global"]["grid_res_techs"]
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

            target = snakemake.config[f"res_target_{year}"][f"{ct}"]
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
        system_res_constraints(n, snakemake)

        if policy == "ref":
            print("no target set")
        elif policy == "cfe":
            print("setting CFE target of", penetration)
            cfe_constraints(n)
            excess_constraints(n, snakemake)
            # vl_constraints(n) if snakemake.config['ci']['spatial_shifting'] else None
            # DSM_constraints(n) if snakemake.config['ci']['temporal_shifting'] else None
            DC_constraints(n)
            DSM_conservation(n) if snakemake.config["ci"]["temporal_shifting"] else None
        elif policy == "res":
            print("setting annual RES target of", penetration)
            res_constraints(n)
            excess_constraints(n, snakemake)
        else:
            print(
                f"'policy' wildcard must be one of 'ref', 'res__' or 'cfe__'. Now is {policy}."
            )
            sys.exit()

    n.consistency_check()

    set_of_options = snakemake.config["solving"]["solver"]["options"]
    solver_options = (
        snakemake.config["solving"]["solver_options"][set_of_options]
        if set_of_options
        else {}
    )
    solver_name = snakemake.config["solving"]["solver"]["name"]

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
            ] = calculate_grid_cfe(n, name=name, node=location)

    grid_cfe_df.to_csv(snakemake.output.grid_cfe)


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        snakemake = mock_snakemake(
            "solve_network",
            year="2025",
            zone="DKDE",
            palette="p1",
            policy="cfe100",
            flexibility="40",
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
    profile_shape = snakemake.config["ci"]["profile_shape"]

    datacenters = snakemake.config["ci"]["datacenters"]
    locations = list(datacenters.keys())
    names = list(datacenters.values())
    flexibility = snakemake.wildcards.flexibility

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
        timescope(year)["costs_projection"],
        snakemake.config["costs"]["USD2013_to_EUR2013"],
        snakemake.config["costs"]["discountrate"],
        Nyears,
        snakemake.config["costs"]["lifetime"],
        year,
    )

    # nhours = 240
    # n.set_snapshots(n.snapshots[:nhours])
    # n.snapshot_weightings[:] = 8760.0 / nhours

    with memory_logger(
        filename=getattr(snakemake.log, "memory", None), interval=30.0
    ) as mem:
        strip_network(n)
        # groupby_assets(n)
        limit_resexp(n, year, snakemake)
        shutdown_lineexp(n)
        nuclear_policy(n)
        coal_policy(n)
        biomass_potential(n)
        cost_parametrization(n)
        co2_policy(n, year)
        load_profile(n, profile_shape, config)

        add_ci(n, year)
        add_vl(n) if snakemake.config["ci"]["spatial_shifting"] else None
        add_dsm(n) if snakemake.config["ci"]["temporal_shifting"] else None
        hack_links(n)

        solve_network(n, policy, penetration, tech_palette)

        n.export_to_netcdf(snakemake.output.network)

    logger.info(f"Maximum memory usage: {mem.mem_usage}")
