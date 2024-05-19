# SPDX-FileCopyrightText: 2023 Iegor Riepin, Tom Brown
#
# SPDX-License-Identifier: MIT


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from solve_network import palette

from matplotlib.lines import Line2D

custom_line = Line2D([0], [0], color="gray", linestyle="--", linewidth=1.0)


def format_column_names(col_tuple):
    # return f"{col_tuple[0]}{col_tuple[1][:2]}"
    return f"{col_tuple[0]}"


def format_country_names(col_tuple):
    # return first two letters of country name
    return f"{col_tuple[:2]}"


def rename_scen(obj, level=0):
    """
    Automatically append a percent sign to numeric string column names if not already present and rename DataFrame columns.

    Returns:
    - None: Modifies the DataFrame in place.
    """
    if isinstance(obj, pd.DataFrame):
        # Handle DataFrame: Modify column names
        modified_dict = {
            key: f"{key}%" if key.isdigit() and "%" not in key else key
            for key in obj.columns.get_level_values(level)
        }
        obj.rename(columns=modified_dict, level=level, inplace=True)

    elif isinstance(obj, pd.Series):
        # Handle Series: Modify index labels
        modified_dict = {
            key: f"{key}%" if "%" not in key else key
            for key in obj.index.get_level_values(level)
        }
        obj.rename(index=modified_dict, level=level, inplace=True)


def prepare_data_frame(df, techs, rename_dict):
    return df.loc[["ci_cap_" + t.replace(" ", "_") for t in techs]].rename(
        {"ci_cap_" + t: t for t in techs}
    )


def add_datacenter_lines(ax, num_columns, num_datacenters):
    space = num_columns / num_datacenters
    for l in range(num_datacenters - 1):
        ax.axvline(x=(space - 0.5) + space * l, color="gray", linestyle="--")


def ci_capacity(df, tech_colors, rename_ci_capacity, preferred_order):
    # Data Preparation
    ldf = pd.concat(
        [
            df.loc[["ci_cap_" + t.replace(" ", "_") for t in clean_techs]].rename(
                {"ci_cap_" + t: t for t in clean_techs}
            ),
            df.loc[["ci_cap_" + t.replace(" ", "_") for t in clean_dischargers]].rename(
                {"ci_cap_" + t: t for t in clean_dischargers}
            ),
            df.loc[["ci_cap_" + t.replace(" ", "_") for t in clean_chargers]].rename(
                {"ci_cap_" + t: t for t in clean_chargers}
            ),
        ]
    )

    # Display discharger capacity for storage technologies with fixed P/E ratio
    to_drop = ["battery_charger", "ironair_charger"]
    ldf = ldf.drop(index=[row for row in to_drop if row in ldf.index])

    # Drop rows with all values less than 0.1
    to_drop = ldf.index[(ldf < 0.1).all(axis=1)]
    ldf.drop(to_drop, inplace=True)

    # Rename columns and indices
    rename_scen(ldf, level=0)
    ldf.rename(index=rename_ci_capacity, level=0, inplace=True)

    # Reorder and sort the final DataFrame
    new_index = preferred_order.intersection(ldf.index).append(
        ldf.index.difference(preferred_order)
    )
    ldf = ldf.loc[new_index].sort_index(
        axis="columns", level=[1, 0], ascending=[False, True]
    )

    # Plotting Enhancements
    fig, ax = plt.subplots(figsize=(8, 6))
    if not ldf.empty:
        ldf.T.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=tech_colors,
            width=0.65,
            edgecolor="black",
            linewidth=0.05,
        )

        ax.set_xticklabels(
            [format_column_names(col) for col in ldf.columns.tolist()], fontsize=12
        )
        ax.grid(alpha=0.8, which="both", linestyle="--")
        ax.set_axisbelow(True)
        ax.set_ylim([0, max(ldf.sum()) * 1.2])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylabel("C&I procurement [MW]", fontsize=14)
        ax.legend(loc="upper left", ncol=2, prop={"size": 12})

        ax.set_title(
            f"C&I portfolio capacity for {int(penetration*100)}% CFE score",
            fontsize=16,
            weight="bold",
        )

        plt.xticks(rotation=0)

    else:
        print("Dataframe to plot is empty")

    fig.tight_layout()
    fig.savefig(snakemake.output.plot, transparent=True, dpi=300)


def ci_costandrev(df, tech_colors, rename_ci_cost, preferred_order):
    techs = clean_techs + [
        "grid",
        "battery_storage",
        "battery_inverter",
        "ironair_storage",
        "ironair_inverter",
        "hydrogen_storage",
        "hydrogen_electrolysis",
        "hydrogen_fuel_cell",
    ]

    # Calculate Costs
    costs = (
        df.loc[["ci_cost_" + t.replace(" ", "_") for t in techs]]
        .rename({"ci_cost_" + t: t for t in techs})
        .multiply(1 / df.loc["ci_demand_total"], axis=1)
        .fillna(0)
    )

    costs.drop(costs.index[(costs < 0.1).all(axis=1)], inplace=True)

    revenues = -df.loc[["ci_average_revenue"]].rename({"ci_average_revenue": "revenue"})
    ldf = pd.concat([costs, revenues])

    rename_scen(ldf, level=0)
    ldf = ldf.groupby(rename_ci_cost).sum()

    # Reorder and sort the DataFrame
    new_index = preferred_order.intersection(ldf.index).append(
        ldf.index.difference(preferred_order)
    )
    ldf = ldf.loc[new_index].sort_index(
        axis="columns", level=[1, 0], ascending=[False, True]
    )

    # treat 0% participation case (NaN and inf to 0.)
    ldf = ldf.fillna(0)
    ldf = ldf.replace([float("inf"), -float("inf")], 0)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    if not ldf.empty:
        ldf.T.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=tech_colors,
            width=0.65,
            edgecolor="black",
            linewidth=0.05,
        )
        ax.set_xticklabels(
            [format_column_names(col) for col in ldf.columns.tolist()],
            fontsize=12,
            rotation=0,
        )
        ax.grid(alpha=0.8, which="both", linestyle="--")
        ax.set_axisbelow(True)
        ax.set_ylabel(r"Average costs [€$\cdot$MWh$^{-1}$]", fontsize=14)
        ax.legend(loc="upper left", ncol=3, prop={"size": 12})
        ax.set_ylim(top=max(ldf.sum()) * 1.5)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.tick_params(axis="y", labelsize=12)

        ax.set_title(
            r"C&I procurement costs and revenue [€$\cdot$MWh$^{-1}$]",
            fontsize=16,
            weight="bold",
        )

        # Add Net Cost Markers
        net_costs = ldf.sum()
        marker_size = 100  # Adjust the size as needed
        marker_style = "o"  # Change marker style as preferred
        marker_color = "black"  # Choose a color that stands out
        for i, cost in enumerate(net_costs):
            ax.scatter(
                x=i, y=cost, color=marker_color, marker=marker_style, s=marker_size
            )
        ax.scatter(
            [],
            [],
            color=marker_color,
            marker=marker_style,
            s=marker_size,
            label="net cost",
        )

    else:
        print("Dataframe to plot is empty")

    # Save Plot
    fig.tight_layout()
    fig.savefig(
        snakemake.output.plot.replace("capacity.pdf", "ci_costandrev.pdf"),
        transparent=True,
        dpi=300,
    )


def ci_generation(df, tech_colors, rename_ci_capacity, preferred_order):
    # Generation and Discharge Calculation
    generation = (
        df.loc[["ci_generation_" + t.replace(" ", "_") for t in clean_techs]].rename(
            {"ci_generation_" + t: t for t in clean_techs}
        )
        / 1000.0
    )
    discharge = (
        df.loc[
            ["ci_generation_" + t.replace(" ", "_") for t in clean_dischargers]
        ].rename({"ci_generation_" + t: t for t in clean_dischargers})
        / 1000.0
    )

    # Concatenate and Drop Rows
    ldf = pd.concat([generation, discharge])
    ldf.drop(ldf.index[(ldf < 0.1).all(axis=1)], inplace=True)

    # Rename and Reorder
    rename_scen(ldf, level=0)
    ldf.rename(index=rename_ci_capacity, level=0, inplace=True)
    new_index = preferred_order.intersection(ldf.index).append(
        ldf.index.difference(preferred_order)
    )
    ldf = ldf.loc[new_index].sort_index(
        axis="columns", level=[1, 0], ascending=[False, True]
    )

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    if not ldf.empty:
        ldf.T.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=tech_colors,
            width=0.65,
            edgecolor="black",
            linewidth=0.05,
        )
        ax.set_xticklabels(
            [format_column_names(col) for col in ldf.columns.tolist()],
            fontsize=12,
            rotation=0,
        )
        ax.grid(alpha=0.8, which="both", linestyle="--")
        ax.set_axisbelow(True)
        ax.set_ylabel("C&I portfolio generation [GWh]", fontsize=14)
        ax.legend(loc="upper left", ncol=2, prop={"size": 12})
        ax.set_ylim([0, max(ldf.sum()) * 1.2])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.tick_params(axis="y", labelsize=12)

        # Existing legend handles and labels
        handles, labels = ax.get_legend_handles_labels()
        # Add the custom line to the existing legend
        handles.append(custom_line)
        labels.append("Total Demand")
        # Update the plot with the modified legend
        ax.legend(
            handles=handles, labels=labels, loc="upper left", ncol=3, prop={"size": 12}
        )

        num_scenarios = len(ldf.columns.get_level_values(0).unique())
        bar_width = 1.0 / num_scenarios
        margin = bar_width * 0.2

        for i, scenario in enumerate(ldf.columns.get_level_values(0).unique()):
            scenario_key = scenario.rstrip("%")
            total_demand = df.xs("ci_demand_total").loc[scenario_key] / 1000
            xmin = (i / num_scenarios) + margin / 2
            xmax = ((i + 1) / num_scenarios) - margin / 2
            ax.axhline(
                y=total_demand[0],
                color="black",
                linestyle="--",
                linewidth=1.5,
                xmin=xmin,
                xmax=xmax,
            )

    else:
        print("Dataframe to plot is empty")

    fig.tight_layout()
    fig.savefig(
        snakemake.output.plot.replace("capacity.pdf", "ci_generation.pdf"),
        transparent=True,
    )


def ci_curtailment(df, ci_res):
    # Data for ci res curtailment across all locations
    ldf = df.loc[["ci_curtailment_" + t for t in ci_res]].rename(
        {"ci_curtailment_" + t: t for t in ci_res}
    )

    # Refine data
    rename_scen(ldf, level=0)
    ldf = pd.DataFrame(ldf.sum(axis=0), columns=["RES curtailment"]).unstack()
    ldf = ldf["RES curtailment"] / 1e3
    ldf.columns = [format_country_names(col) for col in ldf.columns.tolist()]

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    if not ldf.empty:
        ldf.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            width=0.65,
            edgecolor="black",
            linewidth=0.05,
            color=sns.color_palette("rocket", len(ldf.columns)),
        )
        ax.set_xticklabels(
            ldf.index,
            rotation=0,
            fontsize=12,
        )
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_ylabel("C&I portfolio curtailment [GWh]", fontsize=14)
        ax.set_ylim(top=ldf.sum(axis=1).max() * 1.3)
        ax.tick_params(axis="y", labelsize=12)
        ax.legend(loc="upper left", ncol=2, prop={"size": 10}, fancybox=True)
    else:
        print("Dataframe to plot is empty")

    # Save Plot
    fig.tight_layout()
    fig.savefig(
        snakemake.output.plot.replace("capacity.pdf", "ci_curtailment.pdf"),
        transparent=True,
    )


def ci_emisrate(df):
    ldf = df.loc["ci_emission_rate_true"] * 1e3  # convert to gCO2/kWh
    rename_scen(ldf)
    ldf.index = ldf.index.droplevel(1)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    if not ldf.empty:
        ldf.plot(
            kind="bar",
            ax=ax,
            color="#33415c",
            width=0.65,
            edgecolor="black",
            linewidth=0.05,
        )
        ax.set_xticklabels(
            [col for col in ldf.index.tolist()],
            rotation=0,
            fontsize=12,
        )
        ax.grid(alpha=0.8, which="both", linestyle="--")
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylabel(r"C&I emission rate [gCO$_2$$\cdot$kWh$^{-1}$]", fontsize=14)
        ax.set_title(
            f"Emission rate of C&I consumption",
            fontsize=16,
            weight="bold",
        )
    else:
        print("Dataframe to plot is empty")

    # Final Adjustments and Save Plot
    fig.tight_layout()
    fig.savefig(
        snakemake.output.plot.replace("capacity.pdf", "ci_emissions.pdf"),
        transparent=True,
    )


def zone_emissions(df):
    ldf = df.loc["emissions_zone"]
    rename_scen(ldf)
    ldf.index = ldf.index.droplevel(1)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    if not ldf.empty:
        ldf.plot(
            kind="bar",
            ax=ax,
            color="#33415c",
            width=0.65,
            edgecolor="black",
            linewidth=0.05,
        )
        ax.set_xticklabels(
            [col for col in ldf.index.tolist()],
            rotation=0,
            fontsize=12,
        )
        ax.grid(alpha=0.8, which="both", linestyle="--")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylabel(r"Emissions in local zone [MtCO$_2\cdot a^{-1}$]", fontsize=14)
        ax.set_title(
            f"Annual power sector emissions in local zone: {zone}",
            fontsize=16,
            weight="bold",
        )
    else:
        print("Dataframe to plot is empty")

    # Final Adjustments and Save Plot
    fig.tight_layout()
    fig.savefig(
        snakemake.output.plot.replace("capacity.pdf", "zone_emissions.pdf"),
        transparent=True,
    )


def system_investment(df, tech_colors, rename_system_simple, preferred_order, year):
    # Collect data
    gens = df.loc[["system_inv_" + t.replace(" ", "_") for t in exp_generators]].rename(
        {"system_inv_" + t.replace(" ", "_"): t for t in exp_generators}
    )

    links = df.loc[["system_inv_" + t.replace(" ", "_") for t in exp_links]].rename(
        {"system_inv_" + t.replace(" ", "_"): t for t in exp_links}
    )

    dischargers = df.loc[
        ["system_inv_" + t.replace(" ", "_") for t in exp_dischargers]
    ].rename({"system_inv_" + t.replace(" ", "_"): t for t in exp_dischargers})

    chargers = df.loc[
        ["system_inv_" + t.replace(" ", "_") for t in exp_chargers]
    ].rename({"system_inv_" + t.replace(" ", "_"): t for t in exp_chargers})

    chargers = chargers.drop(["battery_charger-%s" % year])

    # Combine all components into a unified DataFrame
    ldf = pd.concat([gens, links, dischargers, chargers], axis=0)

    # Drop rows where all values are below a certain threshold
    to_drop = ldf.index[(ldf < 0.1).all(axis=1)]
    ldf.drop(to_drop, inplace=True)

    # Rename scenario names and group by the simpler names
    rename_scen(ldf, level=0)
    ldf = ldf.groupby(rename_system_simple).sum()

    # Reorder based on preferred order
    new_index = preferred_order.intersection(ldf.index).append(
        ldf.index.difference(preferred_order)
    )
    ldf = ldf.loc[new_index]

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    (ldf / 1e3).T.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=tech_colors,
        width=0.65,
        edgecolor="black",
        linewidth=0.05,
    )

    # Formatting
    ax.set_xticklabels(
        [format_column_names(col) for col in ldf.columns.tolist()], fontsize=12
    )
    ax.grid(alpha=0.8, which="both", linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("System capacity investment [GW]", fontsize=14)
    ax.set_title("System capacity investments", fontsize=16, weight="bold")
    ax.legend(loc="upper left", ncol=2, prop={"size": 12})

    fig.tight_layout()
    fig.savefig(
        snakemake.output.plot.replace("capacity.pdf", "system_investment.pdf"),
        transparent=True,
        dpi=300,
    )


def zone_capacity(df, tech_colors, preferred_order):

    ldf = df.filter(regex="^zone_capacity_", axis=0)

    ldf.index = [col.replace("zone_capacity_", "") for col in ldf.index]

    # Drop rows where all values are below a certain threshold
    to_drop = ldf.index[(ldf < 0.1).all(axis=1)]
    ldf.drop(to_drop, inplace=True)

    # Rename scenario names and group by the simpler names
    rename_scen(ldf, level=0)
    ldf = ldf.groupby(system_techs).sum()

    # Reorder based on preferred order
    new_index = preferred_order.intersection(ldf.index).append(
        ldf.index.difference(preferred_order)
    )
    ldf = ldf.loc[new_index]

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    (ldf / 1e3).T.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=tech_colors,
        width=0.65,
        edgecolor="black",
        linewidth=0.05,
    )

    # Formatting
    ax.set_xticklabels(
        [format_column_names(col) for col in ldf.columns.tolist()], fontsize=12
    )
    ax.grid(alpha=0.8, which="both", linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Zone capacity mix [GW]", fontsize=14)
    ax.set_title("Zone capacity mix ", fontsize=16, weight="bold")
    ax.legend(loc="upper left", ncol=2, prop={"size": 12})

    fig.tight_layout()
    fig.savefig(
        snakemake.output.plot.replace("capacity.pdf", "zone_capacity_mix.pdf"),
        transparent=True,
        dpi=300,
    )


####################################################################################################
if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_summary", year="2030", zone="DE", palette="p4", policy="ref"
        )

    config = snakemake.config
    scaling = int(config["time_sampling"][0])  # temporal scaling -- 3/1 for 3H/1H

    # Wildcards & Settings
    policy = snakemake.wildcards.policy[:3]
    penetration = float(snakemake.wildcards.policy[3:]) / 100 if policy != "ref" else 0
    tech_palette = snakemake.wildcards.palette
    zone = snakemake.wildcards.zone
    year = snakemake.wildcards.year

    datacenters = snakemake.config["ci"]["datacenters"]
    locations = list(datacenters.keys())
    names = list(datacenters.values())
    flexibilities = snakemake.config["ci"]["flexibility"]

    # techs for CFE hourly matching, extracted from palette
    palette_techs = palette(tech_palette)

    (
        clean_techs,
        storage_techs,
        storage_charge_techs,
        storage_discharge_techs,
    ) = palette_techs

    # renaming technologies for plotting
    clean_chargers = [tech.replace(" ", "_") for tech in storage_charge_techs]
    clean_dischargers = [tech.replace(" ", "_") for tech in storage_discharge_techs]

    def tech_names(base_names, year):
        return [f"{name.replace(' ', '_')}-{year}" for name in base_names]

    # expected technology names with year
    exp_generators = tech_names(["offwind-ac", "offwind-dc", "onwind", "solar"], year)
    exp_links = tech_names(["OCGT"], year)
    exp_chargers = tech_names(
        ["battery charger", "H2 Electrolysis", "ironair discharger"], year
    )
    exp_dischargers = tech_names(
        ["battery discharger", "H2 Fuel Cell", "ironair charger"], year
    )

    # Assign colors
    tech_colors = snakemake.config["tech_colors"]

    # Rename mappings
    rename_ci_cost = pd.Series(
        {
            "onwind": "onshore wind",
            "solar": "solar",
            "grid": "grid imports",
            "revenue": "revenue",
            "battery_storage": "battery",
            "battery_inverter": "battery",
            "battery_charger": "battery",
            "ironair_storage": "ironair",
            "ironair_inverter": "ironair",
            "ironair_charger": "ironair",
            "hydrogen_storage": "hydrogen storage",
            "hydrogen_electrolysis": "hydrogen storage",
            "hydrogen_fuel_cell": "hydrogen storage",
            "adv_geothermal": "advanced dispatchable",
            "allam_ccs": "NG-Allam",
        }
    )

    rename_ci_capacity = pd.Series(
        {
            "onwind": "onshore wind",
            "solar": "solar",
            "battery_discharger": "battery",
            "battery_charger": "battery",
            "ironair_discharger": "ironair",
            "ironair_charger": "ironair",
            "H2_Fuel_Cell": "hydrogen fuel cell",
            "H2_Electrolysis": "hydrogen electrolysis",
            "adv_geothermal": "advanced dispatchable",
            "allam_ccs": "NG-Allam",
        }
    )

    system_techs = {
        "offwind-ac": "wind",
        "offwind-dc": "wind",
        "Offshore Wind": "wind",
        "onwind": "wind",
        "Onshore Wind": "wind",
        "solar": "solar PV",
        "Solar": "solar PV",
        "OCGT": "gas OC",
        "CCGT": "gas CC",
        "coal": "hard coal",
        "lignite": "lignite",
        "oil": "oil",
        "urban central solid biomass CHP": "biomass",
        "Pumped Hydro Storage": "PHS",
        "Reservoir & Dam": "hydro",
        "battery": "battery storage",
        "battery_discharger": "battery storage",
        "H2_Fuel_Cell": "hydrogen fuel cell",
        "H2_Electrolysis": "hydrogen electrolysis",
        "ironair_discharger": "ironair storage",
        "ironair": "ironair storage",
    }

    rename_system_simple = {
        f"{key}-%s" % year: value for key, value in system_techs.items()
    }

    preferred_order = pd.Index(
        [
            "advanced dispatchable",
            "NG-Allam",
            "biomass",
            "hard coal",
            "lignite",
            "gas OC",
            "gas CC",
            "oil",
            "offshore wind",
            "onshore wind",
            "wind",
            "solar PV",
            "PHS",
            "battery",
            "battery storage",
            "ironair storage",
            "hydrogen storage",
            "hydrogen electrolysis",
            "hydrogen fuel cell",
        ]
    )

# SUMMARY PLOTS

# %matplotlib inline

df = pd.read_csv(snakemake.input.summary, index_col=0, header=[0, 1])

ci_capacity(
    df=df,
    tech_colors=tech_colors,
    rename_ci_capacity=rename_ci_capacity,
    preferred_order=preferred_order,
)

ci_costandrev(
    df=df,
    tech_colors=tech_colors,
    rename_ci_cost=rename_ci_cost,
    preferred_order=preferred_order,
)

ci_generation(
    df=df,
    tech_colors=tech_colors,
    rename_ci_capacity=rename_ci_capacity,
    preferred_order=preferred_order,
)

# ci_curtailment(
#     df=df, ci_res=snakemake.config["ci"]["res_techs"]
# )

ci_emisrate(df=df)

zone_emissions(df=df)

system_investment(
    df=df,
    tech_colors=tech_colors,
    rename_system_simple=rename_system_simple,
    preferred_order=preferred_order,
    year=year,
)

zone_capacity(df=df, tech_colors=tech_colors, preferred_order=preferred_order)
