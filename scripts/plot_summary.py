import pandas as pd

#allow plotting without Xwindows
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np


def used():

    fig, ax = plt.subplots()
    fig.set_size_inches((4,3))

    typ = ["local","grid"]

    ldf = df.loc[[f"ci_fraction_clean_used_{t}" for t in typ]].rename({f"ci_fraction_clean_used_{t}" : t for t in typ})

    ldf.T.plot(kind="bar",stacked=True,
               ax=ax,
               color=tech_colors)

    ax.set_xlabel("scenario")
    ax.set_ylabel("fraction CFE [per unit]")

    ax.grid()

    fig.tight_layout()

    fig.savefig(snakemake.output.used,
                transparent=True)


def ci_capacity():

    fig, ax = plt.subplots()
    fig.set_size_inches((4,3))

    ldf = df.loc[["ci_cap_" + t for t in clean_techs]].rename({"ci_cap_" + t : t for t in clean_techs})

    ldf.T.plot(kind="bar",stacked=True,
               ax=ax,
               color=tech_colors)

    ax.set_xlabel("scenario")
    ax.set_ylabel("CI capacity [MW]")

    ax.grid()

    fig.tight_layout()

    fig.savefig(snakemake.output.used.replace("used.pdf","ci_capacity.pdf"),
                transparent=True)


def ci_generation():

    fig, ax = plt.subplots()
    fig.set_size_inches((4,3))

    ldf = df.loc[["ci_generation_" + t for t in clean_techs]].rename({"ci_generation_" + t : t for t in clean_techs})/1000.

    ldf.T.plot(kind="bar",stacked=True,
               ax=ax,
               color=tech_colors)

    ax.set_xlabel("scenario")
    ax.set_ylabel("CI generation [GWh]")

    ax.grid()

    fig.tight_layout()

    fig.savefig(snakemake.output.used.replace("used.pdf","ci_generation.pdf"),
                transparent=True)

def global_emissions():

    fig, ax = plt.subplots()
    fig.set_size_inches((4,3))

    (df/1e6).loc['emissions'].plot(kind="bar",
                            ax=ax)

    ax.set_xlabel("scenario")
    ax.set_ylabel("global emissions [MtCO2/a]")

    ax.grid()

    fig.tight_layout()

    fig.savefig(snakemake.output.used.replace("used.pdf","emissions.pdf"),
                transparent=True)


def ci_cost():

    fig, ax = plt.subplots()
    fig.set_size_inches((4,3))

    techs = clean_techs + ["grid",
                           "battery_storage",
                           "battery_inverter",
                           "hydrogen_storage",
                           "hydrogen_electrolysis",
                           "hydrogen_fuel_cell"]

    ldf = df.loc[["ci_cost_" + t for t in techs]].rename({"ci_cost_" + t : t for t in techs}).multiply(1/df.loc["ci_demand_total"],axis=1)

    to_drop = ldf.index[(ldf < 0.1).all(axis=1)]
    ldf.drop(to_drop, inplace=True)

    ldf.T.plot(kind="bar",stacked=True,
               ax=ax,
               color=tech_colors)

    ax.set_xlabel("scenario")
    ax.set_ylabel("CI average cost [EUR/MWh]")

    ax.grid()

    ax.legend(loc="upper left",
              prop={"size":4})


    fig.tight_layout()

    fig.savefig(snakemake.output.used.replace("used.pdf","ci_cost.pdf"),
                transparent=True)

if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('plot_summary')

    # When running via snakemake
    clean_techs = snakemake.config['ci']['clean_techs']
    tech_colors = snakemake.config['tech_colors']

    df = pd.read_csv(snakemake.input.summary,
                     index_col=0)

    used()

    ci_capacity()

    ci_generation()

    ci_cost()

    global_emissions()
