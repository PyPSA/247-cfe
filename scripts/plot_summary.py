import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#allow plotting without Xwindows
import matplotlib
matplotlib.use('Agg')


def used():

    fig, ax = plt.subplots()
    fig.set_size_inches((4,3))

    typ = ["local","grid"]

    ldf = df.loc[[f"ci_fraction_clean_used_{t}" for t in typ]].rename({f"ci_fraction_clean_used_{t}" : t for t in typ})

    ldf.T.plot(kind="bar",stacked=True,
               ax=ax,
               color=tech_colors)

    ax.grid()
    ax.set_axisbelow(True)

    ax.set_xlabel("scenario")
    ax.set_ylabel("fraction CFE [per unit]")
    ax.legend(loc="lower left",
              prop={"size":8})

    fig.tight_layout()

    fig.savefig(snakemake.output.used,
                transparent=True)


def ci_capacity():

    fig, ax = plt.subplots()
    fig.set_size_inches((4,3))

    gen_inv = df.loc[["ci_cap_" + t for t in clean_techs]].rename({"ci_cap_" + t : t for t in clean_techs})
    discharge_inv = df.loc[["ci_cap_" + t for t in clean_dischargers]].rename({"ci_cap_" + t : t for t in clean_dischargers})
    charge_inv = df.loc[["ci_cap_" + t for t in clean_chargers]].rename({"ci_cap_" + t : t for t in clean_chargers})
    charge_inv = charge_inv.drop(['battery_charger']) # display only battery discharger capacity
    
    ldf = pd.concat([gen_inv, charge_inv, discharge_inv])

    ldf.index = ldf.index.map(rename_ci_techs)

    ldf.T.plot(kind="bar",stacked=True,
               ax=ax,
               color=tech_colors)

    ax.grid()
    ax.set_axisbelow(True)

    ax.set_xlabel("scenario")
    ax.set_ylabel("CI capacity [MW]")
    ax.legend(loc="upper left",
              prop={"size":5})

    fig.tight_layout()

    fig.savefig(snakemake.output.used.replace("used.pdf","ci_capacity.pdf"),
                transparent=True)


def ci_generation():

    fig, ax = plt.subplots()
    fig.set_size_inches((4,3))

    generation = df.loc[["ci_generation_" + t for t in clean_techs]].rename({"ci_generation_" + t : t for t in clean_techs})/1000.
    discharge = df.loc[["ci_generation_" + t for t in clean_dischargers]].rename({"ci_generation_" + t : t for t in clean_dischargers})/1000.

    ldf = pd.concat([generation, discharge])

    ldf.index = ldf.index.map(rename_ci_techs)

    ldf.T.plot(kind="bar",stacked=True,
               ax=ax,
               color=tech_colors)

    ax.grid()
    ax.set_axisbelow(True)

    ax.set_xlabel("scenario")
    ax.set_ylabel("CI generation [GWh]")
    ax.legend(loc="upper left",
              prop={"size":5})

    fig.tight_layout()

    fig.savefig(snakemake.output.used.replace("used.pdf","ci_generation.pdf"),
                transparent=True)

def global_emissions():

    fig, ax = plt.subplots()
    fig.set_size_inches((4,3))

    (df/1e6).loc['emissions'].plot(kind="bar",
                            ax=ax)

    ax.grid()
    ax.set_axisbelow(True)

    ax.set_xlabel("scenario")
    ax.set_ylabel("global emissions [MtCO2/a]")

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

    ax.grid()
    ax.set_axisbelow(True)

    ax.set_xlabel("scenario")
    ax.set_ylabel("CI average cost [EUR/MWh]")
    ax.legend(loc="upper left",
              prop={"size":5})


    fig.tight_layout()

    fig.savefig(snakemake.output.used.replace("used.pdf","ci_cost.pdf"),
                transparent=True)

def system_capacity():

    fig, ax = plt.subplots()
    fig.set_size_inches((4,3))

    gens = df.loc[["system_inv_" + t for t in exp_generators]].rename({"system_inv_" + t : t for t in exp_generators})
    links = df.loc[["system_inv_" + t for t in exp_links]].rename({"system_inv_" + t : t for t in exp_links})
    dischargers = df.loc[["system_inv_" + t for t in exp_dischargers]].rename({"system_inv_" + t : t for t in exp_dischargers})
    chargers = df.loc[["system_inv_" + t for t in exp_chargers]].rename({"system_inv_" + t : t for t in exp_chargers})
    chargers = chargers.drop(['battery_charger-2030']) # display only battery discharger capacity

    ldf = pd.concat([gens, links, dischargers, chargers])

    ldf.index = ldf.index.map(rename_system_techs)

    ldf.T.plot(kind="bar",stacked=True,
               ax=ax,
               color=tech_colors)

    ax.grid()
    ax.set_axisbelow(True)

    ax.set_xlabel("scenario")
    ax.set_ylabel("System capacity inv. [MW]")
    ax.legend(loc="upper right",
              prop={"size":5})

    fig.tight_layout()

    fig.savefig(snakemake.output.used.replace("used.pdf","system_capacity.pdf"),
                transparent=True)


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('plot_summary')

    # When running via snakemake
    ci = snakemake.config['ci']
    name = ci['name']
    node = ci['node']
    clean_techs = ci['clean_techs']

    clean_chargers = [g for g in ci['storage_chargers']]
    clean_chargers = [item.replace(' ', '_') for item in clean_chargers]

    clean_dischargers = [g for g in ci['storage_dischargers']]
    clean_dischargers = [item.replace(' ', '_') for item in clean_dischargers]

    exp_generators = snakemake.config['exp_generators']
    exp_generators = [item.replace(' ', '_') for item in exp_generators]

    exp_links = snakemake.config['exp_links']

    exp_chargers = snakemake.config['exp_chargers']
    exp_chargers = [item.replace(' ', '_') for item in exp_chargers]
    
    exp_dischargers = snakemake.config['exp_dischargers']
    exp_dischargers = [item.replace(' ', '_') for item in exp_dischargers]

    tech_colors = snakemake.config['tech_colors']

    rename_ci_techs = {
        'onwind': 'onwind',
        'solar': 'solar',
        'battery_discharger': 'battery_inverter',
        'H2_Electrolysis': 'hydrogen_electrolysis',
        'H2_Fuel_Cell': 'hydrogen_fuel_cell',
        'adv_nuclear': 'adv_nuclear'
    }

    rename_system_techs = {
        'offwind-ac-2030': 'offwind-ac',
        'offwind-dc-2030': 'offwind-dc',
        'onwind-2030': 'onwind',
        'solar-2030': 'solar',
        'solar_rooftop-2030': 'solar_rooftop',
        'OCGT-2030': 'OCGT',
        'battery_discharger-2030': 'battery_inverter',
        'H2_Fuel_Cell-2030': 'hydrogen_fuel_cell',
        'H2_Electrolysis-2030': 'hydrogen_electrolysis'
    }

    df = pd.read_csv(snakemake.input.summary,
                     index_col=0)

    used()

    ci_capacity()

    ci_generation()

    ci_cost()

    global_emissions()

    system_capacity()