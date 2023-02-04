
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
#allow plotting without Xwindows
import matplotlib
from pyrsistent import s
from xarray import align
matplotlib.use('Agg')

from solve_network import palette, geoscope


def ci_capacity():

    fig, ax = plt.subplots()
    fig.set_size_inches((6,4.5))

    gen_inv = df.loc[["ci_cap_" + t for t in clean_techs]].rename({"ci_cap_" + t : t for t in clean_techs})
    discharge_inv = df.loc[["ci_cap_" + t for t in clean_dischargers]].rename({"ci_cap_" + t : t for t in clean_dischargers})
    charge_inv = df.loc[["ci_cap_" + t for t in clean_chargers]].rename({"ci_cap_" + t : t for t in clean_chargers})
    charge_inv = charge_inv.drop(['battery_charger']) # display only battery discharger capacity

    ldf = pd.concat([gen_inv, charge_inv, discharge_inv])
    
    to_drop = ldf.index[(ldf < 0.1).all(axis=1)]
    ldf.drop(to_drop, inplace=True)

    ldf.rename(columns=rename_scen, level=0, inplace=True) 
    ldf.rename(index=rename_ci_capacity, level=0, inplace=True) 
    new_index = preferred_order.intersection(ldf.index).append(ldf.index.difference(preferred_order))
    ldf = ldf.loc[new_index]
    (ldf).T.plot(kind="bar",stacked=True,
                ax=ax, color=tech_colors, width=0.65, edgecolor = "black", linewidth=0.05)

    plt.xticks(rotation=0)
    ax.set_xticklabels([''.join(item) for item in ldf.columns.tolist()])
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim([0,max(ldf.sum())*1.3])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    #ax.set_xlabel("CFE target")
    ax.set_ylabel("DC portfolio capacity [MW]")
    ax.legend(loc="upper left", ncol=2, prop={"size":9})

    fig.tight_layout()
    fig.savefig(snakemake.output.plot, transparent=True)


def ci_costandrev():

    fig, ax = plt.subplots()
    fig.set_size_inches((6,4.5))

    techs = clean_techs + ["grid",
                           "battery_storage",
                           "battery_inverter",
                           "hydrogen_storage",
                           "hydrogen_electrolysis",
                           "hydrogen_fuel_cell"]

    costs = df.loc[["ci_cost_" + t for t in techs]].rename({"ci_cost_" + t : t for t in techs}).multiply(1/df.loc["ci_demand_total"],axis=1)
   
    to_drop = costs.index[(costs < 0.1).all(axis=1)]
    costs.drop(to_drop, inplace=True)

    revenues = - df.loc[["ci_average_revenue"]]
    revenues.index = revenues.index.map({'ci_average_revenue': 'revenue'})
    ldf = pd.concat([costs, revenues])

    ldf.rename(columns=rename_scen, level=0, inplace=True) 
    ldf = ldf.groupby(rename_ci_cost).sum()
    new_index = preferred_order.intersection(ldf.index).append(ldf.index.difference(preferred_order))
    ldf = ldf.loc[new_index]

    ldf.T.plot(kind="bar",stacked=True,
               ax=ax, color=tech_colors, width=0.65, edgecolor = "black", linewidth=0.05)
    netc=ldf.sum()
    x = 0
    for i in range(len(netc)):
        ax.scatter(x = x, y = netc[i], color='black', marker="_")
        x += 1
    ax.scatter([], [], color='black', marker="_", label='net cost')
    #dots = ax.scatter(0, 0, color='black', marker="_")
    #ax.legend(handlers = [dots])

    plt.xticks(rotation=0)
    ax.set_xticklabels([''.join(item) for item in ldf.columns.tolist()])
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylabel("24/7 CFE cost and revenue [€/MWh]")
    ax.legend(loc="upper left", ncol = 3, prop={"size":8})
    ax.set_ylim(top=max(ldf.sum())*1.5)

    fig.tight_layout()
    fig.savefig(snakemake.output.plot.replace("capacity.pdf","ci_costandrev.pdf"), transparent=True)


def ci_generation():

    fig, ax = plt.subplots()
    fig.set_size_inches((6,4.5))

    generation = df.loc[["ci_generation_" + t for t in clean_techs]].rename({"ci_generation_" + t : t for t in clean_techs})/1000.
    discharge = df.loc[["ci_generation_" + t for t in clean_dischargers]].rename({"ci_generation_" + t : t for t in clean_dischargers})/1000.

    ldf = pd.concat([generation, discharge])

    to_drop = ldf.index[(ldf < 0.1).all(axis=1)]
    ldf.drop(to_drop, inplace=True)

    ldf.rename(columns=rename_scen, level=0, inplace=True) 
    ldf.rename(index=rename_ci_capacity, level=0, inplace=True) 
    new_index = preferred_order.intersection(ldf.index).append(ldf.index.difference(preferred_order))
    ldf = ldf.loc[new_index]

    yl = df.loc['ci_demand_total'][0]/1000
    plt.axhline(y = yl, color = 'gray', linestyle="--", linewidth=0.8)

    (ldf).T.plot(kind="bar",stacked=True,
                ax=ax, color=tech_colors, width=0.65, edgecolor = "black", linewidth=0.05)

    plt.xticks(rotation=0)
    ax.set_xticklabels([''.join(item) for item in ldf.columns.tolist()])
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim([0,max(ldf.sum())*1.3])
    #ax.set_xlabel("CFE target")
    ax.set_ylabel("C&I generation [GWh]")
    ax.legend(loc="upper left", ncol=2, prop={"size":9})

    fig.tight_layout()
    fig.savefig(snakemake.output.plot.replace("capacity.pdf","ci_generation.pdf"), transparent=True)



if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('plot_summary', year='2025', zone='IEDK', palette='p1', policy="cfe100")   

    #Wildcards & Settings
    policy = snakemake.wildcards.policy[:3]
    penetration = float(snakemake.wildcards.policy[3:])/100 if policy != "ref" else 0
    tech_palette = snakemake.wildcards.palette
    zone = snakemake.wildcards.zone
    year = snakemake.wildcards.year
    area = snakemake.config['area']
    datacenters = snakemake.config['ci']['datacenters']
    locations = list(datacenters.keys())

    #techs for CFE hourly matching
    clean_techs = palette(tech_palette)[0]
    storage_techs = palette(tech_palette)[1]
    storage_charge_techs = palette(tech_palette)[2]
    storage_discharge_techs = palette(tech_palette)[3]

    #renaming technologies for plotting
    clean_chargers = [g for g in storage_charge_techs]
    clean_chargers = [item.replace(' ', '_') for item in clean_chargers]
    clean_dischargers = [g for g in storage_discharge_techs]
    clean_dischargers = [item.replace(' ', '_') for item in clean_dischargers]

    #Assign colors
    tech_colors = snakemake.config['tech_colors']

    rename_ci_cost = pd.Series({
        "onwind" : "onshore wind",
        "solar" : "solar",
        "grid" : "grid imports",
        'revenue': "revenue",
        "battery_storage" : "battery",
        "battery_inverter" : "battery",
        'battery_discharger':"battery",
        "hydrogen_storage" : "hydrogen storage",
        "hydrogen_electrolysis": "hydrogen storage",
        "hydrogen_fuel_cell": "hydrogen storage",
        'adv_geothermal': "advanced dispatchable",
        'allam_ccs': "NG-Allam"})

    rename_ci_capacity= pd.Series({
        "onwind" : "onshore wind",
        "solar" : "solar",
        'battery_discharger':"battery",
        "H2_Fuel_Cell": "hydrogen fuel cell",
        "H2_Electrolysis": "hydrogen electrolysis",
        'adv_geothermal': "advanced dispatchable",
        'allam_ccs': "NG-Allam"})

    preferred_order = pd.Index([
        "advanced dispatchable",
        "NG-Allam",
        'Gas OC',
        "offshore wind",
        "onshore wind",
        "solar",
        "battery",
        "hydrogen storage",
        "hydrogen electrolysis",
        "hydrogen fuel cell"])

    rename_scen = {'0': '0%\n',
                   '5': '5%\n',
                   '10': '10%\n',
                    '20': '20%\n',
                    '40': '40%\n',
                    }

    df = pd.read_csv(snakemake.input.summary, index_col=0, header=[0,1])


    ci_capacity()
    ci_costandrev()
    ci_generation()

