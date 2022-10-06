
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


def used():

    fig, ax = plt.subplots()
    fig.set_size_inches((6,4.5))

    typ = ["local","grid"]

    ldf = df.loc[[f"ci_fraction_clean_used_{t}" for t in typ]].rename({f"ci_fraction_clean_used_{t}" : t for t in typ})
    ldf.columns = ldf.columns.map(rename_scen)
    ldf.rename({"local" : "PPA", "grid" : "grid imports"}, inplace=True)
    ldf = (100. * ldf).round(1)

    yl_ref = ldf.loc[:,'no\npolicy'].sum()
    yl_100RES = ldf.loc[:,'100%\nRES'].sum()
    plt.axhline(y = yl_ref, color = 'gray', linestyle="--", linewidth=0.8)
    plt.axhline(y = yl_100RES, color = 'gray', linestyle="--", linewidth=0.8)
    plt.axvline(x = 0.5, color = 'gray', linestyle="--")
    plt.text(3.5,yl_ref-6,f'Reference case, fraction CFE={int(round(yl_ref,0))}'+'%', 
            horizontalalignment='left', bbox=dict(facecolor='w', alpha=0.5)) 
    
    #Drop reference scenario before plotting
    ldf.drop(ldf.columns[0], axis=1, inplace=True)

    ldf.T.plot(kind="bar",stacked=True,
                ax=ax, color=tech_colors, width=0.65, edgecolor = "black", linewidth=0.05)

    plt.xticks(rotation=0)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    #ax.set_xlabel("CFE target")
    ax.set_ylabel("fraction CFE [%]")
    ax.set_ylim([0,110])
    ax.legend(loc="upper left", ncol=2, prop={"size":10})

    fig.tight_layout()
    fig.savefig(snakemake.output.used, transparent=True)


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

    #ldf.index = ldf.index.map(rename_ci_cost)
    ldf.columns = ldf.columns.map(rename_scen)
    ldf.index = ldf.index.map(rename_ci_capacity)
    new_index = preferred_order.intersection(ldf.index).append(ldf.index.difference(preferred_order))
    ldf = ldf.loc[new_index]

    plt.axvline(x = 0.5, color = 'gray', linestyle="--")

    #yl_100RES = ldf.loc[:,'100%\nRES'].sum()
    #ax.set_ylim([0,yl_100RES*1.3/1e3])

    #Drop reference scenario before plotting
    ldf.drop(ldf.columns[0], axis=1, inplace=True)

    (ldf/1e3).T.plot(kind="bar",stacked=True,
                ax=ax, color=tech_colors, width=0.65, edgecolor = "black", linewidth=0.05)

    plt.xticks(rotation=0)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim([0,max(ldf.sum())*1.3/1e3])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    #ax.set_xlabel("CFE target")
    ax.set_ylabel("C&I portfolio capacity [GW]")
    ax.legend(loc="upper left", ncol=2, prop={"size":9})

    fig.tight_layout()
    fig.savefig(snakemake.output.used.replace("used.pdf","ci_capacity.pdf"),
                transparent=True)


def ci_generation():

    fig, ax = plt.subplots()
    fig.set_size_inches((6,4.5))

    generation = df.loc[["ci_generation_" + t for t in clean_techs]].rename({"ci_generation_" + t : t for t in clean_techs})/1000.
    discharge = df.loc[["ci_generation_" + t for t in clean_dischargers]].rename({"ci_generation_" + t : t for t in clean_dischargers})/1000.

    ldf = pd.concat([generation, discharge])

    to_drop = ldf.index[(ldf < 0.1).all(axis=1)]
    ldf.drop(to_drop, inplace=True)

    ldf.columns = ldf.columns.map(rename_scen)
    ldf.index = ldf.index.map(rename_ci_capacity)
    new_index = preferred_order.intersection(ldf.index).append(ldf.index.difference(preferred_order))
    ldf = ldf.loc[new_index]

    plt.axvline(x = 0.5, color = 'gray', linestyle="--")

    #Drop reference scenario before plotting
    ldf.drop(ldf.columns[0], axis=1, inplace=True)

    (ldf/1e3).T.plot(kind="bar",stacked=True,
                ax=ax, color=tech_colors, width=0.65, edgecolor = "black", linewidth=0.05)

    plt.xticks(rotation=0)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    #ax.set_xlabel("CFE target")
    ax.set_ylabel("C&I generation [TWh]")
    ax.legend(loc="upper left", ncol=2, prop={"size":9})

    fig.tight_layout()
    fig.savefig(snakemake.output.used.replace("used.pdf","ci_generation.pdf"),
                transparent=True)


def system_emissions():

    fig, ax = plt.subplots()
    fig.set_size_inches((6,4.5))

    ldf = (df/1e6).loc['emissions']
    ldf.index = ldf.index.map(rename_scen)
    
    yl_ref = ldf.loc['no\npolicy']
    yl_100RES = ldf.loc['100%\nRES']
    yl_end = ldf[-1]
    ax.set_ylim([yl_100RES*0.8, yl_100RES*1.1])
    plt.axhline(y = yl_ref, color = 'gray', linestyle="-", linewidth=1.5)
    plt.axhline(y = yl_100RES, color = 'gray', linestyle="--", linewidth=0.8)
    plt.axhline(y = yl_end, color = 'gray', linestyle="--", linewidth=0.8)
    plt.axvline(x = 0.5, color = 'gray', linestyle="--")
    plt.text(3.3, yl_ref+0.3,f'Reference case, emissions {round(yl_ref,1)} [Mt]', 
            horizontalalignment='left', bbox=dict(facecolor='w', alpha=0.5)) 
    
    #Drop reference scenario before plotting
    ldf.drop(ldf.index[0], inplace=True)

    ldf.plot(kind="bar", ax=ax, 
        color='#33415c', width=0.65, edgecolor = "black", linewidth=0.05)
 
    plt.xticks(rotation=0)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    #ax.set_xlabel("CFE target")
    ax.set_ylabel("system emissions [MtCO$_2$/a]")

    fig.tight_layout()
    fig.savefig(snakemake.output.used.replace("used.pdf","system_emissions.pdf"),
                transparent=True)


def zone_emissions():

    fig, ax = plt.subplots()
    fig.set_size_inches((6,4.5))

    ldf = (df).loc['emissions_zone']
    ldf.index = ldf.index.map(rename_scen)
    
    yl_ref = ldf.loc['no\npolicy']
    yl_100RES = ldf.loc['100%\nRES']
    yl_end = ldf[-1]
    ax.set_ylim([yl_100RES*0.8, yl_ref*1.1])
    plt.axhline(y = yl_ref, color = 'gray', linestyle="-", linewidth=1.5)
    plt.axhline(y = yl_100RES, color = 'gray', linestyle="--", linewidth=0.8)
    plt.axhline(y = yl_end, color = 'gray', linestyle="--", linewidth=0.8)
    plt.axvline(x = 0.5, color = 'gray', linestyle="--")
    plt.text(3.3, yl_ref-0.02*yl_ref,f'Reference case, emissions {round(yl_ref,1)} [Mt]', 
            horizontalalignment='left', bbox=dict(facecolor='w', alpha=0.5)) 
    
    #Drop reference scenario before plotting
    ldf.drop(ldf.index[0], inplace=True)

    ldf.plot(kind="bar", ax=ax, 
        color='#33415c', width=0.65, edgecolor = "black", linewidth=0.05)
 
    plt.xticks(rotation=0)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    #ax.set_xlabel("CFE target")
    ax.set_ylabel("Emissions in local zone [MtCO$_2$/a]")

    fig.tight_layout()
    fig.savefig(snakemake.output.used.replace("used.pdf","zone_emissions.pdf"),
                transparent=True)


def ci_emisrate():

    fig, ax = plt.subplots()
    fig.set_size_inches((6,4.5))

    ldf = df.loc['ci_emission_rate_true']
    ldf.index = ldf.index.map(rename_scen)
   
    yl_ref = ldf.loc['no\npolicy']
    yl_100RES = ldf.loc['100%\nRES']
    plt.axhline(y = yl_ref, color = 'gray', linestyle="-", linewidth=1.5)
    plt.axhline(y = yl_100RES, color = 'gray', linestyle="--", linewidth=0.8)
    plt.axvline(x = 0.5, color = 'gray', linestyle="--")
    plt.text(1.5, yl_ref-0.06*yl_ref,f'Reference case, C&I emission rate {round(yl_ref,3)} [t/MWh]', 
            horizontalalignment='left', bbox=dict(facecolor='w', alpha=0.5)) 
    
    #Drop reference scenario before plotting
    ldf.drop(ldf.index[0], inplace=True)

    ldf.plot(kind="bar", ax=ax,
        color='#33415c', width=0.65, edgecolor = "black", linewidth=0.05)
 
    plt.xticks(rotation=0)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    #ax.set_xlabel("CFE target")
    ax.set_ylabel("C&I emission rate [t/MWh]")
    #ax.yaxis.label.set_size(6)

    fig.tight_layout()
    fig.savefig(snakemake.output.used.replace("used.pdf","ci_emisrate.pdf"),
                transparent=True)


def ci_cost():

    fig, ax = plt.subplots()
    fig.set_size_inches((6,4.5))

    techs = clean_techs + ["grid",
                           "battery_storage",
                           "battery_inverter",
                           "hydrogen_storage",
                           "hydrogen_electrolysis",
                           "hydrogen_fuel_cell"]

    ldf = df.loc[["ci_cost_" + t for t in techs]].rename({"ci_cost_" + t : t for t in techs}).multiply(1/df.loc["ci_demand_total"],axis=1)

    to_drop = ldf.index[(ldf < 0.1).all(axis=1)]
    ldf.drop(to_drop, inplace=True)

    ldf.columns = ldf.columns.map(rename_scen)
    ldf = ldf.groupby(rename_ci_cost).sum()
    new_index = preferred_order.intersection(ldf.index).append(ldf.index.difference(preferred_order))
    ldf = ldf.loc[new_index]

    #yl_ref = ldf.loc[:,'no\npolicy'].sum()
    yl_end = ldf.loc[:,ldf.columns[-1]].sum()
    #plt.axhline(y = yl_ref, color = 'gray', linestyle="--", linewidth=0.8)
    plt.axvline(x = 0.5, color = 'gray', linestyle="--")

    #Drop reference scenario before plotting
    ldf.drop(ldf.columns[0], axis=1, inplace=True)

    ldf.T.plot(kind="bar",stacked=True,
               ax=ax, color=tech_colors, width=0.65, edgecolor = "black", linewidth=0.05)

    plt.xticks(rotation=0)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    #ax.set_xlabel("CFE target")
    ax.set_ylabel("24x7 C&I cost [€/MWh]")
    ax.legend(loc="upper left", ncol = 3, prop={"size":9})
    ax.set_ylim(top=yl_end*1.4)

    fig.tight_layout()
    fig.savefig(snakemake.output.used.replace("used.pdf","ci_cost.pdf"),
                transparent=True)


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

    ldf.columns = ldf.columns.map(rename_scen)
    ldf = ldf.groupby(rename_ci_cost).sum()
    new_index = preferred_order.intersection(ldf.index).append(ldf.index.difference(preferred_order))
    ldf = ldf.loc[new_index]

    yl_end = ldf.loc[:,ldf.columns[-1]].sum()
    plt.axhline(y = yl_end, color = 'gray', linestyle="--", linewidth=0.8)
    plt.axhline(y = 0, color = 'black', linestyle="-", linewidth=0.1)
    plt.axvline(x = 0.5, color = 'gray', linestyle="--")
    plt.text(0.6, yl_end+1,f'net cost at 100% 24x7 CFE', 
            horizontalalignment='left') 
    
    #Drop reference scenario before plotting
    ldf.drop(ldf.columns[0], axis=1, inplace=True)

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
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    #ax.set_xlabel("CFE target")
    ax.set_ylabel("24x7 C&I cost and revenue [€/MWh]")
    ax.legend(loc="upper left", ncol = 3, prop={"size":8})
    ax.set_ylim(top=yl_end*1.5)

    fig.tight_layout()
    fig.savefig(snakemake.output.used.replace("used.pdf","ci_costandrev.pdf"),
                transparent=True)


def system_capacity():

    fig, ax = plt.subplots()
    fig.set_size_inches((6,4.5))

    gens = df.loc[["system_inv_" + t for t in exp_generators]].rename({"system_inv_" + t : t for t in exp_generators})
    links = df.loc[["system_inv_" + t for t in exp_links]].rename({"system_inv_" + t : t for t in exp_links})
    dischargers = df.loc[["system_inv_" + t for t in exp_dischargers]].rename({"system_inv_" + t : t for t in exp_dischargers})
    chargers = df.loc[["system_inv_" + t for t in exp_chargers]].rename({"system_inv_" + t : t for t in exp_chargers})
    chargers = chargers.drop(['battery_charger-%s' % year]) # display only battery discharger capacity

    ldf = pd.concat([gens, links, dischargers, chargers])

    to_drop = ldf.index[(ldf < 0.1).all(axis=1)]
    ldf.drop(to_drop, inplace=True)

    ldf.columns = ldf.columns.map(rename_scen)
    ldf = ldf.groupby(rename_system_simple).sum()
    new_index = preferred_order.intersection(ldf.index).append(ldf.index.difference(preferred_order))
    ldf = ldf.loc[new_index]
    
    plt.axvline(x = 0.5, color = 'gray', linestyle="--")

    #Drop reference scenario before plotting
    ldf.drop(ldf.columns[0], axis=1, inplace=True)

    (ldf/1e3).T.plot(kind="bar",stacked=True,
               ax=ax, color=tech_colors, width=0.65, edgecolor = "black", linewidth=0.05)

    plt.xticks(rotation=0)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    #ax.set_xlabel("CFE target")
    ax.set_ylabel("System capacity investment [GW]")
    ax.legend(loc="lower right", ncol=2, prop={"size":9})

    fig.tight_layout()
    fig.savefig(snakemake.output.used.replace("used.pdf","system_capacity.pdf"),
                transparent=True)


def system_capacity_diff():

    fig, ax = plt.subplots()
    fig.set_size_inches((6,4.5))

    gens = df.loc[["system_inv_" + t for t in exp_generators]].rename({"system_inv_" + t : t for t in exp_generators})
    links = df.loc[["system_inv_" + t for t in exp_links]].rename({"system_inv_" + t : t for t in exp_links})
    dischargers = df.loc[["system_inv_" + t for t in exp_dischargers]].rename({"system_inv_" + t : t for t in exp_dischargers})
    chargers = df.loc[["system_inv_" + t for t in exp_chargers]].rename({"system_inv_" + t : t for t in exp_chargers})
    chargers = chargers.drop(['battery_charger-%s' % year]) # display only battery discharger capacity

    ldf = pd.concat([gens, links, dischargers, chargers])
    to_drop = ldf.index[(ldf < 0.1).all(axis=1)]
    ldf.drop(to_drop, inplace=True)
    ldf = ldf.sub(ldf.ref,axis=0)

    ldf.columns = ldf.columns.map(rename_scen)
    ldf = ldf.groupby(rename_system_simple).sum()
    new_index = preferred_order.intersection(ldf.index).append(ldf.index.difference(preferred_order))
    ldf = ldf.loc[new_index]

    plt.axvline(x = 0.5, color = 'gray', linestyle="--")

    #Drop reference scenario before plotting
    ldf.drop(ldf.columns[0], axis=1, inplace=True)

    col_list= list(ldf)
    ldf_pos = ldf[ldf>0]
    ldf_neg = ldf[ldf<0]
    up = ldf_pos[col_list].sum(axis=0).max()/1e3
    lo = ldf_neg[col_list].sum(axis=0).min()/1e3
    ax.set_ylim(bottom=lo*1.1, top=up*2)

    (ldf/1e3).T.plot(kind="bar",stacked=True,
               ax=ax, color=tech_colors, width=0.65, edgecolor = "black", linewidth=0.05)

    plt.xticks(rotation=0)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    #ax.set_xlabel("CFE target")
    ax.set_ylabel(f"System capacity diff. [GW]")
    ax.legend(loc='upper left', ncol=3, prop={"size":8}, fancybox=True)
    fig.tight_layout()
    fig.savefig(snakemake.output.used.replace("used.pdf","system_capacity_diff.pdf"),
                transparent=True)


def total_capacity_diff():

    fig, ax = plt.subplots()
    fig.set_size_inches((6,4.5))

    #system capacity
    gens = df.loc[["system_inv_" + t for t in exp_generators]].rename({"system_inv_" + t : t for t in exp_generators})
    links = df.loc[["system_inv_" + t for t in exp_links]].rename({"system_inv_" + t : t for t in exp_links})
    dischargers = df.loc[["system_inv_" + t for t in exp_dischargers]].rename({"system_inv_" + t : t for t in exp_dischargers})
    chargers = df.loc[["system_inv_" + t for t in exp_chargers]].rename({"system_inv_" + t : t for t in exp_chargers})
    chargers = chargers.drop(['battery_charger-%s' % year]) # display only battery discharger capacity

    ldf_system = pd.concat([gens, links, dischargers, chargers])

    to_drop = ldf_system.index[(ldf_system < 0.1).all(axis=1)]
    ldf_system.drop(to_drop, inplace=True)
    ldf_system = ldf_system.groupby(rename_system_simple).sum()

    #CI capacity
    gen_inv = df.loc[["ci_cap_" + t for t in clean_techs]].rename({"ci_cap_" + t : t for t in clean_techs})
    discharge_inv = df.loc[["ci_cap_" + t for t in clean_dischargers]].rename({"ci_cap_" + t : t for t in clean_dischargers})
    charge_inv = df.loc[["ci_cap_" + t for t in clean_chargers]].rename({"ci_cap_" + t : t for t in clean_chargers})
    charge_inv = charge_inv.drop(['battery_charger']) # display only battery discharger capacity
    
    ldf_ci = pd.concat([gen_inv, charge_inv, discharge_inv])

    to_drop = ldf_ci.index[(ldf_ci < 0.1).all(axis=1)]
    ldf_ci.drop(to_drop, inplace=True)
    ldf_ci.index = ldf_ci.index.map(rename_ci_capacity)

    #Total system capacity
    ldf = ldf_system.add(ldf_ci, fill_value=0)

    #Calculate diff, rename scenarios, order techs
    ldf = ldf.sub(ldf.ref,axis=0)
    ldf.columns = ldf.columns.map(rename_scen)
    new_index = preferred_order.intersection(ldf.index).append(ldf.index.difference(preferred_order))
    ldf = ldf.loc[new_index]

    plt.axvline(x = 0.5, color = 'gray', linestyle="--")

    #Drop reference scenario before plotting
    ldf.drop(ldf.columns[0], axis=1, inplace=True)

    (ldf/1e3).T.plot(kind="bar",stacked=True,
               ax=ax, color=tech_colors, width=0.65, edgecolor = "black", linewidth=0.05)

    plt.xticks(rotation=0)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    #ax.set_xlabel("CFE target")
    ax.set_ylabel(f"System capacity expansion \n diff. to no CFE procurement. [GW]")

    ax.legend(loc="upper left", ncol=2, prop={"size":7})

    fig.tight_layout()
    fig.savefig(snakemake.output.used.replace("used.pdf","total_capacity_diff.pdf"),
                transparent=True)


def objective_rel():

    fig, ax = plt.subplots()
    fig.set_size_inches((6,4.5))

    values = (df/1e9).loc['objective']
    
    #The first scenario in config is reference case -> by default ref
    ref = values[0]
    l, scens = [], {}
    for count, (index, value) in enumerate(values.iteritems()):
        scens[count] = f'{index}'
        l.append((value - ref)/ref*100)

    ldf=pd.Series(l)
    ldf.index = ldf.index.map(scens)
    ldf.index = ldf.index.map(rename_scen)

    plt.axvline(x = 1.5, color = 'gray', linestyle="--")

    ldf.plot(kind="bar", ax=ax,
            color='#4a3a28', width=0.65, edgecolor = "black", linewidth=0.05, alpha=0.95)
 
    plt.xticks(rotation=0)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    #ax.set_xlabel("CFE target")
    ax.set_ylabel(f"obj % increase to reference case")

    fig.tight_layout()
    fig.savefig(snakemake.output.used.replace("used.pdf","system_objective_rel.pdf"),
                transparent=True)


def objective_abs():

    fig, ax = plt.subplots()
    fig.set_size_inches((4,3))

    ldf = (df/1e9).loc['objective']
    ldf.index = ldf.index.map(rename_scen)

    ldf.plot(kind="bar", ax=ax)

    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    #ax.set_xlabel("CFE target")
    ax.set_ylabel("Objective [EUR 1e9]")

    fig.tight_layout()
    fig.savefig(snakemake.output.used.replace("used.pdf","system_objective_abs.pdf"),
                transparent=True)



if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('plot_summary', palette='p3', zone='DK', year='2030', participation='10')   

    #Windcards & Settings
    tech_palette = snakemake.wildcards.palette
    year = snakemake.wildcards.year
    zone = snakemake.wildcards.zone
    participation = snakemake.wildcards.participation
    area = snakemake.config['area']
    name = snakemake.config['ci']['name']
    node = geoscope(zone, area)['node']


    #tech_palette options
    clean_techs = palette(tech_palette)[0]
    storage_techs = palette(tech_palette)[1]
    storage_chargers = palette(tech_palette)[2]
    storage_dischargers = palette(tech_palette)[3]

    #renaming technologies for plotting
    clean_chargers = [g for g in storage_chargers]
    clean_chargers = [item.replace(' ', '_') for item in clean_chargers]
    clean_dischargers = [g for g in storage_dischargers]
    clean_dischargers = [item.replace(' ', '_') for item in clean_dischargers]
    
    exp_generators = ['offwind-ac-%s' % year, 
                    'offwind-dc-%s' % year, 
                    'onwind-%s' % year, 
                    'solar-%s' % year]
    exp_links = ['OCGT-%s' % year]
    exp_chargers = ['battery charger-%s' % year, 'H2 Electrolysis-%s' % year]
    exp_dischargers = ['battery discharger-%s' % year, 'H2 Fuel Cell-%s' % year]

    exp_generators = [item.replace(' ', '_') for item in exp_generators]
    exp_chargers = [item.replace(' ', '_') for item in exp_chargers]
    exp_dischargers = [item.replace(' ', '_') for item in exp_dischargers]

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
        #"adv_nuclear" : "advanced dispatchable",
        #'adv_geothermal': "advanced geothermal",
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

    rename_system_simple = {
        'offwind-ac-%s' % year: 'offshore wind',
        'offwind-dc-%s' % year: 'offshore wind',
        'onwind-%s' % year: 'onshore wind',
        'solar-%s' % year: 'solar',
        'OCGT-%s' % year: 'Gas OC',
        'battery_discharger-%s' % year: 'battery',
        'H2_Fuel_Cell-%s' % year: 'hydrogen fuel cell',
        'H2_Electrolysis-%s' % year: 'hydrogen electrolysis'
    }

    rename_scen = {'ref': 'no\npolicy',
                    'res100': '100%\nRES',
                    'cfe80': '80%',
                    'cfe85':'85%',
                    'cfe90':'90%',
                    'cfe95':'95%',
                    'cfe98':'98%',
                    'cfe99':'99%',
                    'cfe100':'100%'
                    }


    df = pd.read_csv(snakemake.input.summary,
                     index_col=0)

    #ci
    used()
    ci_cost()
    ci_costandrev()
    ci_capacity()
    ci_generation()
    ci_emisrate()

    #system
    zone_emissions()
    #system_emissions()
    system_capacity()
    #objective_abs()
    
    #diffs to reference case
    #objective_rel()
    system_capacity_diff()
    total_capacity_diff()

