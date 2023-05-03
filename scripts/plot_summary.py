import pypsa
import pandas as pd
import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FormatStrFormatter
from matplotlib.transforms import Bbox
import matplotlib.colors as mc 
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as ticker

#allow plotting without Xwindows
matplotlib.use('Agg')

from solve_network import palette, geoscope
from pypsa.plot import add_legend_patches


def format_column_names(col_tuple):
    return f"{col_tuple[0]}{col_tuple[1][:2]}"

def ci_capacity():

    fig, ax = plt.subplots()
    fig.set_size_inches((8,4.5))

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

    #Sort final dataframe before plotting
    ldf.sort_index(axis='columns', level=[1,0], ascending=[False, True], inplace=True)

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

    formatted_columns = [format_column_names(col) for col in ldf.columns.tolist()]
    ax.set_xticklabels(formatted_columns)

    fig.tight_layout()
    fig.savefig(snakemake.output.plot, transparent=True)


def ci_costandrev():

    fig, ax = plt.subplots()
    fig.set_size_inches((8,4.5))

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

    #Sort final dataframe before plotting
    ldf.sort_index(axis='columns', level=[1,0], ascending=[False, True], inplace=True)

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
    ax.set_ylim(top=max(ldf.sum())*2)

    formatted_columns = [format_column_names(col) for col in ldf.columns.tolist()]
    ax.set_xticklabels(formatted_columns)

    fig.tight_layout()
    fig.savefig(snakemake.output.plot.replace("capacity.pdf","ci_costandrev.pdf"), transparent=True)


def ci_generation():

    fig, ax = plt.subplots()
    fig.set_size_inches((8,4.5))

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

    #Sort final dataframe before plotting
    ldf.sort_index(axis='columns', level=[1,0], ascending=[False, True], inplace=True)

    (ldf).T.plot(kind="bar",stacked=True,
                ax=ax, color=tech_colors, width=0.65, edgecolor = "black", linewidth=0.05)

    plt.xticks(rotation=0)
    ax.set_xticklabels([''.join(item) for item in ldf.columns.tolist()])
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim([0,max(ldf.sum())*1.3])
    #ax.set_xlabel("CFE target")
    ax.set_ylabel("DC portfolio generation [GWh]")
    ax.legend(loc="upper left", ncol=2, prop={"size":9})

    formatted_columns = [format_column_names(col) for col in ldf.columns.tolist()]
    ax.set_xticklabels(formatted_columns)

    fig.tight_layout()
    fig.savefig(snakemake.output.plot.replace("capacity.pdf","ci_generation.pdf"), transparent=True)


def zone_emissions():

    fig, ax = plt.subplots()
    fig.set_size_inches((8,4.5))

    ldf = df.loc['emissions_zone'] #.to_frame()

    ldf.index = ldf.index.set_levels(ldf.index.levels[0].map(rename_scen), level=0)

    #Sort final dataframe before plotting
    ldf.sort_index(axis='rows', level=[1,0], ascending=[False, True], inplace=True)
    
    ldf.plot(kind="bar", ax=ax, 
        color='#33415c', width=0.65, edgecolor = "black", linewidth=0.05)
 
    plt.xticks(rotation=0)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    #ax.set_xlabel("CFE target")
    ax.set_ylabel("Emissions in local zone [MtCO$_2$/a]")

    formatted_columns = [format_column_names(col) for col in ldf.index.tolist()]
    ax.set_xticklabels(formatted_columns)

    fig.tight_layout()
    fig.savefig(snakemake.output.plot.replace("capacity.pdf","zone_emissions.pdf"), transparent=True)


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

    #select any location since system-wide values are the same
    ldf = ldf.xs(locations[0], axis='columns', level=1)

    #subtract 0% flex scenario from each column
    ldf = ldf.sub(ldf.iloc[:, 0], axis=0)

    ldf.columns = ldf.columns.map(rename_scen)
    ldf = ldf.groupby(rename_system_simple).sum()
    new_index = preferred_order.intersection(ldf.index).append(ldf.index.difference(preferred_order))
    ldf = ldf.loc[new_index]

    (ldf).T.plot(kind="bar",stacked=True,
               ax=ax, color=tech_colors, width=0.65, edgecolor = "black", linewidth=0.05)

    plt.xticks(rotation=0)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    #ax.set_xlabel("CFE target")
    ax.set_ylabel(f"System capacity diff. [MW]")

    ldf_pos = ldf[ldf>0]
    ldf_neg = ldf[ldf<0]
    up = ldf_pos.sum(axis=0).max()
    lo = ldf_neg.sum(axis=0).min()
    ax.set_ylim(bottom=lo*1.1, top=up*2)

    ax.legend(loc='lower left', ncol=3, prop={"size":8}, fancybox=True)
    fig.tight_layout()
    fig.savefig(snakemake.output.plot.replace("capacity.pdf","system_capacity_diff.pdf"),
                transparent=True)


def ci_curtailment():

    fig, ax = plt.subplots()
    fig.set_size_inches((6,4.5))

    #Data for ci res curtailment across all locations
    ci_res = snakemake.config['ci']['res_techs']
    ldf = df.loc[["ci_curtailment_" + t for t in ci_res]].rename({"ci_curtailment_" + t: t for t in ci_res})

    #Refine data
    ldf.rename(columns=rename_scen, level=0, inplace=True) 
    ldf = pd.DataFrame(ldf.sum(axis=0), columns=['RES curtailment']).unstack()
    ldf = ldf['RES curtailment']/1e3 

    ldf.plot(kind="bar",stacked=True,
                ax=ax, width=0.65, edgecolor = "black", linewidth=0.05)

    plt.xticks(rotation=0)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylabel("DC portfolio RES curtailment [GWh]")
    up = ldf.sum(axis=1).max()
    ax.set_ylim(top=up*1.5)

    ax.legend(loc='upper right', ncol=2, prop={"size":8}, fancybox=True)
    fig.tight_layout()
    fig.savefig(snakemake.output.plot.replace("capacity.pdf","ci_curtailment.pdf"),
                transparent=True)


def system_curtailment():

    fig, ax = plt.subplots()
    fig.set_size_inches((6,4.5))

    #Data for system-wide res curtailment
    system_res = ["offwind","offwind-ac","offwind-dc","onwind", "ror", "solar", "hydro"]
    existing_rows = [t for t in system_res if "system_curtailment_" + t in df.index]

    ldf = df.loc[["system_curtailment_" + t for t in existing_rows]].rename({"system_curtailment_" + t: t for t in existing_rows})
    ldf = ldf.xs(locations[0], axis='columns', level=1)

    #Refine data
    ldf.rename(columns=rename_scen, level=0, inplace=True) 
    ldf = ldf.groupby(rename_curtailment).sum()
    new_index = preferred_order.intersection(ldf.index).append(ldf.index.difference(preferred_order))
    ldf = ldf.loc[new_index]
    ldf = ldf/1e3 

    #subtract 0% flex scenario from each column
    ldf = ldf.sub(ldf.iloc[:, 0], axis=0)

    ldf.T.plot(kind="bar",stacked=True, color=tech_colors,
                ax=ax, width=0.65, edgecolor = "black", linewidth=0.05)

    plt.xticks(rotation=0)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylabel(f"System RES curtailment | diff to 0. flex [GWh]")

    ldf_pos = ldf[ldf>0]
    ldf_neg = ldf[ldf<0]
    up = ldf_pos.sum(axis=0).max()
    lo = ldf_neg.sum(axis=0).min()
    ax.set_ylim(bottom=lo*1.1, top=up*1.5)

    ax.legend(loc='upper right', ncol=2, prop={"size":8}, fancybox=True)
    fig.tight_layout()
    fig.savefig(snakemake.output.plot.replace("capacity.pdf","system_curtailment.pdf"),
                transparent=True)


def ci_abs_costs():

    fig, ax = plt.subplots()
    fig.set_size_inches((6,4.5))

    ldf = (df.loc['ci_total_cost'] - df.loc['ci_revenue_grid'])/1e6
    ldf = ldf.to_frame(name='ci_abs_costs')

    ldf.index = ldf.index.set_levels(ldf.index.levels[0].map(rename_scen), level=0)
    ldf=ldf['ci_abs_costs'].unstack()
    #Sort final dataframe before plotting
    ldf.sort_index(axis='rows', ascending=True, inplace=True)
    
    ldf.plot(kind="bar", stacked=True, ax=ax, 
        width=0.65, edgecolor = "black", linewidth=0.05)
 
    plt.xticks(rotation=0)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    
    value = (ldf.sum(axis=1)[0] -  ldf.sum(axis=1)[-1])
    ax.set_xlabel(f"Cost diff for min/max flex scenarios: {round(value, 1)} MEUR/a")
    plt.axhline(y = ldf.sum(axis=1).max(), color = 'gray', linestyle="--", linewidth=1.5)
    plt.axhline(y = ldf.sum(axis=1).min(), color = 'gray', linestyle="--", linewidth=1.5)

    ax.set_ylabel("24/7 CFE net annual costs [MEUR per year]")

    fig.tight_layout()
    fig.savefig(snakemake.output.plot.replace("capacity.pdf","ci_abs_costs.pdf"), transparent=True)


def objective_abs():

    fig, ax = plt.subplots()
    fig.set_size_inches((6,4.5))

    ldf = df.loc['objective']/1e6
    ldf = ldf.to_frame(name='objective')

    ldf.index = ldf.index.set_levels(ldf.index.levels[0].map(rename_scen), level=0)
    ldf=ldf['objective'].unstack()
    ldf.sort_index(axis='rows', ascending=True, inplace=True)

    ldf = ldf.sub(ldf.iloc[0, :])
    #select any location since system-wide values are the same
    ldf = ldf[locations[0]]

    ldf.plot(kind="bar", ax=ax,
            color='#33415c', width=0.65, edgecolor = "black", linewidth=0.05)

    plt.xticks(rotation=0)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylabel("Objective [MEUR]")

    value = -(ldf[0] - ldf[-1])
    plt.axhline(y = ldf.min(), color = 'gray', linestyle="--", linewidth=1.5)
    ax.set_xlabel(f"Obj. diff for min/max flex scenarios: \n {round(value, 1)} MEUR/a")

    fig.tight_layout()
    fig.savefig(snakemake.output.plot.replace("capacity.pdf","objective.pdf"),
                transparent=True)



### Plotting time-series data

def heatmap(data, month, year, ax):
    data = df[(df["snapshot"].dt.month == month)]

    snapshot = data["snapshot"]
    day = data["snapshot"].dt.day
    value = data["iteration 1"]
    value = value.values.reshape(int(24/scaling), len(day.unique()), order="F") # 8 clusters of 3h in each day
    
    xgrid = np.arange(day.max() + 1) + 1  # The inner + 1 increases the length, the outer + 1 ensures days start at 1, and not at 0
    ygrid = np.arange(int(24/scaling)+1) # hours (sampled or not) + extra 1
    
    ax.pcolormesh(xgrid, ygrid, value, cmap=colormap, vmin=MIN, vmax=MAX)
    # Invert the vertical axis
    ax.set_ylim(int(24/scaling), 0)
    # Set tick positions for both axes
    ax.yaxis.set_ticks([]) #[i for i in range(int(24/scaling))]
    ax.xaxis.set_ticks([])
    # Remove ticks by setting their length to 0
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    
    # Remove all spines
    ax.set_frame_on(False)


def heatmap_utilization(data, month, year, ax, carrier):
    data = df[(df["snapshot"].dt.month == month)]

    snapshot = data["snapshot"]
    day = data["snapshot"].dt.day
    value = data[f"{carrier}"]
    value = value.values.reshape(int(24/scaling), len(day.unique()), order="F") # 8 clusters of 3h in each day
    
    xgrid = np.arange(day.max() + 1) + 1  # The inner + 1 increases the length, the outer + 1 ensures days start at 1, and not at 0
    ygrid = np.arange(int(24/scaling)+1) # hours (sampled or not) + extra 1
    
    ax.pcolormesh(xgrid, ygrid, value, cmap=colormap, vmin=MIN, vmax=MAX)
    # Invert the vertical axis
    ax.set_ylim(int(24/scaling), 0)
    # Set tick positions for both axes
    ax.yaxis.set_ticks([]) #[i for i in range(int(24/scaling))]
    ax.xaxis.set_ticks([])
    # Remove ticks by setting their length to 0
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    
    # Remove all spines
    ax.set_frame_on(False)


def plot_heatmap_cfe():
    # Here 1 row year/variable and 12 columns for month
    fig, axes = plt.subplots(1, 12, figsize=(14, 5), sharey=True)
    plt.tight_layout()

    for i, year in enumerate([2013]):
        for j, month in enumerate(range(1, 13)):
            #print(f'j: {j}, month: {month}')
            heatmap(df, month, year, axes[j])

    # Adjust margin and space between subplots (extra space is on the left for a label)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.9, hspace=0.08, wspace=0) #wspace=0 stacks individual months together but easy to split

    # some room for the legend in the bottom
    fig.subplots_adjust(bottom=0.2)

    # Create a new axis to contain the color bar
    # Values are: (x coordinate of left border, y coordinate for bottom border, width, height)
    cbar_ax = fig.add_axes([0.3, 0.03, 0.4, 0.04])

    # Create a normalizer that goes from minimum to maximum temperature
    norm = mc.Normalize(0, 1) #for CFE, otherwise (MIN, MAX)

    # Create the colorbar and set it to horizontal
    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap=colormap), 
        cax=cbar_ax, # Pass the new axis
        orientation = "horizontal")

    # Remove tick marks and set label
    cb.ax.xaxis.set_tick_params(size=0)
    cb.set_label("Hourly CFE score of electricity supply from grid", size=12)

    # add some figure labels and title
    fig.text(0.5, 0.15, "Days of year", ha="center", va="center", fontsize=14)
    fig.text(0.03, 0.5, 'Hours of a day', ha="center", va="center", rotation="vertical", fontsize=14)
    fig.suptitle(f"Carbon Heat Map | {location}", fontsize=20, y=0.98)

    path = snakemake.output.plot.split('capacity.pdf')[0] + f'heatmaps'
    if not os.path.exists(path):
        os.makedirs(path)

    fig.savefig(path + '/' + f"{flex}_GridCFE_{location}.pdf", 
        bbox_inches = "tight", transparent=True)


def plot_heatmap_utilization(carrier):
    # Here 1 row year/variable and 12 columns for month
    fig, axes = plt.subplots(1, 12, figsize=(14, 5), sharey=True)
    plt.tight_layout()

    for i, year in enumerate([2013]):
        for j, month in enumerate(range(1, 13)):
            #print(f'j: {j}, month: {month}')
            heatmap_utilization(df, month, year, axes[j], carrier=carrier)

    # Adjust margin and space between subplots (extra space is on the left for a label)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.9, hspace=0.08, wspace=0.04) #wspace=0 stacks individual months together but easy to split

    # some room for the legend in the bottom
    fig.subplots_adjust(bottom=0.2)

    # Create a new axis to contain the color bar
    # Values are: (x coordinate of left border, y coordinate for bottom border, width, height)
    cbar_ax = fig.add_axes([0.3, 0.03, 0.4, 0.04])

    # Create a normalizer that goes from minimum to maximum temperature
    norm = mc.Normalize(MIN, MAX)

    # Create the colorbar and set it to horizontal
    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap=colormap), 
        cax=cbar_ax, # Pass the new axis
        orientation = "horizontal")

    # Remove tick marks and set label
    cb.ax.xaxis.set_tick_params(size=0)
    cb.set_label(f"Hourly {carrier} [MW]", size=12)

    # add some figure labels and title
    fig.text(0.5, 0.15, "Days of year", ha="center", va="center", fontsize=14)
    fig.text(0.03, 0.5, 'Hours of a day', ha="center", va="center", rotation="vertical", fontsize=14)
    fig.suptitle(f"Flexibility utilization | {node}", fontsize=20, y=0.98)

    path = snakemake.output.plot.split('capacity.pdf')[0] + f'heatmaps'
    if not os.path.exists(path):
        os.makedirs(path)

    fig.savefig(path + '/' + f"{flex}_{carrier.replace(' ', '_')}_{node}.pdf", 
        bbox_inches = "tight", transparent=True)


def retrieve_nb(n, node):
    '''
    Retrieve nodal energy balance per hour
    This simple function works only for the Data center nodes: 
        -> lines and links are bidirectional AND their subsets are exclusive.
        -> links include fossil gens
    NB {-1} multiplier is a nodal balance sign
    '''

    components=['Generator', 'Load', 'StorageUnit', 'Store', 'Link', 'Line']
    nodal_balance = pd.DataFrame(index=n.snapshots)
    
    for i in components:
        if i == 'Generator':
            node_generators = n.generators.query('bus==@node').index
            nodal_balance = nodal_balance.join(n.generators_t.p[node_generators])
        if i == 'Load':
            node_loads = n.loads.query('bus==@node').index
            nodal_balance = nodal_balance.join(-1*n.loads_t.p_set[node_loads])
        if i == 'Link':
            node_export_links = n.links.query('bus0==@node').index
            node_import_links = n.links.query('bus1==@node').index
            nodal_balance = nodal_balance.join(-1*n.links_t.p0[node_export_links])
            nodal_balance = nodal_balance.join(-1*n.links_t.p1[node_import_links])
            ##################
        if i == 'StorageUnit':
            #node_storage_units = n.storage_units.query('bus==@node').index
            #nodal_balance = nodal_balance.join(n.storage_units_t.p_dispatch[node_storage_units])
            #nodal_balance = nodal_balance.join(n.storage_units_t.p_store[node_storage_units]) 
            continue   
        if i == 'Line':
            continue
        if i == 'Store':
            continue

        nodal_balance = nodal_balance.rename(columns=rename).groupby(level=0, axis=1).sum()

        # Custom groupby function
        def custom_groupby(column_name):
            if column_name.startswith('vcc'):
                return 'spatial shift'
            return column_name
        
        # Apply custom groupby function
        nodal_balance = nodal_balance.groupby(custom_groupby, axis=1).sum()

    return nodal_balance


def plot_balances(n, node, 
            start='2013-03-01 00:00:00', 
            stop='2013-03-08 00:00:00'
            ):

    fig, ax = plt.subplots()
    fig.set_size_inches((8,4.5))

    tech_colors = snakemake.config['tech_colors']
    df = retrieve_nb(n, node)

    #format time & set a range to display
    ldf = df.loc[start:stop,:]
    duration = (pd.to_datetime(stop)-pd.to_datetime(start)).days
    ldf.index = pd.to_datetime(ldf.index, format='%%Y-%m-%d %H:%M:%S').strftime('%m.%d %H:%M')

    #get colors
    for item in ldf.columns:
        if item not in tech_colors:
            print("Warning!",item,"not in config/tech_colors")

    #set logical order
    new_index = preferred_order_balances.intersection(ldf.columns).append(ldf.columns.difference(preferred_order_balances))
    ldf = ldf.loc[:,new_index]
    
    ldf.plot(kind="bar",stacked=True,
            color=tech_colors, 
            ax=ax, width=1, 
            #edgecolor = "black", linewidth=0.01
            )

    #visually ensure net energy balance at the node
    net_balance=ldf.sum(axis=1)
    x = 0
    for i in range(len(net_balance)):
        ax.scatter(x = x, y = net_balance[i], color='black', marker="_")
        x += 1

    plt.xticks(rotation=90)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.set(xlabel=None)
    majors = range(0,duration*24+1,12)
    ax.xaxis.set_major_locator(ticker.FixedLocator(majors))

    ax.set_ylabel("Nodal balance [MW*h/h]")

    add_legend_patches(
        ax,
        colors = [tech_colors[c] for c in ldf.columns],
        labels= ldf.columns,
        legend_kw= dict(bbox_to_anchor=(1, 1), loc = "upper left", frameon = False,
        ))

    fig.tight_layout()

    _start = ldf.index[0].split(' ')[0]
    _stop = ldf.index[-1].split(' ')[0]
    path = snakemake.output.plot.split('capacity.pdf')[0] + f'{_start}-{_stop}'
    if not os.path.exists(path):
        os.makedirs(path)

    fig.savefig(path + '/' + f"{flex}_balance_{node}.pdf")


def utilization_dc(names, flexibilities):

    fig, axs = plt.subplots(len(names), 1, figsize=(5, 1.3*len(names)), sharex=True, sharey=True)
    #fig.suptitle("Mean Spatial Shift for each DC by Flexibility Steps")

    mean_data = []

    for flex in flexibilities:
        n = pypsa.Network(snakemake.input.networks.split('0.nc')[0]+f'/{flex}.nc')
        colormap = "coolwarm"
        mean_flex = []

        if snakemake.config['ci']['spatial_shifting'] == True:
            for node in names:
                spatial_shift = retrieve_nb(n, node).get('spatial shift')

                if spatial_shift is not None:
                    df = spatial_shift
                else:
                    df = pd.Series(0, index=retrieve_nb(n, node).index, name='spatial shift')
                
                mean_flex.append(-df.mean()) #negative spatial shift implies increasing load

        mean_data.append(mean_flex)

    mean_df = pd.DataFrame(mean_data, columns=names, index=[int(f) for f in flexibilities])

    for idx, ax in enumerate(axs):
        mean_df.iloc[:, idx].plot(ax=ax, style='o-')  # Changed from 'bar' to 'o-' for dots with connecting lines
        ax.axhline(y=0.0, color='gray', linestyle='--')  # Dotted line at y=0.0
        ax.set_ylabel(f'{names[idx]}')

    axs[-1].set_xticks([int(f) for f in flexibilities])  # Set xticks positions using integers
    axs[-1].set_xticklabels([f"{f}%" for f in flexibilities])  # Set xticks labels with percentage signs
    axs[-1].set_xlabel('Flexibility Steps')

    plt.tight_layout()  # Adjust the layout to accommodate the suptitle
    #plt.show()
    path = snakemake.output.plot.split('capacity.pdf')[0] + f'utilization_dc'
    if not os.path.exists(path):
        os.makedirs(path)

    fig.savefig(path + '/' + f"utilization_dc.pdf")


####################################################################################################
if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('plot_summary', 
        year='2025', zone='EU', palette='p1', policy="cfe90")   

    config = snakemake.config
    scaling = int(config['time_sampling'][0]) #temporal scaling -- 3/1 for 3H/1H

    #Wildcards & Settings
    policy = snakemake.wildcards.policy[:3]
    penetration = float(snakemake.wildcards.policy[3:])/100 if policy != "ref" else 0
    tech_palette = snakemake.wildcards.palette
    zone = snakemake.wildcards.zone
    year = snakemake.wildcards.year

    datacenters = snakemake.config['ci']['datacenters']
    locations = list(datacenters.keys())
    names = list(datacenters.values())
    flexibilities = snakemake.config['scenario']['flexibility']

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

    rename_curtailment= pd.Series({
        "offwind" : "offshore wind",
        "offwind-ac" : "offshore wind",
        "offwind-dc" : "offshore wind",
        "onwind" : "onshore wind",
        "solar" : "solar",
        "ror" : 'hydroelectricity',
        "hydro": 'hydroelectricity'})

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

    rename_scen = {'0': '0.\n',
                   '5': '0.05\n',
                   '10': '0.1\n',
                    '20': '0.2\n',
                    '40': '0.4\n',
                    }

    rename = {}

    for name in names:
        temp = {
        f'{name} H2 Electrolysis': 'hydrogen storage',
        f'{name} H2 Fuel Cell': 'hydrogen storage',
        f'{name} battery charger': 'battery storage',
        f'{name} battery discharger': 'battery storage',
        f'{name} export': 'grid',
        f'{name} import': 'grid',
        f'{name} onwind': 'wind',	
        f'{name} solar': 'solar',
        f'{name} load': 'load',
        f'{name} adv_geothermal': "clean dispatchable",
        f'{name} allam_ccs': "NG-Allam",
        f'{name} DSM-delayout': 'temporal shift',
        f'{name} DSM-delayin': 'temporal shift',
        }
        rename = rename | temp

    
    preferred_order_balances = pd.Index([
        "clean dispatchable",
        "NG-Allam",
        "solar",
        "wind",
        "load",
        "battery storage",
        "hydrogen storage",
        'grid',
        'spatial shift',
        'temporal shift',
        ])

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


# SUMMARY PLOTS

# %matplotlib inline

df = pd.read_csv(snakemake.input.summary, index_col=0, header=[0,1])

ci_capacity()
ci_costandrev()
ci_generation()
ci_abs_costs()
zone_emissions()
system_capacity_diff()
objective_abs()

ci_curtailment()
system_curtailment()

# utilization_dc(names, flexibilities)


# TIME-SERIES DATA (per flexibility scenario)

if snakemake.config['plot_timeseries'] == True:

    flexibilities = snakemake.config['scenario']['flexibility']

    # CARBON INTENSITY HEATMAPS 
    
    for flex in flexibilities[0]:

        grid_cfe = pd.read_csv(snakemake.input.grid_cfe.split('0.csv')[0]+f'/{flex}.csv', 
                    index_col=0, header=[0,1])
        colormap = "BrBG" # https://matplotlib.org/stable/tutorials/colors/colormaps.html

        for location in locations:

            #location = locations[-1]
            df = grid_cfe[f'{location}']
            df = df.reset_index().rename(columns={'index': 'snapshot'})
            df["snapshot"] = pd.to_datetime(df["snapshot"])

            MIN = df["iteration 1"].min() #case of CFE -> 0
            MAX = df["iteration 1"].max() #case of CFE -> 1

            plot_heatmap_cfe()

    # SPATIAL & TEMPORAL FLEXIBILITY UTILIZATION HEATMAPS

    for flex in flexibilities:

        n = pypsa.Network(snakemake.input.networks.split('0.nc')[0]+f'/{flex}.nc')
        colormap = "coolwarm"

        if snakemake.config['ci']['temporal_shifting'] == True:

            for node in names:
                #node = names[0]
                temporal_shift = retrieve_nb(n, node).get('temporal shift')
                if temporal_shift is not None:
                    df = temporal_shift
                else:
                    df = pd.Series(0, index=retrieve_nb(n, node).index, name='temporal shift')
                df = df.reset_index().rename(columns={'index': 'snapshot'})
                df["snapshot"] = pd.to_datetime(df["snapshot"])
                MIN = df["temporal shift"].min() #case of TEMP or SPATIAL SHIFTS -> flex scenario
                MAX = df["temporal shift"].max() #case of TEMP or SPATIAL SHIFTS -> flex scenario

                plot_heatmap_utilization(carrier="temporal shift")

        if snakemake.config['ci']['spatial_shifting'] == True:

            for node in names:
                #node = names[0]
                spatial_shift = retrieve_nb(n, node).get('spatial shift')
                if spatial_shift is not None:
                    df = spatial_shift
                else:
                    df = pd.Series(0, index=retrieve_nb(n, node).index, name='spatial shift')
                df = df.reset_index().rename(columns={'index': 'snapshot'})
                df["snapshot"] = pd.to_datetime(df["snapshot"])
                MIN = df["spatial shift"].min() #case of TEMP or SPATIAL SHIFTS -> flex scenario
                MAX = df["spatial shift"].max() #case of TEMP or SPATIAL SHIFTS -> flex scenario

                plot_heatmap_utilization(carrier="spatial shift")

    # NODAL BALANCES

        for node in names:

            plot_balances(n, node, '2013-01-01 00:00:00', '2013-01-08 00:00:00')
            plot_balances(n, node, '2013-02-01 00:00:00', '2013-02-08 00:00:00')
            plot_balances(n, node, '2013-03-01 00:00:00', '2013-03-08 00:00:00')
            plot_balances(n, node, '2013-04-01 00:00:00', '2013-04-08 00:00:00')
            plot_balances(n, node, '2013-05-01 00:00:00', '2013-05-08 00:00:00')
        
        print(f'Done for {flex} scen')
