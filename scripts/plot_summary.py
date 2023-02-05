import pypsa
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.transforms import Bbox

from pyrsistent import s
from xarray import align

#allow plotting without Xwindows
matplotlib.use('Agg')

from solve_network import palette, geoscope
from pypsa.plot import add_legend_patches

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

    return nodal_balance


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

    return nodal_balance


def plot_balances(n, node, 
            start='2013-03-01 00:00:00', 
            stop='2013-03-08 00:00:00'
            ):

    fig, ax = plt.subplots()
    fig.set_size_inches((6,4.5))

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
            edgecolor = "black", linewidth=0.05
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
    #ax.set_xlabel("Hours")
    ax.set(xlabel=None)
    ax.xaxis.set_major_locator(plt.MaxNLocator((duration*2+1)))

    ax.set_ylabel("Nodal balance [MW*h/h]")
    #ax.legend(loc="upper left", ncol = 3, prop={"size":8})

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

    fig.savefig(path + '/' + f"{flex}_balance_{node}.pdf", 
                bbox_inches = Bbox([[0,0],[7.7,4.5]])
    )

if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('plot_summary', 
        year='2025', zone='IEDK', palette='p1', policy="cfe100")   

    #Wildcards & Settings
    policy = snakemake.wildcards.policy[:3]
    penetration = float(snakemake.wildcards.policy[3:])/100 if policy != "ref" else 0
    tech_palette = snakemake.wildcards.palette
    zone = snakemake.wildcards.zone
    year = snakemake.wildcards.year
    area = snakemake.config['area']

    datacenters = snakemake.config['ci']['datacenters']
    locations = list(datacenters.keys())
    names = list(datacenters.values())

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
        f'{name} allam_ccs': "NG-Allam"}
        rename = rename | temp

    links = {
        'vcc12': 'spatial shifting',	
        'vcc21': 'spatial shifting'}

    rename = rename | links
    
    preferred_order_balances = pd.Index([
        "clean dispatchable",
        "NG-Allam",
        "solar",
        "wind",
        "load",
        "battery storage",
        "hydrogen storage",
        'grid',
        'spatial shifting',
        ])


df = pd.read_csv(snakemake.input.summary, index_col=0, header=[0,1])

ci_capacity()
ci_costandrev()
ci_generation()

flexibilities = snakemake.config['scenario']['flexibility']

for flex in flexibilities:
    n = pypsa.Network(snakemake.input.networks+f'/{flex}.nc')

    for node in names:
        plot_balances(n, node, '2013-01-01 00:00:00', '2013-01-08 00:00:00')
        plot_balances(n, node, '2013-03-01 00:00:00', '2013-03-08 00:00:00')
        plot_balances(n, node, '2013-05-01 00:00:00', '2013-05-08 00:00:00')
    
    print(f'Done for {flex} scen')