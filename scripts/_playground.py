import pypsa, numpy as np, pandas as pd

def calculate_grid_cfe(n, name, node):

    grid_buses = n.buses.index[~n.buses.index.str.contains(name) & ~n.buses.index.str.contains(node)]
    country_buses = n.buses.index[n.buses.index.str.contains(node)]

    clean_techs = pd.Index(["offwind","offwind-ac","offwind-dc","onwind", "ror", "solar",
                    "hydro", "nuclear", "urban central solid biomass CHP"])
    emitters = pd.Index(["CCGT", "OCGT", "coal", "lignite", "oil"])

    clean_grid_generators = n.generators.index[n.generators.bus.isin(grid_buses) & n.generators.carrier.isin(clean_techs)]
    clean_grid_links = n.links.index[n.links.bus1.isin(grid_buses) & n.links.carrier.isin(clean_techs)]
    clean_grid_storage_units = n.storage_units.index[n.storage_units.bus.isin(grid_buses) & n.storage_units.carrier.isin(clean_techs)]
    dirty_grid_links = n.links.index[n.links.bus1.isin(grid_buses) & n.links.carrier.isin(emitters)]

    clean_country_generators = n.generators.index[n.generators.bus.isin(country_buses) & n.generators.carrier.isin(clean_techs)]
    clean_country_links = n.links.index[n.links.bus1.isin(country_buses) & n.links.carrier.isin(clean_techs)]
    clean_country_storage_units = n.storage_units.index[n.storage_units.bus.isin(country_buses) & n.storage_units.carrier.isin(clean_techs)]
    dirty_country_links = n.links.index[n.links.bus1.isin(country_buses) & n.links.carrier.isin(emitters)]

    clean_grid_gens = n.generators_t.p[clean_grid_generators].sum(axis=1)
    clean_grid_ls = (- n.links_t.p1[clean_grid_links].sum(axis=1))
    clean_grid_sus = n.storage_units_t.p[clean_grid_storage_units].sum(axis=1)
    clean_grid_resources = clean_grid_gens + clean_grid_ls + clean_grid_sus
    
    dirty_grid_resources = (- n.links_t.p1[dirty_grid_links].sum(axis=1))

    #grid_cfe =  clean_grid_resources / n.loads_t.p[grid_loads].sum(axis=1)
    #grid_cfe[grid_cfe > 1] = 1.

    import_cfe =  clean_grid_resources / (clean_grid_resources + dirty_grid_resources)

    clean_country_gens = n.generators_t.p[clean_country_generators].sum(axis=1)
    clean_country_ls = (- n.links_t.p1[clean_country_links].sum(axis=1))
    clean_country_sus = n.storage_units_t.p[clean_country_storage_units].sum(axis=1)
    clean_country_resources = clean_country_gens + clean_country_ls + clean_country_sus

    dirty_country_resources = (- n.links_t.p1[dirty_country_links].sum(axis=1))

    ##################
    # Country imports | 
    # NB lines and links are bidirectional, thus we track imports for both subsets 
    # of interconnectors: where [country] node is bus0 and bus1. Subsets are exclusive.
    
    line_imp_subsetA = n.lines_t.p1.loc[:,n.lines.bus0.str.contains(node)].sum(axis=1)
    line_imp_subsetB = n.lines_t.p0.loc[:,n.lines.bus1.str.contains(node)].sum(axis=1)
    line_imp_subsetA[line_imp_subsetA < 0] = 0.
    line_imp_subsetB[line_imp_subsetB < 0] = 0.

    links_imp_subsetA = n.links_t.p1.loc[:,n.links.bus0.str.contains(node) & 
                        (n.links.carrier == "DC") & ~(n.links.index.str.contains(name))].sum(axis=1)
    links_imp_subsetB = n.links_t.p0.loc[:,n.links.bus1.str.contains(node) & 
                        (n.links.carrier == "DC") & ~(n.links.index.str.contains(name))].sum(axis=1)
    links_imp_subsetA[links_imp_subsetA < 0] = 0.
    links_imp_subsetB[links_imp_subsetB < 0] = 0.

    country_import =   line_imp_subsetA + line_imp_subsetB + links_imp_subsetA + links_imp_subsetB

    grid_supply_cfe = (clean_country_resources + country_import * import_cfe) / \
                    (clean_country_resources + dirty_country_resources + country_import)


    print(f"Grid_supply_CFE for {node} has following stats:")
    print(grid_supply_cfe.describe())

    return grid_supply_cfe


one = '/home/iegor/ensys/247-cfe/results/test-IEGB-one/networks/2025/IE/p1/cfe100/0.csv'
two = '/home/iegor/ensys/247-cfe/results/test-IEGB-two/networks/2025/IE/p1/cfe100/0.csv'
fix = '/home/iegor/ensys/247-cfe/results/test-IEGB-two/networks/2025/IE/p1/cfe100/0.csv'


grid_cfe_one = pd.read_csv(one, index_col=0, parse_dates=True, header=[0,1])
grid_cfe_two = pd.read_csv(two, index_col=0, parse_dates=True, header=[0,1])
grid_cfe_fix = pd.read_csv(fix, index_col=0, parse_dates=True, header=[0,1])


grid_cfe_one.xs("IE5 0", axis=1, level=0)[500:1000].plot()
grid_cfe_two.xs("IE5 0", axis=1, level=0)[500:1000].plot()
grid_cfe_fix.xs("IE5 0", axis=1, level=0)[500:1000].plot()


n = pypsa.Network('../results/test-IEGB-one/networks/2025/IE/p1/cfe100/0.nc')
nn = pypsa.Network('../results/test-IEGB-two/networks/2025/IE/p1/cfe100/0.nc')
f = pypsa.Network('../results/test-IEGB-two-Fix/networks/2025/IE/p1/cfe100/0.nc')

from solve_network import calculate_grid_cfe

CFE_ONE = calculate_grid_cfe(n, name='google', node='IE5 0')
CFE_TWO = calculate_grid_cfe(nn, name='google', node='IE5 0')
CFE_FIX = calculate_grid_cfe(f, name='google', node='IE5 0')

########################################

n.generators[n.generators.p_nom_extendable==True].filter(like='GB0 0', axis=0).T
nn.generators[nn.generators.p_nom_extendable==True].filter(like='GB0 0', axis=0).T


def assign_location(n):
    '''
    Assign bus location per each individual component
    From pypsa-eur-sec/plot_network.py
    '''
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)
        for i in ifind.value_counts().index:
            # these have already been assigned defaults
            if i == -1: continue
            names = ifind.index[ifind == i]
            c.df.loc[names, 'location'] = names.str[:i]


def get_df(network, components=["links", "stores", "storage_units", "generators"]):
    n = network.copy()
    assign_location(n)

    # Drop virtual stores
    n.stores.drop(n.stores[n.stores.index.str.contains("EU")].index, inplace=True)

    # Empty dataframe indexed with electrical buses 
    index = pd.DataFrame(index=n.buses.index)
    df = pd.DataFrame(index=n.buses.index)

    # Fill dataframe with MW of capacity per each component and bus
    for comp in components:
        df_temp = getattr(n, comp)

        attr = "e_nom_opt" if comp == "stores" else "p_nom_opt"

        if (comp == 'generators' or comp == 'links'):
            data = (df_temp[attr]*df_temp['efficiency']).groupby([df_temp.location, df_temp.carrier]).sum().unstack().fillna(0.)
        else:
            data = (df_temp[attr]).groupby([df_temp.location, df_temp.carrier]).sum().unstack().fillna(0.)

        capacities = pd.concat([index, data], axis=1)
        df = df.join(capacities)

    df.drop(list(df.columns[(df == 0.).all()]), axis=1, inplace=True)
    df = df.stack()
    df = df.round(1)

    return df

pd.set_option('display.max_rows', None)
get_df(network=nn)
get_df(network=f)