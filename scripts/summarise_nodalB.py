
"""
Function for plotting a nodal balance considering pypsa-eur-sec component structure
if you can ;)

"""

import pandas as pd
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import yaml
import pypsa

from pypsa.descriptors import Dict


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        #from _helpers import mock_snakemake
        #snakemake = mock_snakemake('plot_summary', palette='p3', zone='DK', year='2030', participation='10')   

        snakemake = Dict()
        with open(f"../config.yaml",'r') as f:
            snakemake.config = yaml.safe_load(f)

        snakemake.input = Dict()
        snakemake.output = Dict()


    folder="../results/test-duo"
    scenario="10/2025/IE/p2"
    policy = 'cfe100'

    #snakemake.input.data = f"{folder}/networks/{scenario}/ref.csv"
    snakemake.output.plot = f"{folder}/balance.png"
    solved_network = f"{folder}/networks/{scenario}/{policy}.nc"

    n = pypsa.Network(solved_network)

node = 'google'
#node = 'GB0 0'
weights = n.snapshot_weightings.generators

rename = {
    'google H2 Electrolysis': 'Hydrogen storage',
    'google H2 Fuel Cell': 'Hydrogen storage',
    'google battery charger': 'Battery storage',
    'google battery discharger': 'Battery storage',
    'google export': 'Import/export',
    'google import': 'Import/export',
    'vcc12': 'Spatial shifting',	
    'vcc21': 'Spatial shifting',
    'google onwind': 'Wind',	
    'google solar': 'Solar PV',
    'google load': 'Load',
}

def retrieve_nb(n, node):
    '''
    Retrieve nodal energy balance per hour
    This simple function works only for the Data center nodes: 
        -> lines and links are bidirectional AND their subsets are exclusive.
        -> links include fossil gens
    '''

    components=['Generator', 'Load', 'StorageUnit', 'Store', 'Link', 'Line']
    nodal_balance = pd.DataFrame(index=n.snapshots)
    
    for i in components:
        if i == 'Generator':
            node_generators = n.generators.query('bus==@node').index
            nodal_balance = nodal_balance.join(n.generators_t.p[node_generators])
        if i == 'Load':
            node_loads = n.loads.query('bus==@node').index
            nodal_balance = nodal_balance.join(n.loads_t.p_set[node_loads])
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

retrieve_nb(n, 'google')