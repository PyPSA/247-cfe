#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 16:32:11 2022

@author: lisa
"""
import pandas as pd
import pypsa
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('summarise_offtake', palette='p1',
                                   zone='IE', year='2025',  participation='10',
                                   policy="cfe")

final = pd.DataFrame()
for network_path in snakemake.input:
    try:
        n = pypsa.Network(network_path)
    except OSError:
            print(network_path, " not solved yet.")
            continue
    price = float(network_path.split("_")[-2].replace("price",""))
    volume = float(network_path.split("_")[-1].replace("volume.nc",""))
    weightings = n.snapshot_weightings.generators
    offtake_p = n.generators_t.p.loc[:,n.generators.carrier=="offtake H2"].mul(weightings, axis=0)
    cols = pd.MultiIndex.from_product([[price], [volume], offtake_p.columns], names=["price", "volume", "node"])
    offtake_p.columns = cols
    final = pd.concat([offtake_p, final], axis=1)

final.to_csv(snakemake.output.csvs)
#%%
LHV_H2 = 33.33 # lower heating value [kWh/kg_H2]
cols = len(final.columns.levels[1])
fig, ax = plt.subplots(cols, sharex=True, sharey=True)
for i, volume in enumerate(final.columns.levels[1]):
    generation = abs(final.xs(200, axis=1, level=1).sum().groupby(level=0).sum()/1e6)
    generation.plot(kind="bar", title=volume, grid=True, ax=ax[i])
plt.ylabel("exported H2 \n TWh_H2")
plt.savefig(snakemake.output.offtake_h2)

fig, ax = plt.subplots(cols, sharex=True, sharey=True)
for i, volume in enumerate(final.columns.levels[1]):
    generation.mul(generation.index/LHV_H2).plot(kind="bar", title=volume, grid=True, ax=ax[i])
plt.ylabel("revenue exported H2 \n [million Euros]")
plt.savefig(snakemake.output.revenue_h2)
