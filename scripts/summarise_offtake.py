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
        import os
        os.chdir("/home/lisa/mnt/247-cfe/scripts")
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('summarise_offtake', palette='p1',
                                   zone='DE', year='2025',  participation='10',
                                   policy="cfe")
        os.chdir("/home/lisa/mnt/247-cfe/")
#%%
final = pd.DataFrame()
cf = pd.DataFrame()
emissions = pd.DataFrame()
for network_path in snakemake.input:
    try:
        n = pypsa.Network(network_path)
    except OSError:
            print(network_path, " not solved yet.")
            continue
    policy = network_path.split("/")[-1].split("_")[0]
    price = float(network_path.split("_")[1].replace("price",""))
    try:
        volume = float(network_path.split("_")[-1].replace("volume.nc",""))
    except ValueError:
        volume = "fix_cap"
    participation = network_path.split("networks/")[1][:2]
    weightings = n.snapshot_weightings.generators
    offtake_p = n.generators_t.p.loc[:,n.generators.carrier=="offtake H2"].mul(weightings, axis=0)
    cols = pd.MultiIndex.from_product([[participation], [policy], [price], [volume], offtake_p.columns],
                                      names=["participation", "policy",
                                             "price", "volume", "node"])
    offtake_p.columns = cols
    final = pd.concat([offtake_p, final], axis=1)

    # capacity factor
    p_nom_opt = n.links[n.links.carrier=="H2 Electrolysis"].p_nom_opt
    df = (n.links_t.p0/n.links.p_nom_opt).loc[:,n.links.carrier=="H2 Electrolysis"]
    df.drop(df.columns[(p_nom_opt<10)], axis=1, inplace=True)
    cols = pd.MultiIndex.from_product([[participation], [policy], [price], [volume], df.columns],
                                      names=["participation", "policy", "price",
                                             "volume", "node"])
    df.columns = cols
    cf = pd.concat([cf, df], axis=1)

    # co2 emissions
    co2_emission = n.stores_t.e["co2 atmosphere"].iloc[-1]
    cols = pd.MultiIndex.from_product([[participation], [policy], [price], [volume]],
                                      names=["participation", "policy", "price", "volume"])
    co2_emission = pd.DataFrame([co2_emission], index=cols)
    emissions = pd.concat([emissions, co2_emission])

final.to_csv(snakemake.output.csvs)
emissions.to_csv(snakemake.output.csvs.split("offtake_summary")[0] + "emissions.csv")
cf.to_csv(snakemake.output.csvs.split("offtake_summary")[0] + "cf.csv")
#%%
LHV_H2 = 33.33 # lower heating value [kWh/kg_H2]
cf = cf.apply(lambda x: x.sort_values(ascending=False).reset_index(drop=True), axis=0)
policies = final.columns.levels[1]
participations = final.columns.levels[0]
for participation in participations:
    for policy in policies:
        df = final[participation][policy]
        cols = len(df.columns.levels[1])
        fig, ax = plt.subplots(cols, sharex=True, sharey=True)
        fig.suptitle(policy)
        for i, volume in enumerate(df.columns.levels[1]):
            generation = abs(df.xs(volume, axis=1, level=1).sum().groupby(level=0).sum()/1e6)
            generation.plot(kind="bar", title=volume, grid=True, ax=ax[i])
            if volume != "fix_cap":
                threshold = (volume * 8760 / 1e6 )
                ax[i].plot([-0.5, 5.5], [threshold, threshold], "k--")
        plt.ylabel("exported H2 \n TWh_H2")
        plt.savefig(snakemake.output.offtake_h2.split(".pdf")[0] + f"_{participation}_{policy}.pdf",
                    bbox_inches="tight")
plt.savefig(snakemake.output.offtake_h2,
            bbox_inches="tight")

        # fig, ax = plt.subplots(cols, sharex=True, sharey=True)
        # fig.suptitle(policy)
        # for i, volume in enumerate(df.columns.levels[1]):
        #     generation.mul(generation.index/LHV_H2).plot(kind="bar", title=volume, grid=True, ax=ax[i])
        # plt.ylabel("revenue exported H2 \n [million Euros]")
        # plt.savefig(snakemake.output.revenue_h2.split(".pdf")[0] + f"_{participation}_{policy}.pdf")

    # # emissions
    # try:
    #     df = emissions.loc[policy]
    # except KeyError:
    #         print(policy, " no electrolysis")
    # cols = len(df.index.levels[1])


    # CF electrolysis
for participation in participations:
    for policy in policies:
        try:
            df = cf[participation][policy]
            cols = len(df.columns.levels[1])
            fig, ax = plt.subplots(cols, sharex=True, sharey=True,
                                   figsize=(5, 6))
            fig.suptitle(policy)
            for i, volume in enumerate(df.columns.levels[1]):
                a = df.xs(volume, axis=1, level=1).groupby(level=0, axis=1).mean()
                a.plot(grid=True, title=volume, ax=ax[i], legend=False)
            plt.xlabel("hours")
            plt.legend(bbox_to_anchor=(1.3,3))
            plt.savefig(snakemake.output.revenue_h2.split("revenue")[0] + f"cf_{participation}_{policy}.pdf",
                        bbox_inches="tight")
        except KeyError:
            print(policy, " no electrolysis")
plt.savefig(snakemake.output.revenue_h2)

# emissions
for participation in participations:
    for policy in policies:
        df = emissions.loc[(participation, policy)]
        cols = len(df.index.levels[1])
        fig, ax = plt.subplots(cols, sharex=True, sharey=True,
                               figsize=(5, 6))
        fig.suptitle( f"emissions participation {participation} {policy}")
        for i, volume in enumerate(df.index.levels[1]):
            round(df.xs(volume, level=1)[0]/1e6, ndigits=6).plot(grid=True, ax=ax[i],
                                                 title =volume)
        plt.ylabel("CO2 emissions \n [Million tonnes of CO2]")
        plt.xlabel("price \n [Eur/kg_H2]")
        plt.savefig(snakemake.output.revenue_h2.split("revenue")[0] + f"emissions_{participation}_{policy}.pdf",
                    bbox_inches="tight")
