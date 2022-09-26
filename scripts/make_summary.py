import pypsa
import numpy as np
import pandas as pd
import yaml


def summarise():

    pols = snakemake.config["scenario"]["policy"]
    df = pd.DataFrame(index=pols)
    df["costs"] = 0.
    df["emissions"] = 0.
    df["onwind"] = 0.
    df["solar"] = 0.
    df["h2 demand"] = 0.
    df["elec demand"] = 0.

    df["onwind gen"] = 0.
    df["solar gen"] = 0.

    df["electrolyser capacity"] = 0.
    df["electrolyser cost"] = 0.

    n = dict()

    for fn in snakemake.input:
        print(fn)

        pol = fn[fn.rfind("/")+1:-3]

        n[pol] = pypsa.Network(fn)

        df.at[pol,"costs"] = n[pol].objective
        df.at[pol,"emissions"] = n[pol].stores_t.e["co2 atmosphere"][-1]

        for gen in ["onwind","solar"]:
            if f"{name} {gen}" in n[pol].generators.index:
                df.at[pol,gen] = n[pol].generators.at[f"{name} {gen}","p_nom_opt"]
                df.at[pol,gen + " gen"] = (n[pol].generators_t.p[f"{name} {gen}"]*n[pol].snapshot_weightings["generators"]).sum()

        if "google H2" in n[pol].loads.index:
            df.at[pol,"h2 demand"] = (n[pol].loads_t.p["google H2"]*n[pol].snapshot_weightings["generators"]).sum()

        if "google H2 Electrolysis" in n[pol].links.index:
            df.at[pol,"elec demand"] = (n[pol].links_t.p0["google H2 Electrolysis"]*n[pol].snapshot_weightings["generators"]).sum()
            df.at[pol,"electrolyser capacity"] = n[pol].links.at["google H2 Electrolysis","p_nom_opt"]
            df.at[pol,"electrolyser cost"] = n[pol].links.at["google H2 Electrolysis","p_nom_opt"]*n[pol].links.at["google H2 Electrolysis","capital_cost"]

    df["h2 cost"] = (df["costs"] - df.at["ref","costs"])/df["h2 demand"]
    df["h2 emissions"] = (df["emissions"] - df.at["ref","emissions"])/df["h2 demand"]

    df.to_csv(snakemake.output["summary"])

if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('make_summary', palette='p1', zone='IE', year='2025',  participation='10')

    name = snakemake.config['ci']['name']

    summarise()
