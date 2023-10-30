# SPDX-FileCopyrightText: 2023 Iegor Riepin, Tom Brown
#
# SPDX-License-Identifier: MIT

import pypsa
import numpy as np
import pandas as pd
import yaml


def create_tuples(scenarios, locations):
    # Use a nested list comprehension to create the list of tuples
    tuples_list = [(str(s), str(l)) for s in scenarios for l in locations]
    return tuples_list


def summarise_scalars():
    fns = list(
        filter(
            lambda k: (
                "{}".format(year + "/" + zone + "/" + tech_palette + "/" + policy)
            )
            in k,
            snakemake.input,
        )
    )

    # create empty dataframe with flex_scenario & locations as multi-columns
    scenarios = [fn[fn.rfind("/") + 1 : -5] for fn in fns]
    cols = pd.MultiIndex.from_tuples(create_tuples(scenarios, list(datacenters.keys())))
    df = pd.DataFrame(columns=cols)

    # fill dataframe by looping over {flex_scenario}.yaml and {location} of data centers
    for fn in fns:
        scenario = [fn[fn.rfind("/") + 1 : -5]][0]

        with open(fn, "r") as f:
            results = yaml.safe_load(f)

        locations = [k for k in results.keys()]

        for location in locations:
            float_keys = [
                k
                for k in results[f"{location}"].keys()
                if isinstance(results[f"{location}"][k], float)
            ]
            df.loc[:, (scenario, location)] = pd.Series(
                {k: results[f"{location}"][k] for k in float_keys}
            )

    df.to_csv(snakemake.output["summary"])


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "make_summary", year="2025", zone="IE", palette="p1", policy="cfe100"
        )

    # When running via snakemake
    tech_palette = snakemake.wildcards.palette
    zone = snakemake.wildcards.zone
    year = snakemake.wildcards.year
    policy = snakemake.wildcards.policy
    datacenters = snakemake.config["ci"]["datacenters"]

    summarise_scalars()
