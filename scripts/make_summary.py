# SPDX-FileCopyrightText: 2023 Iegor Riepin, Tom Brown
#
# SPDX-License-Identifier: MIT

import os
import pandas as pd
import yaml


def make_summary():
    # Filter relevant filenames
    fns = [
        fn
        for fn in snakemake.input
        if "{}".format(year + "/" + zone + "/" + tech_palette + "/" + policy) in fn
    ]

    # Scenarios from filenames
    scenarios = [os.path.splitext(os.path.basename(fn))[0] for fn in fns]

    # Create MultiIndex for columns
    cols = pd.MultiIndex.from_product([scenarios, list(datacenters.keys())])
    df = pd.DataFrame(columns=cols)

    # Process each file
    for fn in fns:
        scenario = os.path.splitext(os.path.basename(fn))[0]

        try:
            with open(fn, "r") as f:
                results = yaml.safe_load(f)
        except IOError:
            print(f"Error reading file {fn}")
            continue

        # Iterate over locations and update DataFrame
        for location, values in results.items():
            for key, value in values.items():
                if isinstance(value, float):
                    df.at[key, (scenario, location)] = value

    df.to_csv(snakemake.output["summary"])


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "make_summary", year="2030", zone="IE", palette="p3", policy="cfe100"
        )

    # When running via snakemake
    tech_palette = snakemake.wildcards.palette
    zone = snakemake.wildcards.zone
    year = snakemake.wildcards.year
    policy = snakemake.wildcards.policy
    datacenters = snakemake.config["ci"]["datacenters"]

    make_summary()
