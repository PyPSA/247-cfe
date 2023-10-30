# SPDX-FileCopyrightText: 2023 Iegor Riepin, Tom Brown
#
# SPDX-License-Identifier: MIT

from shutil import copy

import os

files = ["config.yaml", "Snakefile", "scripts/solve_network.py"]

if __name__ == "__main__":
    if "snakemake" not in globals():
        from helper import mock_snakemake

        snakemake = mock_snakemake("copy_config")

    for f in files:
        copy(
            f,
            os.path.join(
                snakemake.config["results_dir"], snakemake.config["run"], "configs"
            ),
        )
