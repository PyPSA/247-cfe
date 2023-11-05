# SPDX-FileCopyrightText: 2023 Iegor Riepin, Tom Brown
#
# SPDX-License-Identifier: MIT

import os
import pandas as pd
from PyPDF2 import PdfMerger, PdfReader


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "merge_plots", year="2030", zone="IE", palette="p3", policy="cfe100"
        )

# Call the PdfFileMerger
mergedObject = PdfMerger()
path = snakemake.output[0][:-11]

# Loop through existing plots in output repository and append them
for file in os.listdir(path):
    if file.endswith(".pdf"):
        mergedObject.append(PdfReader(path + f"{file}", "rb"))

# Write merged pdf
mergedObject.write(path + "/" + "SUMMARY.pdf")
