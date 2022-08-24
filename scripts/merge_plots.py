import os
import pandas as pd
from PyPDF2 import PdfMerger, PdfReader


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('merge_plots', palette='p1', zone='IE', year='2025', participation='10') 

# Call the PdfFileMerger
mergedObject = PdfMerger()
path = snakemake.output[0][:-11]

# Loop through existing plots in output repository and append them
for file in os.listdir(path):
    mergedObject.append(PdfReader(path+f'{file}', 'rb'))

# Write merged pdf
mergedObject.write(path+"/"+"SUMMARY.pdf")
