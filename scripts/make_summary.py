
import pypsa

import numpy as np

import pandas as pd

import yaml



def summarise_scalars():

    df = pd.DataFrame()

    fns = snakemake.input

    for fn in fns:
        policy = fn[fn.rfind("/")+1:-5]
        print(policy)
        with open(fn,'r') as f:
            results = yaml.safe_load(f)

        float_keys = [k for k in results.keys() if isinstance(results[k],float)]

        df[policy] = pd.Series({k : results[k] for k in float_keys})

    df.to_csv(snakemake.output["summary"])

if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('make_summary')    
    
    # When running via snakemake
    summarise_scalars()