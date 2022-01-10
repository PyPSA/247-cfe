
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
        with open(fn) as f:
            results = yaml.load(f)

        float_keys = [k for k in results.keys() if isinstance(results[k],float)]

        df[policy] = pd.Series({k : results[k] for k in float_keys})

    df.to_csv(snakemake.output["summary"])

if __name__ == "__main__":

    summarise_scalars()
