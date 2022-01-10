import pandas as pd

#allow plotting without Xwindows
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np

colors={"wind" : "b",
        "solar" : "y",
        "nucl" : "r",
        "wind-solar" : "g",
        "wind+solar" : "g",
        "CCGT" : "orange",
        "OCGT" : "wheat",
        "battery storage" : "gray",
        "coal" : "k",
        "lign" : "brown",
        "transmission" : "gray",
        "hydrogen storage" : "m",
        "shed" : "pink"}

def plot_used():

    fig, ax = plt.subplots()
    fig.set_size_inches((4,3))


    df.loc['ci_fraction_clean_used'].plot(kind="bar",
                                          ax=ax)

    ax.set_xlabel("scenario")
    ax.set_ylabel("fraction CFE [per unit]")

    ax.grid()

    fig.tight_layout()

    fig.savefig(snakemake.output.used,
                transparent=True)


if __name__ == "__main__":

    df = pd.read_csv(snakemake.input.summary,
                     index_col=0)

    print(df)

    plot_used()
