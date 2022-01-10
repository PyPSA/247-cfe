import pandas as pd

#allow plotting without Xwindows
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np


clean_techs = snakemake.config['ci']['clean_techs']
tech_colors = snakemake.config['tech_colors']

def used():

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


def ci_capacity():

    fig, ax = plt.subplots()
    fig.set_size_inches((4,3))

    ldf = df.loc[["ci_cap_" + t for t in clean_techs]].rename({"ci_cap_" + t : t for t in clean_techs})

    ldf.T.plot(kind="bar",stacked=True,
               ax=ax,
               color=tech_colors)

    ax.set_xlabel("scenario")
    ax.set_ylabel("CI capacity [MW]")

    ax.grid()

    fig.tight_layout()

    fig.savefig(snakemake.output.used.replace("used.pdf","ci_capacity.pdf"),
                transparent=True)


def ci_generation():

    fig, ax = plt.subplots()
    fig.set_size_inches((4,3))

    ldf = df.loc[["ci_generation_" + t for t in clean_techs]].rename({"ci_generation_" + t : t for t in clean_techs})/1000.

    ldf.T.plot(kind="bar",stacked=True,
               ax=ax,
               color=tech_colors)

    ax.set_xlabel("scenario")
    ax.set_ylabel("CI generation [GWh]")

    ax.grid()

    fig.tight_layout()

    fig.savefig(snakemake.output.used.replace("used.pdf","ci_generation.pdf"),
                transparent=True)



if __name__ == "__main__":

    df = pd.read_csv(snakemake.input.summary,
                     index_col=0)

    used()

    ci_capacity()

    ci_generation()
