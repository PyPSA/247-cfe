# SPDX-FileCopyrightText: 2023 Iegor Riepin, Tom Brown
#
# SPDX-License-Identifier: MIT

"""
Auxiliary plotting functions:
(i) a generation capacity fleet and transmission infrastructure
(ii) data center locations

Keep note that in this project each network has four stages:
1) A (raw) input network from PyPSA-Eur
2) A regionally- and component-wise stripped input network
3) (2) with policy adjustments for nuclear/lignite/etc, i.e.,
    a brownfield fleet for the target year that is an input for optimization.
4) A solved network

This script is not integrated into the Snakemake workflow.
Output: map-fleet.png and map-DCs.png are stored in ../study/images
"""

import pandas as pd
import numpy as np
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import yaml
import pypsa

from pypsa.descriptors import Dict
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches

from cartopy import crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from geopandas.tools import geocode


def assign_location(n):
    """
    Assign bus location per each individual component
    """
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        ifind = pd.Series(c.df.index.str.find(" ", start=4), c.df.index)
        for i in ifind.value_counts().index:
            # these have already been assigned defaults
            if i == -1:
                continue
            names = ifind.index[ifind == i]
            c.df.loc[names, "location"] = names.str[:i]


def plot_map(
    network,
    datacenters,
    components=["links", "stores", "storage_units", "generators"],
    bus_size_factor=6e4,
    transmission=True,
    with_legend=True,
):
    """Plotting generation capacity fleet and transmission infrastructure"""

    tech_colors = snakemake.config["tech_colors"]

    n = network.copy()
    assign_location(n)

    # Drop pypsa-eur-sec artificial buses connected to equator
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    # Drop virtual stores
    n.stores.drop(n.stores[n.stores.index.str.contains("EU")].index, inplace=True)

    # Drop DSM links
    n.links.drop(n.links[n.links.carrier == "dsm"].index, inplace=True)

    # Drop data center nodes
    for name in datacenters:
        if name in n.buses.index:
            n.mremove("Bus", [name])

    # Empty dataframe indexed with electrical buses
    index = pd.DataFrame(index=n.buses.index)
    df = pd.DataFrame(index=n.buses.index)

    # Fill dataframe with MW of capacity per each component and bus
    for comp in components:
        df_temp = getattr(n, comp)

        attr = "e_nom" if comp == "stores" else "p_nom"

        if comp == "generators" or comp == "links":
            data = (
                (df_temp[attr] * df_temp["efficiency"])
                .groupby([df_temp.location, df_temp.carrier])
                .sum()
                .unstack()
                .fillna(0.0)
            )
        else:
            data = (
                (df_temp[attr])
                .groupby([df_temp.location, df_temp.carrier])
                .sum()
                .unstack()
                .fillna(0.0)
            )

        capacities = pd.concat([index, data], axis=1)
        df = df.join(capacities)

    df.drop(list(df.columns[(df == 0.0).all()]), axis=1, inplace=True)

    # Give good names for columns (carriers)
    df.columns = df.columns.map(rename)

    # Check 1: print warning with items missing in the config color map
    for item in df.columns:
        if item not in tech_colors:
            print("Warning!", item, "not in config/tech_colors")

    df = df.stack()

    # Check 2: print warning if any pypsa-eur-sec nodes still sneak into the plot
    check = df.index.levels[0].symmetric_difference(n.buses.index)
    if len(check) != 0:
        print("Warning! ", check, "buses still remain in the network to plot!")

    # Keep only HVDC data in links component
    n.links.drop(n.links.index[(n.links.carrier != "DC")], inplace=True)

    capacity_per_tech = df.groupby(level=1).sum()
    capacity_per_bus = df.groupby(level=0).sum()
    carriers = list(capacity_per_tech.index)

    line_lower_threshold = 500.0
    line_upper_threshold = 1e5
    linewidth_factor = 4e3
    ac_color = "teal"
    dc_color = "purple"

    if transmission == True:
        line_widths = n.lines.s_nom
        link_widths = n.links.p_nom
        linewidth_factor = 3e3
        line_lower_threshold = 0.0
        title = "Transmission grid"

    # line_widths[line_widths < line_lower_threshold] = 0.
    # link_widths[link_widths < line_lower_threshold] = 0.
    # line_widths[line_widths > line_upper_threshold] = line_upper_threshold
    # link_widths[link_widths > line_upper_threshold] = line_upper_threshold

    map_opts = {  #'boundaries': [-11, 30, 34, 71],
        "boundaries": [
            n.buses.x.min() - 3,
            n.buses.x.max() + 3,
            n.buses.y.min() - 3,
            n.buses.y.max() + 3,
        ],
        "color_geomap": {
            "ocean": "lightblue",
            "land": "white",
            "border": "black",
            "coastline": "black",
        },
    }

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.EqualEarth(n.buses.x.mean())})
    fig.set_size_inches(9, 6)

    n.plot(
        bus_sizes=df / bus_size_factor,
        bus_colors=tech_colors,
        line_colors=ac_color,
        link_colors=dc_color,
        line_widths=line_widths / linewidth_factor,
        link_widths=link_widths / linewidth_factor,
        ax=ax,
        **map_opts,
    )

    legend_kwargs = {"loc": "upper left", "frameon": False}

    # big_circle = int(round(capacity_per_bus.mean()/1000,-1))
    # small_circle = int(big_circle / 2)
    sizes = [150, 50]
    labels = [f"{s} GW" for s in sizes]
    sizes = [s / bus_size_factor * 1e3 for s in sizes]

    legend_kw = dict(
        bbox_to_anchor=(1, 1),
        labelspacing=2,
        handletextpad=1.5,
        title="Power generation capacity \n before optimization",
        **legend_kwargs,
    )

    add_legend_circles(
        ax,
        sizes,
        labels,
        srid=n.srid,
        patch_kw=dict(facecolor="royalblue"),
        legend_kw=legend_kw,
    )

    sizes = [10, 5]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / linewidth_factor
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        bbox_to_anchor=(1, 0.75),
        labelspacing=0.8,
        handletextpad=1,
        title=title,
        **legend_kwargs,
    )

    add_legend_lines(
        ax, sizes, labels, patch_kw=dict(color="royalblue"), legend_kw=legend_kw
    )

    legend_kw = dict(bbox_to_anchor=(1, 0), loc="lower left", frameon=False)

    if with_legend:
        colors = [tech_colors[c] for c in carriers] + [ac_color, dc_color]
        labels = carriers + ["HVAC line", "HVDC link"]

        add_legend_patches(
            ax,
            colors,
            labels,
            legend_kw=legend_kw,
        )

    fig.tight_layout()
    fig.savefig(
        snakemake.output.plot,
        facecolor="white",
        bbox_inches=Bbox([[0, 0], [9.5, 6]]),
        dpi=600,
    )


def plot_datacenters(network, datacenters):
    n = network.copy()
    assign_location(n)

    # Drop pypsa-eur-sec artificial buses connected to equator
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    # Drop virtual stores
    n.stores.drop(n.stores[n.stores.index.str.contains("EU")].index, inplace=True)

    # Drop DSM links
    n.links.drop(n.links[n.links.carrier == "dsm"].index, inplace=True)

    # Drop data center nodes
    for name in datacenters:
        if name in n.buses.index:
            n.mremove("Bus", [name])

    # Load the geometries of datacenters
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    # Get the centroids of the specified datacenters
    centroids = geocode(datacenters, provider="nominatim", user_agent="myGeocoder")

    # Extract the geometry (Point) objects from the GeoDataFrame
    centroids = centroids.geometry

    # Set up the figure and axes
    map_opts = {
        "boundaries": [
            n.buses.x.min() - 3,
            n.buses.x.max() + 3,
            n.buses.y.min() - 3,
            n.buses.y.max() + 3,
        ],
        "color_geomap": {
            "ocean": "lightblue",
            "land": "white",
            "border": "black",
            "coastline": "black",
        },
    }

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.EqualEarth(n.buses.x.mean())})
    fig.set_size_inches(9, 6)

    # Plot the country shapes
    ax.set_extent(map_opts["boundaries"], crs=ccrs.PlateCarree())
    ax.add_feature(
        cfeature.OCEAN.with_scale("50m"),
        facecolor=map_opts["color_geomap"]["ocean"],
        zorder=0,
    )
    ax.add_feature(
        cfeature.LAND.with_scale("50m"),
        facecolor=map_opts["color_geomap"]["land"],
        zorder=0,
    )

    # Add borders and coastline with adjusted linewidth and alpha
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        edgecolor=map_opts["color_geomap"]["border"],
        linewidth=0.5,
        alpha=0.5,
        zorder=1,
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        edgecolor=map_opts["color_geomap"]["coastline"],
        linewidth=0.5,
        alpha=0.5,
        zorder=1,
    )

    # Plot the interconnected nodes
    connections = set()
    for i, centroid_i in centroids.iteritems():
        for j, centroid_j in centroids.iteritems():
            if i != j and (j, i) not in connections:
                ax.plot(
                    [centroid_i.x, centroid_j.x],
                    [centroid_i.y, centroid_j.y],
                    color="darkblue",
                    linewidth=1.5,
                    alpha=1,
                    linestyle="dotted",
                    transform=ccrs.Geodetic(),
                )
                connections.add((i, j))
        ax.scatter(
            centroid_i.x,
            centroid_i.y,
            color="darkblue",
            marker="o",
            s=50,
            transform=ccrs.PlateCarree(),
        )

    fig.tight_layout()
    fig.savefig(
        snakemake.output.plot_DC, bbox_inches="tight", facecolor="white", dpi=600
    )


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        # from _helpers import mock_snakemake
        # snakemake = mock_snakemake('plot_summary', palette='p1', zone='EU', year='2030')

        # Emulate snakemake object with dictionary
        snakemake = Dict()
        with open(f"../config.yaml", "r") as f:
            snakemake.config = yaml.safe_load(f)

        snakemake.input = Dict()
        snakemake.output = Dict()

    images = "../study/images"
    results = "../results/" + snakemake["config"]["run"]

    # snakemake.input.data = f"{folder}/networks/{scenario}/ref.csv"
    snakemake.output.plot = f"{images}/map-fleet.png"
    snakemake.output.plot_DC = f"{images}/map-DCs.png"
    original_network = f"../input/v6_elec_s_37_lv1.0__1H-B-solar+p3_2025.nc"
    stripped_network = f"{results}/networks/2025/EU/p1/cfe100/0.nc"

    n = pypsa.Network(stripped_network)

rename = {
    "solar": "solar",
    "offwind": "offshore wind",
    "offwind-ac": "offshore wind",
    "offwind-dc": "offshore wind",
    "onwind": "onshore wind",
    "ror": "hydroelectricity",
    "hydro": "hydroelectricity",
    "PHS": "pumped hydro storage",
    "AC": "transmission lines",
    "DC": "transmission lines",
    "OCGT": "OCGT",
    "CCGT": "CCGT",
    "coal": "hard coal",
    "lignite": "lignite",
    "nuclear": "nuclear",
    "oil": "oil",
    "urban central solid biomass CHP": "solid biomass",
}

datacenters = snakemake["config"]["ci"]["datacenters"]

plot_map(network=n, datacenters=list(datacenters.values()), bus_size_factor=4e4)

plot_datacenters(network=n, datacenters=list(datacenters.values()))
