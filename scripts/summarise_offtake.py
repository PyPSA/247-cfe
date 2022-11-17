#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 16:32:11 2022

@author: lisa
"""
import pandas as pd
import pypsa
import matplotlib.pyplot as plt
from _helpers import override_component_attrs
from resolve_network import geoscope

if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        import os
        os.chdir("/home/lisa/Documents/247-cfe/scripts")
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('summarise_offtake', palette='p1',
                                   zone='DE', year='2025',  participation='10',
                                   policy="ref")
        os.chdir("/home/lisa/Documents/247-cfe/")

LHV_H2 = 33.33 # lower heating value [kWh/kg_H2]

#%%
def calculate_supply_energy(n, label, supply_energy):
    """calculate the total energy supply/consuption of each component at the buses aggregated by carrier"""


    bus_carriers = n.buses.carrier.unique()
    supply_energy = supply_energy.reindex(columns=supply_energy.columns.union(label))

    for i in bus_carriers:
        bus_map = (n.buses.carrier == i)
        # correct CHPs with missing bus2
        n.links.loc[n.links.carrier=="urban central solid biomass CHP", "bus2"] = ""
        bus_map.at[""] = False

        for c in n.iterate_components(n.one_port_components):

            items = c.df.index[c.df.bus.map(bus_map).fillna(False)]

            if len(items) == 0:
                continue

            s = c.pnl.p[items].multiply(n.snapshot_weightings.generators,axis=0).sum().multiply(c.df.loc[items, 'sign']).groupby(c.df.loc[items, 'carrier']).sum()
            s = pd.concat([s], keys=[c.list_name])
            s = pd.concat([s], keys=[i])

            supply_energy = supply_energy.reindex(s.index.union(supply_energy.index))
            supply_energy.loc[s.index, label] = s


        for c in n.iterate_components(n.branch_components):

            for end in [col[3:] for col in c.df.columns if col[:3] == "bus"]:

                items = c.df.index[c.df["bus" + str(end)].map(bus_map, na_action=False)]

                if len(items) == 0:
                    continue

                s = (-1)*c.pnl["p"+end][items].multiply(n.snapshot_weightings.generators,axis=0).sum().groupby(c.df.loc[items, 'carrier']).sum()
                s.index = s.index + end
                s = pd.concat([s], keys=[c.list_name])
                s = pd.concat([s], keys=[i])

                supply_energy = supply_energy.reindex(s.index.union(supply_energy.index))

                supply_energy.loc[s.index, label] = s

    return supply_energy



def calculate_nodal_supply_energy(n, label, supply_energy):
    """calculate the total energy supply/consuption of each component at the buses aggregated by carrier"""


    bus_carriers = n.buses.carrier.unique()
    supply_energy = supply_energy.reindex(columns=supply_energy.columns.union(label))

    for i in bus_carriers:
        bus_map = (n.buses.carrier == i)
        # correct CHPs with missing bus2
        n.links.loc[n.links.carrier=="urban central solid biomass CHP", "bus2"] = ""
        bus_map.at[""] = False

        for c in n.iterate_components(n.one_port_components):

            items = c.df.index[c.df.bus.map(bus_map).fillna(False)]

            if len(items) == 0:
                continue

            s = c.pnl.p[items].multiply(n.snapshot_weightings.generators,axis=0).sum().multiply(c.df.loc[items, 'sign']).groupby([c.df.loc[items].index.str[:2], c.df.loc[items, "carrier"]]).sum()
            s = pd.concat([s], keys=[c.list_name])
            s = pd.concat([s], keys=[i])

            supply_energy = supply_energy.reindex(s.index.union(supply_energy.index))
            supply_energy.loc[s.index, label] = s


        for c in n.iterate_components(n.branch_components):

            for end in [col[3:] for col in c.df.columns if col[:3] == "bus"]:

                items = c.df.index[c.df["bus" + str(end)].map(bus_map, na_action=False)]

                if len(items) == 0:
                    continue

                s = (-1)*c.pnl["p"+end][items].multiply(n.snapshot_weightings.generators,axis=0).sum().groupby([c.df.loc[items, ("bus"+end)].str[:2], c.df.loc[items, "carrier"]]).sum()
                s.rename(index=lambda x: x+end, level=1, inplace=True)
                s = pd.concat([s], keys=[c.list_name])
                s = pd.concat([s], keys=[i])

                supply_energy = supply_energy.reindex(s.index.union(supply_energy.index))

                supply_energy.loc[s.index, label] = s

    return supply_energy


def calculate_h2_generationmix(n, label, gen_mix):
    """calculate the generation mix used for H2 production"""


    # generation_mix = generation_mix.reindex(columns=generation_mix.columns.union(label))
    bus_map = (n.buses.carrier == "AC")
    # correct CHPs with missing bus2
    n.links.loc[n.links.carrier=="urban central solid biomass CHP", "bus2"] = ""
    bus_map.at[""] = False
    zone = snakemake.wildcards.zone
    ci = snakemake.config['ci']
    name = ci['name']

    generation_mix = pd.DataFrame()


    for c in n.iterate_components(n.one_port_components):

        items = c.df.index[c.df.bus.map(bus_map).fillna(False)]

        if len(items) == 0:
            continue

        s = (c.pnl.p[items].multiply(n.snapshot_weightings.generators,axis=0)
             .multiply(c.df.loc[items, 'sign'])
             .groupby([c.df.loc[items].index.str[:2], c.df.loc[items, "carrier"]],
                      axis=1).sum())
        s = s.reindex(columns=[zone, name[:2]], level=0)
        s = pd.concat([s], keys=[c.list_name], axis=1)

        generation_mix = generation_mix.reindex(s.index.union(generation_mix.index))
        generation_mix = generation_mix.reindex(columns= s.columns.union(generation_mix.columns))
        generation_mix.loc[s.index, s.columns] = s


    for c in n.iterate_components(n.branch_components):

        for end in [col[3:] for col in c.df.columns if col[:3] == "bus"]:

            items = c.df.index[c.df["bus" + str(end)].map(bus_map, na_action=False)]

            if len(items) == 0:
                continue

            s = ((-1)*c.pnl["p"+end][items].multiply(n.snapshot_weightings.generators,axis=0)
                 .groupby([c.df.loc[items, ("bus"+end)].str[:2], c.df.loc[items, "carrier"]], axis=1).sum())
            s.rename(columns=lambda x: x+end, level=1, inplace=True)
            s = s.reindex(columns=[zone, name[:2]], level=0)
            s = pd.concat([s], keys=[c.list_name], axis=1)

            generation_mix = generation_mix.reindex(s.index.union(generation_mix.index))
            generation_mix = generation_mix.reindex(columns= s.columns.union(generation_mix.columns))
            generation_mix.loc[s.index, s.columns] = s

    electrolysis = generation_mix["links"][name[:2]]["H2 Electrolysis0"]
    if "import1" in generation_mix.columns.levels[2]:
        imports = generation_mix["links"][name[:2]]["import1"]
    else:
        imports = 0

    if "export0" in generation_mix.columns.levels[2]:
        exports = generation_mix["links"][name[:2]]["export0"]
    else:
        exports = 0
    to_drop = ["AC0", "AC1", "DC0", "DC1", "import0", "export1", "electricity",
               "PHS", "hydro", 'battery charger0', 'battery discharger1',
               "H2 Electrolysis0", "H2 Fuel Cell1"]
    # generation mix in zone
    zone_generation = generation_mix.xs(zone, level=1, axis=1).drop(to_drop, level=1, axis=1, errors="ignore")
    total_generation = zone_generation[zone_generation>0].sum(axis=1)
    # share of each technology in zone for each time step
    share = zone_generation.div(total_generation, axis=0)
    # total imported electricity by carrier
    import_mix = share.mul(imports, axis=0)
    # local generation

    local_generation = generation_mix.loc[:,generation_mix.columns.get_level_values(1)==name[:2]].droplevel(1, axis=1)
    local_gens = local_generation.columns.get_level_values(1).isin(ci["res_techs"])
    if "battery charger0" in local_generation.columns.get_level_values(1).unique():
        battery_charged_by = local_generation.loc[round(local_generation[("links", "battery charger0")])==0, local_gens].sum()
        battery_c_share = battery_charged_by/battery_charged_by.sum()
    local_generation = local_generation.loc[:,local_generation.columns.get_level_values(1).isin(["onwind","solar", "battery discharger1"])]
    if "battery discharger1" in local_generation.columns.get_level_values(1).unique():
        for i in battery_c_share.index:
            local_generation[i] += local_generation[("links", "battery discharger1")] * battery_c_share.loc[i]
        local_generation.drop("battery discharger1", level=1, axis=1, inplace=True)
    local_generation.rename(columns=lambda x: x + " local", level=1, inplace=True)

    total = pd.concat([import_mix, local_generation], axis=1)

    tot_mix = total.div(total.sum(axis=1), axis=0).mul(abs(electrolysis), axis=0).sum()

    gen_mix = gen_mix.reindex(columns=gen_mix.columns.union(label))
    gen_mix = gen_mix.reindex(index=tot_mix.index)
    gen_mix.loc[tot_mix.index, label] = tot_mix

    return gen_mix

opt_name = {
    "Store": "e",
    "Line": "s",
    "Transformer": "s"
}


def assign_locations(n):
    for c in n.iterate_components(n.one_port_components|n.branch_components):
        c.df["location"] = c.df.rename(index=lambda x: "google" if "google" in x else "rest").index


def calculate_nodal_capacities(n, label, nodal_capacities):
    #Beware this also has extraneous locations for country (e.g. biomass) or continent-wide (e.g. fossil gas/oil) stuff
    for c in n.iterate_components(n.branch_components|n.controllable_one_port_components^{"Load"}):
        nodal_capacities_c = c.df.groupby(["location","carrier"])[opt_name.get(c.name,"p") + "_nom_opt"].sum()
        index = pd.MultiIndex.from_tuples([(c.list_name,) + t for t in nodal_capacities_c.index.to_list()])
        nodal_capacities = nodal_capacities.reindex(index.union(nodal_capacities.index))
        nodal_capacities = nodal_capacities.reindex(columns=nodal_capacities.columns.union(label))
        nodal_capacities.loc[index,label] = nodal_capacities_c.values

    return nodal_capacities

ls_dict = {"ref": "-",
           "cfe": ":",
           "exl": "--",
           "grd": "x-",
           "res": "o-",
           "offgrid": "v-"}
color_dict = {"ref": '#377eb8',
           "cfe": '#ff7f00',
           "exl": '#4daf4a',
           "grd":  '#f781bf',
           "res": '#a65628',
           "offgrid": '#984ea3'
    }


def calculate_weighted_prices(n, label, weighted_prices):
    # Warning: doesn't include storage units as loads

    weighted_prices = weighted_prices.reindex(pd.Index([
        "AC",
        "H2"
    ]))
    cols = pd.MultiIndex.from_product([label.levels[0], label.levels[1], label.levels[2], label.levels[3],label.levels[4], ["google", "rest"]])

    weighted_prices = weighted_prices.reindex(columns = weighted_prices.columns.union(cols))
    link_loads = {"AC":  ["battery charger", "H2 Electrolysis"],
                  "H2": ["H2 Fuel Cell"]}

    for carrier in link_loads:

        buses = n.buses.index[n.buses.carrier==carrier]

        if buses.empty:
            continue

        if carrier in ["H2", "gas"]:
            load = pd.DataFrame(index=n.snapshots, columns=buses, data=0.)
        else:
            load = n.loads_t.p_set.loc[:,n.loads.bus.map(n.buses.carrier)==carrier]

        for tech in link_loads[carrier]:

            names = n.links.index[n.links.index.to_series().str[-len(tech):] == tech]

            if names.empty:
                continue

            load = pd.concat([load, n.links_t.p0[names].groupby(n.links.loc[names, "bus0"],axis=1).sum()], axis=1)

        load = load.groupby(load.columns,axis=1).sum()
        if carrier == "AC":
            a = load/load.max() * n.buses_t.marginal_price.reindex(columns=load.columns)
        else:
            a = n.buses_t.marginal_price.loc[:,n.buses.carrier==carrier]
        load_rest = a.loc[:,~a.columns.str.contains("google")].mean(axis=1).mean()
        load_google = a.loc[:,a.columns.str.contains("google")].mean(axis=1).mean()

        weighted_prices.loc[carrier,(label[0][0], label[0][1], label[0][2], label[0][3], label[0][4], "rest")] = load_rest
        weighted_prices.loc[carrier,(label[0][0], label[0][1], label[0][2], label[0][3],label[0][4],"google")]= load_google

    return weighted_prices


def calculate_curtailment(n, label, curtailment):
    curtailment_n = ((n.generators_t.p.sum().groupby([n.generators.carrier, n.generators.location]).sum())
   / ((n.generators_t.p_max_pu * n.generators.p_nom_opt).sum().groupby([n.generators.carrier, n.generators.location]).sum()))
    all_res = n.generators_t.p.groupby(n.generators.carrier,axis=1).sum().sum().loc[["onwind", "offwind-ac", "offwind-dc", "offwind", "solar"]].sum()/((n.generators_t.p_max_pu * n.generators.p_nom_opt).sum().groupby(n.generators.carrier).sum()).loc[["onwind", "offwind-ac", "offwind-dc", "offwind", "solar"]].sum()
    curtailment_n.loc[("all", "all")] = all_res
    curtailment = curtailment.reindex(columns=curtailment.columns.union(label))
    curtailment = curtailment.reindex(index=curtailment.index.union(curtailment_n.index))
    curtailment.loc[curtailment_n.index, label] = curtailment_n
    return curtailment


def calculate_nodal_costs(n, label, nodal_costs):
    #Beware this also has extraneous locations for country (e.g. biomass) or continent-wide (e.g. fossil gas/oil) stuff
    for c in n.iterate_components(n.branch_components|n.controllable_one_port_components^{"Load"}):
        c.df["capital_costs"] = c.df.capital_cost * c.df[opt_name.get(c.name, "p") + "_nom_opt"]
        capital_costs = c.df.groupby(["location", "carrier"])["capital_costs"].sum()
        index = pd.MultiIndex.from_tuples([(c.list_name, "capital") + t for t in capital_costs.index.to_list()])
        nodal_costs = nodal_costs.reindex(index.union(nodal_costs.index))
        nodal_costs = nodal_costs.reindex(columns=nodal_costs.columns.union(label))
        nodal_costs.loc[index,label] = capital_costs.values

        if c.name == "Link":
            p = c.pnl.p0.multiply(n.snapshot_weightings.generators, axis=0).sum()
        elif c.name == "Line":
            continue
        elif c.name == "StorageUnit":
            p_all = c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0)
            p_all[p_all < 0.] = 0.
            p = p_all.sum()
        else:
            p = c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0).sum()

        #correct sequestration cost
        if c.name == "Store":
            items = c.df.index[(c.df.carrier == "co2 stored") & (c.df.marginal_cost <= -100.)]
            c.df.loc[items, "marginal_cost"] = -20.

        c.df["marginal_costs"] = p*c.df.marginal_cost
        marginal_costs = c.df.groupby(["location", "carrier"])["marginal_costs"].sum()
        index = pd.MultiIndex.from_tuples([(c.list_name, "marginal") + t for t in marginal_costs.index.to_list()])
        nodal_costs = nodal_costs.reindex(index.union(nodal_costs.index))

        nodal_costs.loc[index, label] = marginal_costs.values

    return nodal_costs


def calculate_costs(n, label, costs):
    costs = costs.reindex(columns=costs.columns.union(label))
    for c in n.iterate_components(n.branch_components|n.controllable_one_port_components^{"Load"}):
        capital_costs = c.df.capital_cost*c.df[opt_name.get(c.name,"p") + "_nom_opt"]
        capital_costs_grouped = capital_costs.groupby(c.df.carrier).sum()

        capital_costs_grouped = pd.concat([capital_costs_grouped], keys=["capital"])
        capital_costs_grouped = pd.concat([capital_costs_grouped], keys=[c.list_name])

        costs = costs.reindex(capital_costs_grouped.index.union(costs.index))

        costs.loc[capital_costs_grouped.index, label] = capital_costs_grouped

        if c.name == "Link":
            p = c.pnl.p0.multiply(n.snapshot_weightings.generators, axis=0).sum()
        elif c.name == "Line":
            continue
        elif c.name == "StorageUnit":
            p_all = c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0)
            p_all[p_all < 0.] = 0.
            p = p_all.sum()
        else:
            p = c.pnl.p.multiply(n.snapshot_weightings.generators, axis=0).sum()

        #correct sequestration cost
        if c.name == "Store":
            items = c.df.index[(c.df.carrier == "co2 stored") & (c.df.marginal_cost <= -100.)]
            c.df.loc[items, "marginal_cost"] = -20.

        marginal_costs = p*c.df.marginal_cost

        marginal_costs_grouped = marginal_costs.groupby(c.df.carrier).sum()

        marginal_costs_grouped = pd.concat([marginal_costs_grouped], keys=["marginal"])
        marginal_costs_grouped = pd.concat([marginal_costs_grouped], keys=[c.list_name])

        costs = costs.reindex(marginal_costs_grouped.index.union(costs.index))

        costs.loc[marginal_costs_grouped.index,label] = marginal_costs_grouped

    # add back in all hydro
    #costs.loc[("storage_units", "capital", "hydro"),label] = (0.01)*2e6*n.storage_units.loc[n.storage_units.group=="hydro", "p_nom"].sum()
    #costs.loc[("storage_units", "capital", "PHS"),label] = (0.01)*2e6*n.storage_units.loc[n.storage_units.group=="PHS", "p_nom"].sum()
    #costs.loc[("generators", "capital", "ror"),label] = (0.02)*3e6*n.generators.loc[n.generators.group=="ror", "p_nom"].sum()

    return costs

def calculate_hydrogen_cost(n, label, h2_cost):
    if policy in ["ref"]: return h2_cost
    # revenue from selling and costs for buying
    marginal_price = n.buses_t.marginal_price
    # import
    p_import = n.links_t.p0.loc[:,n.links.carrier=="import"]
    p_export = n.links_t.p0.loc[:,n.links.carrier=="export"]
    # marginal price
    name = snakemake.config['ci']['name']
    participation = snakemake.wildcards.participation
    zone = snakemake.wildcards.zone
    area = snakemake.config['area']
    node = geoscope(zone, area)['node']
    weight = n.snapshot_weightings.generators
    price = n.buses_t.marginal_price.loc[:, node].mul(weight)
    # import costs
    import_cost = p_import.mul(price.values, axis=0).sum()
    # export revenue
    export_cost = (-1)* p_export.mul(price.values, axis=0).sum()
    # offtake H2
    offtake = n.loads_t.p.loc[:,"google H2"].mul(weight, axis=0).sum()
    # costs from the others
    try:
        costs = nodal_costs.loc[nodal_costs.index.get_level_values(2)=="google",:][label].groupby(level=[0,3]).sum()
    except KeyError:
        print(label)
        return h2_cost
    if import_cost.empty:
        costs.loc[("links","import"),:] = 0
    else:
        costs.loc[("links","import"),:] = import_cost.values
    if export_cost.empty:
        costs.loc[("links","export"),:] = 0
    else:
        costs.loc[("links","export"),:] = export_cost.values
    costs.loc[("final", "H2 cost"),:] = (costs.drop("offtake H2", level=1, errors="ignore").sum()/abs(offtake))[0]
    costs.loc[("loads", "offtake H2"), label] = offtake
    h2_cost = h2_cost.reindex(index=h2_cost.index.union(costs.index))
    h2_cost = h2_cost.reindex(columns=h2_cost.columns.union(label))
    h2_cost.loc[costs.index, label] = costs

    return h2_cost

def calculate_emission_rate(n, label, emission_rate):
    zone = snakemake.wildcards.zone
    area = snakemake.config['area']
    name = snakemake.config['ci']['name']
    country = geoscope(zone, area)['node']
    grid_clean_techs = snakemake.config['global']['grid_clean_techs']
    emitters = snakemake.config['global']['emitters']

    rest_system_buses = n.buses.index[~n.buses.index.str.contains(name) &
                                      (n.buses.location!=country)]
    country_buses = n.buses[n.buses.location==country].index


    clean_grid_generators =  n.generators.index[n.generators.bus.isin(rest_system_buses)
                                                & n.generators.carrier.isin(grid_clean_techs)]
    clean_grid_links = n.links.index[n.links.bus1.isin(rest_system_buses)
                                     & (n.links.carrier.isin(grid_clean_techs))]
    clean_grid_storage_units = n.storage_units.index[n.storage_units.bus.isin(rest_system_buses)
                                                     & n.storage_units.carrier.isin(grid_clean_techs)]
    dirty_grid_links = n.links.index[n.links.bus1.isin(rest_system_buses)
                                     & n.links.carrier.isin(emitters)]

    clean_country_generators = n.generators.index[n.generators.bus.isin(country_buses) & n.generators.carrier.isin(grid_clean_techs)]
    clean_country_links = n.links.index[n.links.bus1.isin(country_buses) & n.links.carrier.isin(grid_clean_techs)]
    clean_country_storage_units = n.storage_units.index[n.storage_units.bus.isin(country_buses) & n.storage_units.carrier.isin(grid_clean_techs)]
    dirty_country_links = n.links.index[n.links.bus1.isin(country_buses) & n.links.carrier.isin(emitters)]

    country_loads = n.loads.index[n.loads.bus.isin(country_buses)]

    clean_grid_gens = n.generators_t.p[clean_grid_generators].sum(axis=1)
    clean_grid_ls = (- n.links_t.p1[clean_grid_links].sum(axis=1))
    clean_grid_sus = n.storage_units_t.p[clean_grid_storage_units].sum(axis=1)
    clean_grid_resources = clean_grid_gens + clean_grid_ls + clean_grid_sus
    dirty_grid_resources = (- n.links_t.p1[dirty_grid_links].sum(axis=1))

    clean_country_gens = n.generators_t.p[clean_country_generators].sum(axis=1)
    clean_country_ls = (- n.links_t.p1[clean_country_links].sum(axis=1))
    clean_country_sus = n.storage_units_t.p[clean_country_storage_units].sum(axis=1)
    clean_country_resources = clean_country_gens + clean_country_ls + clean_country_sus
    dirty_country_resources = (- n.links_t.p1[dirty_country_links].sum(axis=1))

    line_imp_subsetA = n.lines_t.p1.loc[:,n.lines.bus0.str.contains(country)].sum(axis=1)
    line_imp_subsetB = n.lines_t.p0.loc[:,n.lines.bus1.str.contains(country)].sum(axis=1)
    line_imp_subsetA[line_imp_subsetA < 0] = 0.
    line_imp_subsetB[line_imp_subsetB < 0] = 0.

    links_imp_subsetA = n.links_t.p1.loc[:,n.links.bus0.str.contains(country) &
                        (n.links.carrier == "DC") & ~(n.links.index.str.contains(name))].sum(axis=1)
    links_imp_subsetB = n.links_t.p0.loc[:,n.links.bus1.str.contains(country) &
                        (n.links.carrier == "DC") & ~(n.links.index.str.contains(name))].sum(axis=1)
    links_imp_subsetA[links_imp_subsetA < 0] = 0.
    links_imp_subsetB[links_imp_subsetB < 0] = 0.

    country_import =   line_imp_subsetA + line_imp_subsetB + links_imp_subsetA + links_imp_subsetB

    grid_hourly_emissions = n.links_t.p0[dirty_grid_links].multiply(n.links.efficiency2[dirty_grid_links],axis=1).sum(axis=1)

    grid_emission_rate =  grid_hourly_emissions / (clean_grid_resources + dirty_grid_resources)

    country_hourly_emissions = n.links_t.p0[dirty_country_links].multiply(n.links.efficiency2[dirty_country_links],axis=1).sum(axis=1)

    grid_supply_emission_rate = (country_hourly_emissions + country_import*grid_emission_rate) / \
                                (clean_country_resources + dirty_country_resources + country_import)

    grid_supply_emission_rate_withoutimports = country_hourly_emissions / \
                                (clean_country_resources + dirty_country_resources)
    ci_emissions_t = n.links_t.p0["google import"]*grid_supply_emission_rate
    ci_emissions_t_ni = n.links_t.p0["google import"]*grid_supply_emission_rate_withoutimports

    carbon_intensity_h2 = ci_emissions_t.sum()/abs(n.loads_t.p.loc[:,"google H2"].sum()) * LHV_H2
    carbon_intensity_h2_ni = ci_emissions_t_ni.sum()/abs(n.loads_t.p.loc[:,"google H2"].sum()) * LHV_H2

    emission_rate = emission_rate.reindex(columns = emission_rate.columns.union(label))
    emission_rate.loc["carbon_intensity_H2", label] = carbon_intensity_h2
    emission_rate.loc["carbon_intensity_H2_ni", label] = carbon_intensity_h2_ni
    emission_rate.loc["ci_emissions", label] = ci_emissions_t.sum()
    return emission_rate


def plot_series(network,label, carrier="AC"):

    n = network.copy()
    # assign_location(n)
    # assign_carriers(n)

    buses = n.buses.index[n.buses.carrier.str.contains(carrier)]
    buses = ["google"]

    supply = pd.DataFrame(index=n.snapshots)
    for c in n.iterate_components(n.branch_components):
        n_port = 4 if c.name=='Link' else 2
        for i in range(n_port):
            supply = pd.concat((supply,
                                (-1) * c.pnl["p" + str(i)].loc[:,
                                                               c.df.index[c.df["bus" + str(i)].isin(buses)]].groupby(c.df.carrier,
                                                                                                                     axis=1).sum()),
                               axis=1)

    for c in n.iterate_components(n.one_port_components):
        comps = c.df.index[c.df.bus.isin(buses)]
        supply = pd.concat((supply, ((c.pnl["p"].loc[:, comps]).multiply(
            c.df.loc[comps, "sign"])).groupby(c.df.carrier, axis=1).sum()), axis=1)

    supply = supply.groupby(supply.columns, axis=1).sum()

    both = supply.columns[(supply < 0.).any() & (supply > 0.).any()]

    positive_supply = supply[both]
    negative_supply = supply[both]

    positive_supply[positive_supply < 0.] = 0.
    negative_supply[negative_supply > 0.] = 0.

    supply[both] = positive_supply

    suffix = " charging"

    negative_supply.columns = negative_supply.columns + suffix

    supply = pd.concat((supply, negative_supply), axis=1)

    # 14-21.2 for flaute
    # 19-26.1 for flaute


    threshold = 1

    to_drop = supply.columns[(abs(supply) < threshold).all()]

    if len(to_drop) != 0:
        print("dropping", to_drop)
        supply.drop(columns=to_drop, inplace=True)

    if supply.empty: return

    supply.index.name = None

    supply = supply / 1e3

    supply.rename(columns={"electricity": "electric demand",
                           "heat": "heat demand"},
                  inplace=True)
    supply.columns = supply.columns.str.replace("residential ", "")
    supply.columns = supply.columns.str.replace("services ", "")
    supply.columns = supply.columns.str.replace("urban decentral ", "decentral ")

    preferred_order = pd.Index(["electric demand",
                                "transmission lines",
                                "hydroelectricity",
                                "hydro reservoir",
                                "run of river",
                                "pumped hydro storage",
                                "CHP",
                                "onshore wind",
                                "offshore wind",
                                "solar PV",
                                "solar thermal",
                                "building retrofitting",
                                "ground heat pump",
                                "air heat pump",
                                "resistive heater",
                                "OCGT",
                                "gas boiler",
                                "gas",
                                "natural gas",
                                "methanation",
                                "hydrogen storage",
                                "battery storage",
                                "hot water storage"])

    new_columns = (preferred_order.intersection(supply.columns)
                   .append(supply.columns.difference(preferred_order)))

    year = snakemake.wildcards.year
    supply =  supply.groupby(supply.columns, axis=1).sum()
    snakemake.config["tech_colors"]["PHS charging"] = snakemake.config["tech_colors"]["PHS"]
    snakemake.config["tech_colors"]["electric demand"] = snakemake.config["tech_colors"]["AC"]
    snakemake.config["tech_colors"]["offtake H2"] = "#FFC0CB"
    supply.rename(index=lambda x: x.replace(year = int(year)),
                  inplace=True)

    starts = [f"{year}-03-01", f"{year}-12-21"]
    stops = [f"{year}-03-08", f"{year}-12-28"]

    for i, start in enumerate(starts):
        stop = stops[i]
        fig, ax = plt.subplots()
        fig.set_size_inches((8, 5))

        (supply.loc[start:stop, new_columns]
         .plot(ax=ax, kind="area", stacked=True, linewidth=0.,
               color=[snakemake.config['tech_colors'][i.replace(suffix, "")]
                      for i in new_columns]))

        handles, labels = ax.get_legend_handles_labels()

        handles.reverse()
        labels.reverse()

        new_handles = []
        new_labels = []

        for i, item in enumerate(labels):
            if "charging" not in item:
                new_handles.append(handles[i])
                new_labels.append(labels[i])

        ax.legend(new_handles, new_labels, ncol=3, loc="upper left", frameon=False)
        ax.set_xlim([start, stop])
        # ax.set_ylim([-1300, 1900])
        ax.grid(True)
        ax.set_ylabel("Power [GW]")
        fig.tight_layout()

        fig.savefig("{}/{}/maps/series-{}-{}-{}-{}-{}-{}-{}.pdf".format(
            snakemake.config['results_dir'], snakemake.config['run'],
            label.to_list()[0][1],label.to_list()[0][2],
            label.to_list()[0][3], label.to_list()[0][4], start, stop, year),
            transparent=True)


def plot_nodal_balances(nodal_supply_energy):
    co2_carriers = ["co2", "co2 stored", "process emissions"]


    balances = {i.replace(" ","_"): [i] for i in nodal_supply_energy.index.levels[0]}
    balances["energy"] = [i for i in nodal_supply_energy.index.levels[0] if i not in co2_carriers]

    zone = snakemake.wildcards.zone

    k = "AC"
    v = ["AC"]

    df = nodal_supply_energy.loc[v].xs(zone, level=2)
    df = df.groupby(df.index.get_level_values(2)).sum()

    #convert MWh to TWh
    df = df / 1e6

    #remove trailing link ports
    df.index = [i[:-1] if ((i not in ["co2", "NH3", "H2"]) and (i[-1:] in ["0","1","2","3"])) else i for i in df.index]

    # df = df.groupby(df.index.map(rename_techs)).sum()
    df = df.groupby(df.index).sum()

    df = df.droplevel([0,2], axis=1)

    df = df["ref"].iloc[:,:1]

    to_drop = df.index[df.abs().max(axis=1) < 1] # snakemake.config['plotting']['energy_threshold']/10]

    print("dropping")

    print(df.loc[to_drop])

    df = df.drop(to_drop)

    df.rename(index={"urban central solid biomass CHP": "biomass CHP"}, inplace=True)
    snakemake.config['tech_colors']["biomass CHP"] =  snakemake.config['tech_colors']["urban central solid biomass CHP"]

    for elec_c in ["AC", "DC"]:
        df.rename(lambda x: x.replace(elec_c, "import") if df.loc[x].iloc[0]>0 else x.replace(elec_c, "export"), inplace=True)
    df.rename(index=lambda x: "offshore wind" if "offwind" in x else x, inplace=True)
    df.rename(index=lambda x: "onshore wind" if "onwind" in x else x, inplace=True)
    df = df.groupby(df.index).sum()
    df.columns = ["ref"]

    a = df[df>0]["ref"].dropna()

    fig, ax = plt.subplots()

    a.plot(kind="pie", ax=ax,
                         colors=[snakemake.config['tech_colors'][i] for i in a.index],
                         autopct='%1.1f%%', shadow=True, pctdistance=0.9)
    plt.ylabel("")
    fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"energy-pie-{zone}-{k}.pdf",
                bbox_inches='tight')

    fig, ax = plt.subplots()
    a = a.drop(["import", "export"], errors="ignore")
    a.plot(kind="pie", ax=ax,
                         colors=[snakemake.config['tech_colors'][i] for i in a.index],
                         autopct='%1.1f%%', shadow=True, pctdistance=0.9)
    plt.ylabel("")
    fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"energy-pie-{zone}-{k}-noimports.pdf",
                bbox_inches='tight')


    # fig, ax = plt.subplots(figsize=(12,8))
    # df.T.plot(kind="bar",ax=ax,stacked=True,
    #         color=[snakemake.config['tech_colors'][i] for i in df.index],
    #         grid=True)

    # handles,labels = ax.get_legend_handles_labels()

    # handles.reverse()
    # labels.reverse()

    # if v[0] in co2_carriers:
    #     ax.set_ylabel("CO2 [MtCO2/a]")
    # else:
    #     ax.set_ylabel("Energy [TWh/a]")

    # ax.set_xlabel("")

    # ax.grid(axis="x")

    # ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)


    # fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"nodal_balance-{zone}-{k}.pdf",
    #             bbox_inches='tight')




def plot_balances(balances_df):

    co2_carriers = ["co2", "co2 stored", "process emissions"]


    balances = {i.replace(" ","_"): [i] for i in balances_df.index.levels[0]}
    balances["energy"] = [i for i in balances_df.index.levels[0] if i not in co2_carriers]

    for k, v in balances.items():

        df = balances_df.loc[v]
        df = df.groupby(df.index.get_level_values(2)).sum()

        #convert MWh to TWh
        df = df / 1e6

        #remove trailing link ports
        df.index = [i[:-1] if ((i not in ["co2", "NH3", "H2"]) and (i[-1:] in ["0","1","2","3"])) else i for i in df.index]

        # df = df.groupby(df.index.map(rename_techs)).sum()
        df = df.groupby(df.index).sum()

        df = df.droplevel(0, axis=1)


        to_drop = df.index[df.abs().max(axis=1) < 1] # snakemake.config['plotting']['energy_threshold']/10]


        df = df.drop(to_drop)

        df = df.droplevel(0, axis=1)

        print(df.sum())

        if df.empty:
            continue

        #for price in df.columns.levels[1]:
        for policy in ["grd", "ref", "res"]:
            for volume in df.columns.levels[2]:
                for storage_type in ["nostore", "flexibledemand"]: #df.columns.levels[3]:
                    fig, ax = plt.subplots(figsize=(12,8))

                    balance = df.xs((volume,storage_type),level=[2,3], axis=1)
                    to_drop = ["offgrid", "cfe", "grd", "ref"]
                    if policy=="res": to_drop.append("res")
                    balance.sub(balance[policy], axis=0, level=1).drop(to_drop, errors="ignore",axis=1).T.plot(kind="bar",ax=ax,stacked=True,
                                                            color=[snakemake.config['tech_colors'][i] for i in df.index],
                                                            grid=True,
                                                            title= f"Difference policy - {policy}: volume {volume} storage type {storage_type}")

                    handles,labels = ax.get_legend_handles_labels()

                    handles.reverse()
                    labels.reverse()

                    if v[0] in co2_carriers:
                        ax.set_ylabel("CO2 [MtCO2/a]")
                    else:
                        ax.set_ylabel("Energy [TWh/a]")

                    ax.set_xlabel("")

                    ax.grid(axis="x")

                    ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)


                    fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"balances_difference_{policy}" + k + f"_{volume}volume_{storage_type}.pdf", bbox_inches='tight')

                    # generation diff ############
                    generation = balance[(balance>0)]

                    fig, ax = plt.subplots(figsize=(12,8))
                    generation.sub(generation[policy], axis=0, level=1).drop(to_drop, errors="ignore",axis=1).T.plot(kind="bar",ax=ax,stacked=True,
                                                            color=[snakemake.config['tech_colors'][i] for i in generation.index],
                                                            grid=True,
                                                            title= f"Difference policy - {policy}: volume {volume} storage type {storage_type}")
                    handles,labels = ax.get_legend_handles_labels()

                    handles.reverse()
                    labels.reverse()

                    if v[0] in co2_carriers:
                        ax.set_ylabel("CO2 [MtCO2/a]")
                    else:
                        ax.set_ylabel("Energy [TWh/a]")

                    ax.set_xlabel("")

                    ax.grid(axis="x")

                    ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)


                    fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"balances_difference_generation_{policy}" + k + f"_{volume}volume_{storage_type}.pdf", bbox_inches='tight')


                    # demand diff ############
                    generation = (-1)*balance[(balance<0)]

                    fig, ax = plt.subplots(figsize=(12,8))
                    generation.sub(generation[policy], axis=0, level=1).drop(to_drop, errors="ignore",axis=1).T.plot(kind="bar",ax=ax,stacked=True,
                                                            color=[snakemake.config['tech_colors'][i] for i in generation.index],
                                                            grid=True,
                                                            title= f"Difference policy - {policy}: volume {volume} storage type {storage_type}")
                    handles,labels = ax.get_legend_handles_labels()

                    handles.reverse()
                    labels.reverse()

                    if v[0] in co2_carriers:
                        ax.set_ylabel("CO2 [MtCO2/a]")
                    else:
                        ax.set_ylabel("Energy [TWh/a]")

                    ax.set_xlabel("")

                    ax.grid(axis="x")

                    ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)


                    fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"balances_difference_demand_{policy}" + k + f"_{volume}volume_{storage_type}.pdf", bbox_inches='tight')

        for policy in df.columns.levels[0]:
            for volume in df.columns.levels[2]:
                for storage_type in df.columns.levels[3]:
                    fig, ax = plt.subplots(figsize=(12,8))

                    balance = df.xs((policy,volume,storage_type),level=[0,2,3], axis=1)

                    balance.T.plot(kind="bar",ax=ax,stacked=True,
                                                           color=[snakemake.config['tech_colors'][i] for i in df.index],
                                                           grid=True,
                                                           title= f"policy {policy} volume {volume} storage type {storage_type}")


                    handles,labels = ax.get_legend_handles_labels()

                    handles.reverse()
                    labels.reverse()

                    if v[0] in co2_carriers:
                        ax.set_ylabel("CO2 [MtCO2/a]")
                    else:
                        ax.set_ylabel("Energy [TWh/a]")

                    ax.set_xlabel("")

                    ax.grid(axis="x")

                    ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)


                    fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + "balances_" + k + f"_{policy}policy_{volume}volume_{storage_type}.pdf", bbox_inches='tight')



def plot_cf(df, wished_policies, wished_order, volume, name=""):
    cf_elec = df.loc[wished_policies].xs(volume, level=1)
    cf_elec = cf_elec.reindex(wished_order, level=1)
    cf_elec.rename(index=rename_scenarios,
                   level=0,inplace=True)
    fig, ax = plt.subplots(nrows=1, ncols=len(wished_policies), sharey=True,
                           figsize=(9,3.5))
    nice_names = [rename_scenarios[scen] if scen in rename_scenarios.keys() else scen for scen in wished_policies]
    for i, policy in enumerate(nice_names):
        cf_elec.loc[policy].plot(kind="bar", grid=True, ax=ax[i], title=policy)
        ax[i].set_xlabel("")
    ax[0].set_ylabel("capacity factor")
    fig.savefig(snakemake.output.cf_plot.split(".pdf")[0] + f"{volume}{name}.pdf",
                bbox_inches="tight")

    fig.savefig(snakemake.output.cf_plot,
                bbox_inches="tight")

def plot_consequential_emissions(emissions, wished_policies, wished_order, volume, name=""):
    # consequential emissions
    emissions_v = emissions.loc[wished_policies+["ref"]].xs(float(volume), level=1)
    emissions_v = emissions_v.reindex(wished_order, level=1)
    emissions_v.rename(index=rename_scenarios,
                       level=0,inplace=True)
    nice_names = [rename_scenarios[scen] if scen in rename_scenarios.keys() else scen for scen in wished_policies]
    fig, ax = plt.subplots(nrows=1, ncols=len(wished_policies), sharey=True,figsize=(9,3.5))
    for i, policy in enumerate(nice_names):
        # annually produced H2 in [t_H2/a]
        produced_H2 = float(volume)*8760 / LHV_H2
        em_p = emissions_v.loc[policy].sub(emissions_v.loc["ref"].mean())/ produced_H2
        em_p.iloc[:,0].plot(kind="bar", grid=True, ax=ax[i], title=policy)
        ax[i].set_xlabel("")
    ax[0].set_ylabel("consequential emissions \n [kg$_{CO_2}$/kg$_{H_2}$]")
    fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0]+ f"consequential_emissions_{volume}{name}.pdf",
                bbox_inches="tight")

def plot_attributional_emissions(emission_rate, wished_policies, wished_order, volume, name=""):
    for consider_import in ["carbon_intensity_H2", "carbon_intensity_H2_ni"]:
        em_r = emission_rate.loc[consider_import].unstack().droplevel([0,2]).xs(volume, level=1)
        em_r = em_r.reindex(wished_policies).fillna(0)
        em_r = em_r.reindex(wished_order, axis=1)
        em_r.rename(index=rename_scenarios,
                           level=0,inplace=True)
        nice_names = [rename_scenarios[scen] if scen in rename_scenarios.keys()
                      else scen for scen in wished_policies]
        fig, ax = plt.subplots(nrows=1, ncols=len(wished_policies), sharey=True,figsize=(9,3.5))
        for i, policy in enumerate(nice_names):
            em_r.loc[policy].plot(kind="bar", grid=True, ax=ax[i], title=policy)
            ax[i].set_xlabel("")
            ax[i].axhline(y=1, linestyle="--", color="black")
            ax[i].axhline(y=3, linestyle="--", color="black")
            ax[i].axhline(y=10, linestyle="--", color="black")
        ax[0].set_ylabel("attributional emissions\n [kg$_{CO_2}$/kg$_{H_2}$]")
        ax[2].text(x= 3.6, y=0.8, s='carbon intensity of\nblue hydrogen')
        ax[2].text(x= 3.6, y=9.8, s='carbon intensity of\ngrey hydrogen')
        ax[2].text(x=3.6, y=2.8, s='EU threshold for \nlow-carbon hydrogen')
        if consider_import == "carbon_intensity_H2":
            suffix = ""
        else:
            suffix = "_noimports"
        plt.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"attributional_emissions_{volume}{name}{suffix}.pdf",
                    bbox_inches='tight')

def plot_cost_breakdown(h2_cost, wished_policies, wished_order, volume, name=""):
    costb = h2_cost.xs(volume, level=1, axis=1).droplevel(0)
    costb = costb.stack().reindex(wished_policies, axis=1).fillna(0).unstack()
    costb = costb.reindex(wished_order, level=1, axis=1)
    costb.rename(columns=rename_scenarios,
                       level=0,inplace=True)
    costb.loc["battery"] = costb[costb.index.str.contains("battery")].sum()
    costb.drop(["battery charger", "battery discharger"], inplace=True)
    singlecost = costb.drop(["H2 cost", "offtake H2"]).div(float(volume)*8760)
    # convert EUR/MWh_H2 in Eur/kg_H2
    singlecost = singlecost / 1e3 * LHV_H2
    to_drop = singlecost.loc[abs(singlecost.sum(axis=1))<1].index
    singlecost.drop(to_drop, inplace=True)
    nice_names = [rename_scenarios[scen] if scen in rename_scenarios.keys()
                  else scen for scen in wished_policies]

    fig, ax = plt.subplots(nrows=1, ncols=len(wished_policies), sharey=True,figsize=(9,3.5))
    for i, policy in enumerate(nice_names):
        singlecost.sum().loc[policy].rename("net total").plot(ax=ax[i], style="-o", lw=0, color="black")
        singlecost.T.loc[policy].plot(kind="bar", stacked=True, ax=ax[i], title=policy,
                 color=[snakemake.config['tech_colors'][i] for i in singlecost.index],
                 grid=True, legend=False)
    not_grd = singlecost.columns.levels[0][~singlecost.columns.levels[0].str.contains("grid")]
    ax[0].set_ylim([singlecost[singlecost<0].sum().min()*1.1, singlecost[singlecost>0].sum().loc[not_grd].max()*1.1])
    ax[0].set_ylabel("cost \n [Euro/kg$_{H_2}$]")
    plt.legend(bbox_to_anchor=(1,1))
    fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"costbreakdown_{volume}{name}.pdf",
                bbox_inches='tight')


def plot_shadow_prices(weighted_prices, wished_policies, wished_order, volume,
                       name="", carrier="AC"):
    w_price = weighted_prices.droplevel([0,2], axis=1).xs((volume, "rest"), level=[1,3], axis=1)[wished_policies].loc[carrier]
    w_price.rename(rename_scenarios, level=0, inplace=True)
    nice_names = [rename_scenarios[scen] if scen in rename_scenarios.keys()
                  else scen for scen in wished_policies]
    not_grd = w_price.index.levels[0][~w_price.index.levels[0].str.contains("grid")]

    fig, ax = plt.subplots(nrows=1, ncols=len(wished_policies),
                           sharey=True,figsize=(9,3.5))
    for i, policy in enumerate(nice_names):
        w_price.loc[policy].reindex(wished_order).plot(kind="bar", ax=ax[i],
                                                 title=policy, grid=True)
    ax[0].set_ylabel("price \n [Euro/MWh]")
    ax[0].set_ylim([0, w_price.loc[not_grd].max()*1.1])
    fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"shadowprices_{volume}{name}_{carrier}_croped.pdf",
                bbox_inches='tight')

def plot_h2genmix(h2_gen_mix, wished_policies, wished_order, volume,
                       name=""):
    snakemake.config['tech_colors']["solar local"] = "#ffdd78"
    snakemake.config['tech_colors']["onwind local"] = "#bae3f9"
    snakemake.config['tech_colors']["biomass CHP"] = snakemake.config['tech_colors']["urban central solid biomass CHP"]

    gen_mix = h2_gen_mix.xs(volume, level=1, axis=1)[wished_policies]
    gen_mix.rename(rename_scenarios, level=0, axis=1, inplace=True)
    nice_names = [rename_scenarios[scen] if scen in rename_scenarios.keys()
                  else scen for scen in wished_policies]

    gen_mix = gen_mix[nice_names][gen_mix[nice_names].sum(axis=1)>1]
    gen_mix.rename(index=lambda x: x.replace("urban central solid ", ""), inplace=True)
    gen_mix.loc["offwind",:] = gen_mix[gen_mix.index.str.contains("offwind")].sum()
    gen_mix.drop(["offwind-ac", "offwind-dc"],inplace=True, errors="ignore")
    carrier_order = ["solar local", "onwind local", "solar", "onwind", "offwind",
                     "ror", "biomass CHP", "CCGT", "OCGT", "coal", "lignite", "oil"]
    gen_mix = gen_mix.reindex(carrier_order)
    fig, ax = plt.subplots(nrows=1, ncols=len(wished_policies),
                           sharey=True,figsize=(9,3.5))
    for i, policy in enumerate(nice_names):
        (gen_mix[policy]/1e6).reindex(columns=wished_order).T.plot(kind="bar", stacked=True,
                                                             ax=ax[i],
                                                             color=[snakemake.config['tech_colors'][i] for i in gen_mix.index],
                                                 title=policy, grid=True, legend=False)
    plt.legend(bbox_to_anchor=(1,1))
    ax[0].set_ylabel("generation \n [TWh]")

    fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"h2_generationmix_{volume}{name}.pdf",
                bbox_inches='tight')
#%%
final = pd.DataFrame()
cf = pd.DataFrame()
emissions = pd.DataFrame()
supply_energy = pd.DataFrame()
nodal_capacities = pd.DataFrame()
weighted_prices = pd.DataFrame()
curtailment = pd.DataFrame()
nodal_costs = pd.DataFrame()
costs = pd.DataFrame()
h2_cost = pd.DataFrame()
emission_rate = pd.DataFrame()
nodal_supply_energy = pd.DataFrame()
h2_gen_mix = pd.DataFrame()

for network_path in snakemake.input:
    try:
        n = pypsa.Network(network_path,
                          override_component_attrs=override_component_attrs())
    except OSError:
            print(network_path, " not solved yet.")
            continue
    policy = network_path.split("/")[-1].split("_")[0]
    price = float(network_path.split("price")[0].split("_")[-1])
    try:
        volume = float(network_path.split("/")[-1].split("volume")[0].split("_")[-1])
    except ValueError:
        volume = "fix_cap"
    assign_locations(n)

    participation = network_path.split("networks/")[1][:2]
    weightings = n.snapshot_weightings.generators
    storage_type = network_path.split("_")[-1].replace(".nc", "")

    cols = pd.MultiIndex.from_product([[participation], [policy], [price], [volume], [storage_type]],
                                      names=["participation", "policy", "price", "volume", "storage_type"])
    if policy in ["res", "exl"]:
        plot_series(n, cols)
    # offtake H2
    offtake_p = n.generators_t.p.loc[:,n.generators.carrier=="offtake H2"].mul(weightings, axis=0)
    cols = pd.MultiIndex.from_product([[participation], [policy], [price],
                                       [volume], [storage_type], offtake_p.columns],
                                      names=["participation", "policy",
                                             "price", "volume", "storage_type", "node"])
    offtake_p.columns = cols
    final = pd.concat([offtake_p, final], axis=1)

    # capacity factor
    p_nom_opt = n.links.p_nom_opt
    df = (n.links_t.p0/n.links.p_nom_opt)
    df.drop(df.columns[(p_nom_opt<10)], axis=1, inplace=True)
    df.dropna(axis=1, inplace=True)
    cols = pd.MultiIndex.from_product([[participation], [policy], [price], [volume],
                                       [storage_type], df.columns],
                                      names=["participation", "policy", "price",
                                             "volume", "storage_type", "node"])
    df.columns = cols
    cf = pd.concat([cf, df], axis=1)

    # co2 emissions
    co2_emission = n.stores_t.e["co2 atmosphere"].iloc[-1]
    cols = pd.MultiIndex.from_product([[participation], [policy], [price], [volume], [storage_type]],
                                      names=["participation", "policy", "price", "volume", "storage_type"])
    co2_emission = pd.DataFrame([co2_emission], index=cols)
    emissions = pd.concat([emissions, co2_emission])

    # supply energy
    supply_energy = calculate_supply_energy(n, cols, supply_energy)
    # nodal supply energy
    nodal_supply_energy = calculate_nodal_supply_energy(n, cols, nodal_supply_energy)
    # nodal capacities
    nodal_capacities = calculate_nodal_capacities(n, cols, nodal_capacities)

    weighted_prices = calculate_weighted_prices(n, cols, weighted_prices)

    curtailment = calculate_curtailment(n,cols, curtailment)

    # nodal capacities
    nodal_costs = calculate_nodal_costs(n, cols, nodal_costs)

    # costs
    costs = calculate_costs(n, cols, costs)

    # h2 costs
    h2_cost = calculate_hydrogen_cost(n, cols, h2_cost)

    # carbon intensity H2
    if "import" in n.links.carrier.unique():
        emission_rate = calculate_emission_rate(n, cols, emission_rate)

    # plot_series(n,cols)
    if policy!="ref":
        h2_gen_mix = calculate_h2_generationmix(n, cols, h2_gen_mix)

cf = cf.loc[:,cf.columns.get_level_values(5).str.contains("google H2 Electrolysis")]
#%%
emissions.to_csv(snakemake.output.csvs_emissions)
cf.to_csv(snakemake.output.csvs_cf)
supply_energy.to_csv(snakemake.output.csvs_supply_energy)
nodal_supply_energy.to_csv(snakemake.output.csvs_nodal_supply_energy)
nodal_capacities.to_csv(snakemake.output.csvs_nodal_capacities)
weighted_prices.to_csv(snakemake.output.csvs_weighted_prices)
curtailment.to_csv(snakemake.output.csvs_curtailment)
costs.to_csv(snakemake.output.csvs_costs)
nodal_costs.to_csv(snakemake.output.csvs_nodal_costs)
h2_cost.to_csv(snakemake.output.csvs_h2_costs)
emission_rate.to_csv(snakemake.output.csvs_emission_rate)
h2_gen_mix.to_csv(snakemake.output.csvs_h2_gen_mix)
#%%

# emissions = pd.read_csv(snakemake.output.csvs_emissions ,index_col=[0,1,2,3,4])
# cf = pd.read_csv(snakemake.output.csvs_cf,index_col=0, header=[0,1,2,3,4,5], parse_dates=True)
# supply_energy = pd.read_csv(snakemake.output.csvs_cf, index_col=[0,1,2], header=[0,1,2,3,4])
# nodal_supply_energy = pd.read_csv(snakemake.output.csvs_nodal_supply_energy, index_col=[0,1,2, 3], header=[0,1,2,3,4])
# nodal_capacities = pd.read_csv(snakemake.output.csvs_nodal_capacities, index_col=[0,1,2], header=[0,1,2,3,4])
# weighted_prices = pd.read_csv(snakemake.output.csvs_weighted_prices, index_col=[0], header=[0,1,2,3,4,5])
# curtailment = pd.read_csv(snakemake.output.csvs_curtailment, index_col=[0,1,2], header=[0,1,2,3,4])
# costs = pd.read_csv(snakemake.output.csvs_costs, index_col=[0,1,2], header=[0,1,2,3,4])
# nodal_costs = pd.read_csv(snakemake.output.csvs_nodal_costs, index_col=[0,1,2,3], header=[0,1,2,3,4])
# h2_cost = pd.read_csv(snakemake.output.csvs_h2_costs, index_col=[0,1], header=[0,1,2,3,4])
# emission_rate = pd.read_csv(snakemake.output.csvs_emission_rate, index_col=[0], header=[0,1,2,3,4])
# h2_gen_mix = pd.read_csv(snakemake.output.csvs_h2_gen_mix, index_col=[0,1], header=[0,1,2,3,4])

policies = cf.columns.levels[1]
participations = cf.columns.levels[0]

a = cf.mean().droplevel([0,2, 5])
emissions = emissions.droplevel([0, 2])
h2_cost = h2_cost.droplevel([0,2], axis=1)
h2_gen_mix = h2_gen_mix.rename(index=lambda x: x.replace("1","").replace("0","")).droplevel(0).droplevel([0,2], axis=1)

rename_scenarios = {"res1p0": "annually", "exl1p0": "hourly", "offgrid": "hourly",
                    "grd": "grid",
                    "res1p2": "annually excess 20%", "exl1p2": "hourly excess 20%",
                    "res1p3": "annually excess 30%", "exl1p3": "hourly excess 30%",}

plot_scenarios = {"":["grd", "res1p0", "offgrid"],
                  "_sensi_excess":  ["offgrid", "exl1p2", "exl1p3"],
                  "_sensi_excess_annual": ["res1p0", "res1p2", "res1p3"],
                  "_sensi_monthly": ["res1p0", "monthly", "offgrid"],
                  }
wished_order = ["flexibledemand", "underground", "tank", "nostore"]
#%%
for volume in a.index.get_level_values(1).unique():
    for name, wished_policies in plot_scenarios.items():
        print(name)
        # capacity factors
        plot_cf(a, wished_policies, wished_order, volume, name=name)

        # consequential emissions
        plot_consequential_emissions(emissions, wished_policies, wished_order,
                                     volume, name=name)

        # attributional emissions
        plot_attributional_emissions(emission_rate, wished_policies, wished_order,
                                     volume, name=name)

        # cost breakdown
        plot_cost_breakdown(h2_cost, wished_policies, wished_order, volume, name=name)

        # shadow prices
        plot_shadow_prices(weighted_prices, wished_policies, wished_order, volume,
                               name=name, carrier="AC")

        # generation mix of H2
        plot_h2genmix(h2_gen_mix, wished_policies, wished_order, volume,
                               name=name)

#%%
wished_policies = ["grd", "res1p0", "offgrid"]
res = nodal_capacities.loc[nodal_capacities.index.get_level_values(2).isin(["solar", "onwind"])].droplevel(0, axis=1).drop("cfe", axis=1)
store = nodal_capacities.loc[nodal_capacities.index.get_level_values(2).isin(["H2 Store", "battery"])].droplevel(0, axis=1).drop("cfe", axis=1)
supply_energy.drop("cfe", axis=1, level=1,inplace=True, errors="ignore")
for volume in res.columns.levels[2]:

    caps = res.xs(volume, level=2, axis=1).xs("google", level=1).droplevel(0).droplevel(1, axis=1).reindex(wished_policies, level=0, axis=1).fillna(0)
    caps = caps.stack().reindex(wished_policies, axis=1).fillna(0).unstack()
    caps = caps.reindex(wished_order, level=1, axis=1)
    caps.rename(columns=rename_scenarios,
                       level=0,inplace=True)
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(9,3.5))
    for i, policy in enumerate(['grid', 'annually',  'hourly']):

        (caps[policy].T/1e3).plot(grid=True,
                        title = policy,
                        color=[snakemake.config["tech_colors"][i] for i in caps.index],
                        kind="bar",
                        ax=ax[i],
                        legend=False
                        )
    ax[0].set_ylabel("capacity \n [GW]")
    plt.legend(bbox_to_anchor=(1,1))
    plt.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"capacities_RES_{volume}volume.pdf",
                    bbox_inches='tight')

    caps = store.xs(volume, level=2, axis=1).xs("google", level=1).droplevel(0).droplevel(1, axis=1).reindex(wished_policies, level=0, axis=1).fillna(0)
    caps = caps.stack().reindex(wished_policies, axis=1).fillna(0).unstack()
    caps = caps.reindex(wished_order, level=1, axis=1)
    caps.rename(columns=rename_scenarios,
                       level=0,inplace=True)
    caps.drop("flexibledemand", level=1, axis=1, inplace=True)
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(9,3.5))
    for i, policy in enumerate(['grid', 'annually',  'hourly']):

        (caps[policy].T/1e3).plot(grid=True,
                        title = policy,
                        color=[snakemake.config["tech_colors"][i] for i in caps.index],
                        kind="bar",
                        ax=ax[i],
                        legend=False
                        )
    ax[0].set_ylabel("energy capacity \n [GWh]")
    plt.legend(bbox_to_anchor=(1,1))
    fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"capacities_price_{volume}volume.pdf",
                bbox_inches='tight')





    c = curtailment.xs((volume), level=3, axis=1).droplevel(0, axis=1).droplevel(1, axis=1).reindex(wished_policies, level=0, axis=1).fillna(1)
    res_car = ["solar", "onwind", "all"]
    for carrier in res_car:
        d = c.loc[c.index.get_level_values(0).str.contains(carrier)].groupby(level=1).mean().stack().unstack(-2).dropna(axis=1, how="all")
        d[d<0] = 0
        d = d.reindex(wished_policies,axis=1, level=0)
        d = d.reindex(index=wished_order)
        d.rename(columns=rename_scenarios,
                          inplace=True)
        if carrier=="all":
            d = d.droplevel(1, axis=1)
        else:
            d = d.xs("google", axis=1,level=1)

        fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(9,3.5))
        for i, policy in enumerate(['grid', 'annually',  'hourly']):
            ((1-d[policy])*100).plot(kind="bar", grid=True, ax=ax[i],
                                                 title=policy, legend=False)
        ax[0].set_ylabel("curtailment \n [%]")
        # plt.xlabel("price \n [Eur/kg$_{H2}$]")
        # plt.legend(bbox_to_anchor=(1,1))
        fig.savefig(snakemake.output.cf_plot.split("cf_ele")[0] + f"curtailment_{volume}volume_{carrier}.pdf",
                    bbox_inches='tight')
#%%
plot_nodal_balances(nodal_supply_energy)
