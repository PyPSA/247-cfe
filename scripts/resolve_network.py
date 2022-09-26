


import pypsa
import pandas as pd
import numpy as np
from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints

import logging
logger = logging.getLogger(__name__)

# Suppress logging of the slack bus choices
pypsa.pf.logger.setLevel(logging.WARNING)

from vresutils.benchmark import memory_logger


from _helpers import override_component_attrs


def geoscope(zone, area):
    '''
    basenodes_to_keep -> geographical scope of the model
    country_nodes -> scope for national RES policy constraint
    node -> zone where C&I load is located
    '''
    d = dict();

    if zone == 'IE':
        d['basenodes_to_keep'] = ["IE5 0", "GB0 0", "GB5 0"]
        d['country_nodes'] = ["IE5 0"]
        d['node'] = "IE5 0"

    elif zone == 'DK':
        d['basenodes_to_keep'] = ["DK1 0", "DK2 0", "SE2 0", "NO2 0", "NL1 0", "DE1 0"]
        d['country_nodes'] = ["DK1 0", "DK2 0"]
        d['node'] = "DK1 0"

    elif zone == 'DE':
        d['basenodes_to_keep'] = ['DE1 0', 'BE1 0', 'NO2 0', 'DK1 0', 'DK2 0', 'SE2 0', 'GB0 0',
                                  'FR1 0', 'LU1 0', 'NL1 0', 'PL1 0', 'AT1 0', 'CH1 0', 'CZ1 0']
        d['country_nodes'] = ['DE1 0']
        d['node'] = 'DE1 0'

    elif zone == 'NL':
        d['basenodes_to_keep'] = ['NL1 0', 'GB0 0', 'DK1 0', 'NO2 0', 'BE1 0', 'DE1 0']
        d['country_nodes'] = ['NL1 0']
        d['node'] = 'NL1 0'

    else:
        print(f"'zone' wildcard must be one of 'IE', 'DK', 'DE', 'NL'. Now is {zone}.")
        sys.exit()

    if area == 'EU':
        d['basenodes_to_keep'] = ['AL1 0', 'AT1 0',  'BA1 0',  'BE1 0',  'BG1 0',
            'CH1 0', 'CZ1 0',  'DE1 0',  'DK1 0',  'DK2 0',
            'EE6 0', 'ES1 0',  'ES4 0',  'FI2 0',  'FR1 0',
            'GB0 0', 'GB5 0',  'GR1 0',  'HR1 0',  'HU1 0',
            'IE5 0', 'IT1 0',  'IT3 0',  'LT6 0',  'LU1 0',
            'LV6 0', 'ME1 0',  'MK1 0',  'NL1 0',  'NO2 0',
            'PL1 0', 'PT1 0',  'RO1 0',  'RS1 0',  'SE2 0',
            'SI1 0', 'SK1 0']

    return d

def freeze_capacities(n):

    for name, attr in [("generators","p"),("links","p"),("stores","e")]:
        df = getattr(n,name)
        df[attr + "_nom_extendable"] = False
        df[attr + "_nom"] = df[attr + "_nom_opt"]

    #allow more emissions
    n.stores.at["co2 atmosphere","e_nom"] *=2

def add_H2(n):

    if policy == "ref":
        return None

    n.add("Bus",
          name)

    n.add("Bus",
          f"{name} H2",
          carrier="H2"
          )

    n.add("Link",
          f"{name} H2 Electrolysis",
          bus0=name,
          bus1=f"{name} H2",
          carrier="H2 Electrolysis",
          efficiency=n.links.at[f"{node} H2 Electrolysis"+"-{}".format(year), "efficiency"],
          capital_cost=n.links.at[f"{node} H2 Electrolysis"+"-{}".format(year), "capital_cost"],
          p_nom_extendable=True,
          lifetime=n.links.at[f"{node} H2 Electrolysis"+"-{}".format(year), "lifetime"]
          )

    n.add("Load",
          f"{name} H2",
          carrier=f"{name} H2",
          bus=f"{name} H2",
          p_set=pd.Series(load,index=n.snapshots))

    #cost-less storage to indivcate flexible demand
    n.add("Store",
          f"{name} H2 Store",
          bus=f"{name} H2",
          e_cyclic=True,
          e_nom_extendable=False,
          e_nom=load*8760,
          carrier="H2 Store",
          )

    if policy in ["res","cfe","exl"]:
        n.add("Link",
              name + " export",
              bus0=name,
              bus1=node,
              marginal_cost=0.1, #large enough to avoid optimization artifacts, small enough not to influence PPA portfolio
              p_nom=1e6)

    if policy in ["grd","res"]:
        n.add("Link",
              name + " import",
              bus0=node,
              bus1=name,
              marginal_cost=0.001, #large enough to avoid optimization artifacts, small enough not to influence PPA portfolio
              p_nom=1e6)

    if policy == "grd":
        return None

    #RES generator
    for carrier in ["onwind","solar"]:
        gen_template = node+" "+carrier+"-{}".format(year)
        n.add("Generator",
              f"{name} {carrier}",
              carrier=carrier,
              bus=name,
              p_nom_extendable=True,
              p_max_pu=n.generators_t.p_max_pu[gen_template],
              capital_cost=n.generators.at[gen_template,"capital_cost"],
              marginal_cost=n.generators.at[gen_template,"marginal_cost"])


    if "battery" in ["battery"]:
        n.add("Bus",
              f"{name} battery",
              carrier="battery"
              )

        n.add("Store",
              f"{name} battery",
              bus=f"{name} battery",
              e_cyclic=True,
              e_nom_extendable=True,
              carrier="battery",
              capital_cost=n.stores.at[f"{node} battery"+"-{}".format(year), "capital_cost"],
              lifetime=n.stores.at[f"{node} battery"+"-{}".format(year), "lifetime"]
              )

        n.add("Link",
              f"{name} battery charger",
              bus0=name,
              bus1=f"{name} battery",
              carrier="battery charger",
              efficiency=n.links.at[f"{node} battery charger"+"-{}".format(year), "efficiency"],
              capital_cost=n.links.at[f"{node} battery charger"+"-{}".format(year), "capital_cost"],
              p_nom_extendable=True,
              lifetime=n.links.at[f"{node} battery charger"+"-{}".format(year), "lifetime"]
              )

        n.add("Link",
              f"{name} battery discharger",
              bus0=f"{name} battery",
              bus1=name,
              carrier="battery discharger",
              efficiency=n.links.at[f"{node} battery discharger"+"-{}".format(year), "efficiency"],
              marginal_cost=n.links.at[f"{node} battery discharger"+"-{}".format(year), "marginal_cost"],
              p_nom_extendable=True,
              lifetime=n.links.at[f"{node} battery discharger"+"-{}".format(year), "lifetime"]
              )

def add_dummies(n):
    elec_buses = n.buses.index[n.buses.carrier == "AC"]

    print("adding dummies to",elec_buses)
    n.madd("Generator",
           elec_buses + " dummy",
           bus=elec_buses,
           carrier="dummy",
           p_nom=1e6,
           marginal_cost=1e3)




solver_name = "gurobi"

solver_options = {"method" : 2,
                  #"crossover" : 0,
                  "BarConvTol": 1.e-5}
def solve(policy):

    n = pypsa.Network(snakemake.input.base_network,
                      override_component_attrs=override_component_attrs())

    freeze_capacities(n)

    add_H2(n)

    add_dummies(n)

    def res_constraints(n):

        res_gens = [f"{name} onwind",f"{name} solar"]

        weightings = pd.DataFrame(np.outer(n.snapshot_weightings["generators"],[1.]*len(res_gens)),
                                  index = n.snapshots,
                                  columns = res_gens)
        res = join_exprs(linexpr((weightings,get_var(n, "Generator", "p")[res_gens]))) # single line sum

        electrolysis = get_var(n, "Link", "p")[f"{name} H2 Electrolysis"]

        load = join_exprs(linexpr((-n.snapshot_weightings["generators"],electrolysis)))

        lhs = res + "\n" + load

        con = define_constraints(n, lhs, '=', 0., 'RESconstraints','REStarget')

    def excess_constraints(n):

        res_gens = [f"{name} onwind",f"{name} solar"]

        weightings = pd.DataFrame(np.outer(n.snapshot_weightings["generators"],[1.]*len(res_gens)),
                                  index = n.snapshots,
                                  columns = res_gens)
        res = join_exprs(linexpr((weightings,get_var(n, "Generator", "p")[res_gens]))) # single line sum

        electrolysis = get_var(n, "Link", "p")[f"{name} H2 Electrolysis"]

        allowed_excess = 1.2

        load = join_exprs(linexpr((-allowed_excess*n.snapshot_weightings["generators"],electrolysis)))

        lhs = res + "\n" + load

        con = define_constraints(n, lhs, '<=', 0., 'RESconstraints','REStarget')

    def add_battery_constraints(n):

        chargers_b = n.links.carrier.str.contains("battery charger")
        chargers = n.links.index[chargers_b & n.links.p_nom_extendable]
        dischargers = chargers.str.replace("charger", "discharger")

        if chargers.empty or ('Link', 'p_nom') not in n.variables.index:
            return

        link_p_nom = get_var(n, "Link", "p_nom")

        lhs = linexpr((1,link_p_nom[chargers]),
                      (-n.links.loc[dischargers, "efficiency"].values,
                       link_p_nom[dischargers].values))

        define_constraints(n, lhs, "=", 0, 'Link', 'charger_ratio')

    def extra_functionality(n, snapshots):

        add_battery_constraints(n)

        if policy == "res":
            print("setting annual RES target")
            res_constraints(n)
        elif policy == "exl":
            print("setting excess limit on hourly matching")
            excess_constraints(n)


    result, message = n.lopf(pyomo=False,
           extra_functionality=extra_functionality,
           solver_name=solver_name,
           solver_options=solver_options)

    if result != "ok" or message != "optimal":
        print(f"solver ended with {results} and {message}, so quitting")
        sys.exit()

    return n




if __name__ == "__main__":
    logging.basicConfig(filename=snakemake.log.python,
                    level=snakemake.config['logging_level'])


    policy = snakemake.wildcards.policy
    print(f"solving network for policy: {policy}")

    name = snakemake.config['ci']['name']

    participation = snakemake.wildcards.participation
    print(f"solving with participation: {participation}")

    zone = snakemake.wildcards.zone
    print(f"solving network for bidding zone: {zone}")

    year = snakemake.wildcards.year
    print(f"solving network year: {year}")

    area = snakemake.config['area']
    print(f"solving with geoscope: {area}")

    node = geoscope(zone, area)['node']
    print(f"solving with node: {node}")

    ci_load = snakemake.config['ci_load'][f'{zone}']
    load = ci_load * float(participation)/100  #C&I baseload MW

    print(f"solving with load: {load}")


    with memory_logger(filename=getattr(snakemake.log, 'memory', None), interval=30.) as mem:

        n = solve(policy)
        n.export_to_netcdf(snakemake.output.network)

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
