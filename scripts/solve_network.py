
import pypsa, numpy as np, pandas as pd
from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints

import logging
logger = logging.getLogger(__name__)

# Suppress logging of the slack bus choices
pypsa.pf.logger.setLevel(logging.WARNING)

from vresutils.benchmark import memory_logger

# from https://github.com/PyPSA/pypsa-eur-sec/blob/93eb86eec87d34832ebc061697e289eabb38c105/scripts/solve_network.py
override_component_attrs = pypsa.descriptors.Dict({k : v.copy() for k,v in pypsa.components.component_attrs.items()})
override_component_attrs["Link"].loc["bus2"] = ["string",np.nan,np.nan,"2nd bus","Input (optional)"]
override_component_attrs["Link"].loc["bus3"] = ["string",np.nan,np.nan,"3rd bus","Input (optional)"]
override_component_attrs["Link"].loc["bus4"] = ["string",np.nan,np.nan,"4th bus","Input (optional)"]
override_component_attrs["Link"].loc["efficiency2"] = ["static or series","per unit",1.,"2nd bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["efficiency3"] = ["static or series","per unit",1.,"3rd bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["efficiency4"] = ["static or series","per unit",1.,"4th bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["p2"] = ["series","MW",0.,"2nd bus output","Output"]
override_component_attrs["Link"].loc["p3"] = ["series","MW",0.,"3rd bus output","Output"]
override_component_attrs["Link"].loc["p4"] = ["series","MW",0.,"4th bus output","Output"]


def strip_network(n):
    nodes_to_keep = snakemake.config['basenodes_to_keep'][:]
    for b in snakemake.config['basenodes_to_keep']:
        for s in snakemake.config['node_suffixes_to_keep']:
            nodes_to_keep.append(b + " " + s)

    nodes_to_keep.extend(snakemake.config['additional_nodes'])

    print("keeping nodes:",nodes_to_keep)

    n.buses.drop(n.buses.index.symmetric_difference(nodes_to_keep),
                 inplace=True)

    #make sure lines are kept
    n.lines.carrier = "AC"

    carrier_to_keep = snakemake.config['carrier_to_keep']

    for c in n.iterate_components(["Generator","Link","Line","Store","StorageUnit","Load"]):
        if c.name in ["Link","Line"]:
            location_boolean = c.df.bus0.isin(nodes_to_keep) & c.df.bus1.isin(nodes_to_keep)
        else:
            location_boolean = c.df.bus.isin(nodes_to_keep)
        to_keep = c.df.index[location_boolean & c.df.carrier.isin(carrier_to_keep)]
        to_drop = c.df.index.symmetric_difference(to_keep)
        c.df.drop(to_drop,
                  inplace=True)

def add_ci(n):
    """Add C&I at its own node"""

    #first deal with global policy environment
    gl_policy = snakemake.config['global']
    if gl_policy['policy_type'] == "co2 cap":
        print(type(gl_policy['co2_baseline']),gl_policy['co2_baseline'])
        co2_cap = gl_policy['co2_share']*gl_policy['co2_baseline']
        print("Setting global CO2 cap to ",co2_cap)
        n.global_constraints.at["CO2Limit","constant"] = co2_cap

    #local C&I properties
    ci = snakemake.config['ci']
    name = ci['name']
    node = ci['node']
    load = ci['load']

    n.add("Bus",
          name)

    n.add("Link",
          name + " export",
          bus0=name,
          bus1=node,
          efficiency=0.99,
          p_nom=1e6)

    n.add("Link",
          name + " import",
          bus0=node,
          bus1=name,
          efficiency=0.99,
          p_nom=1e6)

    #baseload clean energy generator
    n.add("Generator",
          name + " green hydrogen OCGT",
          carrier="green hydrogen OCGT",
          bus=name,
          p_nom_extendable=True,
          capital_cost=40000, #based on OGT
          marginal_cost=3/0.033/0.4) #hydrogen at 3 EUR/kg with 0.4 efficiency

    #RES generator
    for carrier in ["onwind","solar"]:
        gen_template = node + " " + carrier
        n.add("Generator",
              f"{name} {carrier}",
              carrier=carrier,
              bus=name,
              p_nom_extendable=True,
              p_max_pu=n.generators_t.p_max_pu[gen_template],
              capital_cost=n.generators.at[gen_template,"capital_cost"],
              marginal_cost=n.generators.at[gen_template,"marginal_cost"])

    n.add("Load",
          name + " load",
          carrier=name,
          bus=name,
          p_set=pd.Series(load,index=n.snapshots))

def solve_network(n, policy, penetration):

    name = snakemake.config['ci']['name']

    if policy == "res":
        res_gens = [name + " " + g for g in snakemake.config['ci']['res_techs']]
    elif policy == "cfe":
        # to replace with time-dependent grid CFE share
        grid_cfe = pd.Series(0,index=n.snapshots)
        clean_gens = [name + " " + g for g in snakemake.config['ci']['clean_techs']]


    def cfe_constraints(n):

        weightings = pd.DataFrame(np.outer(n.snapshot_weightings["generators"],[1.]*len(clean_gens)),
                                  index = n.snapshots,
                                  columns = clean_gens)
        gen_sum = join_exprs(linexpr((weightings,get_var(n, "Generator", "p")[clean_gens]))) # single line sum

        gexport = get_var(n, "Link", "p")[name + " export"] # a series
        gimport = get_var(n, "Link", "p")[name + " import"] # a series
        grid_sum = join_exprs(linexpr((-n.snapshot_weightings["generators"],gexport),
                                      (grid_cfe*n.snapshot_weightings["generators"],gimport))) # single line sum

        lhs = gen_sum + '\n' + grid_sum
        total_load = (n.loads_t.p_set[name + " load"]*n.snapshot_weightings["generators"]).sum() # number
        con = define_constraints(n, lhs, '>=', penetration*total_load, 'CFEconstraints','CFEtarget')

    def res_constraints(n):
        weightings = pd.DataFrame(np.outer(n.snapshot_weightings["generators"],[1.]*len(res_gens)),
                                  index = n.snapshots,
                                  columns = res_gens)
        lhs = join_exprs(linexpr((weightings,get_var(n, "Generator", "p")[res_gens]))) # single line sum
        total_load = (n.loads_t.p_set[name + " load"]*n.snapshot_weightings["generators"]).sum() # number
        con = define_constraints(n, lhs, '>=', penetration*total_load, 'RESconstraints','REStarget')




    def extra_functionality(n, snapshots):

        if policy == "cfe":
            print("setting CFE target of",penetration)
            cfe_constraints(n)
        elif policy == "res":
            print("setting annual RES target of",penetration)
            res_constraints(n)
        else:
            print("no target set")
            sys.exit()

    n.consistency_check()

    formulation = snakemake.config['solving']['options']['formulation']
    solver_options = snakemake.config['solving']['solver']


    n.lopf(pyomo=False,
           extra_functionality=extra_functionality,
           formulation=formulation,
           solver_name=solver_options.pop('name'),
           solver_options=solver_options,
           solver_logfile=snakemake.log.solver)



if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if 'snakemake' not in globals():
        from vresutils.snakemake import MockSnakemake, Dict
        snakemake = MockSnakemake(
            path='',
            wildcards=dict(policy='co2120-trans-storage-wind1040-sola510-nuclNone-lCCSNone',parameter="0"),
            output=dict(network="results/test/0after.nc"),
            log=dict(solver="results/test/log_0after.log")
        )
        import yaml
        with open('config.yaml') as f:
            snakemake.config = yaml.load(f)



    n = pypsa.Network(snakemake.config['network_file'],
                      override_component_attrs=override_component_attrs)

    policy = snakemake.wildcards.policy[:3]
    penetration = float(snakemake.wildcards.policy[3:])/100

    print(f"solving network for policy {policy} and penetration {penetration}")

    with memory_logger(filename=getattr(snakemake.log, 'memory', None), interval=30.) as mem:

        strip_network(n)

        add_ci(n)

        solve_network(n, policy, penetration)

        n.export_to_netcdf(snakemake.output.network)

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
