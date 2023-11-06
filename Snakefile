from shutil import copyfile, move
from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider

HTTP = HTTPRemoteProvider()


configfile: "config.yaml"


wildcard_constraints:
    policy="[\-a-zA-Z0-9\.]+",


RDIR = os.path.join(config["results_dir"], config["run"])
RUN = config["run"]

# Technology data inputs
version = config["technology_data"]["version"]
year = config["technology_data"]["year"]
url = f"https://raw.githubusercontent.com/PyPSA/technology-data/{version}/outputs/costs_{year}.csv"


rule merge_all_plots:
    input:
        expand(
            RDIR + "/plots/{year}/{zone}/{palette}/{policy}/SUMMARY.pdf",
            **config["scenario"]
        ),


rule plot_summary_all_networks:
    input:
        expand(
            RDIR + "/plots/{year}/{zone}/{palette}/{policy}/capacity.pdf",
            **config["scenario"]
        ),


rule make_summary_all_networks:
    input:
        expand(
            RDIR + "/csvs/{year}/{zone}/{palette}/{policy}/summary.csv",
            **config["scenario"]
        ),


rule summarise_all_networks:
    input:
        expand(
            RDIR + "/summaries/{year}/{zone}/{palette}/{policy}/{participation}.yaml",
            **config["scenario"]
        ),


rule solve_all_networks:
    input:
        expand(
            RDIR + "/networks/{year}/{zone}/{palette}/{policy}/{participation}.nc",
            **config["scenario"]
        ),


rule merge_plots:
    input:
        plot=RDIR + "/plots/{year}/{zone}/{palette}/{policy}/capacity.pdf",
        config=RDIR + "/configs/config.yaml",
    output:
        final=RDIR + "/plots/{year}/{zone}/{palette}/{policy}/SUMMARY.pdf",
    threads: 2
    resources:
        mem_mb=2000,
    script:
        "scripts/merge_plots.py"


rule plot_summary:
    input:
        grid_cfe=RDIR + "/networks/{year}/{zone}/{palette}/{policy}/0.csv",
        networks=RDIR + "/networks/{year}/{zone}/{palette}/{policy}/0.nc",
        summary=RDIR + "/csvs/{year}/{zone}/{palette}/{policy}/summary.csv",
        config=RDIR + "/configs/config.yaml",
    output:
        plot=RDIR + "/plots/{year}/{zone}/{palette}/{policy}/capacity.pdf",
    threads: 2
    resources:
        mem_mb=2000,
    script:
        "scripts/plot_summary.py"


rule make_summary:
    input:
        expand(
            RDIR + "/summaries/{year}/{zone}/{palette}/{policy}/{participation}.yaml",
            **config["scenario"]
        ),
    output:
        summary=RDIR + "/csvs/{year}/{zone}/{palette}/{policy}/summary.csv",
    threads: 2
    resources:
        mem_mb=2000,
    script:
        "scripts/make_summary.py"


if config["solve_network"] == "solve":

    rule solve_network:
        input:
            network=config[f'n_{year}_{config["time_sampling"]}'],
            costs="input/costs_{year}.csv",
        output:
            network=RDIR
            + "/networks/{year}/{zone}/{palette}/{policy}/{participation}.nc",
            grid_cfe=RDIR
            + "/networks/{year}/{zone}/{palette}/{policy}/{participation}.csv",
        log:
            solver=RDIR
            + "/logs/{year}/{zone}/{palette}/{policy}/{participation}_solver.log",
            python=RDIR
            + "/logs/{year}/{zone}/{palette}/{policy}/{participation}_python.log",
            memory=RDIR
            + "/logs/{year}/{zone}/{palette}/{policy}/{participation}_memory.log",
        threads: 12
        resources:
            mem=8000,
        script:
            "scripts/solve_network.py"


rule summarise_network:
    input:
        network=RDIR + "/networks/{year}/{zone}/{palette}/{policy}/{participation}.nc",
        grid_cfe=RDIR + "/networks/{year}/{zone}/{palette}/{policy}/{participation}.csv",
    output:
        yaml=RDIR + "/summaries/{year}/{zone}/{palette}/{policy}/{participation}.yaml",
    threads: 2
    resources:
        mem_mb=2000,
    script:
        "scripts/summarise_network.py"


rule copy_config:
    output:
        RDIR + "/configs/config.yaml",
    threads: 1
    resources:
        mem_mb=1000,
    script:
        "scripts/copy_config.py"


if config.get("retrieve_cost_data", True):

    rule retrieve_cost_data:
        input:
            HTTP.remote(url, keep_local=True),
        output:
            f"input/costs_{year}.csv",
        # log: f"logs/{RDIR}retrieve_cost_data_{year}.log"
        resources:
            mem_mb=1000,
        run:
            move(input[0], output[0])


# additional rules for cluster communication -> not included into a workflow
rule sync_solution:
    params:
        cluster=f"iegor.riepin@gateway.hpc.tu-berlin.de:/scratch/iegor.riepin/247-cfe/results/{RUN}",
    shell:
        """
        rsync -uvarh --no-g {params.cluster} results/
        """


rule sync_plots:
    params:
        cluster="iegor.riepin@gateway.hpc.tu-berlin.de:/scratch/iegor.riepin/247-cfe/results/report/plots/",
    shell:
        """
        rsync -uvarh --no-g {params.cluster} report/plots
        """


# illustrate workflow
rule dag:
    message:
        "Plot dependency graph of the workflow."
    output:
        dot="workflow/dag.dot",
        graph="workflow/graph.dot",
        pdf="workflow/graph.pdf",
    shell:
        """
         snakemake --rulegraph > {output.dot}
         sed -e '1,2d' < {output.dot} > {output.graph}
         dot -Tpdf -o {output.pdf} {output.graph}
         """
