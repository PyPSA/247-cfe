configfile: "config.yaml"

wildcard_constraints:
    policy="[\-a-zA-Z0-9\.]+"


RDIR = os.path.join(config['results_dir'], config['run'])
CDIR = config['costs_dir']


rule merge_all_plots:
    input: 
        expand(RDIR + "/plots/{participation}/{year}/{zone}/{palette}/SUMMARY.pdf", **config['scenario'])


rule plot_summary_all_networks:
    input: 
        expand(RDIR + "/plots/{participation}/{year}/{zone}/{palette}/used.pdf", **config['scenario'])


rule make_summary_all_networks:
    input: 
        expand(RDIR + "/csvs/{participation}/{year}/{zone}/{palette}/summary.csv", **config['scenario'])


rule summarise_all_networks:
    input: 
        expand(RDIR + "/summaries/{participation}/{year}/{zone}/{palette}/{policy}.yaml", **config['scenario'])


rule solve_all_networks:
    input: 
        expand(RDIR + "/networks/{participation}/{year}/{zone}/{palette}/{policy}.nc", **config['scenario'])


rule merge_plots:
    input:
        used=RDIR + "/plots/{participation}/{year}/{zone}/{palette}/used.pdf",
        config=RDIR + '/configs/config.yaml'
    output:
        final=RDIR + "/plots/{participation}/{year}/{zone}/{palette}/SUMMARY.pdf"
    threads: 2
    resources: mem_mb=2000
    script:
        'scripts/merge_plots.py'


rule plot_summary:
    input:
        summary=RDIR + "/csvs/{participation}/{year}/{zone}/{palette}/summary.csv",
        config=RDIR + '/configs/config.yaml'
    output:
        used=RDIR + "/plots/{participation}/{year}/{zone}/{palette}/used.pdf"
    threads: 2
    resources: mem_mb=2000
    script:
        'scripts/plot_summary.py'


rule make_summary:
    input:
        expand(RDIR + "/summaries/{participation}/{year}/{zone}/{palette}/{policy}.yaml",
               **config['scenario'])
    output:
        summary=RDIR + "/csvs/{participation}/{year}/{zone}/{palette}/summary.csv"
    threads: 2
    resources: mem_mb=2000
    script: 'scripts/make_summary.py'


if config['solve_network'] == 'solve':
    rule solve_network:
        input:
            network2030 = config['n_2030'],
            network2025 = config['n_2025'],
            costs2030=CDIR + "/costs_2030.csv",
            costs2025=CDIR + "/costs_2025.csv"
        output:
            network=RDIR + "/networks/{participation}/{year}/{zone}/{palette}/{policy}.nc",
            grid_cfe=RDIR + "/networks/{participation}/{year}/{zone}/{palette}/{policy}.csv"
        log:
            solver=RDIR + "/logs/{participation}/{year}/{zone}/{palette}/{policy}_solver.log",
            python=RDIR + "/logs/{participation}/{year}/{zone}/{palette}/{policy}_python.log",
            memory=RDIR + "/logs/{participation}/{year}/{zone}/{palette}/{policy}_memory.log"
        threads: 12
        resources: mem=8000
        script: "scripts/solve_network.py"

rule summarise_network:
    input:
        network=RDIR + "/networks/{participation}/{year}/{zone}/{palette}/{policy}.nc",
	    grid_cfe=RDIR + "/networks/{participation}/{year}/{zone}/{palette}/{policy}.csv"
    output:
        yaml=RDIR + "/summaries/{participation}/{year}/{zone}/{palette}/{policy}.yaml"
    threads: 2
    resources: mem_mb=2000
    script: 'scripts/summarise_network.py'


rule copy_config:
    output: RDIR + '/configs/config.yaml'
    threads: 1
    resources: mem_mb=1000
    script: "scripts/copy_config.py"


# additional rules for cluster communication -> not included into a workflow 
rule sync_solution:
    params:
        cluster="iegor.riepin@gateway.hpc.tu-berlin.de:/scratch/iegor.riepin/247-cfe/results/report"
    shell: 
        """
        rsync -uvarh --no-g {params.cluster} results/
        """

rule sync_plots:
    params:
        cluster="iegor.riepin@gateway.hpc.tu-berlin.de:/scratch/iegor.riepin/247-cfe/results/report/plots/"
    shell: 
        """
        rsync -uvarh --no-g {params.cluster} report/plots
        """


# illustrate workflow
rule dag:
     message: "Plot dependency graph of the workflow."
     output:
         dot="workflow/dag.dot",
         graph="workflow/graph.dot",
         pdf="workflow/graph.pdf"
     shell:
         """
         snakemake --rulegraph > {output.dot}
         sed -e '1,2d' < {output.dot} > {output.graph}
         dot -Tpdf -o {output.pdf} {output.graph}
         """