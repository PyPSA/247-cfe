configfile: "config.yaml"

wildcard_constraints:
    policy="[\-a-zA-Z0-9\.]+"


RDIR = os.path.join(config['results_dir'], config['run'])
CDIR = config['costs_dir']


rule merge_plots:
    input:
        used=RDIR + "/plots/used.pdf",
        config=RDIR + '/configs/config.yaml'
    output:
        final=RDIR + "/plots/SUMMARY.pdf"
    threads: 2
    resources: mem_mb=2000
    script:
        'scripts/merge_plots.py'


rule plot_summary:
    input:
        summary=RDIR + "/csvs/summary.csv",
        config=RDIR + '/configs/config.yaml'
    output:
        used=RDIR + "/plots/used.pdf"
    threads: 2
    resources: mem_mb=2000
    script:
        'scripts/plot_summary.py'


rule make_summary:
    input:
        expand(RDIR + "/summaries/{policy}.yaml",
               **config['scenario'])
    output:
        summary=RDIR + "/csvs/summary.csv"
    threads: 2
    resources: mem_mb=2000
    script: 'scripts/make_summary.py'


rule solve_network:
    input:
        network=config['network_file'],
        costs=CDIR + "/costs_{}.csv".format(config['costs']['projection_year'])
    output:
        network=RDIR + "/networks/{policy}.nc",
	grid_cfe=RDIR + "/networks/{policy}.csv"
    log:
        solver=RDIR + "/logs/{policy}_solver.log",
        python=RDIR + "/logs/{policy}_python.log",
        memory=RDIR + "/logs/{policy}_memory.log"
    threads: 4
    resources: mem=6000
    script: "scripts/solve_network.py"

rule summarise_network:
    input:
        network=RDIR + "/networks/{policy}.nc",
	grid_cfe=RDIR + "/networks/{policy}.csv"
    output:
        yaml=RDIR + "/summaries/{policy}.yaml"
    threads: 2
    resources: mem_mb=2000
    script: 'scripts/summarise_network.py'

rule copy_config:
    output: RDIR + '/configs/config.yaml'
    threads: 1
    resources: mem_mb=1000
    script: "scripts/copy_config.py"
