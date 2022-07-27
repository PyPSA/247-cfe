configfile: "config.yaml"

wildcard_constraints:
    policy="[\-a-zA-Z0-9\.]+"


RDIR = os.path.join(config['results_dir'], config['run'])
CDIR = config['costs_dir']


rule merge_all_plots:
    input: 
        expand(RDIR + "/plots/{palette}/SUMMARY.pdf", **config['scenario'])


rule plot_summary_all_networks:
    input: 
        expand(RDIR + "/plots/{palette}/used.pdf", **config['scenario'])


rule make_summary_all_networks:
    input: 
        expand(RDIR + "/csvs/{palette}/summary.csv", **config['scenario'])


rule summarise_all_networks:
    input: 
        expand(RDIR + "/summaries/{palette}/{policy}.yaml", **config['scenario'])


rule solve_all_networks:
    input: 
        expand(RDIR + "/networks/{palette}/{policy}.nc", **config['scenario'])


rule merge_plots:
    input:
        used=RDIR + "/plots/{palette}/used.pdf",
        config=RDIR + '/configs/config.yaml'
    output:
        final=RDIR + "/plots/{palette}/SUMMARY.pdf"
    threads: 2
    resources: mem_mb=2000
    script:
        'scripts/merge_plots.py'


rule plot_summary:
    input:
        summary=RDIR + "/csvs/{palette}/summary.csv",
        config=RDIR + '/configs/config.yaml'
    output:
        used=RDIR + "/plots/{palette}/used.pdf"
    threads: 2
    resources: mem_mb=2000
    script:
        'scripts/plot_summary.py'


rule make_summary:
    input:
        expand(RDIR + "/summaries/{palette}/{policy}.yaml",
               **config['scenario'])
    output:
        summary=RDIR + "/csvs/{palette}/summary.csv"
    threads: 2
    resources: mem_mb=2000
    script: 'scripts/make_summary.py'


if config['solve_network'] == 'solve':
    rule solve_network:
        input:
            network=config['network_file'],
            costs=CDIR + "/costs_{}.csv".format(config['costs']['projection_year'])
        output:
            network=RDIR + "/networks/{palette}/{policy}.nc",
            grid_cfe=RDIR + "/networks/{palette}/{policy}.csv"
        log:
            solver=RDIR + "/logs/{palette}/{policy}_solver.log",
            python=RDIR + "/logs/{palette}/{policy}_python.log",
            memory=RDIR + "/logs/{palette}/{policy}_memory.log"
        threads: 12
        resources: mem=16000
        script: "scripts/solve_network.py"

rule summarise_network:
    input:
        network=RDIR + "/networks/{palette}/{policy}.nc",
	    grid_cfe=RDIR + "/networks/{palette}/{policy}.csv"
    output:
        yaml=RDIR + "/summaries/{palette}/{policy}.yaml"
    threads: 2
    resources: mem_mb=2000
    script: 'scripts/summarise_network.py'


rule copy_config:
    output: RDIR + '/configs/config.yaml'
    threads: 1
    resources: mem_mb=1000
    script: "scripts/copy_config.py"