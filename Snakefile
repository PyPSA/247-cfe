configfile: "config.yaml"

wildcard_constraints:
    policy="[\-a-zA-Z0-9\.]+"

rule plot_summary:
    input:
        summary=config['results_dir'] + "/" + config['run'] + "/csvs/summary.csv",
        config=config['results_dir'] + "/" + config['run'] + '/configs/config.yaml'
    output:
        used=config['results_dir'] + "/" + config['run'] + "/plots/used.pdf"
    threads: 2
    resources: mem_mb=2000
    script:
        'scripts/plot_summary.py'


rule make_summary:
    input:
        expand(config['results_dir'] + "/" + config['run'] + "/summaries/{policy}.yaml",
               **config['scenario'])
    output:
        summary=config['results_dir'] + "/" + config['run'] + "/csvs/summary.csv"
    threads: 2
    resources: mem_mb=2000
    script: 'scripts/make_summary.py'


rule solve_network:
    output:
        network=config['results_dir'] + "/" + config['run'] + "/networks/{policy}.nc",
	grid_cfe=config['results_dir'] + "/" + config['run'] + "/networks/{policy}.csv"
    log:
        solver=config['results_dir'] + "/" + config['run'] + "/logs/{policy}_solver.log",
        python=config['results_dir'] + "/" + config['run'] + "/logs/{policy}_python.log",
        memory=config['results_dir'] + "/" + config['run'] + "/logs/{policy}_memory.log"
    threads: 4
    resources: mem=6000
    script: "scripts/solve_network.py"

rule summarise_network:
    input:
        network=config['results_dir'] + "/" + config['run'] + "/networks/{policy}.nc",
	grid_cfe=config['results_dir'] + "/" + config['run'] + "/networks/{policy}.csv"
    output:
        yaml=config['results_dir'] + "/" + config['run'] + "/summaries/{policy}.yaml"
    threads: 2
    resources: mem_mb=2000
    script: 'scripts/summarise_network.py'

rule copy_config:
    output: config['results_dir'] + "/" + config['run'] + '/configs/config.yaml'
    threads: 1
    resources: mem_mb=1000
    script: "scripts/copy_config.py"
