
from shutil import copy

import os

files = [
    "config.yaml",
    "Snakefile",
    "scripts/solve_network.py",
    "scripts/resolve_network.py",
    "scripts/summarise_offtake.py"
]

if __name__ == '__main__':
    if 'snakemake' not in globals():
        from helper import mock_snakemake
        snakemake = mock_snakemake('copy_config')

    for f in files:
        copy(f,os.path.join(snakemake.config['results_dir'], snakemake.config['run'], 'configs'))
