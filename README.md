<!--
SPDX-FileCopyrightText: 2023 Iegor Riepin, Tom Brown

SPDX-License-Identifier: CC0-1.0
-->

[![Webpage](https://img.shields.io/badge/-Project%20Webpage-blue?style=flat-square&logo=github)](https://irieo.github.io/247cfe.github.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSES/MIT.txt)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg?style=flat-square)](LICENSES/CC-BY-4.0.txt)
[![Zenodo Study1](https://zenodo.org/badge/DOI/10.5281/zenodo.7180097.svg)](https://zenodo.org/record/7180097)
[![Zenodo Stidy2](https://zenodo.org/badge/DOI/10.5281/zenodo.8185849.svg)](https://zenodo.org/record/8185849)
[![Zenodo Paper1](https://zenodo.org/badge/DOI/10.5281/zenodo.12819516.svg)](https://zenodo.org/records/12819516)
[![Zenodo Paper2](https://zenodo.org/badge/DOI/10.5281/zenodo.10869650.svg)](https://zenodo.org/records/10869650)


# PyPSA code for exploring the 24/7 Carbon-Free Energy procurement

You are welcome to visit a [project webpage](https://irieo.github.io/247cfe.github.io/) for a project overview, publications, media coverage, and more.

## Getting started

Welcome! This project explores the mechanisms, costs, and system-level impacts of 24/7 Carbon-Free Energy (CFE) procurement.

The project comprises five distinct studies, each examining unique aspects of 24/7 CFE. The studies vary in their focus, model formulations, scenarios, and more. Ultimately, we aim to make the entire scientific workflow, from data to final charts, fully reproducible for each study. This repository includes code for three research items linked to GitHub releases. Two other two research papers are hosted in dedicated GitHub repositories with their reproducible workflows.

### 1. System-level impacts of 24/7 carbon-free electricity procurement in Europe
*A study published on Zenodo, October 2022*

- [Study](https://zenodo.org/record/7180097)
- [Code (GitHub release)](https://github.com/PyPSA/247-cfe/tree/v0.1)
- [Code (Github release, math module written in Linopy)](https://github.com/PyPSA/247-cfe/releases/tag/v0.2)

### 2. On the means, costs, and system-level impacts of 24/7 carbon-free energy procurement
*A research paper published in Energy Strategy Reviews, 2024*

- [DOI](https://www.sciencedirect.com/science/article/pii/S2211467X24001950)
- [Code (dedicated GitHub repository)](https://github.com/Irieo/247-procurement-paper)


### 3. The value of space-time load-shifting flexibility for 24/7 carbon-free electricity procurement
*Published on Zenodo, July 2023*

- [Complete study](https://zenodo.org/record/8185850)
- [Code (GitHub release)](https://github.com/PyPSA/247-cfe/tree/v0.3)


### 4. Spatio-temporal load shifting for truly clean computing
*A research paper published in Advances in Applied Energy, 2025*

- [DOI](https://doi.org/10.1016/j.adapen.2024.100202)
- [Code (dedicated GitHub repository)](https://github.com/Irieo/space-time-optimization)


### 5. 24/7 carbon-free electricity matching accelerates adoption of advanced clean energy technologies
*A commentary paper published in Joule, 2025*

- [DOI](https://doi.org/10.1016/j.joule.2024.101808)
- Code (`main` branch)

## How to reproduce results of a specific study?

### Studies #1 and #3

First, clone this repository:

```
git clone https://github.com/PyPSA/247-cfe --branch <tag_name> --single-branch
```
- `--single-branch` option allows for cloning only git history leading to tip of the tag. This saves a lot of unnecessary code from being cloned.

- `tag_name` is v0.2 for [Study 1](#study-1) or v0.3 for [Study 2](#study-2)

Second, install the necessary dependencies using `environment.yml` file. The following commands will do the job:

```
conda env create -f envs/environment.yml
conda activate 247-cfe
```
Third, to run all the scenarios from the study, run the [snakemake](https://snakemake.readthedocs.io/en/stable/) worflow:

```
snakemake --cores <n>
```

where `<n>` is the [number of cores](https://snakemake.readthedocs.io/en/stable/executing/cli.html) to use for the workflow.

- Note that this call requires a high-performance computing environment.

- It is also possible to run a smaller version of the model by adjusting the settings in `config.yaml`. For example, changing the config setting `area` from "EU" to "regions" reduces the regional coverage of the model, making the size of the problem feasible to solve on a private laptop with 8GB RAM.

Finally, when the workflow is complete, the results will be stored in `results` directory. The directory will contain solved networks, plots, summary csvs and logs.

4. At this point, you can also compile the LaTeX project to reproduce the study .pdf file.

### Studies #2 and #4

These research works are maintained in dedicated repositories, each containing an instruction on how to reproduce the results.

### Study #5

1. Clone the repository:

```
git clone git@github.com:PyPSA/247-cfe.git
```

2. Install the necessary dependencies using `environment.yaml` file. The following commands will do the job:

```
conda env create -f envs/environment.yaml
conda activate 247-env
```

3. The results of the paper can be reproduced by running the [snakemake](https://snakemake.readthedocs.io/en/stable/) workflow.  The following commands will run the workflows for the paper:

```
snakemake --cores <n> --configfile config_247cfe
snakemake --cores <n> --configfile config_BackgroundSystem.yaml
```

where `<n>` is the [number of cores](https://snakemake.readthedocs.io/en/stable/executing/cli.html) to use for the workflow.

NB It is possible to reproduce the results on a private laptop with 16GB RAM.

Model results will be stored in the `results` directory. For each workflow, the directory will contain:
- solved networks (.nc) for individual optimization problems
- summary (.yaml) for individual optimization problems
- summary (.csv) for aggregated results
- log files (memory, python, solver)
- detailed plots (.pdf) of the results

4. At this point, a curious reader can reproduce the dashboards from the paper with the jupyter notebooks in the `scripts/` directory. You can also compile the LaTeX project `/manuscript/manuscript.tex` to reproduce the paper .pdf file.

## Data

Code uses pre-processed European electricity system data generated through [PyPSA-Eur](https://github.com/PyPSA/pypsa-eur-sec) workflow using the `myopic` configuration. The data represents brownfield network scenarios. For convenience, sample networks for 2025 and 2030 are provided in the `input/` folder.

Technology data assumptions are automatically retrieved from [technology-data](https://github.com/PyPSA/technology-data) repository for `<year>` and `<version>`, as specified in `config.yaml`.


## Acknowledgments

This research was supported by a grant from Google LLC.

## License

This code is licensed under the open source [MIT License](LICENSES/MIT.txt).
Different open licenses apply to LaTeX files and input data, see [Specifications](.reuse/dep5).
