<!--
SPDX-FileCopyrightText: 2023 Iegor Riepin, Tom Brown

SPDX-License-Identifier: CC0-1.0
-->

[![Webpage](https://img.shields.io/badge/-Project%20Webpage-blue?style=flat-square&logo=github)](https://irieo.github.io/247cfe.github.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSES/MIT.txt)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg?style=flat-square)](LICENSES/CC-BY-4.0.txt)
[![Zenodo Study1](https://zenodo.org/badge/DOI/10.5281/zenodo.7180097.svg)](https://zenodo.org/record/7180097)
[![Zenodo Stidy2](https://zenodo.org/badge/DOI/10.5281/zenodo.8185849.svg)](https://zenodo.org/record/8185849)


# PyPSA code for exploring the 24/7 Carbon-Free Energy procurement in Europe

## Project webpage

You are welcome to visit a [project webpage](https://irieo.github.io/247cfe.github.io/) for detailed information about our research.

## Getting started

This project explores the means, costs and impacts of 24/7 Carbon-Free Energy procurement in Europe.

There are five individual studies planned as deliverables for this research project. Ultimately, we want the entire scientific workflow, from raw data to final charts, to be fully reproducible for each study. This repository will include code for three studies linked to GitHub releases. Additional research papers are hosted in dedicated GitHub repositories with own reproducible workflows. The five studies are all concerned with 24/7 Carbon-Free Energy procurement in Europe, but they differ in their focuses, model formulations, scenarios, etc.

This research is supported by a grant from Google LLC.

### Study 1: System-level impacts of 24/7 carbon-free electricity procurement in Europe (Zenodo, October 2022)

In this study, we investigate both the means and costs of pursuing different clean electricity procurement strategies for companies in a selection of European countries. We also explore how the 24/7 clean energy procurement affects the rest of the European electricity system.

- [Complete study](https://zenodo.org/record/7180097)
- [Twitter thread](https://twitter.com/nworbmot/status/1579810935702982656)
- [Blog post](https://blog.google/around-the-globe/google-europe/how-carbon-free-energy-around-the-clock-can-work/)
- [GitHub release](https://github.com/PyPSA/247-cfe/tree/v0.1)

Study 1 is now also available with [Linopy](https://github.com/pypsa/linopy) integration under the tag [v0.2](https://github.com/PyPSA/247-cfe/releases/tag/v0.2). Linopy is an open-source python package for linear or mixed-integer optimization.

### Study 2: The value of space-time load-shifting flexibility for 24/7 carbon-free electricity procurement (Zenodo, July 2023)

In this study, we explore how and why space-time load-shifting flexibility can be used to meet high 24/7 carbon-free energy targets, as well as what potential benefits it may offer to 24/7 participants and to the rest of the energy system. To answer these questions, we expand the mathematical model of 24/7 CFE procurement developed in the previous work by incorporating spatial and temporal demand flexibility provided by electricity consumers that follow 24/7 carbon-free energy goals.

- [Complete study](https://zenodo.org/record/8185850)
- [GitHub release](https://github.com/PyPSA/247-cfe/tree/v0.3)

### Study 3: (*work in progress*) On the role of 24/7 carbon-free energy matching in accelerating advanced clean energy technologies

In the third study, we argue that the commitment by a small number of companies to round-the-clock matching can create an early market and spur substantial learning of the advanced electricity technologies. We demonstrate these effects for two technologies: long-duration energy storage and clean firm generation. Cost reductions make 24/7 matching more attractive for other actors, leading to a virtuous circle that accelerates the time when the technologies become cost-competitive in the rest of the electricity market. These indirect effects unlock greenhouse gas savings far beyond the direct emission reduction of initial investments.


- [preliminary results](https://iriepin.com/uploads/247Hubtalk_20240521.pdf)

## Research papers with workflows in own GitHub repositories

### On the means, costs, and system-level impacts of 24/7 carbon-free energy procurement (Energy Strategy Reviews, 2024)

- [DOI](https://www.sciencedirect.com/science/article/pii/S2211467X24001950)
- preprint at [arXiv](https://arxiv.org/abs/2403.07876)
- [Code](https://github.com/Irieo/247-procurement-paper)

### Spatio-temporal load shifting for truly clean computing (in review)

- preprint at [arXiv](https://arxiv.org/abs/2405.00036)
- [Code](https://github.com/Irieo/space-time-optimization)


## How to reproduce results of a study?

First, clone the repository:

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
snakemake -call
```

- Note that this call requires a high-performance computing environment.

- It is possible to run a smaller version of the model by adjusting the settings in `config.yaml`. For example, changing the config setting `area` from "EU" to "regions" reduces the regional coverage of the model, making the size of the problem feasible to solve on a private laptop with 8GB RAM.

Finally, when the workflow is complete, the results will be stored in `results` directory. The directory will contain solved networks, plots, summary csvs and logs.

4. At this point, you can also compile the LaTeX project to reproduce the study .pdf file.


## Requirements
### Data

Code is based on the brownfield networks exported from [PyPSA-Eur](https://github.com/PyPSA/pypsa-eur-sec) with `myopic` setting to get brownfield networks for 2025/2030. For convenience, sample networks are provided in the `input/` folder.

Technology data assumptions are automatically retrieved from [technology-data](https://github.com/PyPSA/technology-data) repository for `<year>` and `<version>`, as specified in `config.yaml`.


### Software

The code is known to work with python 3.11, PyPSA 0.25, pandas 1.5.3, numpy 1.24.2, linopy 0.2.7, and gurobi 10.0.1.

The complete list of package requirements is in the [envs/environment.yaml](envs/environment.yaml) file. The environment can be installed and activated using:

```
.../247-cfe % conda env create -f envs/environment.yaml
.../247-cfe % conda activate 247-env
```

If you have troubles with a slow [conda](https://docs.conda.io/en/latest/) installation, we recommend to install [mamba](https://mamba.readthedocs.io/en/latest/):

```
conda install -c conda-forge mamba
```

and then install the environment with a fast drop-in replacement via

```
mamba env create -f envs/environment.yaml
```

## License

This code is licensed under the open source [MIT License](LICENSES/MIT.txt).
Different open licenses apply to LaTeX files and input data, see [Specifications](.reuse/dep5).
