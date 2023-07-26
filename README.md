
# PyPSA code for exploring the 24/7 Carbon-Free Energy procurement in Europe

## Getting started

This project explores the means, costs and impacts of 24/7 Carbon-Free Energy procurement in Europe.

Three invididual studies are planned as deliverables of this research project. The three studies will be linked to GitHub releases and individual Zenodo repositories. All studies are focused on the topic of 24/7 Carbon-Free Energy procurement in Europe; however, studies will differ in terms of their focuses, model formulations, scenarios, etc.

We aim to make the workflow & results for each study to be fully reproducible. See [Requirements](#requirements) and [Software](#software).

This research is supported by a grant from Google Inc.

### Study 1: 

In this study, we investigate both the means and costs of pursuing different clean electricity procurement strategies for companies in a selection of European countries. We also explore how the 24/7 clean energy procurement affects the rest of the European electricity system.

- [Complete study](https://zenodo.org/record/7180097)
- [Twitter thread](https://twitter.com/nworbmot/status/1579810935702982656)
- [Blog post](https://blog.google/around-the-globe/google-europe/how-carbon-free-energy-around-the-clock-can-work/)
- [GitHub release](https://github.com/PyPSA/247-cfe/tree/v0.1)
- [Code is also at Zenodo](https://zenodo.org/record/7181236) 

NB The Study 1 is now also available with [Linopy](https://github.com/pypsa/linopy) integration under the tag [v0.2](https://github.com/PyPSA/247-cfe/releases/tag/v0.2). Linopy is an open-source python package for linear or mixed-integer optimization. 

### Study 2: 

In this study, we explore how and why space-time load-shifting flexibility can be used to meet high 24/7 carbon-free energy targets, as well as what potential benefits it may offer to 24/7 participants and to the rest of the energy system. To answer these questions, we expand the mathematical model of 24/7 CFE procurement developed in the previous work by incorporating spatial and temporal demand flexibility provided by electricity consumers that follow 24/7 carbon-free energy goals.

- [Complete study](https://zenodo.org/record/8185850)
- [GitHub release](https://github.com/PyPSA/247-cfe/tree/v0.3)

## Background

Traditional Power Purchase Agreements (PPAs) for renewable
energy have seen rapid growth in recent years, but they only match
supply and demand on average over a longer period such as a
year. There is increasing interest from corporations such as Google to
match their demand with clean energy supply on a truly 24/7 basis,
whether that is using variable renewables paired with storage, or
using dispatchable clean sources such as geothermal power. In 2020
Google committed to operating entirely on 24/7 carbon-free energy
(CFE) at all of its data centres and campuses worldwide by 2030.  In
this project we will explore different designs for a 24/7 carbon-free
PPA, and how their deployment affects the rest of the energy system.

Further information:

- [Google's 24/7 CFE concept](https://www.gstatic.com/gumdrop/sustainability/247-carbon-free-energy.pdf)
- [Google's 24/7 CFE metrics and methodology](https://www.gstatic.com/gumdrop/sustainability/24x7-carbon-free-energy-methodologies-metrics.pdf)
- [UN's 24/7 Carbon-free Energy Compact](https://www.un.org/en/energy-compacts/page/compact-247-carbon-free-energy)


## Requirements
### Data

Code is based on the brownfield networks exported from [PyPSA-Eur-Sec](https://github.com/PyPSA/pypsa-eur-sec) with `myopic` setting to get brownfield networks for 2025/2030. For convenience, the sample networks are pushed to git and located in the `input` folder.

Parallel to the repository you should also clone the [technology-data](https://github.com/PyPSA/technology-data) repository.


### Software

The code is known to work with python 3.11, PyPSA 0.23, pandas 1.5.3, numpy 1.24.2, linopy 0.1.5, and gurobi 10.0.1.

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

Copyright 2022 Tom Brown, Iegor Riepin. This code is licensed under the open source MIT License.
