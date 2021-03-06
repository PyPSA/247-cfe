
# PyPSA code to explore the impacts of 24/7 Carbon-Free Energy PPAs

Code explores impacts based on a brownfield network exported from [PyPSA-Eur-Sec](https://github.com/PyPSA/pypsa-eur-sec).

The methodology is the same as the [Princeton 24/7 Study](https://acee.princeton.edu/24-7/).

Google sponsored both this project and the Princeton study.

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

The folder `data` should contain PyPSA networks exported from [PyPSA-Eur-Sec](https://github.com/PyPSA/pypsa-eur-sec) built with `myopic` setting to get brownfield networks for 2020/2030. To get started, you can download a [sample network](https://nworbmot.org/energy/elec_s_45_lv1.0__3H-T-H-B-I-solar+p3-offwind+p0.67-dist1-linemaxext10_2020.nc).

Parallel to the repository you should also clone the [technology-data](https://github.com/PyPSA/technology-data) repository.

### Software

The code is known to work with PyPSA 0.18.1, pandas 1.2.4, numpy 1.19.0, vresutils 0.3.1 and gurobi 9.1.2.

## License


Copyright 2022 Tom Brown.

This code is licensed under the open source MIT License.
