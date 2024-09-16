[![DOI](https://zenodo.org/badge/857427966.svg)](https://zenodo.org/doi/10.5281/zenodo.13768181)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

# NTL-TCN
The paper associated with this repository is currently submitted to __JGR: Machine Learning and Computation__. A preprint is availiable in __ESS OPEN ARCHIVE__ as:

_Kaan Cökerim, Henryk Dobslaw, Kyriakos Balidakis, et al. Modeling Non-Tidal Surface Fluid Loading Signatures in Global GNSS Displacements with a Deep Learning Framework. ESS Open Archive . September 15, 2024.
DOI: 10.22541/essoar.172641527.77043060/v1_

# Abstract
"TEXT"

# What this repository contains
This repository contains a minimal example with data to reproduce the results of the paper. In particular it contains: 
- `hdf5` files containing targets, predictions and NTL time series of the stations
- `.pth`-Files containing the trained model
- codes to evaluate the model with the NTL input data and to produce time series figures

# Availibility of Associated Data
The original GNSS displacement time series are obtained from the Nevada Geodetic Laboratory (Blewitt et al., 2018) and are openly available in the NGL data repository under http://geodesy.unr.edu/index.php (Last accessed for download: 01 June 2024). Additional GNSS time series were composed based on the combined IGS repro3 solution (Rebischung et al., 2024) retrieved from `ftp://igs-rf.ign.fr/pub/repro3/`. The geophysical non-tidal surface loading products by Dill and Dobslaw (2013) are available at the ESMGFZ data repository http://esmdata.gfz-potsdam.de:8080/repository/entry/show?entryid=97d15ffe-3b5d-49dc-ba6c-3011851af1de (Last accessed for download: 01 June 2024). The GrAtSiD trajectory modeling software by Bedford and Bevis (2018) is available on the GitHub repository of the Tectonic Geodesy group at Ruhr-University Bochum on https://github.com/TectonicGeodesy-RUB/Gratsid.
