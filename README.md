[![DOI](https://zenodo.org/badge/857427966.svg)](https://zenodo.org/doi/10.5281/zenodo.13768181)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

# NTL-TCN
The paper associated with this repository is currently submitted to __JGR: Machine Learning and Computation__. A preprint is availiable in __ESS OPEN ARCHIVE__ as:

Kaan Cökerim, Henryk Dobslaw, Kyriakos Balidakis, Laura Jensen, Carlos Pena and Jonathan Bedford. _Modeling Non-Tidal Surface Fluid Loading Signatures in Global GNSS Displacements with a Deep Learning Framework_. ESS Open Archive . September 15, 2024. DOI: 10.22541/essoar.172641527.77043060/v1

# Abstract
"TEXT"

# What this repository contains
This repository contains a minimal example with data to reproduce the results of the paper. In particular it contains: 
- `hdf5` files containing targets, predictions and NTL time series of the stations
- `.pth`-Files containing the trained model
- codes to evaluate the model with the NTL input data and to produce time series figures:
  - `plot_predictions_from_hdf5.ipynb`: notebook to plot time series of targets, predictions, and NTL input data
  - `predict.py`: script to predict time series at a selected station
  - `NGL_download.py`: script to scrape [NGL](http://geodesy.unr.edu/index.php) for time series (run at your own risk; may cause heavy download traffic)
  - `decompse.py`: script to decompose [NGL](http://geodesy.unr.edu/index.php) time series into _a priori_ non-tectonic time series targets using [GrAtSiD](https://github.com/TectonicGeodesy-RUB/Gratsid)

# Availibility of Associated Data
The original GNSS displacement time series are obtained from the Nevada Geodetic Laboratory (Blewitt et al., 2018) and are openly available in the NGL data repository under http://geodesy.unr.edu/index.php (Last accessed for download: 01 June 2024). 

Additional GNSS time series were composed based on the combined IGS repro3 solution (Rebischung et al., 2024) retrieved from `ftp://igs-rf.ign.fr/pub/repro3/`. Please visit the IGN webpage to retrieve their data.

The geophysical non-tidal surface loading products by Dill and Dobslaw (2013) are available at the ESMGFZ data repository http://esmdata.gfz-potsdam.de:8080/repository/entry/show?entryid=97d15ffe-3b5d-49dc-ba6c-3011851af1de (Last accessed for download: 01 June 2024). Please follow the manual on their webpage to retrieve the data accordingly.

The GrAtSiD trajectory modeling software by Bedford and Bevis (2018) is available on the GitHub repository of the Tectonic Geodesy group at Ruhr-University Bochum on https://github.com/TectonicGeodesy-RUB/Gratsid.
