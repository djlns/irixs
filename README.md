# IRIXS reduction routines

Scripts for analysis of data collected on the IRIXS spectrometer, Beamline P01, Synchrotron Petra-III, DESY

## Overview

`irixs.irixs`: reduction class for the Rowland circle spectrometer
`irixs.spectrograph`: reduction class for spectrograph
- extracts raw collected images, transforms them into spectra and loads them to text files for analysis.
- basic plotting and fitting functionality

`P01PLOT`: standalone GUI program for quick plotting and fitting for experiments on P01 and P09

## Installation

`pip install irixs`

## Usage

To run P01PLOT simply run `P01PLOT` from a terminal

Example reduction script for `irixs.irixs`

```python
from irixs import irixs

expname = 'irixs_11009137'
a = irixs(expname, y0=667, roix=[160, 1500], roih=[-200, 200])

elastic_runs = [1713, 1719]
spectra_runs = [1710, 1711, 1712, 1722, 1723]

a.condition(0.006, elastic_runs, fit=True)
a.condition(0.02, spectra_runs)

fig, ax = plt.subplots)
a.plot(elastic_runs, ax=ax)
a.plot(spectra_runs, ax=ax)
```

## License

Copyright (C) Max Planck Institute for Solid State Research 2019-2021
MIT License
