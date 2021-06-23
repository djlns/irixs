# IRIXS reduction routines

Scripts for analysis of data collected on the IRIXS spectrometer, Beamline P01, Synchrotron Petra-III, DESY

- [IRIXS: a resonant inelastic X-ray scattering instrument dedicated to X-rays in the intermediate energy range](https://doi.org/10.1107/S1600577519017119)  
- [IRIXS Spectrograph: an ultra high-resolution spectrometer for tender RIXS](https://doi.org/10.1107/S1600577521003805)

## Overview

### Reduction Classes
`IRIXS.irixs`: reduction class for the Rowland circle spectrometer  
`IRIXS.spectrograph`: reduction class for spectrograph
- extracts raw collected images, transforms them into spectra and loads them to text files for analysis.
- basic plotting and fitting functionality

### Scripts
`p01plot`: GUI application for quick plotting and fitting for experiments on P01 and P09  
`irixs_oneshot`: check detector images from a specific measurement

## Installation

Environment: Python 3.8+ w/ scipy + matplotlib + scikit-image + PyQT5

from PyPI:
1. `pip install IRIXS`

If using an anaconda/miniconda distribution, it is suggested to install dependencies separately:
1. `pip install IRIXS --no-deps`
2. then e.g. `conda install pyqt`

To symlink to the source folder instead:
1. Clone repository to a prefered location
2. Enter root directory
3. `pip install -e .`


## Usage

### IRIXS.irixs
Example reduction script for `IRIXS.irixs`

```python
from IRIXS import irixs

expname = 'irixs_11009137'
a = irixs(expname, y0=667, roix=[160, 1500], roih=[-200, 200])

elastic_runs = [1713, 1719]
spectra_runs = [1710, 1711, 1712, 1722, 1723]

a.condition(0.006, elastic_runs, fit=True)
a.condition(0.02, spectra_runs)

fig, ax = plt.subplots()
a.plot(elastic_runs, ax=ax)
a.plot(spectra_runs, ax=ax)
```

### IRIXS.spectrograph
Example reduction script for `IRIXS.spectrograph` todo

### p01plot
```
p01plot [directory] [--remote -r] [--help -h]
directory : location to look for .fio data files
            defaults to /gpfs/current/raw, then current directory
--remote : remove cursor to speed up remote connections
--help : show this menu
```

### irixs_oneshot

```
irixs_oneshot [number of run]
```

## License

Copyright (C) Max Planck Institute for Solid State Research 2019-2021  
GNU General Public License v3.0
