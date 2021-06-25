# IRIXS routines

Scripts for the IRIXS spectrometer, Beamline P01, Synchrotron Petra-III, DESY

- [IRIXS: a resonant inelastic X-ray scattering instrument dedicated to X-rays in the intermediate energy range](https://doi.org/10.1107/S1600577519017119)  
- [IRIXS Spectrograph: an ultra high-resolution spectrometer for tender RIXS](https://doi.org/10.1107/S1600577521003805)

## Overview

### Classes
`IRIXS.irixs`: reduction class for the Rowland circle spectrometer  
`IRIXS.spectrograph`: reduction class for spectrograph
- extracts raw collected images, transforms them into spectra and loads them to text files for analysis.
- basic plotting and fitting functionality

`IRIXS.sixc`: six-circle diffractometer simulator class for experiment planning

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
todo: example script for `IRIXS.spectrograph`

### IRIXS.sixc
Example script for `IRIXS.sixc`

```python
from IRIXS import sixc

# initialise UB-matrix using experimental conditions
unit_cell = [5.37, 5.60, 19.35, 90, 90, 90]
ref0 = (0, 0, 4)
ref1 = (1, 0, 0)
th0 = 29.85
tth0 = 53.70
chi0 = 2.0
f = sixc(unit_cell, ref0, ref1, th0, tth0, chi0, energy=2838.5)

# print hkl for values from grazing to normal with detector fixed at tth=90
for th in range(0, 95, 5):
    print(th, f.hkl(th))
```

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
