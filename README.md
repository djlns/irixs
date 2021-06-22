# IRIXS reduction routines

Scripts for analysis of data collected on the IRIXS spectrometer, Beamline P01, Synchrotron Petra-III, DESY

## Overview

### Reduction Classes
`IRIXS.irixs`: reduction class for the Rowland circle spectrometer  
`IRIXS.spectrograph`: reduction class for spectrograph
- extracts raw collected images, transforms them into spectra and loads them to text files for analysis.
- basic plotting and fitting functionality

### Applications
`P01PLOT`: GUI application for quick plotting and fitting for experiments on P01 and P09  
`oneshot`: check detector images from a specific measurement

## Installation

`pip install irixs`

Environment: Python 3.8+ with numpy + scipy + matplotlib + skimage + PyQT5

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
Example reduction script for `IRIXS.spectrograph`

### P01PLOT
```
p01plot [directory] [--remote -r] [--help -h]
directory : location to look for .fio data files
            defaults to /gpfs/current/raw, then current directory
--remote : remove cursor to speed up remote connections
--help : show this menu
```

### oneshot

```
oneshot [number of run]
```

## License

Copyright (C) Max Planck Institute for Solid State Research 2019-2021  
GNU General Public License v3.0
