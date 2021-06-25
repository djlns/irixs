import os
import numpy as np
from numpy import sin, cos, sqrt, log, radians, arccos, pi
from scipy.optimize import curve_fit


def unit_cell_volume(a1, a2, a3, alpha1, alpha2, alpha3):
    """calculate unit cell volume (angles in radians)"""
    V = (
        a1 * a2 * a3 * sqrt(
            1 - cos(alpha1)**2 - cos(alpha2)**2 - cos(alpha3)**2 -
            2 * cos(alpha1) * cos(alpha2) * cos(alpha3)
        )
    )
    return V


def reciprocol_lattice(a1, a2, a3, alpha1, alpha2, alpha3):
    """calculate reciprocol lattice parameters (angles in radians)"""

    beta1 = arccos(
        (cos(alpha2) * cos(alpha3) - cos(alpha1)) / (sin(alpha2) * sin(alpha3))
    )

    beta2 = arccos(
        (cos(alpha1) * cos(alpha3) - cos(alpha2)) / (sin(alpha1) * sin(alpha3))
    )

    beta3 = arccos(
        (cos(alpha1) * cos(alpha2) - cos(alpha3)) / (sin(alpha1) * sin(alpha2))
    )

    V = unit_cell_volume(a1, a2, a3, alpha1, alpha2, alpha3)

    b1 = 2 * pi * a2 * a3 * sin(alpha1) / V
    b2 = 2 * pi * a1 * a3 * sin(alpha2) / V
    b3 = 2 * pi * a1 * a2 * sin(alpha3) / V

    return b1, b2, b3, beta1, beta2, beta3


def calc_dspacing(hkl, cell):
    """calculate dspacing from crystal lattice and orientation"""
    h, k, l = hkl
    a1, a2, a3, alpha1, alpha2, alpha3 = cell
    alpha1, alpha2, alpha3 = radians(alpha1), radians(alpha2), radians(alpha3)
    V = unit_cell_volume(a1, a2, a3, alpha1, alpha2, alpha3)

    S11 = a2**2 * a3**2 * sin(alpha1)**2
    S22 = a1**2 * a3**2 * sin(alpha2)**2
    S33 = a1**2 * a2**2 * sin(alpha3)**2
    S12 = a1 * a2 * a3**2 * (cos(alpha1) * cos(alpha2) - cos(alpha3))
    S23 = a2 * a3 * a1**2 * (cos(alpha2) * cos(alpha3) - cos(alpha1))
    S13 = a1 * a3 * a2**2 * (cos(alpha3) * cos(alpha1) - cos(alpha2))

    invD2 = (
        S11 * h ** 2
        + S22 * k ** 2
        + S33 * l ** 2
        + 2 * S12 * h * k
        + 2 * S23 * k * l
        + 2 * S13 * h * l
    )
    invD2 *= 1 / V ** 2
    d = 1 / sqrt(invD2)
    return d


def binning(x, y, n, photon_counting=False):
    """
    binning routine that also calculates error
    n: bin size
        - if integer, then simply divide array into steps of n
        - if float, then n corresponds to step size in eV
    photon_counting:
        - True: return events per bin
        - False: return intensity per bin averaged by counts
    """
    e = np.sqrt(y)
    if isinstance(n, int):  # strides
        xbin = int(len(x) / n)
    else:  # regularly spaced (in eV)
        xbin = np.arange(min(x), max(x), n)
    yi, xi = np.histogram(x, bins=xbin, weights=y)
    ei, _ = np.histogram(x, bins=xbin, weights=e ** 2)

    if not photon_counting:
        count, _ = np.histogram(x, bins=xbin)
        with np.errstate(divide="ignore", invalid="ignore"):
            yi = yi / count
            ei = np.sqrt(ei) / count

    xi = (xi[1:] + xi[:-1]) / 2
    return xi, yi, ei


def peak(x, a, sl, x0, f, bgnd):
    """basic pseudovoight profile with flat background"""
    m = np.full(len(x), bgnd)
    sg = sl / np.sqrt(2 * log(2))
    m += (
        (1 - f)
        * (a / (sg * np.sqrt(2.0 * np.pi)))
        * np.exp(-1.0 * (x - x0) ** 2.0 / (2.0 * sg ** 2.0))
    )
    m += f * (a / np.pi) * (sl / ((x - x0) ** 2.0 + sl ** 2.0))
    return m


def peak_fit(x, y):
    """
    peak fit routine that guesses initial values
    returns fitted peak xf,yf and parameters p = [amp, sig, cen, fra, bgnd]
    """
    fra = 0.5
    bgnd = float(np.min(y))
    height = np.max(y) - bgnd
    half = x[np.abs(y - height / 2 + bgnd).argmin()]
    cen = x[np.abs(y - np.max(y)).argmin()]
    fwhm = np.abs(half - cen) * 2
    sig = fwhm / 2
    amp = height * (sig * np.sqrt(2.0 * np.pi))

    xf = np.linspace(x.min(), x.max(), 1000)
    p0 = [amp, sig, cen, fra, bgnd]
    bounds = [(0, 1e-6, -1e9, 0, 0), (1e9, 1e6, 1e9, 1, 1e9)]
    p, _ = curve_fit(peak, x, y, p0, bounds=bounds)
    yi = peak(xf, *p)

    return xf, yi, p


def load_fio(run, exp, datdir):
    """
    .fio loader - returns everything as a dict
    data stored as a structured array with headers extracted from the fio
    contains a few helpers for P01 (qh, qk, ql, t_sample -> T)
    looks for '! Acquisition ended' to see if scan has been completed
    """
    a = {}
    head = []
    data = []
    complete = False
    fio_file = "{0}_{1:05d}.fio".format(exp, run)
    path = os.path.join(datdir, fio_file)
    if not os.path.isfile(path):
        print("#{0:<4} -- no .fio".format(run))
        return
    with open(path) as f:
        for line in f:
            l = line.strip()
            if l.startswith("%c"):
                command = next(f)[:-1].split()
                _, date = next(f).split(" started at ", maxsplit=1)
            elif line.startswith("%p"):
                break
        for line in f:
            if line.startswith("!"):
                break
            else:
                p, v = line.strip().split("=")
                try:
                    a[p.strip()] = float(v)
                except TypeError:
                    a[p.strip()] = v.strip()
        for line in f:
            l = line.strip().split()
            if not l:
                break
            if l[0] == "Col":
                head.append(l[2])
            else:
                try:
                    data.append([float(x) for x in l])
                except ValueError:
                    if line.startswith("! Acquisition ended"):
                        complete = True
    if head and data:

        data = [x for x in data if x]
        data = np.array(data)
        pnts = data.shape[0]
        data = data.view(dtype=[(n, float) for n in head])
        data = data.reshape(len(data))

        a["data"] = data
        a["auto"] = head[0]
        a["pnts"] = pnts
        if head[0] == "exp_dmy01":
            a["EF"] = np.full(pnts, a["rixs_ener"])
        else:
            a["EF"] = a["data"][head[0]]
        a["EI"] = a["dcm_ener"]
        a["th"] = a["rixs_th"]
        a["chi"] = a["rixs_chi"]

        if "q_h" in head:
            a["qh"] = np.average(data["q_h"])
            a["qk"] = np.average(data["q_k"])
            a["ql"] = np.average(data["q_l"])
        else:
            a["qh"], a["qk"], a["ql"] = 0.0, 0.0, 0.0

        if "t_coldhead" in head:
            a["t_coldhead"] = np.round(np.average(data["t_coldhead"]))
        if "t_sample" in head:
            a["t_sample"] = np.round(np.average(data["t_sample"]))
            a["T"] = a["t_sample"]
        else:
            a["T"] = 0.0

        a["command"] = command
        a["date"] = date.strip()
        a["time"] = float(command[-1])
        a["no"] = run
        a["complete"] = complete

        return a


def flatten(*n):
    """flattens a lists of lists/ranges/tuples for loading"""
    return [
        e
        for a in n
        for e in (flatten(*a) if isinstance(a, (tuple, list, range)) else (a,))
    ]
