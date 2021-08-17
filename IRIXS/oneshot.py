#!/usr/bin/env python

""" quick view of detector images for a specified measurement

usage: irixs_oneshot [number of run]

"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

from glob import glob
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.pyplot import imread

plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['font.size'] = 8
plt.rcParams['axes.titlesize'] = 'medium'
plt.rcParams['figure.titlesize'] = 'medium'


def find_nearest(array, value):
    return (np.abs(array-value)).argmin()


def peak(x, a, sl, x0, f, bgnd):
    m = np.full(len(x), bgnd)
    sg = sl/np.sqrt(2*np.log(2))
    m += (1-f)*(a/(sg*np.sqrt(2.*np.pi))) * np.exp(-1.*(x-x0)**2./(2.*sg**2.))
    m += f * (a/np.pi) * (sl / ((x-x0)**2. + sl**2.))
    return m


def peak_fit(x, y):
    fra = 0.5
    bgnd = float(np.min(y))
    height = np.max(y)-bgnd
    half = x[find_nearest(y, height/2+bgnd)]
    cen = x[find_nearest(y, np.max(y))]
    fwhm = np.abs(half-cen)*2
    sig = fwhm / 2
    amp = height*(sig*np.sqrt(2.*np.pi))

    xf = np.linspace(x.min(), x.max(), 1000)
    p0 = [amp, sig, cen, fra, bgnd]
    bounds = [(0, 1E-6, -1E9, 0, 0),(1E9, 1E6, 1E9, 1, 1E9)]
    p, _ = curve_fit(peak, x, y, p0, bounds=bounds)
    yi = peak(xf, *p)

    return xf, yi, p


def load_fio(run, exp, datdir):
    path = '{0}/{1}_{2:05d}.fio'.format(datdir, exp, run)
    a = {}
    head = []
    data = []
    with open(path) as f:
        for line in f:
            l = line.strip()
            if line.startswith('%d'):
                break
        for line in f:
            l = line.strip().split()
            if not l:
                break
            if l[0] == 'Col':
                head.append(l[2])
            else:
                try:
                    data.append([float(x) for x in l])
                except ValueError:
                    break
    if head and data:
        data = [x for x in data if x]
        data = np.array(data)
        data = data.view(dtype=[(n, float) for n in head])
        data = data.reshape(len(data))
        a['data'] = data
        a['auto'] = head[0]
        a['pnts'] = data.shape[0]
        return a


def load_tiff(run, exp, datdir, no, indicator=True):
    path = '{0}/{1}_{2:05d}/andor/{1}_{2:05d}_{3:04d}.tiff'.format(
            datdir, exp, run, no)
    with warnings.catch_warnings():
        try:
            img = imread(path)
            if indicator:
                print('*', end=' ', flush=True)
        except (Warning, FileNotFoundError, OSError):
            print('\nload_tiff: failed to load {}'.format(path))
            return
    return img


def load(run, exp, datdir, detfac, to, co):
    a = load_fio(run, exp, datdir)
    imgtest = load_tiff(run, exp, datdir, 0, indicator=False)
    if imgtest is None:
        return
    else:
        a['img'] = []
    print('#{} ({} points)'.format(run, a['pnts']), end=' ')
    for i,_ in enumerate(a['data'][a['auto']]):
        if i > len(a['img'])-1:
            img = load_tiff(run, exp, datdir, i)
            if img is not None:
                img -= detfac
                img[~np.logical_and(img > to, img < co)] = 0
                a['img'].append(img)
    print()
    return a


def detector(run, exp, datdir, vmax=10, threshold=1010, cutoff=1800, detfac=935):

    to = threshold - detfac
    co = cutoff - detfac

    a = load(run, exp, datdir, detfac, to, co)
    if a is None:
        return

    step = a['auto']
    if step == 'exp_dmy01':
        oneshot = True
    else:
        oneshot = False

    imgarr = np.atleast_3d(np.array(a['img']))
    imtotal = np.nansum(imgarr, axis=0) / imgarr.shape[0]

    if oneshot:
        x = np.arange(imgarr.shape[1])
        y = np.nansum(imgarr, axis=(0,2)) / imgarr.shape[0]
    else:
        x = []
        y = []
        for i, ef in enumerate(a['data'][a['auto']]):
            try:
                img = imgarr[i]
            except IndexError:
                continue
            yi = np.nansum(img)
            xi = ef
            x.append(xi)
            y.append(yi)
        x, y = np.array(x), np.array(y)

    p = False
    if oneshot:  # try fitting the signal if the measurement was a oneshot
        try:
            xf, yf, p = peak_fit(x, y)
            print('  amp: {:.2f}'.format(p[0]), end='  ')
            print('fwhm: {:.3f}'.format(p[1]*2), end='  ')
            print('cen: {:.4f}'.format(p[2]), end='  ')
            print('fra: {:3.1f}'.format(p[3]), end='  ')
            print('bg: {:.2f}'.format(p[3]), end='')
        except:
            pass
    print()

    fig, ax = plt.subplots(1, 2, figsize=(8.5, 4))
    fig.subplots_adjust(0.06, 0.15, 0.98, 0.93)

    plt.suptitle(f'#{run}', ha='left', va='top', x=0.005, y=0.995)

    im = ax[0].imshow(imtotal, origin='lower', vmax=vmax,
                      cmap=plt.get_cmap('bone_r'),
                      interpolation='hanning')

    div = make_axes_locatable(ax[0])
    cax = div.append_axes('right', size='4%', pad=0.1)
    fig.colorbar(im, cax=cax)

    ax[0].set_title('Summed Detector Map')
    ax[0].set_xlabel('x-pixel')
    ax[0].set_ylabel('y-pixel')
    ax[0].tick_params(which='both', direction='out', length=2)
    ax[0].xaxis.set_major_locator(plt.MultipleLocator(400))
    ax[0].yaxis.set_major_locator(plt.MultipleLocator(400))
    ax[0].xaxis.set_minor_locator(plt.MultipleLocator(100))
    ax[0].yaxis.set_minor_locator(plt.MultipleLocator(100))

    # total counts
    ax[1].plot(x, y, lw=1, color='#001F3F')
    if oneshot:
        ax[1].set_xlabel('y-pixel')
        ax[1].set_title('Integrated')
    else:
        ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax[1].set_title('Counts')

    if p is not False:
        ax[1].plot(xf, yf,
                   color='#F012BE', dashes=(2,8), lw=0.5)
        fr = 'fwhm: {:.3f}\ncen: {:.2f}'.format(
             p[1]*2,p[2])
        ax[1].text(0.025, 0.975, fr, va='top',
                   transform=ax[1].transAxes,
                   fontsize='small', linespacing=1.3)

    for axi in ax[1:]:
        axi.minorticks_on()
        if not oneshot:
            axi.set_xlabel(step)
            for l in axi.get_xmajorticklabels():
                l.set_rotation(30)


def find_run(run):
    datdir = '/gpfs/current/raw'
    search = datdir + '/*_{0:05d}.fio'.format(run)
    path = sorted(glob(search), key=os.path.getmtime)
    if not path:
        datdir = '/gpfs/commissioning/raw'
        search = datdir + '/*_{0:05d}.fio'.format(run)
        path = sorted(glob(search), key=os.path.getmtime)
    if path:
        path = path[-1]
        _,f = path.rsplit('/',1)
        exp,_ = f.rsplit('_',1)
        return exp, datdir
    else:
        return False, False


def main():
    try:
        run = int(sys.argv[1])
    except (IndexError, ValueError):
        print("irixs_oneshot [number of run]")
        sys.exit(2)
    exp, datdir = find_run(run)
    if exp:
        detector(run, exp, datdir)
        plt.show()
    else:
        print('failed to load #{}'.format(run))


if __name__ == "__main__":
    main()
