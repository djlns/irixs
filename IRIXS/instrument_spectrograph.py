""" reduction class for spectrograph """

import os
import shutil
import copy
import warnings
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from skimage import io
from matplotlib.offsetbox import AnchoredText

from .tools import load_fio, flatten, peak_fit, binning


def bias_correct_4output(rawimg):
    """
    for EFGH output
    divide matrix up into four quadrants and subtract median from each quadrant
    """
    r, h = rawimg.shape[1], rawimg.shape[0]
    nrows,ncols = h//2, r//2
    img = (rawimg
           .reshape((h//nrows,nrows,-1,ncols))
           .swapaxes(1,2)
           .reshape((-1, nrows, ncols)))
    for i,im in enumerate(img):
        cen = int(np.median(im))
        img[i] -= cen
    img = np.block([[img[0], img[1]], [img[2], img[3]]])
    return img


def load_tiff(filename, run_no, exp, datdir, localdir, correct=True):
    """ load greateyes detector tiff file
    correct - correct for bias of 4 quadrants when using EFGH output
    """
    tiffpath = '{dir}/{exp}_{run_no:05d}/greateyes/{filename}'
    path = tiffpath.format(dir=datdir, exp=exp, run_no=run_no, filename=filename)
    with warnings.catch_warnings():
        if localdir:
            path2 = tiffpath.format(dir=localdir, exp=exp, run_no=run_no, filename=filename)
            try:
                img = io.imread(path2)
            except (Warning, FileNotFoundError, OSError):
                try:
                    os.makedirs(os.path.dirname(path2), exist_ok=True)
                    shutil.copyfile(path, path2)
                    img = io.imread(path2)
                except IOError:
                    return
        else:
            try:
                img = io.imread(path)
            except (Warning, FileNotFoundError, OSError):
                return
    if correct:
        img = bias_correct_4output(img)
    return img


class spectrograph:

    def __init__(
        self,
        exp,
        detfac=0,
        threshold=15,
        cutoff=2000,
        roix=[0,2048],
        roih=2064,
        roic=2064//2,
        bias_correct=True,
        vmax_average=100,
        vmax_single=100,
        vmin=1,
        datdir_remote="/gpfs/current/raw",
        datdir_local="raw",
        savedir="processed",
    ):

        self.exp = exp
        self.detfac = detfac
        self.threshold = threshold
        self.cutoff = cutoff
        self.roix = roix
        self.roih = roih
        self.roic = roic
        self.bias_correct = bias_correct
        self.vmax_average = vmax_average
        self.vmax_single = vmax_single
        self.vmin = vmin
        self.datdir = datdir_remote
        self.localdir = datdir_local
        self.savedir = savedir

        os.makedirs(self.localdir, exist_ok=True)
        os.makedirs(self.savedir, exist_ok=True)

        self.runs = {}

    def extract(self, run_nos):

        if not isinstance(run_nos, (list, tuple, range)):
            run_nos = [run_nos]
        run_nos = flatten(run_nos)

        for n in run_nos:
            if n not in self.runs.keys():
                self.runs[n] = None

        for run_no in run_nos:
            if self.runs[run_no] and self.runs[run_no]["complete"]:
                continue

            fio_file = "{0}_{1:05d}.fio".format(self.exp, run_no)
            fio_remote = os.path.join(self.datdir, fio_file)
            fio_local = os.path.join(self.localdir, fio_file)

            if not os.path.isfile(fio_local):
                a = load_fio(run_no, self.exp, self.datdir)
                if a and a['complete']:
                    shutil.copyfile(fio_remote, fio_local)
            else:
                a = load_fio(run_no, self.exp, self.localdir)

            a["img"] = []
            print(f"{run_no}", end="")

            # since we don't have any control over image filename, sort by collection time
            # default to remote folder if it exists
            if os.path.exists(self.datdir):
                img_root = self.datdir
            else:
                img_root = self.localdir
            img_folder = os.path.join(img_root, f"{self.exp}_{run_no:05d}", "greateyes")
            filepaths = sorted(glob(img_folder+'*.tiff'), key=os.path.getctime)
            filenames = [os.path.basename(f) for f in filepaths]

        for i,f in enumerate(filenames):
            img = load_tiff(f, run_no, self.exp, self.datdir, self.localdir, self.bias_correct)
            if img is not None:
                img -= self.detfac
                img[~np.logical_and(img > self.threshold, img < self.cutoff)] = 0
                a['img'].append(img)
            else:
                break
        a['x'] = a['x'][:i+1]  # ensure length of x is same as number of images
        print(f'/{len(a["img"])}')

        self.runs[run_no] = a

    def transform(self, run_nos, ysca=1, fit=True):

        self.extract(run_nos)
        if not isinstance(run_nos, (list, tuple, range)):
            run_nos = [run_nos]

        for run_no in run_nos:

            # sum up images if given a list of run_nos
            if isinstance(run_no, (list, tuple)):
                a = self.runs[run_no[0]]
                img = np.atleast_3d(np.array(a['img'])) / len(run_no)
                for r in run_no[1:]:
                    b = self.runs[r]
                    img = img + np.atleast_3d(np.array(b['img'])) / len(run_no)
            # single image
            else:
                a = self.runs[run_no]
                img = np.atleast_3d(np.array(a['img']))
                img = img * ysca

            x = a['x']
            if len(x) != len(img):
                x = x[:img.shape[0]]

            y, roi, imgr = [], [], []
            rx, ry, imgx, imgy = [], [], [], []

            for im, xi in zip(img, x):
                try:
                    # centre of roi as a lambda function
                    rc = self.roic(xi)
                except TypeError:
                    # fixed value
                    rc = self.roic
                roiy = rc - (self.roih//2),rc+(self.roih//2)
                roi.append([self.roix[0], self.roix[1], roiy[0], roiy[1]])

                ri = im[roiy[0]:roiy[1], self.roix[0]:self.roix[1]]
                imgr.append(ri)
                y.append(np.nansum(ri))

                rx.append(np.arange(roiy[0], roiy[1]))
                ry.append(np.arange(self.roix[0], self.roix[1]))

                imgx.append(np.nansum(ri, axis=1))
                imgy.append(np.nansum(ri, axis=0))

            y = np.array(y)

            r1 = min(i[0] for i in roi)
            r2 = max(i[1] for i in roi)
            r3 = min(i[2] for i in roi)
            r4 = max(i[3] for i in roi)
            x_totx = np.arange(r3, r4)
            x_toty = np.arange(r1, r2)

            imgroi = np.atleast_3d(np.array(imgr))
            tot = np.nansum(imgroi, axis=0) / img.shape[0]
            totx = np.nansum(tot, axis=1)
            toty = np.nansum(tot, axis=0)

            if fit:
                try:
                    xfx, yfx, px = peak_fit(x_totx, totx)
                    xfy, yfy, py = peak_fit(x_toty, toty)
                    txtx = f'fwhm: {px[1]*2:.4f}\ncen: {px[2]:.1f}\namp: {px[0]:.1f}\nfra: {px[3]:.1f}'
                    txty = f'fwhm: {py[1]*2:.4f}\ncen: {py[2]:.1f}\namp: {py[0]:.1f}\nfra: {py[3]:.1f}'
                except:
                    fit = False

            a["x"], a["y"] = x, y
            a["roi"], a["imgr"] = roi, imgroi
            a["rx"], a["ry"], a["imgx"], a["imgy"] = rx, ry, imgx, imgy
            a["extent"] = [r1, r2, r3, r4]
            a["tot"], a["totx"], a["toty"] = tot, totx, toty
            a["x_totx"] = x_totx
            a["x_toty"] = x_toty
            a["fitted"] = fit
            if fit:
                a["xfx"], a["yfx"], a["px"], a["txtx"] = xfx, yfx, px, txtx
                a["xfy"], a["yfy"], a["py"], a["txty"] = xfy, yfy, py, txty

    def detector(self, run_no):

        self.extract(run_no)

        if "x" not in self.runs[run_no]:
            self.transform(run_no)
        a = self.runs[run_no]

        fig = plt.figure(figsize=(16.5, 8.5))
        gs0 = fig.add_gridspec(
            1, 2, left=0.05, right=0.99, top=0.99, bottom=0.04, width_ratios=[1, 1.35]
        )

        # averaged total

        gs1 = gs0[0].subgridspec(
            3, 2, height_ratios=[1, 0.2, 0.4], width_ratios=[0.2, 1], hspace=0.1, wspace=0.1
        )
        ax1 = fig.add_subplot(gs1[0,1])
        ax2 = fig.add_subplot(gs1[0,0])
        ax3 = fig.add_subplot(gs1[1,1])
        ax4 = fig.add_subplot(gs1[2,:])

        fig.canvas.set_window_title(f'#{run_no}')

        ax1.imshow(
            a["tot"],
            origin='lower',
            interpolation='hanning',
            vmin=self.vmin,
            vmax=self.vmax_average,
            cmap=plt.get_cmap('bone_r'),
            extent=a["extent"],
            aspect='auto'
        )

        ax2.plot(a["totx"], a["x_totx"], color='k', lw=0.75)
        ax3.plot(a["x_toty"], a["toty"], color='k', lw=0.75)

        if a["fitted"]:
            ax2.plot(a["yfx"], a["xfx"], color='k',lw=0.5)
            ax3.plot(a["xfy"], a["yfy"], color='k',lw=0.5)
            ax2.add_artist(AnchoredText(a["txtx"], loc=1, frameon=False, pad=1))
            ax3.add_artist(AnchoredText(a["txty"], loc=2, frameon=False, pad=1))

        r1, r2, r3, r4 = a["extent"]
        ax2.set_ylim(r3, r4)
        ax3.set_xlim(r1, r2)

        # interactive singles

        img, roi, x = a["img"], a["roi"], a["x"]
        imgx, imgy, rx, ry = a["imgx"], a["imgy"], a["rx"], a["ry"]

        gs2 = gs0[1].subgridspec(
            2, 2, height_ratios=[1, 0.2], width_ratios=[0.2, 1], hspace=0.05, wspace=0.05
        )
        ax1 = fig.add_subplot(gs2[0,1])
        ax2 = fig.add_subplot(gs2[0,0])
        ax3 = fig.add_subplot(gs2[1,1])

        ax4.plot(a["x"], a["y"], color='k', lw=0.75)
        ax4.set_xlabel(a['auto'])
        ax4.set_ylabel('summed intensity')

        i = {'no':0}
        im = img[0]
        r = roi[0]

        cmap = copy.copy(plt.cmap.winter)
        cmap.set_under('w')
        cmap.set_over('#F012BE')

        i['a'] = ax1.imshow(
            im, origin='lower', vmin=self.vmin, vmax=self.vmax_single, cmap=cmap, aspect='auto'
        )
        i['b'], = ax2.plot(imgx[0], rx[0], color='k',lw=0.75)
        i['c'], = ax3.plot(ry[0], imgy[0], color='k',lw=0.75)
        i['v'] = ax4.axvline(x[0],color='r',lw=0.5)
        i['t'] = ax4.text(0.05, 0.9, x[0], transform=ax4.transAxes)

        ax1.set_xlim(r[0], r[1])
        ax1.set_ylim(r[2], r[3])
        ax2.set_ylim(r[2], r[3])
        ax3.set_xlim(r[0], r[1])

        def do_plot(i):
            im = img[i['no']]
            imx = imgx[i['no']]
            imy = imgy[i['no']]
            irx = rx[i['no']]
            iry = ry[i['no']]
            r = roi[i['no']]
            i['a'].set_data(im)
            i['b'].set_xdata(imx)
            i['b'].set_ydata(irx)
            i['c'].set_xdata(iry)
            i['c'].set_ydata(imy)
            i['v'].set_xdata(x[i['no']])
            i['t'].set_text(x[i['no']])
            ax2.relim()
            ax3.relim()
            ax2.autoscale(axis='x')
            ax3.autoscale(axis='y')
            plt.draw()

        def press(event):
            if event.key == 'left':
                if i['no'] > 0:
                    i['no'] -= 1
                    do_plot(i)
            elif event.key == 'right':
                if i['no'] < img.shape[0]-1:
                    i['no'] += 1
                    do_plot(i)

        def click(event):
            if not event.dblclick and event.button == 1:
                if event.inaxes in [ax4]:
                    i['no'] = np.abs(x-event.xdata).argmin()
                    do_plot(i)

        fig.canvas.mpl_connect('button_press_event', click)
        fig.canvas.mpl_connect('key_press_event', press)

    def condition(
        self,
        run_nos,
        bins=None,
        oneshot_x=False,
        oneshot_y=False,
        oneshot_no=None,
        fit=False,
        xsca=1,
        x0=0
    ):

        if not isinstance(run_nos, (list, tuple, range)):
            run_nos = [run_nos]

        self.transform(run_nos)

        for run_no in run_nos:
            if "x" not in self.runs[run_no]:
                self.transform(run_no)
            a = self.runs[run_no]

            if oneshot_x:
                if oneshot_no:
                    x = a['rx'][oneshot_no]
                    y = a['imgx'][oneshot_no]
                else:
                    x = a['x_totx']
                    y = a['totx']
                a["cond_type"] = "oneshot_x"
            elif oneshot_y:
                if oneshot_no:
                    x = a['ry'][oneshot_no]
                    y = a['imgy'][oneshot_no]
                else:
                    x = a['x_toty']
                    y = a['toty']
                a["cond_type"] = "oneshot_y"
            else:
                x, y = a['x'], a['y']
                a["cond_type"] = a["auto"]

            if bins:
                x, y, _ = binning(x, y, bins)

            x = (x - x0) * xsca

        a['cond_x'], a['cond_y'] = x, y

        if fit:
            try:
                xf, yf, p = peak_fit(x, y)
                r = '#{0}  '.format(a['no'])
                r += 'cen:{0:.4f}   '.format(p[2])
                r += 'amp:{0:.2f}   '.format(p[0])
                r += 'sig:{0:.4f}   '.format(p[1])
                r += 'fwhm:{0:.4f}  '.format(p[1]*2)
                r += 'fra:{0:.1f}   '.format(p[3])
                a['cond_xf'], a['cond_yf'], a['cond_p'], a['cond_r'] = xf, yf, p, r
                print(r)
            except:
                pass

        header = f"run: {a['no']}\nexp: {self.exp}\nbins: {bins}\n"
        if oneshot_x:
            header += "      y-pixel       intensity"
        elif oneshot_y:
            header += "      x-pixel       intensity"
        else:
            header += f"{a['auto']}    Intensity"
        np.savetxt(
            os.path.join(self.savedir, f"{self.exp}_{a['no']}_b{bins}.dat"),
            np.array([x, y]).T,
            fmt="% .8e",
            header=header
        )

    def plot(
        self,
        run_nos,
        ax=None,
        motor=None,
        norm=False,
        fit=False,
        ystep=0,
        plot_trend=False,
        **kwargs
    ):

        if not isinstance(run_nos, (list, tuple, range)):
            run_nos = [run_nos]

        if not ax:
            if plot_trend:
                fig = plt.figure(figsize=(7, 9), constrained_layout=True)
                gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 1.0])
                axes = [
                    fig.add_subplot(gs[0,:]),
                    fig.add_subplot(gs[1,0]),
                    fig.add_subplot(gs[1,1]),
                    fig.add_subplot(gs[1,2])
                ]
            else:
                _, ax = plt.subplots(constrained_layout=True)

        if plot_trend:
            ax = axes[0]

        trend_x, trend_x0, trend_I, trend_fw = [], [], [], []
        for i, run_no in enumerate(run_nos):
            a = self.runs[run_no]
            x, y = a['cond_x'], a['cond_y']
            if "cond_xf" in a and fit:
                xf, yf, p = a['cond_xf'], a['cond_yf'], a['cond_p']
                if norm:
                    yf = yf/np.max(y)
            else:
                xf = None
            if norm:
                y = y/np.max(y)

            lab = f"#{run_no}"
            if motor:
                lab += f" {motor}: {a[motor]:.2f}"

            if fit and xf:
                trend_x.append(run_no)
                trend_x0.append(p[2])
                trend_fw.append(p[1]*2)
                trend_I.append(p[0])

            l, = ax.plot(x, y+i*ystep, label=lab, **kwargs)
            if fit and xf:
                ax.plot(xf, yf+i*ystep, color=l.get_color(), lw=0.5)

        ax.set_ylabel('Intensity')

        if a["cond_type"] == "oneshot_x":
            ax.set_xlabel('y-pixel')
        elif a["cond_type"] == "oneshot_y":
            ax.set_xlabel('x-pixel')
        else:
            ax.set_xlabel(a["cond_type"])

        if plot_trend:
            axes[1].plot(trend_x, trend_x0, '.-')
            axes[2].plot(trend_x, trend_fw, '.-')
            axes[3].plot(trend_x, trend_I, '.-')
            axes[1].set_ylabel('Center')
            axes[2].set_ylabel('FWHM')
            axes[3].set_ylabel('Amplitude')
            axes[3].set_ylim(0)
            for axi in axes:
                axi.set_xlabel("run number")

        ax.legend(fontsize='small')

    def track_signal(self, run_no, fit=True, bins=None, maxsig=1200, vert=False, title='no',
                     plot=True, plot_trend=False, ystep=0):

        a = self.runs[run_no]

        rx, ry, imgx, imgy = a['rx'], a['ry'], a['imgx'], a['imgy']

        if vert:
            r = rx
            im = imgx
        else:
            r = ry
            im = imgy

        txt_title = f'#{a["no"]}'
        if title != 'no':
            txt_title += f' {title}: {a[title]}'

        if plot:
            fig, ax = plt.subplots(figsize=(4,9), constrained_layout=True)
            ax.text(0.04, 0.98, txt_title, transform=ax.transAxes)

        trend_x0, trend_I, trend_fw = [], [], []
        for i, (x, y, xpos) in enumerate(zip(r, im, a['x'])):

            if bins:
                x, y, _ = binning(x, y, bins)

            if plot:
                l, = ax.plot(x, y+i*ystep, lw=0.75)
                ax.text(x[0], y[0]+i*ystep, f'{xpos:.4f}', fontsize='small')

            if fit:
                try:
                    xf, yf, p = peak_fit(x, y)
                    if plot:
                        ax.plot(xf, yf+i*ystep, color=l.get_color(), lw=0.5)

                    if (p[1] < maxsig) and (p[2] > 0):
                        trend_I.append(p[0])
                        trend_x0.append(p[2])
                        trend_fw.append(p[1]*2)
                    else:
                        trend_I.append(np.nan)
                        trend_x0.append(np.nan)
                        trend_fw.append(np.nan)    
                except:
                    trend_I.append(np.nan)
                    trend_x0.append(np.nan)
                    trend_fw.append(np.nan)
            else:
                com = np.sum(x*y)/np.sum(y)
                height = np.max(y) - np.min(y)
                half = x[np.abs(y-(height/2+np.min(y))).argmin()]
                wid = np.abs(half-com)*2
                trend_I.append(np.sum(y))
                trend_x0.append(com)
                trend_fw.append(wid)

        trend_x = a['x']

        if plot_trend:
            fig,ax = plt.subplots(1,3,constrained_layout=True,figsize=(10,4))

            ax[0].plot(trend_x, trend_I)
            ax[1].plot(trend_x, trend_x0)
            ax[2].plot(trend_x, trend_fw)

            ax[0].set_ylabel('intensity')
            ax[1].set_ylabel('centre')
            ax[2].set_ylabel('fwhm')

            ax[0].text(0.05, 0.95, txt_title, transform=ax[0].transAxes)

        # table = np.array([range(len(trend_x)),trend_x, trend_I, trend_x0, trend_fw]).T
        # print(txt_title)
        # print(tabulate(table,headers=['no', a['auto'], 'intensity', 'centre', 'fwhm']))
        # print()

        return np.array([trend_x, trend_I, trend_x0, trend_fw])
