""" reduction class for rowland circle spectrometer """

import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
import warnings
import scipy.ndimage

from numpy import pi, cos, tan, arcsin
from matplotlib.pyplot import imread
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy
from glob import iglob

from .tools import calc_dspacing, binning, peak_fit, load_fio, flatten

PIX_EN_CONV = 13.5e-6  # andor detector pixel size
SR_LIMIT = 50  # minimum ring current in mA to identify beam dump

plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["font.size"] = 8
plt.rcParams["axes.titlesize"] = "medium"
plt.rcParams["figure.titlesize"] = "medium"


def pix_to_E(energy, dspacing):
    """convert y-axis pixel position to energy dispersion from analyser"""
    wl = 12398.4193 / energy
    th = arcsin(wl / (2 * dspacing))
    l = 2 * cos(pi / 2 - th)
    dE = energy * PIX_EN_CONV / (l * tan(th))
    return dE


def load_tiff(run, no, exp, datdir, localdir):
    """
    loads a tiff image and returns as a numpy array
    - if localdir defined, looks to local directory first,
      otherwise copies the remote file once loaded
    - tries to catch corrupt files
    """
    tiff_file = '{0}_{1:05d}_{2:04d}.tiff'.format(exp, run, no)
    tiff_path = os.path.join('{0}_{1:05d}'.format(exp, run), "andor", tiff_file)
    path = os.path.join(datdir, tiff_path)

    with warnings.catch_warnings():
        if localdir:
            path2 = os.path.join(localdir, tiff_path)
            try:
                img = imread(path2)
            except (Warning, FileNotFoundError, OSError):
                try:
                    os.makedirs(os.path.dirname(path2), exist_ok=True)
                    shutil.copyfile(path, path2)
                    img = imread(path2)
                except IOError:
                    return
        else:
            try:
                img = imread(path)
            except (Warning, FileNotFoundError, OSError):
                return
    return img


class irixs:
    def __init__(
        self,
        exp,
        y0=None,
        y0_fallback=None,
        roix=[0, 2048],
        roiy=[0, 2048],
        roih=None,
        threshold=1010,
        cutoff=1800,
        detfac=935,
        photon_factor=750,
        E0=None,
        analyser=None,
        photon_counting=False,
        photon_event_threshold=400,
        photon_max_events=0,
        datdir_remote="/gpfs/current/raw",
        datdir_local="raw"
    ):
        """
        exp -- experiment filename prefix

        y0 -- corresponding vertical pixel position on detector
        - can be given as single value or as a dict for each run
        y0_fallback -- if a dict is used for y0, use y0_fallback for runs not included the dict
        E0 -- elastic energy: typically use None to extract from .fio file

        roix -- detector region of interest
        roiy -- detector region of interest
        roih -- define height of roi instead of roiy, using y0 as centre (preferred)

        threshold -- minimum to kill readout noise (use histogram to refine)
        cutoff -- upper limit to kill cosmic rays (use histogram to refine)
        detfac -- andor detector factor (0 if andor bgnd sub is enabled, 935 otherwise)

        analyser -- analyser crystal lattice and reflection.
        - None defaults to quartz (1,0,2)
        photon_factor -- conversion between detector intensitiy and photon count
        - should be = 788, seems closer to 750
        - used to calculate correct intensities and systematic error

        photon_counting -- enable photon counting mode
        photon_event_threshold -- threshold intensity for a contigous detector event
        photon_max_events -- maximum multiple events to correct for (0 to disable correction)
        - photon_counting only works if the count rate on the detector is low
        """

        self.exp = exp
        self.runs = {}

        self.E0 = E0
        self.y0 = y0
        self.roix = roix
        self.roiy = roiy
        self.roih = roih

        self.threshold = threshold
        self.cutoff = cutoff
        self.detfac = detfac
        self.photon_factor = photon_factor

        self.photon_counting = photon_counting
        self.event_min = photon_event_threshold
        self.max_events = photon_max_events

        quartz = [(1, 0, 2), (4.9133, 4.9133, 5.4053, 90, 90, 120)]
        self.analyser = quartz if analyser is None else analyser
        self.dspacing = calc_dspacing(self.analyser[0], self.analyser[1])

        self.datdir = datdir_remote
        self.localdir = datdir_local
        if self.localdir:
            os.makedirs(self.localdir, exist_ok=True)

        self.savedir_dat = "dat"
        self.savedir_con = "con"
        self.savedir_det = "det"
        self.savedir_fig = "fig"
        os.makedirs(self.savedir_dat, exist_ok=True)
        os.makedirs(self.savedir_con, exist_ok=True)
        os.makedirs(self.savedir_det, exist_ok=True)
        os.makedirs(self.savedir_fig, exist_ok=True)

        self.corr_shift = False  # distortion correction

    def load(self, run_nos, tiff=True):
        """
        load data & parameters from fio and tiff files
        - stores runs as dicts in the self.runs dict
        - only copies to local directory when a run is completed
        - .fio files loaded first
        - performs tiff thresholding and cutoff (and stores values used)
        - stores parameters to be used for data conditioning later
        - should be smart enough to only reload/recondition runs when necessary

        runs -- numbers of runs, can be given as single run or list or list of lists
        tiff -- only load image data if True
        """

        if not isinstance(run_nos, (list, tuple, range)):
            run_nos = [run_nos]
        run_nos = flatten(run_nos)

        for n in run_nos:
            if n not in self.runs.keys():
                self.runs[n] = None

        for n in run_nos:
            if self.runs[n] and self.runs[n]["complete"]:
                continue
            path = "{0}/{1}_{2:05d}.fio".format(self.datdir, self.exp, n)
            if self.localdir:
                path2 = "{0}/{1}_{2:05d}.fio".format(self.localdir, self.exp, n)
                if not os.path.isfile(path2):
                    a = load_fio(n, self.exp, self.datdir)
                    if a and a["complete"]:
                        os.makedirs(self.localdir, exist_ok=True)
                        shutil.copyfile(path, path2)
                else:
                    a = load_fio(n, self.exp, self.localdir)
            else:
                a = load_fio(n, self.exp, self.datdir)
            self.runs[n] = a

        if not tiff:
            return

        to = self.threshold - self.detfac
        co = self.cutoff - self.detfac

        for n in run_nos:

            a = self.runs[n]
            if not a:
                continue

            if isinstance(self.y0, dict):
                try:
                    y0 = self.y0[n]
                except KeyError:
                    try:
                        y0 = self.y0_fallback
                    except AttributeError:
                        print("#{0:<4} -- y0 not given".format(n))
                        continue
            else:
                y0 = self.y0
            if self.roih:
                try:
                    roiy = [y0 + self.roih[0], y0 + self.roih[1]]
                except IndexError:
                    roiy = [y0 - self.roih // 2, y0 + self.roih // 2]
            else:
                roiy = self.roiy

            a["roix"], a["roiy"], a["y0"] = self.roix, roiy, y0

            if "to" in a and to == a["to"] and co == a["co"] and a["complete"]:
                continue

            if "img" not in a or a["img"] is None:
                imtest = load_tiff(n, 0, self.exp, self.datdir, self.localdir)
                if imtest is None:
                    print("#{0:<4} -- no images".format(n))
                    a["img"] = None
                    continue
                else:
                    a["img"] = []

            for i, _ in enumerate(a["EF"]):
                if i > len(a["img"]) - 1:
                    img = load_tiff(n, i, self.exp, self.datdir, self.localdir)
                    if img is None:
                        print("!!!")
                        break
                    if img is not None:
                        img -= self.detfac
                        img[~np.logical_and(img > to, img < co)] = 0
                        a["img"].append(img)
                    sys.stdout.write(
                        "\r#{0:<4} {1:<3}/{2:>3} ".format(n, i + 1, a["pnts"])
                    )
                    if i + 1 == a["pnts"]:
                        sys.stdout.write("\n")
                    sys.stdout.flush()

            a["threshold"] = self.threshold
            a["cutoff"] = self.cutoff
            a["detfac"] = self.detfac
            a["to"], a["co"] = to, co

    def logbook(
        self,
        nstart=None,
        nend=None,
        run_nos=None,
        extras=["th"],
        hkl=False,
        date=False,
        only_rixs=True,
    ):
        """
        tabulate runs
        - shows if ring current was too low (SR_LIMIT)
        - shows if rixs_ener is different to dcm_ener

        nstart,nend -- range of runs
        runs -- specific runs
        extras,hkl,date -- extra parameters to display
        only_rixs -- only tabulate rixs runs
        """
        if not nstart and not run_nos:
            run_nos = list(self.runs.keys())

        if nstart:
            if nend is None:
                try:
                    latest = max(
                        iglob(os.path.join(self.datdir, "*.fio")), key=os.path.getctime
                    )
                except ValueError:
                    print("Using Local Directory")
                    latest = max(
                        iglob(os.path.join(self.localdir, "*.fio")),
                        key=os.path.getctime,
                    )
                try:
                    latest = latest[:-4].split("_")[-1]
                    run_nos = range(nstart, int(latest) + 1)
                except ValueError:
                    return
            else:
                run_nos = range(nstart, nend + 1)

        self.load(run_nos, False)
        for run_no in run_nos:
            out = ""
            a = self.runs[run_no]
            if a is None:
                continue
            try:
                command = a["command"]
                scantype, motor = command[:2]
                m1, m2, pnt, t = [float(c) for c in command[2:]]
            except IndexError:
                continue
            if np.any(a["data"]["sr_current"] < SR_LIMIT):
                dump = "*"
            else:
                dump = " "
            out += "#{0:<4}{1}{2:>13} ".format(run_no, dump, motor)
            if motor in ["rixs_ener"]:
                out += "{0:>+6.2f} > {1:+4.2f}".format(m1, m2)
            else:
                if only_rixs:
                    continue
                out += " {0:<13}".format("")
            out += " {0:3.0f}pnt {1:4.0f}s  ".format(pnt, t)
            out += "{0:7.1f}eV".format(a["dcm_ener"])
            if a["dcm_ener"] != a["rixs_ener"]:
                out += "* "
            else:
                out += "  "
            out += "{0:3.0f}K  ".format(a["T"])
            if hkl:
                qh = np.round(a["qh"], 2) + 0
                qk = np.round(a["qk"], 2) + 0
                ql = np.round(a["ql"], 2) + 0
                if not qh and not qk and not ql:
                    out += " " * 18
                else:
                    out += "({0: 3.1f} {1: 3.1f} {2:4.1f})  ".format(qh, qk, ql)
            for ex in extras:
                out += "{0}:{1:6.2f}  ".format(ex, a[ex])
            if date:
                out += " " + a["date"]
            print(out)
        print()

    def detector(
        self,
        run_nos,
        oneshot=None,
        fit=False,
        plot=True,
        com=False,
        vmax=10,
        savefig=False,
        use_distortion_corr=True,
    ):
        """
        data reduction without energy binning
        - if oneshot = False, returns summed intensity of ROI window vs step
        - if oneshot = True, returns total summed and flattened counts vs detector x-axis

        runs -- numbers of runs
        oneshot -- sum detector mode
        fit -- try and fit a peak
        plot -- plots all detector images summed together, as well as conditioned data
        com -- also plot (and store) center of mass along x- and y- detector axes
        vmax -- colormap threshold (10 is pretty good in most cases)
        """
        roic = "#F012BE"
        self.load(run_nos)

        if not isinstance(run_nos, (list, tuple, range)):
            run_nos = [run_nos]

        for run_no in run_nos:
            a = self.runs[run_no]
            if a is None or a["img"] is None:
                continue

            savefile = "{0}/{1}_{2:05d}_det.txt".format(
                self.savedir_det, self.exp, run_no
            )

            step, roix, roiy, y0 = a["auto"], a["roix"], a["roiy"], a["y0"]

            if oneshot is None:
                if step == "exp_dmy01":
                    oneshot = True
                else:
                    oneshot = False

            x = []
            y = []
            comV = []
            comH = []

            imgarr = np.atleast_3d(np.array(a["img"]))
            imtotal = np.nansum(imgarr, axis=0) / imgarr.shape[0]
            imgarr = imgarr[:, roiy[0] : roiy[1], roix[0] : roix[1]]

            if use_distortion_corr and self.corr_shift is not False:
                for sh, (c1, c2) in zip(self.corr_shift, self.corr_regions):
                    c1, c2 = c1 + roix[0], c2 + roix[0]
                    imtotal[:, c1:c2] = np.roll(imtotal[:, c1:c2], sh, axis=0)

            if oneshot:
                x = np.arange(roiy[0], roiy[1])
                y = np.nansum(imgarr, axis=(0, 2)) / imgarr.shape[0]
            else:
                for i, ef in enumerate(a["EF"]):
                    try:
                        img = imgarr[i]
                    except IndexError:
                        continue
                    yi = np.nansum(img)
                    xi = ef
                    x.append(xi)
                    y.append(yi)
                    if com:
                        rx = range(roiy[0], roiy[1])
                        ry = range(roix[0], roix[1])
                        cv = np.nansum(rx * np.nansum(img, axis=1)) / yi
                        ch = np.nansum(ry * np.nansum(img, axis=0)) / yi
                        comV.append(cv)
                        comH.append(ch)
                x, y = np.array(x), np.array(y)

            a["x"], a["y"], a["e"] = x, y, False
            a["label"] = run_no

            header = "experiment: {0}\n".format(self.exp)
            header += "run: {0}\n".format(run_no)
            header += "date: {}\n".format(a["date"])
            header += "command: {0}\n".format(" ".join(a["command"]))
            header += "dcm_ener: {0}\n".format(a["dcm_ener"])
            header += "rixs_ener: {0}\n".format(a["rixs_ener"])
            header += "det_threshold: {0}\n".format(a["threshold"])
            header += "det_cutoff: {0}\n".format(a["cutoff"])
            header += "det_factor: {0}\n".format(a["detfac"])
            header += "det_roix: {0}\n".format(roix)
            header += "det_roiy: {0}\n\n".format(roiy)

            if oneshot:
                header += "{0:>24}{1:>24}".format("y-pixel", "counts")
                save_array = np.array([x, y]).T
            elif com:
                header += "{0:>24}{1:>24}{2:>24}{3:>24}".format(
                    step, "roi-counts", "vert-COM", "horiz-COM"
                )
                comV, comH = np.array(comV), np.array(comH)
                a["comV"], a["comH"] = comV, comH
                save_array = np.array([x, y, comV, comH]).T
            else:
                header += "{0:>24}{1:>24}".format(step, "roi-counts")
                save_array = np.array([x, y]).T
            np.savetxt(savefile, save_array, header=header)

            if fit:
                a["xf"], a["yf"], a["p"] = peak_fit(x, y)
                report = "#{0:<4} (det)  ".format(run_no)
                report += "cen:{0:.4f}   ".format(a["p"][2])
                report += "amp:{0:.2f}   ".format(a["p"][0])
                report += "fwhm:{0:.3f}   ".format(a["p"][1] * 2)
                report += "fra:{0:.1f}   ".format(a["p"][3])
                report += "bg:{0:.3f}".format(a["p"][3])
                print(report)
            else:
                a["p"] = False

            a["oneshot"] = oneshot
            a.pop("E0", None)

            if plot:
                if com:
                    fig, ax = plt.subplots(
                        2, 2, constrained_layout=True, figsize=(8.5, 8)
                    )
                    ax = ax.flatten()
                else:
                    fig, ax = plt.subplots(1, 2, figsize=(8.5, 4))
                    fig.subplots_adjust(0.06, 0.15, 0.98, 0.93)

                plt.suptitle(
                    "#{}".format(a["label"]), ha="left", va="top", x=0.005, y=0.995
                )

                im = ax[0].imshow(
                    imtotal,
                    origin="lower",
                    vmax=vmax,
                    cmap=plt.get_cmap("bone_r"),
                    interpolation="hanning",
                )

                rect = Rectangle(
                    (roix[0], roiy[0]),  # xy origin, width, height
                    roix[1] - roix[0],
                    roiy[1] - roiy[0],
                    linewidth=0.5,
                    linestyle="dashed",
                    edgecolor=roic,
                    fill=False,
                )
                ax[0].add_patch(rect)
                ax[0].axhline(y0, color=roic, lw=0.5, dashes=(2, 2))

                div = make_axes_locatable(ax[0])
                cax = div.append_axes("right", size="4%", pad=0.1)
                fig.colorbar(im, cax=cax)

                ax[0].set_title("Summed Detector Map")
                ax[0].set_xlabel("x-pixel")
                ax[0].set_ylabel("y-pixel")
                ax[0].tick_params(which="both", direction="out", length=2)
                ax[0].xaxis.set_major_locator(plt.MultipleLocator(400))
                ax[0].yaxis.set_major_locator(plt.MultipleLocator(400))
                ax[0].xaxis.set_minor_locator(plt.MultipleLocator(100))
                ax[0].yaxis.set_minor_locator(plt.MultipleLocator(100))

                ax[1].plot(x, y, lw=1, color="#001F3F")
                if oneshot:
                    ax[1].set_xlabel("y-pixel")
                    ax[1].set_title("Integrated")
                else:
                    ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                    ax[1].set_title("Counts in ROI")

                if a["p"] is not False:
                    ax[1].plot(a["xf"], a["yf"], color="#001F3F", dashes=(2, 8), lw=0.5)
                    fr = "fwhm: {:.3f}\ncen: {:.2f}".format(a["p"][1] * 2, a["p"][2])
                    ax[1].text(
                        0.025,
                        0.975,
                        fr,
                        va="top",
                        transform=ax[1].transAxes,
                        fontsize="small",
                        linespacing=1.3,
                    )

                if com:
                    ax[2].plot(a["x"], a["comH"], lw=1, color="#0074D9")
                    ax[2].set_title("Horizontal COM")
                    ax[2].set_ylabel("x-pixel")
                    ax[2].set_ylim(roix[0], roix[1])

                    ax[3].plot(a["x"], a["comV"], lw=1, color="#FF4136")
                    ax[3].set_title("Vertical COM")
                    ax[3].set_ylabel("y-pixel")
                    ax[3].set_ylim(roiy[0], roiy[1])

                for axi in ax[1:]:
                    axi.minorticks_on()
                    if not oneshot:
                        axi.set_xlabel(step)
                        for l in axi.get_xmajorticklabels():
                            l.set_rotation(30)

                if savefig:
                    if savefig is True:
                        savename = "{0}/det_s{1}_{2}.pdf".format(
                            self.savedir_fig, a["label"], a["auto"]
                        )
                    else:
                        savename = "{0}/{1}_s{2}.pdf".format(
                            self.savedir_fig, savefig, a["label"]
                        )
                    plt.savefig(savename, dpi=300)

    def condition(
        self, bins, run_nos, fit=False, use_distortion_corr=True, drop_beamdump=False
    ):
        """
        main data reduction routine, returns signal vs energy
        - stores result in self.runs[run]['x'] and self.runs[run]['y']
        - elastic line set using y0 and E0 (E0 normally set by hrm_ener)
        - saves binned and unbinned datasets to file
        - reloads data before starting condition (which also update parameters if changed)

        bins -- 0 to disable binning, set in eV (float) or in array stride (integer)
        runs -- numbers of runs
        - single value or list of runs
        - list of list will stitch runs under the first run number
        fit -- fit peak
        use_distortion_correction -- run calc_distortion to determine correction first
        drop_beamdump -- ignore runs measured during a beamdump
        """
        self.load(run_nos)
        if isinstance(run_nos, int):
            run_nos = [run_nos]

        for run_no in run_nos:

            if isinstance(run_no, int):
                run_no = [run_no]

            x, y, ns = [], [], []
            for n in run_no:
                a = self.runs[n]
                if a is None or "img" not in a or a["img"] is None:
                    continue
                if a["auto"] not in ["rixs_ener", "dcm_ener", "exp_dmy01"]:
                    continue
                ns.append(n)

                roix, roiy, y0 = a["roix"], a["roiy"], a["y0"]
                xinit = np.arange(roiy[0], roiy[1])
                if self.photon_counting:
                    xinit = np.tile(xinit, (roix[1] - roix[0], 1)).T

                for ef, img, sr in zip(a["EF"], a["img"], a["data"]["sr_current"]):

                    if drop_beamdump and sr < SR_LIMIT:
                        continue

                    img = deepcopy(img[:, roix[0] : roix[1]])

                    if use_distortion_corr and self.corr_shift is not False:
                        for sh, (c1, c2) in zip(self.corr_shift, self.corr_regions):
                            img[:, c1:c2] = np.roll(img[:, c1:c2], sh, axis=0)

                    img = img[roiy[0] : roiy[1]]
                    if self.photon_counting:
                        lbl, nlbl = scipy.ndimage.label(img)
                        try:
                            yi = scipy.ndimage.labeled_comprehension(
                                img, lbl, range(1, nlbl + 1), np.sum, float, 0
                            )
                            xi = scipy.ndimage.labeled_comprehension(
                                xinit, lbl, range(1, nlbl + 1), np.mean, float, 0
                            )
                            xi = (xi - y0) * pix_to_E(ef, self.dspacing) + ef
                            xi = xi[yi > self.event_min]
                            yi = yi[yi > self.event_min]
                        except ValueError:
                            pass
                    else:
                        yi = np.sum(img, axis=1)
                        xi = (xinit - y0) * pix_to_E(ef, self.dspacing) + ef
                    x.extend(xi)
                    y.extend(yi)
            if not x:
                continue

            n = ns[0]
            a = self.runs[n]
            a["label"] = ",".join([str(ni) for ni in ns])

            x, y = np.array(x), np.array(y)
            y = y[np.argsort(x)]
            x = np.sort(x)
            if self.E0 is None:
                en = a["EI"]
            else:
                en = self.E0
            x -= en

            a["E0"] = en
            a.pop("oneshot", None)

            header = "experiment: {0}\n".format(self.exp)
            header += "run: {0}\n".format(n)
            header += "date: {}\n".format(a["date"])
            header += "command: {0}\n".format(" ".join(a["command"]))
            header += "dcm_ener: {0}\n".format(a["dcm_ener"])
            header += "rixs_ener: {0}\n".format(a["rixs_ener"])
            if "t_coldhead" in a:
                header += "t_coldhead: {0}\n".format(a["t_coldhead"])
            if "t_sample" in a:
                header += "t_sample: {0}\n".format(a["t_sample"])
            header += "rixs_th: {0}\n".format(a["rixs_th"])
            header += "rixs_chi: {0}\n".format(a["rixs_chi"])
            header += "rixs_sam_x: {0}\n".format(a["rixs_sam_x"])
            header += "rixs_sam_y: {0}\n".format(a["rixs_sam_y"])
            header += "rixs_sam_z: {0}\n".format(a["rixs_sam_z"])
            header += "q_hkl: {0:.4f} {1:.4f} {2:.4f}\n".format(
                a["qh"], a["qk"], a["ql"]
            )
            header += "det_threshold: {0}\n".format(a["threshold"])
            header += "det_cutoff: {0}\n".format(a["cutoff"])
            header += "det_factor: {0}\n".format(a["detfac"])
            header += "det_roix: {0}\n".format(roix)
            header += "det_roiy: {0}\n".format(roiy)
            header += "E0_ypixel: {0}\n".format(y0)
            header += "E0_offset: {0}\n".format(en)

            if self.photon_counting:
                savefile = "{0}/{1}_pc_{2:05d}.txt".format(
                    self.savedir_dat, self.exp, n
                )
            else:
                savefile = "{0}/{1}_{2:05d}.txt".format(self.savedir_dat, self.exp, n)
            np.savetxt(
                savefile,
                np.array([x, y]).T,
                header=header + "\n{0:>24}{1:>24}".format(a["auto"], "counts"),
            )

            y = y / self.photon_factor
            if self.photon_counting and self.max_events:
                x = np.delete(x, np.where(y > self.max_events + 0.5))
                y = np.delete(y, np.where(y > self.max_events + 0.5))
                for i in range(self.max_events, 1, -1):
                    cnts = np.where(np.logical_and(y >= i - 0.5, y < i + 0.5))
                    y[cnts] /= i
                    if i > 2:
                        y = np.append(y, np.tile(y[cnts], i - 1))
                        x = np.append(x, np.tile(x[cnts], i - 1))
                    else:
                        y = np.append(y, y[cnts])
                        x = np.append(x, x[cnts])
                y = y[np.argsort(x)]
                x = np.sort(x)
            if bins:
                x, y, e = binning(x, y, bins, self.photon_counting)
            else:
                e = np.sqrt(y)
            y[~np.isfinite(y)] = 0
            a["x"], a["y"], a["e"] = x, y, e

            if fit:
                a["xf"], a["yf"], a["p"] = peak_fit(x, y)
                report = "#{0:<4} (bin: {1})  ".format(n, bins)
                report += "cen:{0:8.4f}   ".format(a["p"][2])
                report += "amp:{0:6.2f}   ".format(a["p"][0])
                report += "fwhm:{0:6.3f}   ".format(a["p"][1] * 2)
                report += "fra:{0:4.1f}   ".format(a["p"][3])
                report += "bg:{0:6.3f}".format(a["p"][3])
                print(report)
            else:
                a["p"] = False

            header += "bin_size: {0}\n".format(bins)
            header += "\n{0:>24}{1:>24}{2:>24}".format(a["auto"], "counts", "stderr")
            if bins < 5:
                savefile = "{0}/{1}_{2:05d}_b{3:.1f}meV.txt".format(
                    self.savedir_con, self.exp, n, bins * 1000
                )
            else:
                savefile = "{0}/{1}_{2:05d}_b{3}.txt".format(
                    self.savedir_con, self.exp, n, bins
                )
            np.savetxt(savefile, np.array([x, y, e]).T, header=header)

    def plot(
        self,
        run_nos,
        ax=None,
        step="run",
        labels=None,
        sort=False,
        rev=False,
        norm=False,
        ysca=None,
        ystp=0,
        yoff=None,
        xoff=None,
        show_fit=True,
        stderr=False,
        cmap=None,
        fmt="-",
        lw=1,
        vline=[0],
        leg=0,
        title=None,
        savefig=True,
        xlim=None,
        ylim=None,
    ):

        if not isinstance(run_nos, (list, tuple, range)):
            run_nos = [run_nos]

        run_nos = [n[0] if isinstance(n, list) else n for n in run_nos]
        runs = [
            self.runs[n] for n in run_nos if n in self.runs and self.runs[n] is not None
        ]
        if not runs:
            return

        if sort:
            runs.sort(key=lambda x: x[step])
        if rev:
            runs = runs[::-1]

        if labels:
            if not isinstance(labels, (list, tuple)):
                labels = [labels]

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        if cmap:
            if isinstance(cmap, str):
                cmap = plt.get_cmap(cmap)

        for i, a in enumerate(runs):

            xf, yf, p = False, False, False
            if "x" in a:
                x, y, e = deepcopy(a["x"]), deepcopy(a["y"]), deepcopy(a["e"])
                if a["p"] is not False:
                    xf, yf, p = a["xf"], a["yf"], a["p"]
            else:
                print("#{0}: nothing to plot".format(a["run"]))
                continue

            if xoff is not None:
                if isinstance(xoff, list):
                    x += xoff[i]
                else:
                    x += xoff

            if norm:
                if norm == "time":
                    maxy = a["time"]
                if isinstance(norm, (tuple, list)):
                    maxy = np.mean(y[(x > min(norm)) & (x < max(norm))])
                else:
                    maxy = max(y)
                y = y / maxy
                if e is not False:
                    e = e / maxy
                if p is not False:
                    yf = yf / maxy

            if ysca is not None:
                if isinstance(ysca, list):
                    ys = ysca[i]
                else:
                    ys = ysca
                y *= ys
                if e is not False:
                    e *= ys
                if p is not False:
                    yf *= ys

            if yoff is not None:
                if isinstance(yoff, list):
                    yo = yoff[i]
                else:
                    yo = yoff
                y += yo
                if p is not False:
                    yf += yo

            if cmap:
                c = cmap(i / len(runs))
            else:
                c = None

            label = "#{0}".format(a["label"])
            if labels:
                label += " {0}".format(labels[i])
            elif step == "T":
                label += " {:.0f}K".format(a[step])
            elif step == "rixs_th":
                label += " th: {:.2f}".format(a[step])
            elif step == "hkl":
                label += " {:.2f} {:.2f} {:.2f}".format(a["qh"], a["qk"], a["ql"])
            elif step != "run":
                try:
                    label += " {}: {:.3f}".format(step, a[step])
                except TypeError:
                    label += " {}: {}".format(step, a[step])

            (l,) = ax.plot(x, y + i * ystp, fmt, color=c, lw=lw, label=label)

            if stderr and e is not None and not norm:
                ax.errorbar(x, y + i * ystp, e, fmt="none", color=l.get_color(), lw=lw)

            if show_fit and p is not False:
                amp, sig, cen, fra, bgnd = p

                ax.plot(xf, yf + i * ystp, color=l.get_color(), lw=0.5)

                if len(runs) == 1 and len(ax.texts) == 0:
                    fr = "amp: {:.2f}\nfwhm: {:.3f}\ncen: {:.4f}".format(
                        amp, sig * 2, cen
                    )
                    ax.text(
                        0.05,
                        0.95,
                        fr,
                        va="top",
                        linespacing=1.3,
                        transform=ax.transAxes,
                    )

        ax.minorticks_on()

        if norm == "time":
            ax.set_ylabel("Intensity (per second)")
        elif norm:
            ax.set_ylabel("Intensity (normalised)")
        else:
            ax.set_ylabel("Intensity")

        if "E0" in a:
            ax.set_xlabel("Energy Transfer (eV)")
        elif a["oneshot"]:
            ax.set_xlabel("y-pixel")
        else:
            ax.set_xlabel(a["auto"])

        if leg is not False:
            if len(runs) > 20:
                ncol = 2
            else:
                ncol = 1
            ax.legend(
                loc=leg,
                handlelength=1.5,
                labelspacing=0.3,
                handletextpad=0.5,
                ncol=ncol,
                fontsize="small",
            )

        if "E0" not in a:
            vline = False
        if vline is not False:
            if not isinstance(vline, (tuple, list)):
                vline = [vline]
            for v in vline:
                ax.axvline(v, color="k", lw=0.5)

        if title:
            ax.set_title(title)
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)

        if savefig:
            if not isinstance(savefig, str):
                sc = "_".join(str(n) for n in run_nos)
                savefig = "s{}_{}".format(sc, a["auto"])
            plt.savefig("{}/{}.pdf".format(self.savedir_fig, savefig), dpi=300)

    def check_run(
        self,
        run_no,
        hist=False,
        no=0,
        vmin=0,
        vmax=10,
        interp="hanning",
        photon_counting=False,
    ):

        """Step through each detector image of a run.
        run -- run number
        no -- starting step number (default is the first)
        hist -- plot histogram rather than detector map
        vmin -- colourmap minimum
        vmax -- colourmap maximum
        interp -- interpolation mode for image plot ('nearest' to disable)
        photon_counting -- check photon counting algorithm
        """

        self.load(run_no)
        a = self.runs[run_no]

        pnts = a["pnts"]
        to = a["threshold"] - a["detfac"]
        co = a["cutoff"] - a["detfac"]

        i = {}
        i["idx"] = no
        fig, i["ax"] = plt.subplots()

        imdat = a["img"][no]
        imgdim = imdat.shape

        if hist:
            b, c = np.histogram(imdat, bins=range(to, co, 1))
            i["ax"].bar(c[:-1], b)
        elif photon_counting:
            imdat, _ = scipy.ndimage.label(imdat)
            # cmap = plt.get_cmap('prism')
            vals = np.linspace(0, 1, 256)
            np.random.shuffle(vals)
            cmap = plt.cm.colors.ListedColormap(plt.get_cmap("jet")(vals))
            cmap.set_under("k")
            i["im"] = i["ax"].imshow(
                imdat, origin="lower", cmap=cmap, interpolation="nearest", vmin=0.1
            )
        else:
            cmap = plt.get_cmap("bone_r")
            i["im"] = i["ax"].imshow(
                imdat,
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation=interp,
            )
        i["ax"].set_title("#{} no {}".format(run_no, i["idx"]))

        def do_plot(i):
            try:
                imdat = a["img"][i["idx"]]
            except IndexError:
                imdat = np.zeros(imgdim)
                print("check run: no tiff for step {}".format(i["idx"]))
            if hist:
                b, c = np.histogram(imdat, bins=range(to, co, 1))
                i["ax"].cla()
                i["ax"].bar(c[:-1], b)
            else:
                if photon_counting:
                    imdat, _ = scipy.ndimage.label(imdat)
                i["im"].set_data(imdat)
            i["ax"].set_title("#{} no {}".format(run_no, i["idx"]))
            plt.draw()

        def press(event):
            if event.key == "left":
                if i["idx"] > 0:
                    i["idx"] -= 1
                    do_plot(i)
            elif event.key == "right":
                if i["idx"] < pnts - 1:
                    i["idx"] += 1
                    do_plot(i)

        fig.canvas.mpl_connect("key_press_event", press)

    def calc_distortion(
        self, run_no, slices=8, oneshot=True, no=0, plot=False, vmin=0, vmax=10
    ):

        self.load(run_no)
        a = self.runs[run_no]
        if "img" not in a:
            print("detector images not loaded")
            return

        if oneshot:
            img = np.atleast_3d(np.array(a["img"]))
            img = np.sum(img, axis=0)
        else:
            img = a["img"][no]
        img = img[self.roiy[0] : self.roiy[1], self.roix[0] : self.roix[1]]

        y = np.sum(img, axis=1)
        x = np.arange(self.roiy[0], self.roiy[1])
        _, _, pinit = peak_fit(x, y)
        y0 = int(round(pinit[2]))
        print("fitted y0: {}".format(y0))
        print("initial fwhm: {:.4f}".format(pinit[1] * 2))

        slice_width = img.shape[1] / slices
        shift = []
        regions = []

        for i in range(slices):
            c1, c2 = int(i * slice_width), int(i * slice_width + slice_width)
            yi = np.sum(img[:, c1:c2], axis=1)
            try:
                _, _, pi = peak_fit(x, yi)
                cen = int(round(pi[2]))
            except RuntimeError:
                cen = y0
            shift.append(y0 - cen)
            regions.append([c1, c2])

        self.corr_shift = shift
        self.corr_regions = regions

        imgcorr = deepcopy(img)
        for sh, (c1, c2) in zip(shift, regions):
            yi = np.sum(imgcorr[:, c1:c2], axis=1)
            imgcorr[:, c1:c2] = np.roll(imgcorr[:, c1:c2], sh, axis=0)

        ycorr = np.sum(imgcorr, axis=1)
        _, _, pfinal = peak_fit(x, ycorr)
        print("final fwhm: {:.4f}".format(pfinal[1] * 2))

        if plot:
            _, ax = plt.subplots(1, 3, figsize=(10, 4), constrained_layout=True)
            ax[0].plot(x, y, lw=0.5)
            ax[0].plot(x, ycorr, lw=0.5)

            ax[1].imshow(
                img,
                origin="lower",
                cmap=plt.get_cmap("bone_r"),
                aspect="auto",
                extent=(self.roix[0], self.roix[1], self.roiy[0], self.roiy[1]),
                vmin=vmin,
                vmax=vmax,
                interpolation="hanning",
            )

            ax[2].imshow(
                imgcorr,
                origin="lower",
                cmap=plt.get_cmap("bone_r"),
                aspect="auto",
                extent=(self.roix[0], self.roix[1], self.roiy[0], self.roiy[1]),
                vmin=vmin,
                vmax=vmax,
                interpolation="hanning",
            )

            ax[1].axhline(y0, color="#F012BE", lw=0.5)
            ax[2].axhline(y0, color="#F012BE", lw=0.5)
            for sh, (c1, c2) in zip(shift, regions):
                ax[1].axvline(c1 + self.roix[0], color="#F012BE", lw=0.5)
                ax[2].axvline(c1 + self.roix[0], color="#F012BE", lw=0.5)
