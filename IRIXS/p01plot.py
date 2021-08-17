#!/usr/bin/env python

"""P01-PLOT: quick plotting for experiments on P01 and P09 at DESY"""

__author__ = 'Joel Bertinshaw'
__credits__ = 'Martin Sundermann, Hlynur Gretarsson'
__copyright__ = 'Copyright 2021, Max Planck Institute for Solid State Research'
__license__ = 'GPL v3'

import os
import sys
import glob
import re
import tempfile
import subprocess
import platform

try:
    import fcntl

    nonblockread = True
except ImportError:
    nonblockread = False

import numpy as np

np.seterr(all="raise")

import matplotlib

matplotlib.use("Qt5Agg")
matplotlib.rcParams["xtick.top"] = True
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.right"] = True
matplotlib.rcParams["ytick.direction"] = "in"
matplotlib.rcParams["xtick.labelsize"] = "small"
matplotlib.rcParams["ytick.labelsize"] = "small"
matplotlib.rcParams["axes.titlesize"] = "small"
matplotlib.rcParams["axes.labelsize"] = "small"
matplotlib.rcParams["axes.formatter.useoffset"] = False

from PyQt5 import QtCore, QtWidgets, QtGui, QtPrintSupport
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import Cursor, RectangleSelector
from matplotlib.offsetbox import AnchoredText
from os.path import basename, splitext
from scipy.optimize import least_squares
from datetime import datetime as dt
from copy import deepcopy

progname = "P01-PLOT"

REMOTE = False
FOLDER = None

HIDE = [
    "enc_dcm_ener",
    "epoch",
    "timestamp",
    "qh",
    "qk",
    "ql",
    "q_h",
    "q_k",
    "q_l",
    "e4cctrl_h/position",
    "e4cctrl_k/position",
    "e4cctrl_l/position",
    "h_position",
    "k_position",
    "l_position",
    "abs_attenfactor",
    "abs_position",
    "dettimesattenfactor_counts",
    "sumvfcs_counts",
]

TEMPCNTR = [
    "lks336_outputchannela",
    "lks336_outputchannelb",
    "lks336_outputchannelc",
    "lks336_outputchanneld",
    "t_coldhead",
    "t_sample",
]

TIMECNTR = [
    "exp_t01",
    "exp_t02",
    "eh1_t01",
    "eh1_t02",
    "oh2_t01",
    "oh2_t02",
    "eh2_t01",
    "eh2_t02",
    "eh3_t01",
    "eh3_t02",
]

ENERCNTR = ["enc_dcm_ener"]

HIDE += TIMECNTR

PRECISION = 4


def open_external(path, diff=False):
    if diff:
        paramdiff = fio_param_diff(path)
        path = os.path.join(tempfile.gettempdir(), "p01plotdiff.txt")
        with open(path, "w") as f:
            f.write(paramdiff)
    if platform.system() == "Darwin":
        subprocess.Popen(("open", path))
    elif platform.system() == "Windows":
        os.startfile(path)
    else:
        subprocess.Popen(["gedit", "-s", path])


def fio_param_diff(paths):
    with open(paths[0], "r") as f:
        al = f.readlines()
    with open(paths[1], "r") as f:
        bl = f.readlines()
    out = [[basename(p) for p in paths]]
    ml = len(out[0][0])
    for a, b in zip(al, bl):
        a, b = a.strip(), b.strip()
        if a.startswith("Col"):
            break
        elif a.startswith("user"):
            continue
        elif a != b or "scan" in a:
            out.append([a, b])
            if a.split()[0] in [
                "ascan",
                "a2scan",
                "a3scan",
                "dscan",
                "d2scan",
                "d3scan",
                "hklscan",
            ]:
                out.append(["-" * len(p) for p in [a, b]])
            if len(a) > ml:
                ml = len(a)
    out = ["{0}  |  {1}".format(l[0].ljust(ml), l[1]) for l in out]
    return "\n".join(out)


def find_nearest(array, value):
    return (np.abs(array - value)).argmin()


def get_numor(s):
    return int(re.findall(r"\d+", s)[-1])


def peak(x, amp, sig, cen, bgnd, frac):
    base = np.full_like(x, bgnd)
    sigg = sig / np.sqrt(2 * np.log(2))
    try:
        g = (
            (1 - frac)
            * (amp / (sigg * np.sqrt(2 * np.pi)))
            * np.exp(-1 * (x - cen) ** 2 / (2 * sigg ** 2))
        )
    except FloatingPointError:
        g = 0
    l = (frac) * (amp / np.pi) * (sig / ((x - cen) ** 2 + sig ** 2))
    return base + g + l


def peak_stats(x, y, deriv=False):
    try:
        centre = x[find_nearest(y, np.max(y))]
        msg = "max position: ( {0:.{2}f}, {1:.0f} )   ".format(
            centre, np.max(y), PRECISION
        )
        com = np.sum(x * y) / np.sum(y)
        msg += "com: {0:.{1}f}   ".format(com, PRECISION)
        if deriv:
            bgnd = 0
            miny, maxy = np.min(y), np.max(y)
            height = miny if abs(miny) > maxy else maxy
        else:
            bgnd = np.min(y)
            height = np.max(y) - bgnd
        half = x[find_nearest(y, height / 2 + bgnd)]
        wid = np.abs(half - com) * 2
        msg += "est. width: {0:.{1}f}".format(wid, PRECISION)
        return msg
    except (ValueError, FloatingPointError):
        return ""


def peak_fit(x, y, deriv=False):

    errfunc = lambda p, x, y: peak(x, *p) - y
    bounds = ([-np.inf, 0, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf, 1])

    def init_params(x, y):
        bgnd = 0 if deriv else np.min(y)
        frac = 0.5
        height = np.max(y) - bgnd
        half = x[find_nearest(y, height / 2 + bgnd)]
        centre = x[find_nearest(y, np.max(y))]
        fwhm = np.abs(half - centre) * 2
        sigma = fwhm / 2
        amplitude = height * sigma * np.pi
        return [amplitude, sigma, centre, bgnd, frac]

    # least squares fit
    p0 = init_params(x, y)
    p = least_squares(errfunc, p0, args=(x, y), bounds=bounds)

    # try inverted peak
    p0inv = init_params(x, y * -1)
    pinv = least_squares(errfunc, p0inv, args=(x, y * -1), bounds=bounds)

    # compare residuals
    msg_pos = "top"
    if np.sum((y * -1 - peak(x, *pinv.x)) ** 2) < np.sum((y - peak(x, *p.x)) ** 2):
        p = pinv
        p.x[0] *= -1
        p.x[3] *= -1
        msg_pos = "bottom"

    msg = "amp: {0:.1f}\ncen: {1:.{3}f}\nfwhm: {2:.{3}f}".format(
        p.x[0], p.x[2], p.x[1] * 2, PRECISION
    )
    xi = np.linspace(min(x), max(x), 1000)
    yi = peak(xi, *p.x)

    return xi, yi, [msg, msg_pos]


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5.8, height=4.8, dpi=100, replot=None):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        self.replot = replot
        self.zoomed_plot = False
        FigureCanvas.setSizePolicy(
            self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        FigureCanvas.updateGeometry(self)
        self.mpl_connect("key_press_event", self.press)
        self.mpl_connect("button_press_event", self.click)
        self.mpl_connect("button_release_event", self.release)
        self.cleanup()

    def reset(self):
        self.zoomed_plot = False
        self.axes.clear()

    def cleanup(
        self, plot_count=0, xlabel=None, ylabel=None, fitmsg=None, modemsg=None
    ):

        if xlabel:
            self.axes.set_xlabel(", ".join(sorted(set(xlabel))))
        if ylabel:
            ylabel = ", ".join(sorted(set(ylabel)))
            self.axes.set_ylabel(ylabel)
        if 0 < plot_count < 3:
            self.axes.legend(
                loc="lower right",
                fontsize="small",
                handlelength=1.5,
                ncol=2,
                labelspacing=0.3,
                handletextpad=0.5,
                borderaxespad=0.1,
                frameon=False,
                borderpad=0,
                bbox_to_anchor=(1, 1),
                bbox_transform=self.axes.transAxes,
            )
        elif 0 < plot_count < 21:
            self.axes.legend(
                loc=0,
                fontsize="small",
                handlelength=1.5,
                labelspacing=0.3,
                handletextpad=0.5,
            )

        if REMOTE:
            self.axes.format_coord = lambda x, y: ""
        else:
            self.cursor = Cursor(self.axes, lw=0.5, color="0.6", useblit=True)

        self.RS = RectangleSelector(
            self.axes,
            self.rect_callback,
            button=3,
            drawtype="box",
            useblit=True,
            rectprops=dict(fc="black", alpha=0.05),
        )
        self.RS.set_active(True)

        self.txt = AnchoredText(
            "",
            loc="upper left",
            bbox_to_anchor=(-0.12, -0.07),
            bbox_transform=self.axes.transAxes,
            frameon=False,
            borderpad=0,
            prop=dict(fontsize="small"),
        )
        self.axes.add_artist(self.txt)

        if modemsg:
            modemsg = ", ".join(modemsg)
            modemsg = AnchoredText(
                modemsg,
                loc="lower left",
                bbox_to_anchor=(0, 1),
                bbox_transform=self.axes.transAxes,
                frameon=False,
                borderpad=0,
                pad=0.1,
                prop=dict(fontsize="small"),
            )
            self.axes.add_artist(modemsg)

        if fitmsg:
            if fitmsg[1] == "bottom":
                loc = "lower left"
            elif fitmsg[1] == "top":
                loc = "upper left"
            fitmsg = AnchoredText(
                fitmsg[0], loc, prop=dict(fontsize="small"), frameon=False, pad=0.8
            )
            self.axes.add_artist(fitmsg)

        self.draw_idle()

    def press(self, event):
        if event.key == "l":
            if self.axes.get_yscale() == "linear":
                self.axes.set_yscale("log", nonposy="mask")
            else:
                self.axes.set_yscale("linear")
            self.draw_idle()

    def click(self, event):
        if not event.inaxes:
            return
        if event.button == 1:
            try:
                self.v.remove()
                self.h.remove()
            except:
                pass
            self.txt.txt.set_text(
                "( {0:.{2}f}, {1:.0f} )".format(event.xdata, event.ydata, PRECISION)
            )
            self.v = self.axes.axvline(event.xdata, color="k", lw=0.5)
            self.h = self.axes.axhline(event.ydata, color="k", lw=0.5)
            self.draw_idle()
        elif event.button == 3 and not REMOTE:
            self.cursor.disconnect_events()

    def release(self, event):
        if event.button == 3 and not REMOTE:
            self.cursor.connect_event("motion_notify_event", self.cursor.onmove)
            self.cursor.connect_event("draw_event", self.cursor.clear)

    def rect_callback(self, eclick, erelease):
        if eclick.xdata != erelease.xdata:
            xlim = sorted([eclick.xdata, erelease.xdata])
            ylim = sorted([eclick.ydata, erelease.ydata])
            if xlim[1] - xlim[0] > 1e-8 and ylim[1] - ylim[0] > 1e-8:
                self.axes.set_xlim(*xlim)
                self.axes.set_ylim(*ylim)
                self.zoomed_plot = True
        else:
            self.replot()
        self.draw_idle()


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")

        self.exp_name = ""
        self.folder = FOLDER

        file_menu = QtWidgets.QMenu("File", self)
        file_menu.addAction("Filter experiments", self.select_exp_name)
        file_menu.addAction("Reload counters", self.reinit)
        file_menu.addSeparator()
        file_menu.addAction(
            "Save plot", self.save_on_click, QtCore.Qt.CTRL + QtCore.Qt.Key_S
        )
        file_menu.addAction(
            "Print plot", self.print_on_click, QtCore.Qt.CTRL + QtCore.Qt.Key_P
        )
        file_menu.addSeparator()
        file_menu.addAction("About", self.about)
        file_menu.addAction("Quit", self.fileQuit, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)

        tool_menu = QtWidgets.QMenu("Options", self)
        self.autoCheck = QtWidgets.QAction("Follow experiment", self, checkable=True)
        self.motoCheck = QtWidgets.QAction(
            "Mark original motor position", self, checkable=True
        )
        self.timeCheck = QtWidgets.QAction("Scale by count time", self, checkable=True)
        self.motoCheck.triggered.connect(lambda: self.plot())
        self.timeCheck.triggered.connect(lambda: self.plot(dofit=False, deriv=False))
        tool_menu.addAction(self.autoCheck)
        tool_menu.addAction(self.motoCheck)
        tool_menu.addSeparator()
        tool_menu.addAction(self.timeCheck)

        self.menuBar().addMenu(file_menu)
        self.menuBar().addMenu(tool_menu)

        self.main_widget = QtWidgets.QWidget(self)
        l = QtWidgets.QHBoxLayout(self.main_widget)
        v = QtWidgets.QVBoxLayout()
        h1 = QtWidgets.QHBoxLayout()
        h2 = QtWidgets.QHBoxLayout()
        h3 = QtWidgets.QHBoxLayout()
        width = 202

        self.selectScans = QtWidgets.QListWidget()
        self.selectScans.setFixedWidth(width)
        self.selectScans.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.selectScans.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.selectScans.itemSelectionChanged.connect(
            lambda: self.plot(dofit=False, deriv=False, norm=False)
        )
        self.selectScans.customContextMenuRequested.connect(self.scansRightClick)

        self.selectCounters = QtWidgets.QListView()
        self.selectCounters.setFixedSize(width, 100)
        self.counters = QtGui.QStandardItemModel(self.selectCounters)
        self.counters.itemChanged.connect(
            lambda: self.plot(dofit=False, deriv=False, norm=False)
        )

        self.xCombo = QtWidgets.QComboBox(self)
        self.xCombo.addItem("Primary motor", "cstp")
        self.xCombo.addItem("Step number", "xstp")
        self.xCombo.addItem("Elapsed time", "tstp")
        self.xCombo.addItem("Incident energy", "estp")
        self.xCombo.activated.connect(self.pick_xaxis)

        self.plotButton = QtWidgets.QPushButton("Reset")
        self.plotButton.clicked.connect(
            lambda: self.plot(deriv=False, norm=False, dofit=False)
        )
        self.plotButton.setEnabled(False)

        self.deriButton = QtWidgets.QPushButton("Derivative")
        self.deriButton.clicked.connect(lambda: self.plot(deriv=True, norm=None))
        self.deriButton.setEnabled(False)

        self.normButton = QtWidgets.QPushButton("Minmax")
        self.normButton.clicked.connect(lambda: self.plot(norm=True, deriv=None))
        self.normButton.setEnabled(False)

        self.fitButton = QtWidgets.QPushButton("Fit")
        self.fitButton.clicked.connect(
            lambda: self.plot(dofit=True, norm=None, deriv=None)
        )
        self.fitButton.setEnabled(False)

        self.mplCanvas = MplCanvas(self.main_widget, replot=self.plot)
        self.mplCanvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.mplCanvas.setFocus()

        h1.addWidget(self.xCombo)
        h2.addWidget(self.normButton)
        h2.addWidget(self.deriButton)
        h3.addWidget(self.plotButton)
        h3.addWidget(self.fitButton)
        v.addWidget(self.selectScans)
        v.addWidget(self.selectCounters)
        v.addLayout(h1)
        v.addLayout(h2)
        v.addLayout(h3)
        l.addLayout(v)
        l.addWidget(self.mplCanvas)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.status_message = QtWidgets.QLabel()
        self.statusBar().addWidget(self.status_message)

        self.initialise()

        def check_changes():
            try:
                self.plot(auto=True)
            finally:
                QtCore.QTimer.singleShot(2000, check_changes)

        QtCore.QTimer.singleShot(2000, check_changes)

    def initialise(self):
        self.fit_on = False
        self.deriv_on = False
        self.norm_on = False
        self.loaded = {}
        self.fio_counters = []
        self.flist = []
        self.ban_list = []
        self.pop_block = False
        self.plotted_scans = {}
        self.epoch_init = None
        self.xaxis = "cstp"

        self.populate()
        QtCore.QTimer.singleShot(500, lambda: self.selectScans.scrollToBottom())

        for counter in sorted(self.fio_counters):
            item = QtGui.QStandardItem(counter)
            item.setCheckable(True)
            self.counters.appendRow(item)
        self.selectCounters.setModel(self.counters)

        self.status_message.setText(" select a scan...")

    def populate(self):
        if self.pop_block:
            return
        self.pop_block = True
        fiotxt = "{}*.fio".format(self.exp_name)
        flist = glob.glob(os.path.join(self.folder, fiotxt))
        if nonblockread:
            flist.sort(key=lambda x: os.path.getmtime(x))
        else:
            flist.sort()

        if not flist:
            print("no .fio files found in {}\n".format(self.folder))
            print(HELP)
            sys.exit(2)

        for b in self.ban_list:
            try:
                flist.remove(b)
            except ValueError:
                pass
        pop_list = list(set(flist).difference(self.flist))
        if nonblockread:
            pop_list.sort(key=lambda x: os.path.getmtime(x))
        else:
            pop_list.sort()
        latest = flist[-1]
        item = False
        for i, f in enumerate(pop_list):
            n, a = self.load_fio(f)
            if not n:
                flist.remove(f)
                if f != latest:
                    self.ban_list.append(f)
                continue
            title = "{}  –  {}".format(n, a)
            item = QtWidgets.QListWidgetItem(title)
            item.setData(QtCore.Qt.UserRole, f)
            self.selectScans.addItem(item)
            if self.autoCheck.isChecked():
                if i == len(pop_list) - 1:
                    self.selectScans.selectionModel().clear()
                    self.selectScans.setCurrentItem(item)
        if item:
            rect = self.selectScans.viewport().contentsRect()
            bottom = self.selectScans.indexAt(rect.bottomLeft())
            if self.selectScans.item(bottom.row()) in [None, item]:
                self.selectScans.scrollToBottom()
        self.flist = flist
        self.pop_block = False

    def load_fio(self, path):
        head = []
        data = []
        complete = False
        with open(path) as f:

            if nonblockread:
                fd = f.fileno()
                flag = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flag | os.O_NONBLOCK)
                flag = fcntl.fcntl(fd, fcntl.F_GETFL)

            filename = splitext(basename(path))[0]
            numor = get_numor(filename)
            a = {}
            for line in f:
                if line[:-1] == "%c":
                    scantype, _ = next(f).split(maxsplit=1)
                    _, date = next(f).split(" started at ", maxsplit=1)
                    try:
                        date = dt.strptime(date[:-1], "%a %b %d %H:%M:%S %Y")
                        date = date.timestamp()
                    except:
                        pass
                elif line.startswith("%p"):
                    break
            for line in f:
                if line[:-1] == "!":
                    break
                else:
                    p, v = line.split(" = ")
                    try:
                        a[p] = np.float(v[:-1])
                    except:
                        a[p] = v[:-1]
            for line in f:
                if line[:-1] == "%d":
                    break
            for line in f:
                if line.startswith("! Acquisition ended"):
                    complete = True
                    break
                elif not line[:-1]:
                    break
                l = line[:-1].split()
                if l[0] == "Col":
                    head.append(l[2])
                else:
                    data.append(l)

        if head and data:
            try:
                data = np.asarray(data, dtype=np.float)
            except ValueError:
                try:
                    data = np.asarray(data[:-1], dtype=np.float)
                except ValueError:
                    return False, False

            pnts = data.shape[0]
            data = data.view(dtype=[(n, np.float) for n in head])
            data = data.reshape(pnts)
            a["data"] = data
            a["auto"] = head[0]
            a["pnts"] = pnts
            a["scantype"] = scantype
            a["numor"] = numor
            a["path"] = path
            a["expname"] = filename[:-6]
            a["complete"] = complete
            a["date"] = date

            self.loaded[path] = a

            if scantype in ["a3scan", "d3scan", "hklscan"]:
                counters = [c for c in head[3:] if c not in HIDE]
                motors = ", ".join(head[:3]).replace("e6cctrl_", "q")
            elif scantype in ["a2scan", "d2scan"]:
                counters = [c for c in head[2:] if c not in HIDE]
                motors = ", ".join(head[:2]).replace("e6cctrl_", "q")
            else:
                counters = [c for c in head[1:] if c not in HIDE]
                motors = head[0].replace("e6cctrl_", "q")
            self.fio_counters = list(set(counters + self.fio_counters))
            return numor, motors

        return False, False

    def plot(self, dofit=None, auto=False, deriv=None, norm=None):

        self.populate()

        scans = []
        changed = False
        for s in self.selectScans.selectedItems():
            scans.append(s.data(QtCore.Qt.UserRole))
        for no in scans:
            if not self.loaded[no]["complete"]:
                self.load_fio(no)
                changed = True
        if auto:
            if changed:
                self.update_only(scans)
            return

        counter_count = 0
        for idx in range(self.counters.rowCount()):
            item = self.counters.item(idx)
            if item.checkState():
                counter_count += 1
        if not counter_count:
            self.mplCanvas.reset()
            self.mplCanvas.draw_idle()
            self.status_message.setText(" select a counter...")
            return

        if dofit is True:
            self.fit_on = True
        elif dofit is False:
            self.fit_on = False
        if deriv is True:
            self.deriv_on = True
        elif deriv is False:
            self.deriv_on = False
        if norm is True:
            self.norm_on = True
        elif norm is False:
            self.norm_on = False

        self.mplCanvas.reset()
        ax = self.mplCanvas.axes

        self.plotted_scans = {}
        self.epoch_init = None
        plot_count = 0
        counter_found = True
        fitmsg = None
        xlabel = []
        ylabel = []
        ctime = []
        for no in scans:
            l = self.loaded[no]
            motor = l["auto"]
            self.plotted_scans[no] = {}

            for idx in range(self.counters.rowCount()):
                item = self.counters.item(idx)
                counter = item.text()
                if counter in l["data"].dtype.names:
                    if item.checkState():

                        x, y, xi, yi, fitmsg, ps, tc = self.gen_xy(l, motor, counter)
                        ctime.append(tc)

                        if counter_count > 1:
                            lab = "#{} {}".format(l["numor"], counter)
                        else:
                            lab = "#{}".format(l["numor"])

                        (d,) = ax.plot(
                            x, y, ".-", mfc="w", lw=1, mew=1, ms=4, label=lab
                        )

                        if self.fit_on:
                            ax.plot(xi, yi, "--", lw=0.5, color=d.get_color())

                        if self.xaxis == "tstp":
                            xlabel.append("Elapsed Time (seconds)")
                        elif self.xaxis == "xstp":
                            xlabel.append("step")
                        elif self.xaxis == "estp":
                            xlabel.append("Incident Energy (eV)")
                        else:
                            xlabel.append(motor)
                        ylabel.append(counter)
                        self.plotted_scans[no][counter] = d
                        plot_count += 1

                elif item.checkState():
                    counter_found = False

        if plot_count == 0:
            if not scans:
                self.status_message.setText(" select a scan...")
            elif not counter_found:
                self.status_message.setText(" counter(s) not in .fio...")
        elif plot_count == 1:
            self.status_message.setText(" " + ps)
        else:
            self.status_message.setText("")

        if plot_count == 1:
            self.fitButton.setEnabled(True)
        else:
            self.fitButton.setEnabled(False)

        if plot_count > 0 and not self.deriv_on:
            self.deriButton.setEnabled(True)
        else:
            self.deriButton.setEnabled(False)

        if not self.norm_on:
            self.normButton.setEnabled(True)
        else:
            self.normButton.setEnabled(False)

        if self.fit_on or self.norm_on or self.deriv_on:
            self.plotButton.setEnabled(True)
        else:
            self.plotButton.setEnabled(False)

        if (
            self.motoCheck.isChecked()
            and plot_count > 0
            and len(scans) == 1
            and self.xaxis == "cstp"
            and motor in l
            and "_dmy" not in motor
        ):
            ax.axvline(l[motor], lw=0.5, color="0.7")

        ct = list(set(ctime))
        if len(ct) == 1 and not self.norm_on and not self.deriv_on:
            modemsg = ["{}s".format(ct[0])]
        else:
            modemsg = []
        if self.deriv_on:
            modemsg += ["Derivative"]
        if self.norm_on:
            modemsg += ["Normalised"]
        self.mplCanvas.cleanup(plot_count, xlabel, ylabel, fitmsg, modemsg)

    def update_only(self, scans):
        plot_count = 0
        for no in scans:
            l = self.loaded[no]
            motor = l["auto"]
            for idx in range(self.counters.rowCount()):
                item = self.counters.item(idx)
                counter = item.text()
                if counter in l["data"].dtype.names:
                    if item.checkState():
                        x, y, _, _, _, ps, _ = self.gen_xy(l, motor, counter)
                        self.plotted_scans[no][counter].set_data(x, y)
                        plot_count += 1
        if not plot_count:
            return
        if not self.mplCanvas.zoomed_plot:
            self.mplCanvas.axes.relim(visible_only=True)
            self.mplCanvas.axes.autoscale()
        self.mplCanvas.draw_idle()
        if plot_count == 1:
            self.status_message.setText(" " + ps)

    def gen_xy(self, l, motor, counter):
        y = deepcopy(l["data"][counter])
        if self.xaxis == "cstp":
            x = deepcopy(l["data"][motor])
        elif self.xaxis == "tstp":
            try:
                x = deepcopy(l["data"]["epoch"])
            except ValueError:
                x = deepcopy(l["data"]["timestamp"]) + l["date"]
            if not self.epoch_init:
                self.epoch_init = x[0]
            x -= self.epoch_init
        elif self.xaxis == "xstp":
            x = np.arange(len(y)) + 1
        elif self.xaxis == "estp":
            for EC in ENERCNTR:
                if EC in l["data"].dtype.names:
                    x = deepcopy(l["data"][EC])
        if self.deriv_on:
            y = y[np.argsort(x)]
            x = x[np.argsort(x)]
            y = np.diff(y)
            x = x[:-1]
        xi, yi, fitmsg = None, None, None
        if self.fit_on:
            try:
                xi, yi, fitmsg = peak_fit(x, y, self.deriv_on)
            except:
                self.fit_on = False
        if self.norm_on:
            try:
                if self.fit_on:
                    yi = (yi - min(y)) / (max(y) - min(y))
                y = (y - min(y)) / (max(y) - min(y))
            except:
                pass
        count_time = None
        for tc in TIMECNTR:
            if tc in l["data"].dtype.names:
                count_time = l["data"][tc][0]
                break
        if self.timeCheck.isChecked() and counter not in TEMPCNTR and not self.norm_on:
            if count_time:
                y /= count_time
                if self.fit_on:
                    yi /= count_time
                count_time = 1.0
            else:
                self.timeCheck.setChecked(False)
        ps = peak_stats(x, y, self.deriv_on)
        return x, y, xi, yi, fitmsg, ps, count_time

    def pick_xaxis(self):
        self.xaxis = self.xCombo.itemData(self.xCombo.currentIndex())
        self.plot(dofit=False, deriv=False, norm=False)

    def save_on_click(self):
        f, t = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save plot", "", "*.pdf;;*.png;;*"
        )
        if f:
            figname, figext = os.path.splitext(f)
            if not figext and t != "*":
                f = figname + t[1:]
            figsize = self.mplCanvas.fig.get_size_inches()
            self.mplCanvas.fig.set_size_inches(5.5, 4.5)
            self.mplCanvas.fig.savefig(f, dpi=150)
            self.mplCanvas.fig.set_size_inches(*figsize)
            self.mplCanvas.draw()

    def print_on_click(self):
        printer = QtPrintSupport.QPrinter(QtPrintSupport.QPrinter.HighResolution)
        dialog = QtPrintSupport.QPrintDialog(printer, self)
        if dialog.exec_() == QtPrintSupport.QPrintDialog.Accepted:
            painter = QtGui.QPainter(printer)
            figfile = os.path.join(tempfile.gettempdir(), "p01plot.png")
            figsize = self.mplCanvas.fig.get_size_inches()
            self.mplCanvas.fig.set_size_inches(5.5, 4.5)
            self.mplCanvas.fig.savefig(figfile, dpi=900)
            self.mplCanvas.fig.set_size_inches(*figsize)
            self.mplCanvas.draw()
            image = QtGui.QImage(figfile)
            painter.drawImage(0, 0, image)
            painter.end()

    def scansRightClick(self, QPos):
        self.listMenu = QtWidgets.QMenu()
        menu_openfio = self.listMenu.addAction("Open .fio")
        menu_diffcheck = self.listMenu.addAction("Param. diff")
        if len(self.selectScans.selectedItems()) == 0:
            menu_openfio.setDisabled(True)
        if len(self.selectScans.selectedItems()) != 2:
            menu_diffcheck.setDisabled(True)
        menu_openfio.triggered.connect(self.openFioClicked)
        menu_diffcheck.triggered.connect(self.diffcheckClicked)
        parentPosition = self.selectScans.mapToGlobal(QtCore.QPoint(0, 0))
        self.listMenu.move(parentPosition + QPos)
        self.listMenu.show()

    def openFioClicked(self):
        for s in self.selectScans.selectedItems():
            no = s.data(QtCore.Qt.UserRole)
            open_external(self.loaded[no]["path"])

    def diffcheckClicked(self):
        fiofiles = []
        for s in self.selectScans.selectedItems():
            no = s.data(QtCore.Qt.UserRole)
            fiofiles.append(self.loaded[no]["path"])
        open_external(sorted(fiofiles), diff=True)

    def select_exp_name(self):
        self.pop_block = True
        title = "Filter Experiments"
        header = "Select experiment (filename prefix)" + " " * 25

        flist = glob.glob(os.path.join(self.folder, "*.fio"))
        flist = [os.path.basename(f).rsplit("_", 1)[0] for f in flist]
        flist = [""] + list(set(flist))
        idx = flist.index(self.exp_name)
        flist[0] = "All experiments"
        txt, ok = QtWidgets.QInputDialog.getItem(self, title, header, flist, idx, False)
        self.pop_block = False
        if ok:
            self.exp_name = txt if txt != "All experiments" else ""
            self.reinit()

    def reinit(self):
        self.counters.removeRows(0, self.counters.rowCount())
        self.selectScans.clear()
        self.initialise()

    def fileQuit(self):
        self.close()

    def about(self):
        pixmap = QtGui.QPixmap()
        pixmap.loadFromData(QtCore.QByteArray.fromBase64(ICON))
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowIcon(QtGui.QIcon(pixmap))
        msg.setIconPixmap(pixmap)
        msg.setText(ABOUTMSG)
        msg.setWindowTitle("About")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.show()


HELP = """
p01plot [directory] [--remote -r] [--help -h]
directory : location to look for .fio data files
            defaults to /gpfs/current/raw, then current directory
--remote : remove cursor to speed up remote connections
--help : show this menu
"""

ABOUTMSG = """<h3>{0}</h3>
<p><i>Quick plotting for experiments on P01 (and P09) at DESY.</i></p>
<p>Created by {1}<br>Thanks to {2}</p><p>{3}</p><p><pre>{4}</pre></p>
""".format(progname, __author__, __credits__, __copyright__, HELP)


ICON = b"iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAYAAADDPmHLAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAG5wAABucBl0YRPQAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAABJNSURBVHja7Z1peFPHuccT2vT2ts1t8/T2udvT5waDwZDg3cbGsi3Llixbi+UF7wYbG0KTkIUkkKRJ02yQ0pvm3oTctCE0S5twS0JKNpomDZYN3vEmvCDJlvd9AWIggMG6M/KReyykmXOkI/kcaT78v9jnzDln3p9m3nnnnZmbzGbzTUS+K1IJBABSCQQAIgIAEQGAiABARADwAa1bHv9DoDVA0UB+QP9IAPAuA98GlAX0ENCLQIeAKoCMQBeBzHY0DdQG9AXQW0B7gO4FSgK6hQDAf6P/B9A9QF8CzTowsrM6B/QuBdUPCAD8MTpsxh8Dqgea49jojvQN0CdApUA/IQB43ujfAioDOuMhg6N0HegroFgCgGeMrwRq54Hh7ekjoAACgHsMHwFUzlPD03UN6LdA/0oA4MbwyykPfk4AxqfrAtAvheAw8tXwy4CeBbrCpWEC/cRT4QFSQ3yE5lS6vPjE5uwdWqWkoEoUomoOXZXUQxmOSxBGgNIIAOyMfyvQxy5W/GxsqLrpqcd/XXHss+O65taOgaHR8cvj02fNOPUODM/U1Tebjrx/rGl7ySPlYQFSo4vvAluvxwgAzIx/O5DOyYq+KBVl1770X6+fNPb0TzMxNlOdajzdu/vBZ7XRgak6yvN35v3+APQPBADHxhcBjbOt2A1Bita3Dh6uHRgZu8Sl0R2p02gaf+bJlyqCV0qGnICgGuhfCAA3Gr+YbX8fujqpGxreE0a3p8GR8W92PfCMFvgV51lC0AcURACYN/zNQPvYVGDQioTx55/+n8qRiclZdxq4u6f/7Lt//HP90aNftPQODM04us5o6pvalH2vliXAF/jiHC41AM+yqLRL9259rLxvcPiCu3/dz/7iJThxdGlh9LA8/puyooe0g8OjDruZpua2/hRxfjUbRxVI7LMAgI/PYVpZwSsThsvLazo90bQfPvxpo6P3kAEnE3f/vj2vVrJoDSbhdLTPAQA+Ooz+C0Npa/HDPW2dxklP9e3iSE0D6n0+BcNKXBmffPy31sAV4kmGEMAp6Ft9BgDwsf8GNMikch7fvbejprHVPDY17THnLtQ/sRf1To/c97SWSTkw9gCDTgwhgLOLy7weADgOBqplMsv22qtvd1SfajE36jo86t3DKCHq3fY883Il07Kgv5K0IauWIQQv+AIAf2ASOfvgyLGuqoZmM1SbocujAMChpcNQ8grxlK5NP2S9dmR8sn/f3v89IY/P689IKe768Mjn5eDvi5xU0HrNZaaWVjKEoMBrAaDSq7CV8Lvf/tFgNT5UV++Ax8f4j+96QbvOT2ybOjZ75INjTQtxgLGJ7rjwtBv6+VzNXZ1jk1MT9PKGxyeuxgQrWxgmm9zpdQCAj/oR0BSuAvbuebWXbnyowdHxJQn0tLUbR2B/r5ZtOvnko/u0LaBPp///yd37ahx9hyqxqGdkYnKIfr2hu3cqeKWEie9zzBsBeAH34bkZ2/tsjQ99AE86gGwUF6GZQH1PhrzEODZ9dlEAqaKyTs9wxlHsNQBQyZrIIV9ogHQKGHvWFgBPO4DsnMV07LzF7p3PwbjBHP2+A797r4ZBfkOdNwHwBi4i9pW2ZszW+EvhALLR3udeOcnEp/m/946W2957d+luJtlNWYIHgMravYb60Afv+UUTbOrtAbAUDiBTjU5NjSfH5Q4wgaCppb3a1ikM8U/sw9wHu4tvCx2Ao+gQr+RibWPrrD3jL6UDyFRwGJgizu/HARCxRnZxYHhUT7/3tf1vM5k32C5YAMDLb8B94IEDh3ocGZ/PDuCilmByelQmysbmBuRn/KwDXH+dHh+IXJvcySCl7PtCBeAw6uPk8bnnHRmf7w6grfqHRjrhrxwHwcmqhkVRxA/e/6yRQSuwRXAAgJf+DtDXqA97/8ixIRQA7Tx2AO2psbkN26TDwBH45Z9fNJwMS8NB8JEQAZCjPip8jWy4prHFjAKAzw6gI737zodY7x5eQ7/n+PHqTsywEA6hvyc0AF5DVcIrL7/VhjK+EBxAB7qmkW3qRn27KEw9NT41vShDGaalY8BJEwwAVJoX0imqrm++gDK+UBxAZ7uCurqWk/R7noBzD+h7fi8kACJQHxMfoelzNO5fcABPdwjS+JTmMlO3INcSFG28t802dwADwLi78gXcAcBzyOb/lbeMuOZfaA6grVp1nTW4VsA2LgBGEXrMPSKhAHAa9SGVtacu4QAQogNo2wrAaWFUPXz+14pFmUX3lD2KcyB/zXsAqIkfxNg/bxJnfKjeoRGhA2BuazcgM4F+vuuFGvr1MOkVA4BOCAAgo397n99vYAbAsOABgJ5+6Oqky47qImF9+qjtPZjVRl8LAYAsFAB//uiLISYA9Ax6AQBAZUU7W1H1AecS6Ncnx+bifIcf8B2A+1AfcPxE3ddMADANDHkFAO+9exTZr59u0y9aZ1CQeXcFBoBVfAdgL+oDquqbrjEBoE3go4CF1UIt7ciYQH2DblE84OH7nsY5gvF8B+BtRy8feUeymYnxrYGg0YlJwQPQcaa7HmXQisraRZNDcGk7BoA8vgPwpaOXF0emzzIFAOpMd6/gATCa+pAh3r98rtWynB18iO8AONzBSy3bPM4GgNrGVnP/yKigAegZGGpDGZRaR7BwfVVNYxcGgBf5DsBZRy+/Of+BHjYAQMEZQyHHBKCXj5wS/9Nn5bZL0jEAHOItAHA/XWTu346nDGwBsPoDXX0DQp0cuha1LsVhXkSnvrvB9h7w96uIevyKzwB8FwXAww884xQAf28NWs26M0ZLjGBUQDAcO3bc7kxfce59p8H/Z+0AcBlRj1qfBcCej3BK125uaT9jGTbqTb3m7v5Bc9/wKMy65VWL8cnHX8INps5TW9XNwTCw7fIxAoAbVNess+QW6joN5g6DyWzo6TP39A9ZnMsRzw8zrw2MjnfZpoQRAJZQ0L+ob2kzN7d1mk+D7qWzq8ds7O23zEUAYy1JV0MA4JlgV9PQSnU1+vmupquP6mrGuO9qCAACVF3TfFfT2kF1NSaqqxlm39V4LQAaeXHfm28e7vnNb94wPvHYr/Tby3YbSzft7PIFbcrbYVAlFbXHhqlbYkKUNwj8vQnucqpJLj6BWUonXACIOBGvAbiDGMjt+prahv4nvAEAvMwKahbwGjGQx3SJWnuxcskAgKdiAB10wwldROzOLNrvSqaQs8ZXObOzN5HbBPc2lLkdALhGDbfsi2hJ9Xu2p6CyMf6/U9uakormt2Bi6Y85BQAU+J9A3U6d07Mi4WJUhNIgkeYMJinyp6XKgosyVeGsVFUwJ1UVmq2SazZfS8vZdtkXpc7eOqPQlPTJUot0EklOrShaow0JSKpHHHOLE1xltJwTAEBB/kD9rIzuJ/5GJEo3AIOfnzdwASOpssvMmry7iChBOJJk+Y0R6+QV1CaSbA+sCnQJAFDAT4GGmf/axRfi4jO7ZOqC60yNTleyusiclreNGN+OVJml4+tDUrUsTzYbxJ1hiNvYuY7pw2Ki1HqZunDOGcPTlZpR4sGK3TbreWO69kzVxrLpsDuTT7CAoBZ1WBUKgNcZHuFyITElb9xZg4N7x8IDk02w2wj2T5yJCleaZCn5OncaQRSTURG4MqFz3XIxPNjhbMiqxMZk5aZ2tzXluduux0SlaYNWiGHffDXQL34qZHVSfYp6s8HZMsXijVW4bXhoeocVANTJ2NiCAYm9lEPnlPFjYtIc7Kcvvi6K0VS4wRhzoQFSR8uvZuNEGZXu+MUHr5I4SPcWX0lI2FjtbNmK9JJ+4GTrGUJwFyMAqAMdZrCndt0pA8YvcLrJT0rJm0A/Q3wlRVWk59IYsTHpmKVX4oupmuJeLp8ZE6HGrfY5p8ooHXGhS5hiCMG0vTkEewC8iTX+HdJ+qbrwuit9PWj2scNK4P1y+otc5yfuwT0TOlqcPpNBxBRC4pJfkFU2CRzwLgYQHEQCAC4Ix21kHLxaMgaa/euuOnugz8eeGRTsL+GsX1ZvLDvLpKkM9k9o4cxzzygZZfLMkIDEWi5GCQBw3IaV0LbRKACqcC87P7Z3zfgwNrBuhRjbzUQGybtU2aXcAJCz9QL0LbDGAA4hh9BNMQEgdI20movnSZPzmxjsRF5lFwDwj/W4FxXFphtdN/68IoBxcc+LjcswyjiMDYBWB3sQdFSospzjLmAYP4RO46zbiQxK0TKALtQeAAfQzVTSsFTNjfEtUhScQwaVVoovWEcYqRnFnFQONXRym0PmjOMJhoRjqqzSaQ6jh5cC8b7OwUUAwPlknOcvSc4b4cz4lOITsrrtNlkrxJcSU/In6NeCJpyTCgqfD6ve+I1+8RdhHN4dQ8+wtTJHy77PJcnyGrl+Znx85kkGCSW30QFAjvuDVkkm2MT0WbYEM1GRKn3I6qTR0LXSgZgYjR486+oNYWJNkVmTy00FJUpz60MDkmqB0UdgtxBxp7xSoSnpd2fwSSLJrgldnVgPnjcKh20RgckV0El0U+DpKnAIRzAQ3E8H4BPUxeLE7D63GJ+l5JpN8OPI3ACT+EOkGucLWA6nsm7tOu3wQj/xVSkHMX6uQUgBfkFqJlSJWZG1xazcWAqDIpYZRTUU6DIgLGm5vgkAGIGcw6w0hmsVl2GzeYG33s0n4zsrmbrQnJxWBLTJApBcs9mckr7Z4mDCCSiFFSQgFYBpAaKcbVSrI7yWJ8hf0oppBUIgANuQe/smZJm8AQCulGwBqdAsBzBZWiIAkaU1olokCJGCgki5ABFsjYAT6+HuKzpMiQtD348N/XIT+CFaJACRLG2+RbK0RumbKZCoFmmhNSqd79qobi2N6tYYB4bkhbjTSt+BAHyODP7EZXTFiTNNRPxSvDizKz4uQx8nSm+PjdHoRNFpTTHr1Q0bIpQ10eGKk9GhCi2DoNBfIAANJJHSZ9Vwk7PJnkReIRNyZy8i719reBPLJEMi79I0BOAEqQif1V8hADJSET67sDTKOheQAzRBKsVnBCeKVLb5AMuosLDYSSFbkohgebdUWThjXwUzUkXBWamicEyqyB+QphSYgM4A6aTyghZfEiZ+3+KCfayCNr7Z41vEREeoDM5GzeBUsDVur8ikJn5YRsWEIq/dJMppAHCTPCoYVp0Pqc4DUgIA2bIwmSO0SRwCgFtmAv8ec4eQKLNKFmLsliljAoB3A8B8lo/W1WTNtySWiZicbZxlIxEABJ1TUDA/g0dNAVtzCCz5AzllnPgkQgbgZlRiKcwrFInSjV6vGI1+Q5RaFxWmOBUZlFIFVzcxFbXqF5XXf4TvJ4aQqKJ79XO+A/DfxEhulZzvAMQxWJpE5Hz07lYhnB6+nxjLLVIK5fh4uJdgJzEYp3rdHcZ3CwAUBP9EbVpIjOea4DZxO+ixe0EAQANBAfQnknbGSnAyqJnao8nfnfZxOwA2MNwGFEBpJ6oSHiouKD/zt+dNVkXdkdiOuP4KrVyrTI43tRKP0st++cntlbjTOu2U73AjhrBVCd308vc9shV3IvhTtHJXoXb0EjQANjDkoirlxUe3Vc6Z9putig2SovLbL9sp/4zDZe4rEwbpZX96YNcpjIEetVN+r8NNLdYk6unlv//KTtxWe9uXwgZLDUAGqlJ+uaNES6/E9WsTO1CJjXbK1zlsAfzEk9e7989Zyz700oM1GAPttFO+wSFg/gl99Hc/uGcHbqn2Fl8EYCVyo6a1iZ1WI4FmtBsTW6i2U/57qPI/e2PXKauBkqNTazEGSrJT/ofIY+EPPdFqLV8SLsetu4j0RQDQK5Lnf0n9afGqKgYbJu+3U/7DmHuupMYoaiICJAYGmyr9yE75T2Duu6yMVVSHr5bgtsGBh2181+cAoCrxC4685hI7ZSdwVLbBwbuncFR+61Iaf6kBuJuDCoRrGn5qp+zvs9nkGqF9Dt79hxydmPK0LwOwjIPZwx2YY21cKRvuKPY9RPkbXSz/NNB3fBYAmjPo7KEIWlyEDC5/diFnXsTg/Q87Wf4sfas2nwWAtiZhhmUFwmHe7QyDTxVOROIeYPju/wxHISzLhwc/lPHB+LwAgKrI24HKGVQePJvweTZNJzXiuJ/aGg1XPgw4BTnRlT2CSeWi792/mi/G5w0ANEOVUHMHJptfZCN1WlmEC+WvohJWqm1ggGHdoxQkt7hQ/hqgl6lDm+jHu8Djdo4A/QzoW3wyPq8AsFOhP4abGLkjNg7K/DY8Twduje+md78FtiS441oIAEQEACICABEBgIgAQEQAICIAEHlY/w9UFWeMlK+HNQAAAABJRU5ErkJggg=="


def main():

    global REMOTE
    global FOLDER

    for arg in sys.argv:
        if arg in ["--help", "-h"]:
            print(HELP)
            sys.exit()
        if arg in ["--remote", "-r"]:
            REMOTE = True
    for arg in ["--remote", "-r", "--help", "-h"]:
        if arg in sys.argv:
            sys.argv.remove(arg)

    try:
        FOLDER = os.path.abspath(sys.argv[1])
        if not os.path.isdir(FOLDER):
            print("cannot find {}\n".format(FOLDER))
            print(HELP)
            sys.exit()
    except IndexError:
        if os.path.isdir("/gpfs/current/raw"):
            FOLDER = "/gpfs/current/raw"
        else:
            FOLDER = os.getcwd()

    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.setWindowTitle(progname + " - " + FOLDER)

    pixmap = QtGui.QPixmap()
    pixmap.loadFromData(QtCore.QByteArray.fromBase64(ICON))
    aw.setWindowIcon(QtGui.QIcon(pixmap))

    aw.show()
    sys.exit(qApp.exec_())


if __name__ == "__main__":
    main()
