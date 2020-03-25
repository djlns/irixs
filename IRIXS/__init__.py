import os, sys, shutil
import numpy as np
import matplotlib.pyplot as plt
import warnings
import scipy.ndimage

from numpy import pi,sin,cos,tan,arccos,arcsin,arctan,sqrt,log,radians,degrees
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy
from glob import iglob

try:
    from tifffile import imread
except ImportError:
    from matplotlib.pyplot import imread

#### Default Variables ########################################################

defs = {
'datdir': '/gpfs/current/raw',

'savedir_dat': './dat',  #reduced routine savedir
'savedir_det': './det',  #detector routine savedir
'savedir_fig': './fig',  #plot routine savedir
'savedir_raw': './raw',  #local storarge of raw data, False to disable
'savedir_con': './binned',  #binned data

'fit_pv': True,  #pseudovoight or lorentzian
}

PIX_EN_CONV = 13.5E-6

#### Matplotlib Settings ######################################################

plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['font.size'] = 8
plt.rcParams['axes.titlesize'] = 'medium'
plt.rcParams['figure.titlesize'] = 'medium'

#### Helper Routines ##########################################################

def calc_dspacing(hkl, cell):
    h, k, l = hkl
    a, b, c, al, be, ga = cell
    al, be, ga = radians(al), radians(be), radians(ga)
    S11 = b**2 * c**2 * sin(al)**2
    S22 = a**2 * c**2 * sin(be)**2
    S33 = a**2 * b**2 * sin(ga)**2
    S12 = a*b*c**2 * (cos(al) * cos(be) - cos(ga))
    S23 = b*c*a**2 * (cos(be) * cos(ga) - cos(al))
    S13 = a*c*b**2 * (cos(ga) * cos(al) - cos(be))
    V = a*b*c*sqrt(1-cos(al)**2-cos(be)**2-cos(ga)**2-2*cos(al)*cos(be)*cos(ga))
    invD2 = (S11*h**2 + S22*k**2 + S33*l**2 + 2*S12*h*k + 2*S23*k*l + 2*S13*h*l)
    invD2 *= 1/V**2
    d = 1/sqrt(invD2)
    return d


def pix_to_E(energy, dspacing):
    wl = 12398.4193/energy
    th = arcsin(wl/(2*dspacing))
    l = 2 * cos(pi/2 - th)
    dE = energy*PIX_EN_CONV/(l*tan(th))
    return dE


def binning(x, y, n, photon_counting=False):
    e = np.sqrt(y)
    if isinstance(n,int): #strides
        xbin = int(len(x)/n)
    else: #regularly spaced (in eV)
        xbin = np.arange(min(x), max(x), n)
    yi,xi = np.histogram(x, bins=xbin, weights=y)
    ei,_ = np.histogram(x, bins=xbin, weights=e**2)
    
    if not photon_counting:
        count,_ = np.histogram(x, bins=xbin)
        with np.errstate(divide='ignore', invalid='ignore'):
            yi = yi / count
            ei = np.sqrt(ei) / count
    
    xi = (xi[1:]+xi[:-1])/2
    return xi, yi, ei


def find_nearest(array, value):
    return (np.abs(array-value)).argmin()


def peak(x, a, sl, x0, f, bgnd):
    m = np.full(len(x), bgnd)
    sg = sl/np.sqrt(2*log(2))
    m += (1-f)*(a/(sg*np.sqrt(2.*np.pi))) * np.exp(-1.*(x-x0)**2./(2.*sg**2.))
    m += f * (a/np.pi) * (sl / ((x-x0)**2. + sl**2.))
    return m


def lorentzian(x, a, sl, x0, bgnd):
    m = np.full(len(x), bgnd)
    m += (a/np.pi) * (sl / ((x-x0)**2. + sl**2.))
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
    if defs['fit_pv']:
        p0 = [amp, sig, cen, fra, bgnd]
        bounds = [(0, 1E-6, -1E9, 0, 0),(1E9, 1E6, 1E9, 1, 1E9)]
        p, _ = curve_fit(peak, x, y, p0, bounds=bounds)
        yi = peak(xf, *p)
    else:
        p0 = [amp, sig, cen, bgnd]
        bounds = [(0, 1E-6, -1E9, 0),(1E9, 1E6, 1E9, 1E9)]
        p, _ = curve_fit(lorentzian, x, y, p0, bounds=bounds)
        yi = lorentzian(xf, *p)
    return xf, yi, p


#### Load Files ###############################################################

def load_fio(run, exp, datdir):
    a = {}
    head = []
    data = []
    complete = False
    path = '{0}/{1}_{2:05d}.fio'.format(datdir, exp, run)
    if not os.path.isfile(path):
        print('#{0:<4} -- no .fio'.format(run))
        return
    with open(path) as f:
        for line in f:
            l = line.strip()
            if l.startswith('%c'):
                command = next(f)[:-1].split()
                _,date = next(f).split(' started at ',maxsplit=1)
            elif line.startswith('%p'):
                break
        for line in f:
            if line.startswith('!'):
                break
            else:
                p, v = line.strip().split('=')
                try:
                    a[p.strip()] = float(v)
                except:
                    a[p.strip()] = v.strip()
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
                    if line.startswith('! Acquisition ended'):
                        complete = True
    if head and data:
        
        data = [x for x in data if x]
        data = np.array(data)
        pnts = data.shape[0]
        data = data.view(dtype=[(n, float) for n in head])
        data = data.reshape(len(data))

        a['data'] = data
        a['auto'] = head[0]
        a['pnts'] = pnts
        if head[0] == 'exp_dmy01':
            a['EF'] = np.full(pnts,a['rixs_ener'])
        else:
            a['EF'] = a['data'][head[0]]
        a['EI'] = a['dcm_ener']
        a['th'] = a['rixs_th']
        a['chi'] = a['rixs_chi']

        if 'q_h' in head:
            a['qh'] = np.average(data['q_h'])
            a['qk'] = np.average(data['q_k'])
            a['ql'] = np.average(data['q_l'])
        else:
            a['qh'], a['qk'], a['ql'] = 0.0, 0.0, 0.0

        if 't_coldhead' in head:
            a['t_coldhead'] = np.round(np.average(data['t_coldhead']))
        if 't_sample' in head:
            a['t_sample'] = np.round(np.average(data['t_sample']))
            a['T'] = a['t_sample']
        else:
            a['T'] = 0.0

        a['command'] = command
        a['date'] = date[:-9].strip()
        a['time'] = float(command[-1])
        a['numor'] = run
        a['complete'] = complete

        return a


def load_tiff(run, no, exp, datdir, localdir):
    path = '{0}/{1}_{2:05d}/andor/{1}_{2:05d}_{3:04d}.tiff'.format(
            datdir, exp, run, no)
    with warnings.catch_warnings():
        if localdir:
            path2 = '{0}/{1}_{2:05d}/andor/{1}_{2:05d}_{3:04d}.tiff'.format(
                    localdir, exp, run, no)
            try:
                img = imread(path2)
            except (Warning, FileNotFoundError, OSError):
                try:
                    os.makedirs(os.path.dirname(path2), exist_ok=True)
                    shutil.copyfile(path, path2)
                    img = imread(path2)
                except:
                    return
        else:
            try:
                img = imread(path)
            except (Warning, FileNotFoundError, OSError):
                return
    return img


#### Processing Routine #######################################################

class irixs:

    def __init__(self, exp,
                 y0=None, roix=[0,2048], roiy=[0,2048], roih=None,
                 threshold=1010, cutoff=1800, detfac=935,
                 photon_factor = 750, E0=None, analyser=None, datdir=None,
                 photon_event_threshold=400, photon_max_events=0):
        '''
        datdir -- define datdir to work locally
        exp -- experiment title / data filename prefix
        E0 -- elastic energy: typically use None to extract from .fio file
        y0 -- corresponding vertical pixel position on detector
        roix -- detector region of interest
        roiy -- detector region of interest
        roih -- define height of roi instead of roiy, using y0 as centre
        threshold -- minimum to kill readout noise (use histogram to refine)
        cutoff -- upper limit to kill cosmic rays (use histogram to refine)
        detfac -- andor detector factor (0 if andor bgnd sub is enabled, 935 otherwise)
        analyser -- analyser crystal lattice and reflection
        photon_factor -- conversion between detector intensitiy and photon count (should be = 788)
        photon_event_threshold -- threshold intensity for a contigous detector event
        photon_max_events -- maximum multiple events to correct for (0 to disable correction)
        '''

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
        self.event_min = photon_event_threshold
        self.max_events = photon_max_events

        quartz = ([(1, 0, 2) , (4.9133, 4.9133, 5.4053, 90, 90, 120)])
        self.analyser = quartz if analyser is None else analyser
        self.dspacing = calc_dspacing(self.analyser[0], self.analyser[1])

        if datdir:
            self.datdir = datdir
            self.localdir = False
        else:
            self.datdir = defs['datdir']
            self.localdir = defs['savedir_raw']
        
        if self.localdir:
            os.makedirs(self.localdir, exist_ok=True)

        self.savedir_dat = defs['savedir_dat']
        self.savedir_con = defs['savedir_con']
        self.savedir_det = defs['savedir_det']
        self.savedir_fig = defs['savedir_fig']
        os.makedirs(self.savedir_dat, exist_ok=True)
        os.makedirs(self.savedir_con, exist_ok=True)
        os.makedirs(self.savedir_det, exist_ok=True)
        os.makedirs(self.savedir_fig, exist_ok=True)

        self.corr_shift = False  # distortion correction


    def load(self, numors, tiff=True):

        if not isinstance(numors, (list, tuple, range)):
            numors = [numors]

        flatten = lambda *n: (e for a in n
            for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))
        numors = list(flatten(numors))

        for n in numors:
            if n not in self.runs.keys():
                self.runs[n] = None

        for numor in numors:
            path = '{0}/{1}_{2:05d}.fio'.format(self.datdir, self.exp, numor)
            if self.localdir:
                path2 = '{0}/{1}_{2:05d}.fio'.format(
                        self.localdir, self.exp, numor)
                if not os.path.isfile(path2):
                    a = load_fio(numor, self.exp, self.datdir)
                    if a and a['complete']:
                        os.makedirs(self.localdir, exist_ok=True)
                        shutil.copyfile(path, path2)
                else:
                    a = load_fio(numor, self.exp, self.localdir)
            else:
                a = load_fio(numor, self.exp, self.datdir)
            self.runs[numor] = a

        if not tiff:
            return

        to = self.threshold - self.detfac
        co = self.cutoff - self.detfac

        for numor in numors:
            
            a = self.runs[numor]

            if not a:
                continue

            if 'to' in a and to == a['to'] and co == a['co'] and a['complete']:
                continue

            if 'img' not in a or a['img'] is None:
                imtest = load_tiff(numor, 0, self.exp, self.datdir, self.localdir)
                if imtest is None:
                    print('#{0:<4} -- no images'.format(numor))
                    a['img'] = None
                    continue
                else:
                    a['img'] = []

            for i,_ in enumerate(a['EF']):
                if i > len(a['img'])-1:
                    img = load_tiff(numor, i, self.exp, self.datdir, self.localdir)
                    if img is None:
                        print('!!!')
                        break
                    if img is not None:
                        img -= self.detfac
                        img[~np.logical_and(img>to, img<co)] = 0
                        a['img'].append(img)
                    sys.stdout.write('\r#{0:<4} {1:<3}/{2:>3} '.format(numor, i+1, a['pnts']))
                    if i+1 == a['pnts']:
                        sys.stdout.write('\n')
                    sys.stdout.flush()
           
            a['threshold'] = self.threshold
            a['cutoff'] = self.cutoff
            a['detfac'] = self.detfac
            a['to'], a['co'] = to, co


    def logbook(self, numors=None, nend=None, extras=['th'],
                hkl=False, date=False, only_rixs=True):
        if numors is None:
            numors = self.runs.keys()
        elif isinstance(numors,(list,tuple,range)):
            numors = list(numors)
        elif nend is None:
            try:
                latest = max(iglob(os.path.join(self.datdir,'*.fio')),
                             key=os.path.getctime)
            except ValueError:
                print('Using Local Directory')
                latest = max(iglob(os.path.join(self.localdir,'*.fio')),
                             key=os.path.getctime)
            try:
                latest = latest[:-4].split('_')[-1]
                numors = range(numors,int(latest)+1)
            except ValueError:
                return
        else:
            numors = range(numors,nend+1)
        self.load(numors, False)
        for numor in numors:
            out = ''
            a = self.runs[numor]
            if a is None:
                continue
            try:
                command = a['command']
                scantype, motor = command[:2]
                m1, m2, pnt, t = [float(c) for c in command[2:]]
            except:
                continue
            out += '#{0:<4}{1:>13} '.format(numor,motor)
            if motor in ['rixs_ener']:
                out += '{0:>+6.2f} > {1:+4.2f}'.format(m1, m2)
            else:
                if only_rixs:
                    continue
                out += ' {0:<12}'.format('')
            out += ' {0:3.0f}pnt {1:4.0f}s  '.format(pnt, t)
            out += '{0:7.1f}eV'.format(a["dcm_ener"])
            if a["dcm_ener"] != a["rixs_ener"]:
                out += '* '
            else:
                out += '  '
            out += '{0:3.0f}K  '.format(a['T'])
            if hkl:
                qh = np.round(a['qh'],2)+0
                qk = np.round(a['qk'],2)+0
                ql = np.round(a['ql'],2)+0
                if not qh and not qk and not ql:
                    out += ' '*18
                else:
                    out += '({0: 3.1f} {1: 3.1f} {2:4.1f})  '.format(qh, qk, ql)
            for ex in extras:
                out += '{0}:{1:6.2f}  '.format(ex,a[ex])
            if date:
                out += ' ' + a['date']
            print(out)
        print()


    def detector(self, numors, com=False, fit=False,
                 plot=True, vmax=10, savefig=False,
                 use_distortion_corr=False):

        roic = '#F012BE'
        roix, roiy = self.roix, self.roiy

        self.load(numors)
        if not isinstance(numors,(list,tuple,range)):
            numors = [numors]
        for numor in numors:
            a = self.runs[numor]
            if a is None or a['img'] is None:
                continue
            
            savefile = '{0}/{1}_{2:05d}_det.txt'.format(
                       self.savedir_det, self.exp, numor)

            step = a['auto']
            if step == 'exp_dmy01':
                oneshot = True
                com = False
            else:
                oneshot = False

            x = []
            y = []
            comV = []
            comH = []

            imgarr = np.atleast_3d(np.array(a['img']))
            imtotal = np.nansum(imgarr, axis=0) / imgarr.shape[0]
            imgarr = imgarr[:,roiy[0]:roiy[1], roix[0]:roix[1]]

            if use_distortion_corr and self.corr_shift is not False:
                for sh,(c1,c2) in zip(self.corr_shift,self.corr_regions):
                    c1,c2 = c1+roix[0],c2+roix[0]
                    imtotal[:,c1:c2] = np.roll(imtotal[:,c1:c2],sh,axis=0)

            if oneshot:
                x = np.arange(roiy[0], roiy[1])
                y = np.nansum(imgarr, axis=(0,2)) / imgarr.shape[0]
            else:
                for i, ef in enumerate(a['EF']):
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
                        cv = np.nansum(rx*np.nansum(img, axis=1))/yi
                        ch = np.nansum(ry*np.nansum(img, axis=0))/yi
                        comV.append(cv)
                        comH.append(ch)
                x, y = np.array(x), np.array(y)

            a['xd'], a['yd'] = x, y

            header = 'experiment: {0}\n'.format(self.exp)
            header+= 'run: {0}\n'.format(numor)
            header+= 'command: {0}\n'.format(' '.join(a['command']))
            header+= 'dcm_ener: {0}\n'.format(a['dcm_ener'])
            header+= 'rixs_ener: {0}\n'.format(a['rixs_ener'])
            header+= 'det_threshold: {0}\n'.format(a['threshold'])
            header+= 'det_cutoff: {0}\n'.format(a['cutoff'])
            header+= 'det_factor: {0}\n'.format(a['detfac'])
            header+= 'det_roix: {0}\n'.format(roix)
            header+= 'det_roiy: {0}\n\n'.format(roiy)

            if oneshot:
                header+= '{0:>24}{1:>24}'.format('y-pixel','counts')
                save_array = np.array([x, y]).T
            elif com:
                header+= '{0:>24}{1:>24}{2:>24}{3:>24}'.format(step,'roi-counts','vert-COM','horiz-COM')
                comV, comH = np.array(comV), np.array(comH)
                a['comV'], a['comH'] = comV, comH
                save_array = np.array([x, y, comV, comH]).T
            else:
                header+= '{0:>24}{1:>24}'.format(step,'roi-counts')
                save_array = np.array([x, y]).T
            np.savetxt(savefile, save_array, header=header)

            if fit:
                try:
                    a['xfd'], a['yfd'], a['pd'] = peak_fit(x, y)
                    report  = '#{0:<4} (det)  '.format(numor)
                    report += 'cen:{0:.4f}   '.format(a['pd'][2])
                    report += 'amp:{0:.2f}   '.format(a['pd'][0])
                    report += 'fwhm:{0:.3f}   '.format(a['pd'][1]*2)
                    if defs['fit_pv']:
                        report += 'fra:{0:.1f}   '.format(a['pd'][3])
                    report += 'bg:{0:.3f}\n'.format(a['pd'][3])
                    print(report)
                except:
                    a['pd'] = False
            else:
                a['pd'] = False

            if plot:
                if com:
                    fig, ax = plt.subplots(2, 2, constrained_layout=True,
                                           figsize=(8.5, 8))
                    ax = ax.flatten()
                else:
                    fig, ax = plt.subplots(1, 2, figsize=(8.5, 4))
                    fig.subplots_adjust(0.06, 0.15, 0.98, 0.93)

                plt.suptitle('#{}'.format(a['numor']), ha='left', va='top',
                            x=0.005, y=0.995)

                im = ax[0].imshow(imtotal, origin='lower', vmax=vmax,
                                  cmap=plt.get_cmap('bone_r'),
                                  interpolation='hanning')
                
                rect = Rectangle((roix[0], roiy[0]), #xy origin, width, height
                                roix[1]-roix[0], roiy[1]-roiy[0], 
                                linewidth=0.5, linestyle='dashed',
                                edgecolor=roic, fill=False)
                ax[0].add_patch(rect)
                if isinstance(self.y0, dict):
                    y0 = self.y0[numor]
                else:
                    y0 = self.y0         
                ax[0].axhline(y0, color=roic, lw=0.5, dashes=(2, 2))
                
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

                ax[1].plot(x, y, lw=1, color='#001F3F')
                if oneshot:
                    ax[1].set_xlabel('y-pixel')
                    ax[1].set_title('Integrated')
                else:
                    ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                    ax[1].set_title('Counts in ROI')

                if a['pd'] is not False:
                    ax[1].plot(a['xfd'], a['yfd'],
                               color='#001F3F', dashes=(2,8), lw=0.5)
                    fr = 'fwhm: {:.3f}\ncen: {:.2f}'.format(
                         a['pd'][1]*2,a['pd'][2])
                    ax[1].text(0.025, 0.975, fr, va='top',
                               transform=ax[1].transAxes,
                               fontsize='small', linespacing=1.3)

                if com:
                    ax[2].plot(a['xd'], a['comH'], lw=1, color='#0074D9')
                    ax[2].set_title('Horizontal COM')
                    ax[2].set_ylabel('x-pixel')
                    ax[2].set_ylim(roix[0], roix[1])

                    ax[3].plot(a['xd'], a['comV'], lw=1, color='#FF4136')
                    ax[3].set_title('Vertical COM')
                    ax[3].set_ylabel('y-pixel')
                    ax[3].set_ylim(roiy[0], roiy[1])

                for axi in ax[1:]:
                    axi.minorticks_on()
                    if not oneshot:
                        axi.set_xlabel(step)
                        for l in axi.get_xmajorticklabels():
                            l.set_rotation(30)

                if savefig:
                    if savefig is True:
                        savename = '{0}/det_s{1}_{2}.pdf'.format(
                                self.savedir_fig, a['numor'], a['auto'])
                    else:
                        savename = '{0}/{1}_s{2}.pdf'.format(
                                self.savedir_fig, savefig, a['numor']) 
                    plt.savefig(savename, dpi=300)


    def condition(self, bins, numors, fit=False,
                  photon_counting=False, use_distortion_corr=True):

        self.load(numors)
        roix = self.roix
        if isinstance(numors, int):
            numors = [numors]

        for numor in numors:
            
            if isinstance(numor,int):
                numor = [numor]
            
            x,y,ns = [],[],[]
            for n in numor:
                a = self.runs[n]
                if a is None or a['img'] is None:
                    continue
                if a['auto'] not in ['rixs_ener','dcm_ener','exp_dmy01']:
                    continue
                
                ns.append(n)
                if isinstance(self.y0,dict):
                    y0 = self.y0[n]
                else:
                    y0 = self.y0
                if self.roih:
                    try:
                        roiy = [y0+self.roih[0], y0+self.roih[1]]
                    except IndexError:
                        roiy = [y0-self.roih//2, y0+self.roih//2]
                else:
                    roiy = self.roiy
                xinit = np.arange(roiy[0], roiy[1])
                if photon_counting:
                    xinit = np.tile(xinit,(roix[1]-roix[0],1)).T

                for ef, img in zip(a['EF'],a['img']):

                    img = deepcopy(img[:,roix[0]:roix[1]])

                    if use_distortion_corr and self.corr_shift is not False:
                        for sh,(c1,c2) in zip(self.corr_shift,self.corr_regions):
                            img[:,c1:c2] = np.roll(img[:,c1:c2],sh,axis=0)

                    img  = img[roiy[0]:roiy[1]]
                    if photon_counting:
                        lbl,nlbl = scipy.ndimage.label(img)
                        try:
                            yi = scipy.ndimage.labeled_comprehension(img,lbl,range(1,nlbl+1),
                                                                 np.sum, float, 0)
                            xi = scipy.ndimage.labeled_comprehension(xinit,lbl,range(1,nlbl+1),
                                                                 np.mean, float, 0)
                            xi = (xi - y0) * pix_to_E(ef, self.dspacing) + ef
                            xi = xi[yi>self.event_min]
                            yi = yi[yi>self.event_min]
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
            a['label'] = ','.join([str(ni) for ni in ns])

            x, y = np.array(x), np.array(y)
            y = y[np.argsort(x)]
            x = np.sort(x)
            if self.E0 is None:
                en = a['EI']
            else:
                en = self.E0
            x -= en

            a['roix'], a['roiy'], a['y0'], a['E0'] = roix, roiy, y0, en

            header = 'experiment: {0}\n'.format(self.exp)
            header+= 'run: {0}\n'.format(n)
            header+= 'command: {0}\n'.format(' '.join(a['command']))
            header+= 'dcm_ener: {0}\n'.format(a["dcm_ener"])
            header+= 'rixs_ener: {0}\n'.format(a["rixs_ener"])
            if 't_coldhead' in a:
                header+= 't_coldhead: {0}\n'.format(a["t_coldhead"])
            if 't_sample' in a:
                header+= 't_sample: {0}\n'.format(a["t_sample"])
            header+= 'rixs_th: {0}\n'.format(a["rixs_th"])
            header+= 'rixs_chi: {0}\n'.format(a["rixs_chi"])
            header+= 'q_hkl: {0:.4f} {1:.4f} {2:.4f}\n'.format(a["qh"], a["qk"], a["ql"])
            header+= 'det_threshold: {0}\n'.format(a['threshold'])
            header+= 'det_cutoff: {0}\n'.format(a['cutoff'])
            header+= 'det_factor: {0}\n'.format(a['detfac'])
            header+= 'det_roix: {0}\n'.format(roix)
            header+= 'det_roiy: {0}\n'.format(roiy)
            header+= 'E0_ypixel: {0}\n'.format(y0)
            header+= 'E0_offset: {0}\n'.format(en)

            if photon_counting:
                savefile = '{0}/{1}_pc_{2:05d}.txt'.format(self.savedir_dat, self.exp, n)
            else:
                savefile = '{0}/{1}_{2:05d}.txt'.format(self.savedir_dat, self.exp, n)            
            np.savetxt(savefile, np.array([x, y]).T,
                        header=header+'\n{0:>24}{1:>24}'.format(a['auto'],'counts'))

            y = y / self.photon_factor
            if photon_counting and self.max_events:
                x = np.delete(x,np.where(y>self.max_events+0.5))
                y = np.delete(y,np.where(y>self.max_events+0.5))
                for i in range(self.max_events,1,-1):
                    cnts = np.where(np.logical_and(y>=i-0.5,y<i+0.5))
                    y[cnts] /= i
                    if i > 2:
                        y = np.append(y,np.tile(y[cnts],i-1))
                        x = np.append(x,np.tile(x[cnts],i-1))
                    else:
                        y = np.append(y,y[cnts])
                        x = np.append(x,x[cnts])
                y = y[np.argsort(x)]
                x = np.sort(x)
            if bins:
                x, y, e = binning(x, y, bins, photon_counting)
            else:
                e = np.sqrt(y)
            y[~np.isfinite(y)] = 0
            a['x'], a['y'], a['e'] = x, y, e

            if fit:
                a['xf'], a['yf'], a['p'] = peak_fit(x, y)
                report  = '#{0:<4} (bin: {1})  '.format(n, bins)
                report += 'cen:{0:8.4f}   '.format(a['p'][2])
                report += 'amp:{0:6.2f}   '.format(a['p'][0])
                report += 'fwhm:{0:6.3f}   '.format(a['p'][1]*2)
                if defs['fit_pv']:
                    report += 'fra:{0:4.1f}   '.format(a['p'][3])
                report += 'bg:{0:6.3f}\n'.format(a['p'][3])
                print(report)
            else:
                a['p'] = False

            header+= 'bin_size: {0}\n'.format(bins)
            header+= '\n{0:>24}{1:>24}{2:>24}'.format(a['auto'],'counts','stderr')
            if bins < 5:
                savefile = '{0}/{1}_{2:05d}_b{3:.1f}meV.txt'.format(
                    self.savedir_con, self.exp, n, bins*1000)
            else:
                savefile = '{0}/{1}_{2:05d}_b{3}.txt'.format(
                    self.savedir_con, self.exp, n, bins)
            np.savetxt(savefile, np.array([x, y, e]).T, header=header)



    def plot(self, numors, ax=None, step='numor', labels=None, sort=False, rev=False,
             norm=False, ysca=None, ystp=0, yoff=None, xoff=None,
             show_fit=True, stderr=False, cmap=None, fmt='-', lw=1,
             vline=[0], leg=0, title=None, savefig=True,
             plot_det=False, xlim=None, ylim=None):

        if not isinstance(numors,(list,tuple,range)):
            numors = [numors]

        numors = [n[0] if isinstance(n,list) else n for n in numors]
        runs = [self.runs[n] for n in numors if n in self.runs and self.runs[n] is not None]
        if not runs:
            return

        if sort:
            runs.sort(key=lambda x: x[step])
        if rev:
            runs = runs[::-1]

        if labels:
            if not isinstance(labels,(list,tuple)):
                labels = [labels]

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        if cmap:
            if isinstance(cmap, str):
                cmap = plt.get_cmap(cmap)

        for i, a in enumerate(runs):

            xf, yf, p = False, False, False
            if 'x' in a and not plot_det:
                x, y, e = deepcopy(a['x']), deepcopy(a['y']), deepcopy(a['e'])
                if a['p'] is not False:
                    xf, yf, p = a['xf'], a['yf'], a['p']
            elif 'xd' in a and plot_det:
                x, y = deepcopy(a['xd']), deepcopy(a['yd'])
                e = False
                if a['pd'] is not False:
                    xf, yf, p = a['xfd'], a['yfd'], a['pd']
            else:
                print('#{0}: nothing to plot'.format(a['numor']))
                continue

            if xoff is not None:
                if isinstance(xoff, list):
                    x+=xoff[i]
                else:
                    x+=xoff

            if norm:
                if norm == 'time':
                    maxy = a['time']
                if isinstance(norm,(tuple,list)):
                    maxy = np.mean(y[(x > min(norm)) & (x < max(norm))])
                else:
                    maxy = max(y)
                y = y/maxy
                if e is not False:
                    e = e/maxy
                if p is not False:
                    yf = yf/maxy

            if ysca is not None:
                if isinstance(ysca, list):
                    ys = ysca[i]
                else:
                    ys = ysca
                y*=ys
                if e is not False:
                    e*=ys
                if p is not False:
                    yf*=ys

            if yoff is not None:
                if isinstance(yoff, list):
                    yo = yoff[i]
                else:
                    yo = yoff
                y+=yo
                if p is not False:
                    yf+=yo

            if cmap:
                c = cmap(i/len(numors))
            else:
                c = None

            label = '#{0}'.format(a['label'])
            if labels:
                label += ' {0}'.format(labels[i])
            elif step == 'T':
                label += ' {:.0f}K'.format(a[step])
            elif step == 'rixs_th':
                label += ' th: {:.2f}'.format(a[step])
            elif step == 'hkl':
                label += ' {:.2f} {:.2f} {:.2f}'.format(a['qh'],a['qk'],a['ql'])
            elif step != 'numor':
                try:
                    label += ' {}: {:.3f}'.format(step, a[step])
                except TypeError:
                    label += ' {}: {}'.format(step, a[step])

            l, = ax.plot(x, y+i*ystp, fmt, color=c, lw=lw, label=label)
            if stderr and not plot_det and not norm:
                ax.errorbar(x,y+i*ystp,e,fmt='none',color=l.get_color(),lw=lw)

            if show_fit and p is not False:
                if defs['fit_pv']:
                    amp, sig, cen, fra, bgnd = p
                else:
                    amp, sig, cen, bgnd = p
                
                ax.plot(xf, yf+i*ystp, color=l.get_color(), lw=0.5)
                
                if len(numors) == 1:
                    fr = 'amp: {:.2f}\nfwhm: {:.3f}\ncen: {:.4f}'.format(
                        amp, sig*2, cen)
                    ax.text(0.05, 0.95, fr, va='top', linespacing=1.3,
                            transform=ax.transAxes)

        ax.minorticks_on()
        
        if norm == 'time':
            ax.set_ylabel('Intensity (per second)')
        elif norm:
            ax.set_ylabel('Intensity (normalised)')
        else:
            ax.set_ylabel('Intensity')
        
        if 'x' in a:
            ax.set_xlabel('Energy Transfer (eV)')
        else:
            ax.set_xlabel(a['auto'])

        if leg is not False:
            if len(runs) > 20:
                ncol = 2
            else:
                ncol = 1
            ax.legend(loc=leg, handlelength=1.5, labelspacing=0.3, handletextpad=0.5,
                        ncol=ncol, fontsize='small')
        
        if vline is not False:
            if not isinstance(vline,(tuple,list)):
                vline = [vline]
            for v in vline:
                ax.axvline(v, color='k', lw=0.5)
        
        if title:
            ax.set_title(title)
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        
        if savefig:
            if not isinstance(savefig, str):
                sc = '_'.join(str(n) for n in numors)
                savefig = 's{}_{}'.format(sc, a['auto'])
            plt.savefig('{}/{}.pdf'.format(self.savedir_fig, savefig), dpi=300)


    def check_run(self, numor, hist=False, no=0, vmin=0, vmax=10,
                    interp='hanning', photon_counting=False):
        
        '''Step through each detector image of a run.
        run -- run number
        numor -- starting step number (default is the first)
        hist -- plot histogram rather than detector map
        vmin -- colourmap minimum
        vmax -- colourmap maximum
        interp -- interpolation mode for image plot ('nearest' to disable)
        photon_counting -- check photon counting algorithm
        '''

        self.load(numor)
        a = self.runs[numor]
     
        pnts = a['pnts']
        to = a['threshold'] - a['detfac']
        co = a['cutoff'] - a['detfac']

        i = {}
        i['idx'] = no
        fig, i['ax'] = plt.subplots()

        imdat = a['img'][no]
        imgdim = imdat.shape

        if hist:
            b, c = np.histogram(imdat, bins=range(to, co, 1))
            i['ax'].bar(c[:-1], b)
        elif photon_counting:
            imdat,_ = scipy.ndimage.label(imdat)
            #cmap = plt.get_cmap('prism')
            vals = np.linspace(0,1,256)
            np.random.shuffle(vals)
            cmap = plt.cm.colors.ListedColormap(plt.get_cmap('jet')(vals))
            cmap.set_under('k')
            i['im'] = i['ax'].imshow(imdat, origin='lower', cmap=cmap,
                                        interpolation='nearest',vmin=0.1)
        else:
            cmap = plt.get_cmap('bone_r')
            i['im'] = i['ax'].imshow(imdat, origin='lower', cmap=cmap,
                                 vmin=vmin, vmax=vmax, interpolation=interp)
        i['ax'].set_title('#{} no {}'.format(numor, i['idx']))

        def do_plot(i):
            try:
                imdat = a['img'][i['idx']]
            except IndexError:
                imdat = np.zeros(imgdim)
                print('check run: no tiff for step {}'.format(i['idx']))
            if hist:
                b, c = np.histogram(imdat, bins=range(to, co, 1))
                i['ax'].cla()
                i['ax'].bar(c[:-1], b)
            else:
                if photon_counting:
                    imdat,_ = scipy.ndimage.label(imdat)
                i['im'].set_data(imdat)
            i['ax'].set_title('#{} no {}'.format(numor, i['idx']))
            plt.draw()

        def press(event):
            if event.key == 'left':
                if i['idx'] > 0:
                    i['idx'] -= 1
                    do_plot(i)
            elif event.key == 'right':
                if i['idx'] < pnts-1:
                    i['idx'] += 1
                    do_plot(i)

        fig.canvas.mpl_connect('key_press_event', press)


    def calc_distortion(self, numor, no=0, slices=8, force_y0=False,
                            plot=True, vmin=0, vmax=10):

        self.load(numor)
        a = self.runs[numor]

        img = a['img'][no]
        img = img[self.roiy[0]:self.roiy[1], self.roix[0]:self.roix[1]]

        y = np.sum(img,axis=1)
        x = np.arange(self.roiy[0],self.roiy[1])
        _,_,pinit = peak_fit(x, y)
        y0 = int(round(pinit[2]))
        print('fitted y0: {}'.format(y0))
        print('initial fwhm: {:.4f}'.format(pinit[1]*2))

        if force_y0:
            y0 = self.y0

        slice_width = img.shape[1]/slices
        shift = []
        regions = []
        
        for i in range(slices):
            c1, c2 = int(i*slice_width), int(i*slice_width+slice_width)
            yi = np.sum(img[:,c1:c2],axis=1)
            try:
                _,_,pi = peak_fit(x,yi)
                cen = int(round(pi[2]))
            except RuntimeError:
                cen = y0
            shift.append(y0-cen)
            regions.append([c1,c2])

        self.corr_shift = shift
        self.corr_regions = regions

        imgcorr = deepcopy(img)
        for sh,(c1,c2) in zip(shift,regions):
            yi = np.sum(imgcorr[:,c1:c2],axis=1)
            imgcorr[:,c1:c2] = np.roll(imgcorr[:,c1:c2],sh,axis=0)

        ycorr = np.sum(imgcorr,axis=1)
        _,_,pfinal = peak_fit(x, ycorr)
        print('final fwhm: {:.4f}'.format(pfinal[1]*2))

        if plot:
            _, ax = plt.subplots(1,3, figsize=(10, 4),constrained_layout=True)
            ax[0].plot(x,y,lw=0.5)
            ax[0].plot(x,ycorr,lw=0.5)

            ax[1].imshow(img, origin='lower', cmap=plt.get_cmap('bone_r'), aspect='auto',
                        extent=(self.roix[0],self.roix[1],self.roiy[0],self.roiy[1]),
                        vmin=vmin, vmax=vmax, interpolation='hanning')

            ax[2].imshow(imgcorr, origin='lower', cmap=plt.get_cmap('bone_r'), aspect='auto',
                        extent=(self.roix[0],self.roix[1],self.roiy[0],self.roiy[1]),
                        vmin=vmin, vmax=vmax, interpolation='hanning')

            ax[1].axhline(y0,color='#F012BE',lw=0.5)
            ax[2].axhline(y0,color='#F012BE',lw=0.5)
            for sh,(c1,c2) in zip(shift,regions):
                ax[1].axvline(c1+self.roix[0],color='#F012BE',lw=0.5)
                ax[2].axvline(c1+self.roix[0],color='#F012BE',lw=0.5)

