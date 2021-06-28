""" basic six-circle diffractometer simulator """
import numpy as np

from numpy import pi, sin, cos, radians, degrees
from numpy.linalg import norm, inv, multi_dot
from scipy.optimize import least_squares
from copy import deepcopy

from .tools import reciprocol_lattice

np.seterr(invalid='raise')  # catch failed UB-matrix initialisation due to parallel reflections

def rotation_matricies(mu, nu, chi, eta=0, delta=0, phi=0):
    """ matrices corresponding to the rotation of the circles """

    MU = np.array([
        [1, 0, 0],
        [0, cos(mu), -sin(mu)],
        [0, sin(mu), cos(mu)]
    ])

    NU = np.array([
        [1, 0, 0],
        [0, cos(nu), -sin(nu)],
        [0, sin(nu), cos(nu)]
    ])

    CHI = np.array([
        [cos(chi), 0, sin(chi)],
        [0, 1, 0],
        [-sin(chi), 0, cos(chi)]
    ])

    ETA = np.array([
        [cos(eta), sin(eta), 0],
        [-sin(eta), cos(eta), 0],
        [0, 0, 1]
    ])

    DELTA = np.array([
        [cos(delta), sin(delta), 0],
        [-sin(delta), cos(delta), 0],
        [0, 0, 1]
    ])

    PHI = np.array([
        [cos(phi), sin(phi), 0],
        [-sin(phi), cos(phi), 0],
        [0, 0, 1]
    ])

    return MU, NU, CHI, ETA, DELTA, PHI


def q_phi(mu, nu, chi, eta=0, delta=0, phi=0):
    """Calculate hkl in phi frame, in units of 2*pi/lambda"""

    MU, NU, CHI, ETA, DELTA, PHI = rotation_matricies(mu, nu, chi, eta, delta, phi)

    q_lab = NU.dot(DELTA) - np.eye(3)
    q_lab = q_lab.dot(np.array([[0], [1], [0]]))

    Z = multi_dot([inv(PHI), inv(CHI), inv(ETA), inv(MU), q_lab])
    return Z


def q_hkl(wl, UB, mu, nu, chi, eta=0, delta=0, phi=0):
    """Calculate miller indicies from six circle angles with UB-matrix"""
    
    MU, NU, CHI, ETA, DELTA, PHI = rotation_matricies(mu, nu, chi, eta, delta, phi)

    q_lab = NU.dot(DELTA) - np.eye(3)
    q_lab = q_lab.dot(np.array([[0], [2 * pi / wl], [0]]))

    hkl = multi_dot([inv(UB), inv(PHI), inv(CHI), inv(ETA), inv(MU), q_lab])

    return hkl.ravel()


class sixc:
    """ six-circle diffractometer simulator

    Uses methodology described by
    H. You, 4S+2D six-circle diffractometer, J. Appl. Cryst 32, 614, (1999)
    https://doi.org/10.1107/S0021889899001223

    Basic approach follows diffcalc (Diamond Light Source)
    https://github.com/DiamondLightSource/diffcalc

    UB matrix setup limited to IRIXS geometry (horizontal scattering angles)

    Circle definitions
    ------------------
    mu -- horiz th
    nu -- horiz tth
    eta -- vert th
    delta -- vert tth
    chi -- sample rocking
    phi -- sample rotation

    Attributes
    ----------
    UB : (3x3) np.array
        UB matrix based on crystal lattice (B) + instrument orientation (U)
    orientation : list
        reflection values for U orientation matrix [hkl0, hkl1, angles0, angles1]
    cell : list
        lattice parameters [a(Å), b(Å), c(Å), alpha(rad), beta(rad), gamma(rad)]
    recip_cell : list
        reciprocol lattice [a*, b*, c*, alpha*, beta*, gamma*]
    energy : float
        x-ray energy (eV)
    wl : float
        x-ray wavelength (Å)

    Methods
    -------
    hkl(th, tth, chi) -> tuple(h, k, l)
        Return miller indicies for given IRIXS angles
    find_hk_angles(hk_list) -> list([th, chi]):
        Return th and chi angles for list of HK values, for tth=90°
    set_B(a1, a2, a3, alpha1, alpha2, alpha3)
        Set lattice parameters and B matrix
    set_UB(hkl0, hkl1, angles0, angles1)
        Set U orientation Matrix and initalise UB-matrix
    """

    def __init__(self, cell, hkl0, hkl1, angles0, angles1=None, energy=2838.5):
        """
        Parameters
        ----------
        cell : list
            lattice parameters [a(Å), b(Å), c(Å), alpha(°), beta(°), gamma(°)]
        hkl0 : list
            first alignment reflection [h, k, l]
        hkl1 : list
            perp. second reflection [h, k, l]
        angles0 : list
            horiz angles for hkl0  [th0(°), tth0(°), chi0(°)]
        angles1 : list (optional -- otherwise assume hkl1 is precisely th+90° from hkl0)
            horiz angles for hkl1  [th1(°), tth1(°), chi1(°)]
        energy : float
            x-ray energy (eV)
        """
        self.energy = energy
        self.wl = 12398 / energy  # eV -> Å
        self.set_B(*cell)        
        self.set_UB(hkl0, hkl1, angles0, angles1)


    def set_B(self, a1, a2, a3, alpha1, alpha2, alpha3):
        """calculate B matrix from crystal lattice parameters"""

        alpha1, alpha2, alpha3 = radians(alpha1), radians(alpha2), radians(alpha3)
        self.cell = [a1, a2, a3, alpha1, alpha2, alpha3]

        b1, b2, b3, beta1, beta2, beta3 = reciprocol_lattice(*self.cell)
        self.recip_cell = [b1, b2, b3, beta1, beta2, beta3]

        self.B = np.array([
            [b1, b2 * cos(beta3), b3 * cos(beta2)],
            [0.0, b2 * sin(beta3), -b3 * sin(beta2) * cos(alpha1)],
            [0.0, 0.0, 2 * pi / a3]
        ])

    def set_UB(self, hkl0, hkl1, angles0, angles1):
        """calculate the U Matrix using two reflections"""

        # [th, tth, chi] degrees -> [mu, nu, chi] radians
        if angles1 is None:
            angles1 = deepcopy(angles0)
            angles1[0] += 90
        angles0 = [radians(ang) for ang in angles0]
        angles1 = [radians(ang) for ang in angles1]
        self.orientation = [hkl0, hkl1, angles0, angles1]

        # HKL orientation vectors
        h1 = np.atleast_2d(np.array(hkl0)).T
        h2 = np.atleast_2d(np.array(hkl1)).T

        h1c = self.B.dot(h1)
        h2c = self.B.dot(h2)

        t1c = h1c
        t3c = np.cross(h1c.ravel(), h2c.ravel()).reshape(3, 1)
        t2c = np.cross(t3c.ravel(), t1c.ravel()).reshape(3, 1)

        # Reflection vectors in phi frame
        u1p = q_phi(*angles0)
        u2p = q_phi(*angles1)

        t1p = u1p
        t3p = np.cross(u1p.ravel(), u2p.ravel()).reshape(3, 1)
        t2p = np.cross(t3p.ravel(), t1p.ravel()).reshape(3, 1)

        try:
            Tc = np.hstack([t1c/norm(t1c), t2c/norm(t2c), t3c/norm(t3c)])
            Tp = np.hstack([t1p/norm(t1p), t2p/norm(t2p), t3p/norm(t3p)])
        except FloatingPointError:
            raise BaseException("Reflections are parallel. Could not initialise UB-Matrix.")

        U = Tp.dot(inv(Tc))
        self.UB = U.dot(self.B)

    def hkl(self, th, tth, chi):
        """return miller indices for given IRIXS angles"""
        hkl = q_hkl(self.wl, self.UB, radians(th), radians(tth), radians(chi))
        return tuple(hkl.round(3))

    def find_hk_angles(self, hk_list, printout=True):
        """find th and chi angles for list of HK values (ignoring L) with detector at 90°"""
        
        def fitfun(angles, hval, kval):
            hkl = q_hkl(self.wl, self.UB, angles[0], pi/2, angles[1])
            return hkl[0]-hval, hkl[1]-kval

        x0 = [pi/4, self.orientation[2][2]]  # init with th=45, chi=chi0
        
        angle_list = []
        for hk in hk_list:
            result = least_squares(fitfun, x0, args=hk)
            angle = [degrees(x) for x in result.x]
            angle_list.append(angle)
            if printout:
                hkl = q_hkl(self.wl, self.UB, result.x[0], pi/2, result.x[1])
                print(f"({hkl[0]: 5.2f} {hkl[1]: 5.2f} {hkl[2]: 5.2f})", end=" ")
                print(f"th{angle[0]: 9.4f}  chi{angle[1]: 9.4f}")

        return angle_list

if __name__ == "__main__":

    # quick test based on Ca3Ru2O7
    cell = [5.368, 5.599, 19.35, 90, 90, 90]
    f = sixc(cell, [0,0,4], [1,0,0], [29.845, 53.6905, 2.0], energy=2838.5)

    # print hkl for values from grazing to normal
    for th in range(0, 95, 5):
        print(th, f.hkl(th, 90, 2.0))
    print()

    # find angles for given hk and detector at 90° from grazing to normal
    hk_list = [(x, 0) for x in np.arange(-1.5, 1.6, 0.1)]
    angle_list = f.find_hk_angles(hk_list)

    # check it's working as expected
    assert(np.all(f.hkl(45, 90, 2)) == np.all([-0.091, 0, 6.257]))
