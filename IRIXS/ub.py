""" basic six-circle diffractometer simulator """
import numpy as np
from numpy import pi, sin, cos, radians
from numpy.linalg import norm, inv, multi_dot
from tools import reciprocol_lattice


def rotation_matricies(mu, nu, chi, eta=0, delta=0, phi=0):
    """ matrices corresponding to the rotation of the circles """

    mu = radians(mu)
    MU = np.array([
        [1, 0, 0],
        [0, cos(mu), -sin(mu)],
        [0, sin(mu), cos(mu)]
    ])

    nu = radians(nu)
    NU = np.array([
        [1, 0, 0],
        [0, cos(nu), -sin(nu)],
        [0, sin(nu), cos(nu)]
    ])

    chi = radians(chi)
    CHI = np.array([
        [cos(chi), 0, sin(chi)],
        [0, 1, 0],
        [-sin(chi), 0, cos(chi)]
    ])

    eta = radians(eta)
    ETA = np.array([
        [cos(eta), sin(eta), 0],
        [-sin(eta), cos(eta), 0],
        [0, 0, 1]
    ])

    delta = radians(delta)
    DELTA = np.array([
        [cos(delta), sin(delta), 0],
        [-sin(delta), cos(delta), 0],
        [0, 0, 1]
    ])

    phi = radians(phi)
    PHI = np.array([
        [cos(phi), sin(phi), 0],
        [-sin(phi), cos(phi), 0],
        [0, 0, 1]
    ])

    return MU, NU, CHI, ETA, DELTA, PHI


def q_phi(mu, nu, chi, eta=0, delta=0, phi=0):
    """Calculate hkl in the phi frame, in units of 2*pi/lambda."""

    MU, NU, CHI, ETA, DELTA, PHI = rotation_matricies(mu, nu, chi, eta, delta, phi)

    q_lab = NU.dot(DELTA) - np.eye(3)
    q_lab = q_lab.dot(np.array([[0], [1], [0]]))

    Z = multi_dot([inv(PHI), inv(CHI), inv(ETA), inv(MU), q_lab])
    return Z


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
        reflection values for U orientation matrix [hkl0, hkl1, th0, tth0, chi0]
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
    hkl(mu, nu, chi, eta, delta, phi) -> np.array((h, k, l))
        Return miller indicies for given circle angles
    set_B(a1, a2, a3, alpha1, alpha2, alpha3)
        Set lattice parameters and B matrix
    set_UB(hkl0, hkl1, th0, tth0, chi0)
        Set U orientation Matrix and UB-matrix
    """

    def __init__(self, cell, hkl0, hkl1, th0, tth0, chi0, energy=2838.5):
        """
        Parameters
        ----------
        cell : list
            lattice parameters [a(Å), b(Å), c(Å), alpha(°), beta(°), gamma(°)]
        hkl0 : list
            first reflection [h, k, l]
        hkl1 : list
            perp. second reflection [h, k, l]
        th0 : float
            horiz angle for hkl0 (°)
        tth0 : float
            horiz two-theta for hkl0 (°)
        chi0 : float
            chi for hkl0 (°)
        energy : float
            x-ray energy (eV)
        """
        self.energy = energy
        self.wl = 12398 / energy  # eV -> Å
        self.set_B(*cell)
        self.set_UB(hkl0, hkl1, th0, tth0, chi0)

    def set_B(self, a1, a2, a3, alpha1, alpha2, alpha3):
        """calculate the B matrix from crystal lattice parameters"""

        alpha1, alpha2, alpha3 = radians(alpha1), radians(alpha2), radians(alpha3)
        self.cell = [a1, a2, a3, alpha1, alpha2, alpha3]

        b1, b2, b3, beta1, beta2, beta3 = reciprocol_lattice(*self.cell)
        self.reciprocol_cell = [b1, b2, b3, beta1, beta2, beta3]

        self.B = np.array([
            [b1, b2 * cos(beta3), b3 * cos(beta2)],
            [0.0, b2 * sin(beta3), -b3 * sin(beta2) * cos(alpha1)],
            [0.0, 0.0, 2 * pi / a3]
        ])

    def set_UB(self, hkl0, hkl1, th0, tth0, chi0):
        """calculate the U Matrix using two reflections"""

        self.orientation = [hkl0, hkl1, th0, tth0, chi0]

        # HKL orientation vectors
        h1 = np.atleast_2d(np.array(hkl0)).T
        h2 = np.atleast_2d(np.array(hkl1)).T

        h1c = self.B.dot(h1)
        h2c = self.B.dot(h2)

        t1c = h1c
        t3c = np.cross(h1c.ravel(), h2c.ravel()).reshape(3, 1)
        t2c = np.cross(t3c.ravel(), t1c.ravel()).reshape(3, 1)

        # Reflection vectors in phi frame
        u1p = q_phi(th0, tth0, chi0)
        u2p = q_phi(th0+90.0, tth0, chi0)

        t1p = u1p
        t3p = np.cross(u1p.ravel(), u2p.ravel()).reshape(3, 1)
        t2p = np.cross(t3p.ravel(), t1p.ravel()).reshape(3, 1)

        t1c = t1c / norm(t1c)
        t2c = t2c / norm(t2c)
        t3c = t3c / norm(t3c)

        t1p = t1p / norm(t1p)
        t2p = t2p / norm(t2p)
        t3p = t3p / norm(t3p)

        Tc = np.hstack([t1c, t2c, t3c])
        Tp = np.hstack([t1p, t2p, t3p])

        U = Tp.dot(inv(Tc))
        self.UB = U.dot(self.B)

    def hkl(self, mu, nu=90, chi=None, eta=0, delta=0, phi=0):
        """return miller indices for given circle angles"""

        if chi is None:
            chi = self.orientation[-1]  # use alignment chi

        MU, NU, CHI, ETA, DELTA, PHI = rotation_matricies(mu, nu, chi, eta, delta, phi)

        q_lab = NU.dot(DELTA) - np.eye(3)
        q_lab = q_lab.dot(np.array([[0], [2 * pi / self.wl], [0]]))

        hkl = multi_dot([inv(self.UB), inv(PHI), inv(CHI), inv(ETA), inv(MU), q_lab])

        return np.round(hkl.ravel(), 3)


if __name__ == "__main__":

    # quick test based on Ca3Ru2O7
    cell = [5.368, 5.599, 19.35, 90, 90, 90]
    f = sixc(cell, [0,0,4], [1,0,0], 29.845, 53.6905, 2.0, 2838.5)

    # print hkl for values from grazing to normal
    for th in range(0, 95, 5):
        print(th, f.hkl(th))

    # check it's working as expected
    assert(np.all(f.hkl(45)) == np.all([-0.091, 0, 6.257]))
