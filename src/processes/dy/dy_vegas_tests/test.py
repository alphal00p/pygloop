import math
import time
from pyexpat.errors import XML_ERROR_SUSPENDED

import numpy as np
import vegas
from symbolica import E, S


def sp4(v, w):
    return v[0] * w[0] - v[1] * w[1] - v[2] * w[2] - v[3] * w[3]


def sq4(v):
    return sp4(v, v)


def en(v, msq):
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2 + msq)


def h(t):
    return 2 * math.exp(-(t**2)) / math.pi ** (1 / 2)


class bubble(object):
    def __init__(self, p1, p2, zmin, eq=1.0, g=1.0, tf=0.5):
        self.p1 = p1
        self.p2 = p2
        self.zmin = zmin
        self.eq = eq
        self.g = g
        self.tf = tf
        self.s = sq4(self.p1 + self.p2)

    def integrand(self, k, z):

        msq = z * self.s
        k4 = [-en(k, msq), k[0], k[1], k[2]]
        numerator = -16 * (
            -(sp4(self.p1, self.p2))
            * (2 * sp4(k4, self.p1) + 2 * sp4(self.p1, self.p2))
        )
        """
        -16 * (
            sp4(k4, self.p2) * (sp4(self.p1, self.p1) - sp4(self.p2, self.p2))
            - (sp4(self.p1, self.p2) + sp4(self.p2, self.p2))
            * (
                2 * sp4(k4, self.p1)
                + sp4(self.p1, self.p1)
                + 2 * sp4(self.p1, self.p2)
                + sp4(self.p2, self.p2)
            )
        )"""
        e0 = en(k, 0)
        em = en(k, msq)
        if e0 <= 1e-15 or em <= 1e-15:
            return 0.0
        inverse_energies = 1 / (2 * e0 * 2 * em)
        propagators = 1 / self.s**2

        return numerator * inverse_energies * propagators

    def observable(self, z):
        if z > self.zmin and z < 1:
            return 1
        else:
            return 0

    def rescaled_integrand(self, k, z):
        msq = z * self.s
        e0 = en(k, 0)
        em = en(k, msq)
        if e0 + em <= 1e-15:
            return 0.0
        t = math.sqrt(self.s) / (e0 + em)
        # print("t: ", t)
        kt = t * k
        zt = t**2 * z
        jacobian = t**5 / (e0 + em) * h(t)
        # print("theta: ", self.observable(zt), " argument: ", zt, " ", 1 - zt)
        return jacobian * self.integrand(kt, zt) * self.observable(zt)

    def gaussian(self, k, z):
        return np.exp(-(en(k, 0) ** 2)) / math.pi ** (3 / 2)

    def test_z(self, k, z):
        e0 = en(k, 0)
        t = math.sqrt(self.s) / (e0)
        kt = t * k
        zt = t**2 * z
        jacobian = 1 / e0

        return t**5 * self.observable(zt) * h(t) * jacobian

    def x_parametrise(self, x):
        ecm = math.sqrt(self.s)
        z = x[3] / (1 - x[3])
        r = x[0] / (1 - x[0]) * ecm
        th = 2 * math.pi * x[1]
        ph = math.pi * x[2]

        spherical_jacobian = r**2 * math.sin(ph)
        x_jacobian = 2 * math.pi**2 / (1 - x[0]) ** 2 / (1 - x[3]) ** 2 * ecm

        k = r * np.array([
            math.cos(th) * math.sin(ph),
            math.sin(th) * math.sin(ph),
            math.cos(ph),
        ])

        # print("momentum input: ", k, " ", z)

        return (
            spherical_jacobian * x_jacobian * self.rescaled_integrand(k, z)
        )  # self.rescaled_integrand(k, z)


def main():
    p1 = np.array([1.0, 0.0, 0.0, 1.0])
    p2 = np.array([1.0, 0.0, 0.0, -1.0])
    zmin = 0
    b = bubble(p1, p2, zmin)
    bx = bubble(p2, p1, zmin)

    def f(x):
        return b.x_parametrise(x) + bx.x_parametrise(x)

    # integ = vegas.Integrator([[0, 1], [0, 1], [0, 1], [0, 1]])
    # res = integ(f, nitn=5, neval=1000000)
    # print(res.mean, res.sdev)
    #

    t0 = time.perf_counter()

    t = S("t")
    e = E(
        "(4*z*t^2*p(1,3)^2+t^2*k(0,1)^2+t^2*k(0,2)^2+t^2*k(0,3)^2)^(1/2)+(2*t*k(0,3)*p(1,3)+2*t*k(0,3)*p(2,3)+2*p(1,3)*p(2,3)+t^2*k(0,1)^2+t^2*k(0,2)^2+t^2*k(0,3)^2+p(1,3)^2+p(2,3)^2)^(1/2)-(4*p(1,3)^2)^(1/2)"
    )

    vals = {
        E("k(0,1)"): 0.1,
        E("k(0,2)"): 0.2,
        E("k(0,3)"): 0.3,
        E("p(1,1)"): 0.0,
        E("p(1,2)"): 0.0,
        E("p(1,3)"): 1,
        E("p(2,1)"): 0.0,
        E("p(2,2)"): 0.0,
        E("p(2,3)"): -1,
        E("z"): 0.6,
    }

    t1 = time.perf_counter()
    print("input preparation (ms): ", t1 - t0)

    for k, v in vals.items():
        e = e.replace(k, E(f"{float(v):.16e}"))

    t2 = time.perf_counter()
    print("filling in values (ms): ", t2 - t1)

    print(e.nsolve(t, 1))

    t3 = time.perf_counter()
    print("solving in t (ms): ", t3 - t2)

    #
    # ks = np.array([
    #    6.0240036246689332e-02,
    #    -2.7254303355454151e-01,
    #    1.1859758597063783e00,
    # ])
    # zs = 0.8608284805105424
    # print(b.rescaled_integrand(ks, zs))


#
# xs = np.array([
#    0.29597685062575274,
#    0.4711680056411622,
#    0.42223863759598107,
#    0.00941049709105633,
# ])
# print(b.x_parametrise(xs) + bx.x_parametrise(xs))


if __name__ == "__main__":
    main()
