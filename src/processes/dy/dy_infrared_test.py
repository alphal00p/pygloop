import math
import random as rndm
from copy import deepcopy
from typing import List

from symbolica import E  # pyright: ignore

from processes.dy.dy_integrand import (
    RoutedIntegrand,  # ruff: ignore
    evaluate_integrand,
)


class infrared_test(object):
    def __init__(self, L, process, routed_integrands: List[RoutedIntegrand]):
        self.routed_integrands = routed_integrands
        self.L = L
        self.process = process
        self.evaluators = [
            evaluate_integrand(L, process, deepcopy(rout)) for rout in routed_integrands
        ]

    def concretise_scalar_products(self, integrand):

        if self.process == "DY":
            integrand = integrand.replace(E("m(a)^2"), E("4*z*sp3D(p(1),p(1))"))

        integrand = integrand.replace(
            E("sp3D(w_(x_),z_(y_))"),
            E("w_(x_,1)*z_(y_,1)+w_(x_,2)*z_(y_,2)+w_(x_,3)*z_(y_,3)"),
        )

        return integrand

    def approach_limits(self, sqrts_s):
        replacements = []
        for routed_integrand in self.routed_integrands:
            these_replacements = routed_integrand.replacements
            if len(these_replacements) > 0:
                replacements.append(these_replacements)

        replacements = [list(t) for t in dict.fromkeys(tuple(x) for x in replacements)]

        if len(replacements) > 0:
            for r in replacements:
                patt = r[0]
                kc = r[0].replace(r[0], r[1]) + E("10^(-ep)*vperp")
                kc = self.concretise_scalar_products(kc)
                kc_comps = [
                    kc.replace(E("x_(y_)"), E(f"x_(y_,{i})")).replace(
                        E("vperp"), E(f"vperp({i})")
                    )
                    for i in range(1, 4)
                ]
                id_c = patt.replace(E("k(x_)"), E("x_"))
                vp = [1 / math.sqrt(2), 1 / math.sqrt(2), 0]
                ks = []
                for j in range(0, self.L):
                    ks.append([rndm.uniform(-sqrts_s, sqrts_s) for rr in range(0, 3)])
                ks = [[0.1, 0.2, -0.3]]
                p1 = [0, 0, 1]
                p2 = [0, 0, -1]

                for i, k in enumerate(ks):
                    for j in range(0, 3):
                        for n in range(0, 3):
                            kc_comps[n] = kc_comps[n].replace(
                                E(f"k({i},{j + 1})"), k[j]
                            )

                for j in range(0, 3):
                    for n in range(0, 3):
                        kc_comps[n] = kc_comps[n].replace(E(f"p(1,{j + 1})"), p1[j])
                        kc_comps[n] = kc_comps[n].replace(E(f"p(2,{j + 1})"), p2[j])
                        kc_comps[n] = kc_comps[n].replace(E(f"vperp({j + 1})"), vp[j])

                ks[int(str(id_c))] = [kc_comps[i] for i in range(0, 3)]

                z = 0.6  # rndm.uniform(0, 1)

                print(
                    "\033[31mApproaching limit:\033[0m",
                    f"\033[31m{r[0]}\033[0m",
                    "\033[31m->\033[0m",
                    f"\033[31m{r[1]}\033[0m",
                )

                for ep in range(3, 4):
                    print(
                        f"\033[33mep: {10 ** (-ep)} - k: {k} - p1: {p1} - p2: {p2} - z: {z}\033[0m"
                    )
                    print(
                        "\033[34mApproaching limit:\033[0m",
                        f"\033[34m{r[0]}\033[0m",
                        "\033[34m->\033[0m",
                        f"\033[34m{r[1]}\033[0m",
                    )

                    for cut_graph, cut_graph_evaluator in zip(
                        self.routed_integrands, self.evaluators
                    ):
                        # if len(cut_graph.cut_graph.partition[1]) == 1:
                        k = [
                            [E(str(kij)).replace(E("ep"), ep) for kij in ki]
                            for ki in ks
                        ]
                        print("------------------------------")
                        print(k)
                        print(
                            f"\033[32m{cut_graph.cut_graph.graph.get_name()} : {cut_graph_evaluator.eval(k, p1, p2, z)}\033[0m"
                        )

        else:
            print("The diagram is infrared finite")
