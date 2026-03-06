# src/processes/dy/dy_compiled_bundle.py
from __future__ import annotations

import json
import math
import os
import shutil
import time
from copy import deepcopy
from dataclasses import dataclass

from symbolica import E, Expression, S

from utils.utils import (
    EVALUATORS_FOLDER,
    ParamBuilder,
    PygloopEvaluator,
    expr_to_string,
    pygloopException,
)
from utils.vectors import Vector

pjoin = os.path.join

from itertools import product

from processes.dy.dy_graph_utils import (
    _strip_quotes,
)
from processes.dy.dy_integrand import (
    RoutedIntegrand,  # ruff: ignore
)


def heaviside_theta(x):
    if x > 0:
        return 1
    else:
        return 0


class evaluate_integrand(object):
    def impose_rest_frame(self, integrand):
        return integrand.replace(E("p(x_,1)"), E("0")).replace(E("p(x_,2)"), E("0"))

    def concretise_scalar_products(self, integrand):

        if self.process == "DY":
            integrand = integrand.replace(E("m(a)^2"), E("4*z*sp3D(p(1),p(1))"))

        integrand = integrand.replace(
            E("sp3D(w_(x_),z_(y_))"),
            E("w_(x_,1)*z_(y_,1)+w_(x_,2)*z_(y_,2)+w_(x_,3)*z_(y_,3)"),
        )

        return integrand

    def t_parametrise(self, integrand):

        t = S("t")
        integrand = integrand.replace(
            E("k(x___,y___)"),
            t * E("k(x___,y___)"),
        )

        ## NEW: rescale z
        integrand = integrand.replace(
            E("z"),
            t**2 * E("z"),
        )

        return integrand

    def set_e_surface(self):
        final_moms = []
        e_surface = E(
            "-(4*p(1,3)^2)^(1/2)"
        )  # -E("s^(1/2)")  # E("-4*p(1,3)^2")  # check

        for ep in self.routed_integrand.cut_graph.final_cut:
            ep_atts = ep.get_attributes()
            id = ep_atts["id"]

            for e in self.routed_integrand.cut_graph.graph.get_edges():
                e_atts = e.get_attributes()
                if e_atts["id"] == id:
                    k_keys = ["routing_k" + str(i) for i in range(0, self.L)]
                    loop_coeff = [E(e_atts[rout]) for rout in k_keys]
                    mass = (
                        E("0")
                        if _strip_quotes(str(e_atts["particle"])) != "a"
                        else E("m(a)")
                    )
                    final_mom = (
                        sum(loop_coeff[i] * E(f"k({i})") for i in range(0, self.L))
                        + E(e_atts["routing_p1"]) * E("p(1)")
                        + E(e_atts["routing_p2"]) * E("p(2)")
                    )
                    final_moms.append([
                        final_mom,
                        mass,
                    ])
                    e_surface += (self.sp3D(final_mom, final_mom) + mass**2) ** E("1/2")

        e_surface = self.concretise_scalar_products(e_surface)
        if len(self.routed_integrand.replacements) > 0:
            patts = [
                self.concretise_scalar_products(
                    self.routed_integrand.replacements[0]
                ).replace(E("x_(y_)"), E(f"x_(y_,{i})"))
                for i in range(1, 4)
            ]
            repls = [
                self.concretise_scalar_products(
                    self.routed_integrand.replacements[1]
                ).replace(E("x_(y_)"), E(f"x_(y_,{i})"))
                for i in range(1, 4)
            ]
            for i in range(0, 3):
                e_surface = e_surface.replace(patts[i], repls[i])

        e_surface = self.concretise_scalar_products(e_surface)
        e_surface = self.concretise_scalar_products(e_surface)
        e_surface = self.impose_rest_frame(e_surface)

        rescaled_e_surface = e_surface.replace(E("k(0,x_)"), E("t*k(0,x_)"))

        ## NEW: RESCALE z
        rescaled_e_surface = rescaled_e_surface.replace(E("z"), E("t^2*z"))

        # print(rescaled_e_surface)

        return rescaled_e_surface

    def __init__(
        self,
        L,
        process,
        routed_integrand,
        n_hornerscheme_iterations,  #: int | None = None,
        n_cpe_iterations,  #: int | None = None,
        observable_params,
    ):
        self.L = L
        self.process = process
        self.routed_integrand = routed_integrand

        self.symbols = []
        for i in range(self.L):
            for j in range(1, 4):
                self.symbols.append(E(f"k({i},{j})"))
        for i in range(1, 3):
            for j in range(1, 4):
                self.symbols.append(E(f"p({i},{j})"))

        if process == "DY":
            self.symbols.append(E("z"))

        self.symbols.append(E("t"))
        # MUV
        self.symbols.append(E("mUV"))

        self.routed_integrand.integrand = self.concretise_scalar_products(
            self.routed_integrand.integrand
        )
        self.routed_integrand.integrand = self.impose_rest_frame(
            self.routed_integrand.integrand
        )
        self.routed_integrand.integrand = self.t_parametrise(
            self.routed_integrand.integrand
        )

        self.routed_integrand.integrand = self.routed_integrand.integrand.replace(
            E("TR"), E("1/2")
        )
        self.routed_integrand.integrand = self.routed_integrand.integrand.replace(
            E("GC_11"), E("1")
        )
        self.routed_integrand.integrand = self.routed_integrand.integrand.replace(
            E("GC_1"), E("1")
        )

        self.observable_params = observable_params

        theta_x = self.routed_integrand.integrand.match(E("Θ(x_)"))
        self.theta_expressions: list[Expression] = []
        self.theta_val = []

        if theta_x is not None:
            for th in theta_x:
                theta_expr = th[E("x_")]
                self.theta_expressions.append(theta_expr)
                self.theta_val.append(theta_expr.evaluator({}, {}, self.symbols))

        theta_zmin_expr = E("t^2*z") - E(str(observable_params["zmin"]))
        self.theta_expressions.append(theta_zmin_expr)
        self.theta_val.append(theta_zmin_expr.evaluator({}, {}, self.symbols))

        theta_zmax_expr = E(str(observable_params["zmax"])) - E("t^2*z")
        self.theta_expressions.append(theta_zmax_expr)
        self.theta_val.append(theta_zmax_expr.evaluator({}, {}, self.symbols))

        self.routed_integrand.integrand = self.routed_integrand.integrand.replace(
            E("Θ(x_)"), E("1")
        )

        self.sp3D = S("sp3D", is_linear=True, is_symmetric=True)

        self.e_surface = self.set_e_surface()

        ht_prefactor = 2.0 / math.sqrt(math.pi)
        ht = (-(E("t") ** 2)).exp() * E(f"{ht_prefactor:.16e}")
        jacobian = E("t") ** 5 / self.e_surface.derivative(E("t"))

        self.routed_integrand.integrand = (
            self.routed_integrand.integrand * ht * jacobian
        )

        ## ADD THETA OF t^2 z

        self.evaluator = self.routed_integrand.integrand.evaluator(
            {},
            {},
            self.symbols,
            iterations=n_hornerscheme_iterations,
            cpe_iterations=n_cpe_iterations,
        )

        # self.is_rescaling_necessary = True
        # if self.e_surface.derivative(E("t")) == E("0"):
        #    self.is_rescaling_necessary = False
        #    self.routed_integrand.integrand = self.routed_integrand.integrand.replace(
        #        E("z"), E("1")
        #    )

    def set_t_value(self, k, p1, p2, z):

        if self.process == "DY":
            # if (
            #    len(self.routed_integrand.cut_graph.final_cut) > 1
            # ):
            s = 4 * p1[2] ** 2

            e_surface = self.e_surface
            print(e_surface)

            for j in range(0, self.L):
                for i in range(0, 3):
                    e_surface = e_surface.replace(E(f"k({j},{i + 1})"), k[j][i])
            for i in range(0, 3):
                e_surface = e_surface.replace(E(f"p(1,{i + 1})"), p1[i])
            for i in range(0, 3):
                e_surface = e_surface.replace(E(f"p(2,{i + 1})"), p2[i])
            e_surface = e_surface.replace(E("s"), s)
            e_surface = e_surface.replace(E("z"), z)

            t_sol = e_surface.nsolve(E("t"), 1.0)
            return t_sol
            # else:
            #    return 1

        return 1

    def debug_printout(self, k, p1, p2, z):
        momenta = []

        tstar = self.set_t_value(k, p1, p2, z)

        # print(k)
        input_k = {
            E(f"k({i},{j + 1})"): k[i][j]
            for (i, j) in product(range(0, self.L), range(0, 3))
        }
        input_p1 = {E(f"p(1,{j + 1})"): p1[j] for j in range(0, 3)}
        input_p2 = {E(f"p(2,{j + 1})"): p2[j] for j in range(0, 3)}
        input = input_k | input_p1 | input_p2
        input[E("z")] = z
        input[E("t")] = tstar

        for e in self.routed_integrand.cut_graph.graph.get_edges():
            e_atts = e.get_attributes()

            k_keys = ["routing_k" + str(i) for i in range(0, self.L)]
            loop_coeff = [E(e_atts[rout]) for rout in k_keys]

            mass_sq = (
                E("0")
                if _strip_quotes(str(e_atts["particle"])) != "a"
                else E("4*z*t^2*p(1,3)^2")
            )
            mom = (
                sum(loop_coeff[i] * E(f"k({i})") for i in range(0, self.L))
                + E(e_atts["routing_p1"]) * E("p(1)")
                + E(e_atts["routing_p2"]) * E("p(2)")
            )
            momenta.append([mom, mass_sq, e_atts["id"]])

        energies = {}
        masses = {}
        qmomenta = {}
        for mom, mass_sq, id in momenta:
            rep = self.routed_integrand.replacements
            if len(rep) > 0:
                patt = rep[0]
                repl = rep[1]
                mom = mom.replace(patt, repl)

            mom = self.concretise_scalar_products(mom)
            mom = self.t_parametrise(mom)
            # print(mom)
            mom3d = [
                mom.replace(E("k(x_)"), E(f"k(x_,{i})")).replace(
                    E("p(x_)"), E(f"p(x_,{i})")
                )
                for i in range(1, 4)
            ]

            for key, val in input.items():
                for i in range(0, 3):
                    mom3d[i] = mom3d[i].replace(key, val)
                    mass_sq = mass_sq.replace(key, val)

            energies[E(f"E({id})")] = (
                mom3d[0] ** 2 + mom3d[1] ** 2 + mom3d[2] ** 2 + mass_sq
            ) ** E("1/2")
            masses[E(f"m({id})^2")] = mass_sq
            qmomenta[E(f"q({id})")] = mom3d

        print(input)
        print(energies)
        print(masses)
        # print(self.routed_integrand.integrand)
        emr_int = deepcopy(self.routed_integrand.emr_integrand)
        print(emr_int)

        # for key, val in energies.items():
        #    emr_int = emr_int.replace(key, val)

        # print(emr_int)

    def param_builder(self, k, p1, p2, z):
        param_list = []
        for i in range(self.L):
            for j in range(0, 3):
                param_list.append(k[i][j])
        for j in range(0, 3):
            param_list.append(p1[j])
        for j in range(0, 3):
            param_list.append(p2[j])
        if self.process == "DY":
            param_list.append(z)

        t_sol = self.set_t_value(k, p1, p2, z)

        param_list.append(t_sol)

        # MUV
        param_list.append(1)

        return param_list

    def eval(self, k, p1, p2, z):
        param_list = self.param_builder(k, p1, p2, z)
        param_list = [float(str(v)) for v in param_list]

        theta = 1
        for th in self.theta_val:
            print("x,1-x: ", th.evaluate(param_list)[0][0])
            theta *= heaviside_theta(th.evaluate(param_list)[0][0])

        print(param_list)
        self.debug_printout(k, p1, p2, z)
        # print(self.routed_integrand.integrand)

        return self.evaluator.evaluate(param_list) * theta


@dataclass
class DYCompiledTerm:
    evaluator_name: str
    e_surface: Expression
    theta_expressions: list[Expression]
    t_initial_guess: float


class DYCompiledBundle:
    METADATA_FILE = "bundle_metadata.json"

    def __init__(
        self,
        process: str,
        integrand_name: str,
        n_loops: int,
        terms: list[DYCompiledTerm],
        evaluators: dict[str, PygloopEvaluator],
    ):
        self.process = process
        self.integrand_name = integrand_name
        self.n_loops = n_loops
        self.terms = terms
        self.evaluators = evaluators
        self.t_symbol = E("t")

    @staticmethod
    def _bundle_dir(process: str, integrand_name: str) -> str:
        return pjoin(EVALUATORS_FOLDER, process, integrand_name)

    @staticmethod
    def _build_param_builder(symbols: list[Expression]) -> ParamBuilder:
        pb = ParamBuilder()
        pb.add_parameter_list((E("dummy"),), len(symbols))
        pb.order = []
        pb.positions = {}
        for i, s in enumerate(symbols):
            head = (s,)
            pb.order.append(head)
            pb.positions[head] = (i, i + 1)
            pb.np[i] = 0.0
        return pb

    # src/processes/dy/dy_evaluators.py

    @staticmethod
    def _normalize_symbol_key(key: str) -> str:
        # drop namespace prefix if present, keep tail symbol form
        if "::" in key:
            key = key.split("::")[-1]
        return key

    @classmethod
    def create_from_evaluators(
        cls,
        process: str,
        integrand_name: str,
        n_loops: int,
        observable: str,
        evaluators: list,
    ) -> "DYCompiledBundle":
        if len(evaluators) == 0:
            raise pygloopException(
                "Cannot create DYCompiledBundle from empty evaluator list."
            )

        out_dir = cls._bundle_dir(process, integrand_name)
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        loaded_evaluators: dict[str, PygloopEvaluator] = {}
        terms: list[DYCompiledTerm] = []

        for i, ev in enumerate(evaluators):
            evaluator_name = f"term_{i}_integrand"
            pb = cls._build_param_builder(ev.symbols)

            pe = PygloopEvaluator(
                evaluator=ev.evaluator,
                param_builder=pb,
                name=evaluator_name,
                output_length=1,
                additional_data={
                    "process": process,
                    "integrand_name": integrand_name,
                    "observable": observable,
                    "term_id": i,
                },
                complexified=False,
            )
            pe.compile(
                out_dir,
                optimization_level=3,  # max in your current setup
                native=True,
                inline_asm="default",
            )
            pe.save(out_dir)

            loaded_evaluators[evaluator_name] = PygloopEvaluator.load(
                out_dir, evaluator_name
            )

            terms.append(
                DYCompiledTerm(
                    evaluator_name=evaluator_name,
                    e_surface=ev.e_surface,
                    theta_expressions=getattr(ev, "theta_expressions", []),
                    t_initial_guess=1.0,
                )
            )

        metadata = {
            "process": process,
            "integrand_name": integrand_name,
            "n_loops": n_loops,
            "terms": [
                {
                    "evaluator_name": t.evaluator_name,
                    "e_surface": t.e_surface.to_canonical_string(),
                    "theta_expressions": [
                        th.to_canonical_string() for th in t.theta_expressions
                    ],
                    "t_initial_guess": t.t_initial_guess,
                }
                for t in terms
            ],
        }
        with open(pjoin(out_dir, cls.METADATA_FILE), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return cls(process, integrand_name, n_loops, terms, loaded_evaluators)

    @classmethod
    def load(cls, process: str, integrand_name: str) -> "DYCompiledBundle":
        out_dir = cls._bundle_dir(process, integrand_name)
        metadata_path = pjoin(out_dir, cls.METADATA_FILE)
        if not os.path.isfile(metadata_path):
            raise pygloopException(f"Missing bundle metadata: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        terms: list[DYCompiledTerm] = []
        evaluators: dict[str, PygloopEvaluator] = {}

        for t in metadata["terms"]:
            name = t["evaluator_name"]
            evaluators[name] = PygloopEvaluator.load(out_dir, name)
            terms.append(
                DYCompiledTerm(
                    evaluator_name=name,
                    e_surface=E(t["e_surface"]),
                    theta_expressions=[E(x) for x in t["theta_expressions"]],
                    t_initial_guess=float(t.get("t_initial_guess", 1.0)),
                )
            )

        return cls(
            process=metadata["process"],
            integrand_name=metadata["integrand_name"],
            n_loops=int(metadata["n_loops"]),
            terms=terms,
            evaluators=evaluators,
        )

    @staticmethod
    def _strip_python_namespace(expr: Expression) -> Expression:
        # Avoid constructing namespaced symbols with potentially wrong attributes.
        canon = expr.to_canonical_string().replace("python::{}::", "")
        return E(canon)

    @staticmethod
    def _replace_values(expr: Expression, values: dict[str, float]) -> Expression:
        out = DYCompiledBundle._strip_python_namespace(expr)
        for k, v in values.items():
            out = out.replace(E(k), E(f"{v:.16e}"))
        return out

    @staticmethod
    def _set_inputs(pe: PygloopEvaluator, values: dict[str, float]) -> None:
        for head in pe.param_builder.order:
            raw_key = head[0].to_canonical_string()
            key = DYCompiledBundle._normalize_symbol_key(raw_key)

            if key not in values:
                raise pygloopException(
                    f"Missing value for symbol '{raw_key}' (normalized '{key}') in compiled evaluator '{pe.name}'."
                )
            pe.param_builder.set_parameter_values(head, [values[key]])

    def evaluate(
        self,
        loop_momenta: list[Vector],
        p1: Vector,
        p2: Vector,
        z: float,
        m_uv: float = 1.0,
    ) -> complex:

        t0 = time.perf_counter()

        vals: dict[str, float] = {}
        for i, k in enumerate(loop_momenta):
            kx, ky, kz = k.to_list()
            vals[f"k({i},1)"] = float(kx)
            vals[f"k({i},2)"] = float(ky)
            vals[f"k({i},3)"] = float(kz)

        p1x, p1y, p1z = p1.to_list()
        p2x, p2y, p2z = p2.to_list()
        vals["p(1,1)"] = float(p1x)
        vals["p(1,2)"] = float(p1y)
        vals["p(1,3)"] = float(p1z)
        vals["p(2,1)"] = float(p2x)
        vals["p(2,2)"] = float(p2y)
        vals["p(2,3)"] = float(p2z)
        vals["z"] = float(z)
        vals["mUV"] = float(m_uv)

        eq_vals = deepcopy(dict(vals))
        # eq_vals["s"] = 4.0 * float(p1.to_list()[2]) ** 2

        total = 0.0 + 0.0j
        # before loop over terms
        total = 0.0 + 0.0j

        t1 = time.perf_counter()

        print("input building time: ", t1 - t0)

        # Sum over all cut graphs
        for term in self.terms:
            t3 = time.perf_counter()

            eq = self._replace_values(term.e_surface, eq_vals)

            try:
                t_sol = float(str(eq.nsolve(self.t_symbol, 1.0)))
                if not math.isfinite(t_sol):
                    continue
            except Exception as e:
                # Treat failed t-solve as zero contribution at this sample
                raise ValueError("error in t solving: ", e)

            t4 = time.perf_counter()

            print("t solving time: ", t4 - t3)

            vals_with_t = dict(vals)
            vals_with_t["t"] = t_sol
            eq_vals_with_t = dict(eq_vals)
            eq_vals_with_t["t"] = t_sol

            theta = 1
            for th in term.theta_expressions:
                try:
                    th_str = th.to_canonical_string()
                    th_args = {k: v for k, v in eq_vals_with_t.items() if k in th_str}
                    th_val = self._replace_values(th, eq_vals_with_t)
                    print("theta expr:", th_str)
                    print("theta args:", th_args)
                    print("theta value:", th_val)

                    # th_val = self._replace_values(th, eq_vals_with_t)
                    if float(str(th_val)) <= 0.0:
                        theta = 0
                        break
                except Exception:
                    theta = 0
                    break
            if theta == 0:
                continue

            t44 = time.perf_counter()

            print("theta adding time: ", t44 - t4)

            pe = self.evaluators[term.evaluator_name]
            try:
                self._set_inputs(pe, vals_with_t)
                total += complex(pe.evaluate(eager=False)[0])

            except Exception:
                continue

            t5 = time.perf_counter()
            print("integrand evaluation time: ", t5 - t4)

        print("total")
        print(loop_momenta[0])
        print(p1)
        print(p2)
        print(z)
        print(t_sol)
        print(total)

        return total


class compile_integrands(object):
    def __init__(self, L, process, name, observable, evaluators):
        self.L = L
        self.process = process
        self.observable = observable
        self.evaluators = evaluators
        self.name = name

    def save_compiled_integrand(self):

        DYCompiledBundle.create_from_evaluators(
            process=self.process,
            integrand_name=self.name,
            n_loops=self.L,
            observable=self.observable,
            evaluators=self.evaluators,
        )
