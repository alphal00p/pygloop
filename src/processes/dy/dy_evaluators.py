# src/processes/dy/dy_compiled_bundle.py
from __future__ import annotations

import json
import math
import os
import shutil
import time
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import lru_cache

from symbolica import E, Expression, S

from utils.utils import (
    EVALUATORS_FOLDER,
    ParamBuilder,
    PygloopEvaluator,
    pygloopException,
)
from utils.vectors import Vector

pjoin = os.path.join

from itertools import product

from processes.dy.dy_graph_utils import (
    _strip_quotes,
)

# MT = 0.69200000000000000000000000000000  # s=500
# MT = 0.46133333333333333333333333333333  # s=750
# MT = 0.346  # s=1000
# MT = 0.173  # s=2000
MT = 173


def heaviside_theta(x):
    if x > 0:
        return 1
    else:
        return 0


def _coerce_numeric_param(value):
    if isinstance(value, (int, float)):
        return float(value)

    value_str = str(value).strip()
    try:
        return float(value_str)
    except ValueError:
        return float(Fraction(value_str))


def _is_ttbar_process(process: str) -> bool:
    return process in {"tt~", "ttbar"}


def _format_float_expr(value: float) -> str:
    return f"{value:.17g}"


def _complex_to_symbolica_expr(value: complex) -> Expression:
    value = complex(value)
    real = 0.0 if abs(value.real) < 1.0e-15 else value.real
    imag = 0.0 if abs(value.imag) < 1.0e-15 else value.imag

    if real == 0.0 and imag == 0.0:
        return E("0")

    if imag == 0.0:
        return E(_format_float_expr(real))

    imag_coeff = E(_format_float_expr(abs(imag)))
    imag_expr = imag_coeff * E("1i")
    if imag < 0.0:
        imag_expr = -imag_expr

    if real == 0.0:
        return imag_expr

    return E(_format_float_expr(real)) + imag_expr


@lru_cache(maxsize=1)
def _sm_ttbar_couplings() -> dict[str, Expression]:
    from ufo_model_loader.commands import load_model

    model, _ = load_model(
        "sm",
        None,
        simplify_model=True,
        wrap_indices_in_lorentz_structures=False,
    )

    coupling_names = {"GC_1", "GC_10", "GC_11"}
    couplings: dict[str, Expression] = {}
    for coupling in model.couplings:
        if coupling.name in coupling_names:
            couplings[coupling.name] = _complex_to_symbolica_expr(coupling.value)

    missing = coupling_names.difference(couplings)
    if missing:
        raise pygloopException(
            "Missing SM ttbar couplings in UFO model load: "
            + ", ".join(sorted(missing))
        )

    return couplings


class evaluate_integrand:
    def _replace_couplings(self, expr: Expression, include_tr: bool) -> Expression:
        if include_tr:
            expr = expr.replace(E("TR"), E("1/2"))

        if _is_ttbar_process(self.process):
            couplings = _sm_ttbar_couplings()
            for coupling_name, coupling_value in couplings.items():
                expr = expr.replace(E(coupling_name), coupling_value)
            return expr

        expr = expr.replace(E("GC_11"), E("1"))
        expr = expr.replace(E("GC_1"), E("1"))
        expr = expr.replace(E("GC_10"), E("1"))
        return expr

    def impose_rest_frame(self, integrand):
        return integrand.replace(E("p(x_,1)"), E("0")).replace(E("p(x_,2)"), E("0"))

    def concretise_scalar_products(self, integrand):

        if self.process == "DY":
            integrand = integrand.replace(E("m(a)^2"), E("4*z*sp3D(p(1),p(1))"))

        if self.process == "tt~":
            integrand = integrand.replace(E("m(t)"), E(str(MT)))

        return integrand.replace(
            E("sp3D(w_(x_),z_(y_))"),
            E("w_(x_,1)*z_(y_,1)+w_(x_,2)*z_(y_,2)+w_(x_,3)*z_(y_,3)"),
        )

    @staticmethod
    def drop_exact_zero_sqrts(expr: Expression) -> Expression:
        out = expr
        while True:
            changed = False
            for match in list(out.match(E("x_^(1/2)"))):
                radicand = match[E("x_")]
                try:
                    if radicand.expand().to_canonical_string() == "0":
                        out = out.replace(radicand ** E("1/2"), E("0"))
                        changed = True
                except Exception:
                    continue
            if not changed:
                return out

    def t_parametrise(self, integrand):

        t = S("t")
        integrand = integrand.replace(
            E("k(x___,y___)"),
            t * E("k(x___,y___)"),
        )

        if self.process == "DY":
            integrand = integrand.replace(
                E("z"),
                t**2 * E("z"),
            )

        return integrand

    def set_e_surface(self):
        final_moms = []
        e_surface = E(
            "-(4*(p(1,1)^2+p(1,2)^2+p(1,3)^2))^(1/2)"
        )  # -E("s^(1/2)")  # E("-4*p(1,3)^2")  # check

        for ep in self.routed_integrand.cut_graph.final_cut:
            ep_atts = ep.get_attributes()
            id = ep_atts["id"]

            for e in self.routed_integrand.cut_graph.graph.get_edges():
                e_atts = e.get_attributes()
                if e_atts["id"] == id:
                    k_keys = ["routing_k" + str(i) for i in range(self.L)]
                    loop_coeff = [E(e_atts[rout]) for rout in k_keys]
                    particle_type = _strip_quotes(str(e_atts["particle"]))
                    mass = (
                        E("0")
                        if particle_type in ["d", "d~", "g", "ghG", "ghG~"]
                        else E(f"m({particle_type})")
                    )
                    final_mom = (
                        sum(loop_coeff[i] * E(f"k({i})") for i in range(self.L))
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
            # Do a simultaneous component substitution. Sequential component
            # replacement rewrites inside already-substituted expressions and
            # breaks rotational covariance for the collinear terms.
            patt = self.concretise_scalar_products(
                self.routed_integrand.replacements[0]
            )
            repl = self.concretise_scalar_products(
                self.routed_integrand.replacements[1]
            )

            tmp_keys = [E(f"__tmp_kcomp_{i}") for i in range(1, 4)]
            patt_comps = [
                patt.replace(E("x_(y_)"), E(f"x_(y_,{i})")) for i in range(1, 4)
            ]
            repl_comps = [
                repl.replace(E("x_(y_)"), E(f"x_(y_,{i})")) for i in range(1, 4)
            ]

            for i in range(3):
                e_surface = e_surface.replace(patt_comps[i], tmp_keys[i])
            for i in range(3):
                e_surface = e_surface.replace(tmp_keys[i], repl_comps[i])

        e_surface = self.concretise_scalar_products(e_surface)
        # e_surface = self.impose_rest_frame(e_surface)

        rescaled_e_surface = e_surface
        for i_loop in range(self.L):
            rescaled_e_surface = rescaled_e_surface.replace(
                E(f"k({i_loop},x_)"), E(f"t*k({i_loop},x_)")
            )
        if self.process == "DY":
            rescaled_e_surface = rescaled_e_surface.replace(E("z"), E("t^2*z"))
        # Reversal note: before this cleanup pass, we returned the rescaled
        # surface directly and left exact sqrt(0) factors unsimplified.
        return self.drop_exact_zero_sqrts(rescaled_e_surface)

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

        self.routed_integrand.integrand = self.concretise_scalar_products(
            self.routed_integrand.integrand
        )
        # self.routed_integrand.integrand = self.impose_rest_frame(
        #    self.routed_integrand.integrand
        # )
        #
        self.routed_integrand.integrand = self.t_parametrise(
            self.routed_integrand.integrand
        )

        self.routed_integrand.integrand = self._replace_couplings(
            self.routed_integrand.integrand,
            include_tr=True,
        )
        self.routed_integrand.integrand = self.routed_integrand.integrand.replace(
            E("m(t)"), E(str(MT))
        )
        self.routed_integrand.integrand = self.routed_integrand.integrand.replace(
            E("MT"), E(str(MT))
        )
        self.routed_integrand.integrand = self.routed_integrand.integrand.replace(
            E("Lambdasq"), E(str(observable_params["Lambdasq"]))
        )
        self.routed_integrand.integrand = self.routed_integrand.integrand.replace(
            E("mUV"), E(str(observable_params.get("mUV", 1.0)))
        )

        self.observable_params = observable_params

        theta_x = self.routed_integrand.integrand.match(E("Θ(x___)"))
        self.theta_expressions: list[Expression] = []
        self.theta_val = []

        if theta_x is not None:
            for th in theta_x:
                theta_expr = th[E("x___")]
                self.theta_expressions.append(theta_expr)
                self.theta_val.append(theta_expr.evaluator({}, {}, self.symbols))

        if len(self.routed_integrand.cut_graph.final_cut) > 1 and self.process == "DY":
            theta_zmin_expr = E("t^2*z") - E(str(observable_params["zmin"]))
            self.theta_expressions.append(theta_zmin_expr)
            self.theta_val.append(theta_zmin_expr.evaluator({}, {}, self.symbols))

        if len(self.routed_integrand.cut_graph.final_cut) > 1 and self.process == "DY":
            theta_zmax_expr = E(str(observable_params["zmax"])) - E("t^2*z")
            self.theta_expressions.append(theta_zmax_expr)
            self.theta_val.append(theta_zmax_expr.evaluator({}, {}, self.symbols))

        self.routed_integrand.integrand = self.routed_integrand.integrand.replace(
            E("Θ(x___)"), E("1")
        )

        self.sp3D = S("sp3D", is_linear=True, is_symmetric=True)

        self.e_surface = self.set_e_surface()

        # print("esurface for:     ", self.routed_integrand.approximation_type)
        # print(self.e_surface)

        ht_prefactor = 2.0 / math.sqrt(math.pi)
        ht = (-(E("t") ** 2)).exp() * E(f"{ht_prefactor:.16e}")

        ## NEW: H FUNCTION
        ht_prefactor = (
            1.0 / 0.1199377719680614473680365016367935162194504519102290907562408570
        )
        ht = (-(E("t") ** 2) - 1 / (E("t") ** 2)).exp() * E(f"{ht_prefactor:.16e}")
        if self.process == "DY":
            jacobian = E("t") ** 5 / self.e_surface.derivative(E("t"))
        if self.process == "tt~":
            jacobian = E("t") ** (3 * self.L) / self.e_surface.derivative(E("t"))

        self.routed_integrand.integrand = (
            self.routed_integrand.integrand * ht * jacobian
        )
        self.integrand_expression = self.routed_integrand.integrand
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

        s = 4 * (p1[0] ** 2 + p1[1] ** 2 + p1[2] ** 2)

        e_surface = self.e_surface

        for j in range(self.L):
            for i in range(3):
                e_surface = e_surface.replace(E(f"k({j},{i + 1})"), k[j][i])
        for i in range(3):
            e_surface = e_surface.replace(E(f"p(1,{i + 1})"), p1[i])
        for i in range(3):
            e_surface = e_surface.replace(E(f"p(2,{i + 1})"), p2[i])
        e_surface = e_surface.replace(E("s"), s)
        if self.process == "DY":
            e_surface = e_surface.replace(E("z"), z)

        return e_surface.nsolve(E("t"), 1.0)

    def debug_printout(self, k, p1, p2, z):
        momenta = []

        tstar = self.set_t_value(k, p1, p2, z)

        input_k = {
            E(f"k({i},{j + 1})"): k[i][j] for (i, j) in product(range(self.L), range(3))
        }
        input_p1 = {E(f"p(1,{j + 1})"): p1[j] for j in range(3)}
        input_p2 = {E(f"p(2,{j + 1})"): p2[j] for j in range(3)}
        input = input_k | input_p1 | input_p2
        input[E("z")] = z
        input[E("t")] = tstar

        for e in self.routed_integrand.cut_graph.graph.get_edges():
            e_atts = e.get_attributes()

            k_keys = ["routing_k" + str(i) for i in range(self.L)]
            loop_coeff = [E(e_atts[rout]) for rout in k_keys]
            particle_type = _strip_quotes(str(e_atts["particle"]))
            mass_sq = E("0")

            if particle_type == "a":
                mass_sq = E("t") ** 2 * E(f"m({particle_type})") ** 2

            if particle_type == "t":
                mass_sq = E(f"m({particle_type})") ** 2

            mom = (
                sum(loop_coeff[i] * E(f"k({i})") for i in range(self.L))
                + E(e_atts["routing_p1"]) * E("p(1)")
                + E(e_atts["routing_p2"]) * E("p(2)")
            )
            momenta.append([mom, mass_sq, e_atts["id"]])

        energies = {}
        masses = {}
        qmomenta = {}

        eval_emr_int = self.routed_integrand.emr_integrand
        eval_emr_int = eval_emr_int.replace(
            E("sp3(x_,y_)"), self.sp3D(E("x_"), E("y_"))
        )
        eval_emr_int = self.concretise_scalar_products(eval_emr_int)
        eval_emr_int = self._replace_couplings(eval_emr_int, include_tr=False)
        eval_emr_int = eval_emr_int.replace(E("TR"), E("1"))

        for mom, mass_sq, id in momenta:
            rep = self.routed_integrand.replacements
            if len(rep) > 0:
                patt = rep[0]
                repl = rep[1]
                mom = mom.replace(patt, repl)

            mom_old = self.concretise_scalar_products(mom)
            mom = self.t_parametrise(mom_old)
            mom3d = [
                mom.replace(E("k(x_)"), E(f"k(x_,{i})")).replace(
                    E("p(x_)"), E(f"p(x_,{i})")
                )
                for i in range(1, 4)
            ]

            for key, val in input.items():
                for i in range(3):
                    mom3d[i] = mom3d[i].replace(key, val)
                    mass_sq = mass_sq.replace(E("m(t)"), E(str(MT))).replace(key, val)

            energies[E(f"E({id})")] = (
                mom3d[0] ** 2 + mom3d[1] ** 2 + mom3d[2] ** 2 + mass_sq
            ) ** E("1/2")
            masses[E(f"m({id})^2")] = mass_sq
            qmomenta[E(f"q({id})")] = mom3d
            for i in range(1, 4):
                eval_emr_int = eval_emr_int.replace(E(f"q({id},{i})"), mom3d[i - 1])
            eval_emr_int = eval_emr_int.replace(E(f"E({id})"), energies[E(f"E({id})")])
            eval_emr_int = eval_emr_int.replace(E("MT"), E(str(MT)))

        ht_prefactor = (
            1.0 / 0.1199377719680614473680365016367935162194504519102290907562408570
        )
        ht = (-(E("t") ** 2) - 1 / (E("t") ** 2)).exp() * E(f"{ht_prefactor:.16e}")
        jacobian = self.e_surface.derivative(E("t"))

        for i in range(1, 4):
            jacobian = jacobian.replace(E(f"p(1,{i})"), p1[i - 1])
            jacobian = jacobian.replace(E(f"p(2,{i})"), p2[i - 1])

        jacobian = jacobian.replace(E("z"), z)

        print(self.routed_integrand.cut_graph.graph)
        print("input parameters: ", input)
        print("energies:", energies)
        print("masses: ", masses)
        print("momenta: ", qmomenta)
        # print(self.routed_integrand.integrand)
        emr_int = deepcopy(self.routed_integrand.emr_integrand)
        print("EMR: ", emr_int)
        print("e_surface : ", self.e_surface)
        print("h(t): ", ht.replace(E("t"), tstar))
        print(jacobian)
        print("delta jacobian : ", jacobian.replace(E("t"), tstar))
        print("Evaluated EMR: ", eval_emr_int)
        # for key, val in energies.items():
        #    emr_int = emr_int.replace(key, val)

        # print("Evaluated EMR: ", emr_int)

    def param_builder(self, k, p1, p2, z):
        param_list = []
        for i in range(self.L):
            for j in range(3):
                param_list.append(k[i][j])  # noqa:PERF401
        for j in range(3):
            param_list.append(p1[j])  # noqa:PERF401
        for j in range(3):
            param_list.append(p2[j])  # noqa:PERF401
        if self.process == "DY":
            param_list.append(z)

        t_sol = self.set_t_value(k, p1, p2, z)

        param_list.append(t_sol)

        return param_list

    def _evaluate_expression_arb(
        self,
        expr: Expression,
        values: dict[Expression, str],
        decimal_digit_precision: int,
    ) -> Decimal | None:
        try:
            value = expr.evaluate_with_prec(values, {}, decimal_digit_precision)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return None

        try:
            decimal_value = DYCompiledBundle._decimal_from_number(value)
        except (InvalidOperation, ValueError, pygloopException):
            return None

        if decimal_value.is_nan() or not decimal_value.is_finite():
            return None
        return decimal_value

    def eval(self, k, p1, p2, z, mode="compiled", decimal_digit_precision=80):
        param_list = self.param_builder(k, p1, p2, z)
        if mode == "arb":
            string_values = {
                symbol: repr(_coerce_numeric_param(value))
                for symbol, value in zip(self.symbols, param_list)
            }

            for th in self.theta_expressions:
                th_value = self._evaluate_expression_arb(
                    th, string_values, decimal_digit_precision
                )
                if th_value is None or th_value <= 0:
                    return Decimal(0)

            value = self._evaluate_expression_arb(
                self.integrand_expression, string_values, decimal_digit_precision
            )
            return value
        if mode != "compiled":
            raise pygloopException(f"Unsupported evaluate_integrand mode '{mode}'.")

        param_list = [_coerce_numeric_param(v) for v in param_list]

        theta = 1
        for th in self.theta_val:
            print("x,1-x: ", th.evaluate(param_list)[0][0])
            theta *= heaviside_theta(th.evaluate(param_list)[0][0])  # th_tol + 1.0e-10)

        # self.debug_printout(k, p1, p2, z)

        if theta == 1:
            return self.evaluator.evaluate(param_list) * theta
        else:
            return 0


@dataclass
class DYCompiledTerm:
    evaluator_name: str
    e_surface: Expression
    theta_expressions: list[Expression]
    integrand_expression: Expression | None
    t_initial_guess: float
    graph_group_name: str | None = None


class DYCompiledBundle:
    METADATA_FILE = "bundle_metadata.json"
    BUNDLE_FORMAT_VERSION = 2

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
        self._t_key = self.t_symbol
        self._z_key = E("z")
        self._muv_key = E("mUV")
        self._p11 = E("p(1,1)")
        self._p12 = E("p(1,2)")
        self._p13 = E("p(1,3)")
        self._p21 = E("p(2,1)")
        self._p22 = E("p(2,2)")
        self._p23 = E("p(2,3)")
        self._k_keys = [
            (E(f"k({i},1)"), E(f"k({i},2)"), E(f"k({i},3)")) for i in range(n_loops)
        ]

        self._value_key_by_name: dict[str, Expression] = {
            self._normalize_symbol_key(self._p11.to_canonical_string()): self._p11,
            self._normalize_symbol_key(self._p12.to_canonical_string()): self._p12,
            self._normalize_symbol_key(self._p13.to_canonical_string()): self._p13,
            self._normalize_symbol_key(self._p21.to_canonical_string()): self._p21,
            self._normalize_symbol_key(self._p22.to_canonical_string()): self._p22,
            self._normalize_symbol_key(self._p23.to_canonical_string()): self._p23,
            self._normalize_symbol_key(self._z_key.to_canonical_string()): self._z_key,
            self._normalize_symbol_key(
                self._muv_key.to_canonical_string()
            ): self._muv_key,
            self._normalize_symbol_key(self._t_key.to_canonical_string()): self._t_key,
        }
        for ks in self._k_keys:
            for k_expr in ks:
                self._value_key_by_name[
                    self._normalize_symbol_key(k_expr.to_canonical_string())
                ] = k_expr

        self._input_plans: dict[str, list[tuple[tuple[Expression], Expression]]] = {}
        self._input_index_plans: dict[str, list[tuple[int, Expression]]] = {}
        for evaluator_name, pe in self.evaluators.items():
            plan: list[tuple[tuple[Expression], Expression]] = []
            index_plan: list[tuple[int, Expression]] = []
            for head in pe.param_builder.order:
                key = self._normalize_symbol_key(head[0].to_canonical_string())
                if key not in self._value_key_by_name:
                    raise pygloopException(
                        f"Missing runtime key mapping for symbol '{head[0].to_canonical_string()}' "
                        f"(normalized '{key}') in compiled evaluator '{pe.name}'."
                    )
                plan.append((head, self._value_key_by_name[key]))
                pos = pe.param_builder.positions[head][0]
                index_plan.append((pos, self._value_key_by_name[key]))
            self._input_plans[evaluator_name] = plan
            self._input_index_plans[evaluator_name] = index_plan

        self._t_guess_by_term = {
            t.evaluator_name: float(t.t_initial_guess) for t in self.terms
        }
        graph_group_terms: dict[str, list[DYCompiledTerm]] = {}
        for term in self.terms:
            group_name = term.graph_group_name or self._graph_group_name_from_term(term)
            graph_group_terms.setdefault(group_name, []).append(term)
        self._graph_group_names = sorted(
            graph_group_terms.keys(), key=self._graph_group_sort_key
        )
        self._graph_group_terms = {
            group_name: graph_group_terms[group_name]
            for group_name in self._graph_group_names
        }

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

    @staticmethod
    def _graph_group_name_from_evaluator_name(evaluator_name: str) -> str | None:
        if not evaluator_name.startswith("graph_"):
            return None
        graph_prefix, separator, _rest = evaluator_name.partition("_cut_")
        if separator == "":
            return None
        return graph_prefix

    @staticmethod
    def _graph_group_sort_key(group_name: str) -> tuple[int, int | str]:
        if group_name.startswith("graph_"):
            suffix = group_name.removeprefix("graph_")
            if suffix.isdigit():
                return (0, int(suffix))
        return (1, group_name)

    def _graph_group_name_from_term(self, term: DYCompiledTerm) -> str:
        evaluator = self.evaluators.get(term.evaluator_name)
        if evaluator is not None:
            group_name = evaluator.additional_data.get("graph_group_name")
            if group_name is not None:
                return str(group_name)
        parsed_group_name = self._graph_group_name_from_evaluator_name(
            term.evaluator_name
        )
        if parsed_group_name is not None:
            return parsed_group_name
        if evaluator is not None:
            source_graph_name = evaluator.additional_data.get("source_graph_name")
            if source_graph_name is not None:
                return str(source_graph_name)
        return term.evaluator_name

    def graph_channel_names(self) -> list[str]:
        return list(self._graph_group_names)

    def graph_channel_count(self) -> int:
        return len(self._graph_group_names)

    def terms_for_channel(self, channel_selector: int | None) -> list[DYCompiledTerm]:
        if channel_selector is None:
            return self.terms
        if channel_selector < 0 or channel_selector >= self.graph_channel_count():
            raise pygloopException(
                f"DY graph channel {channel_selector} out of range for bundle "
                f"'{self.integrand_name}' with {self.graph_channel_count()} channels."
            )
        return self._graph_group_terms[self._graph_group_names[channel_selector]]

    @classmethod
    def create_from_evaluators(
        cls,
        process: str,
        integrand_name: str,
        n_loops: int,
        observable: str,
        evaluators: list,
    ) -> DYCompiledBundle:
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
            evaluator_name = getattr(ev, "compiled_name", f"term_{i}_integrand")
            if evaluator_name in loaded_evaluators:
                raise pygloopException(
                    f"Duplicate compiled evaluator name '{evaluator_name}'."
                )
            pb = cls._build_param_builder(ev.symbols)
            graph_group_name = cls._graph_group_name_from_evaluator_name(evaluator_name)
            additional_data = {
                "process": process,
                "integrand_name": integrand_name,
                "observable": observable,
                "term_id": i,
            }
            if graph_group_name is not None:
                additional_data["graph_group_name"] = graph_group_name
            source_graph_name = getattr(ev, "source_graph_name", None)
            if source_graph_name is not None:
                additional_data["source_graph_name"] = source_graph_name
            routed_graph_name = getattr(ev, "routed_graph_name", None)
            if routed_graph_name is not None:
                additional_data["routed_graph_name"] = routed_graph_name

            pe = PygloopEvaluator(
                evaluator=ev.evaluator,
                param_builder=pb,
                name=evaluator_name,
                output_length=1,
                additional_data=additional_data,
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
                    integrand_expression=getattr(ev, "integrand_expression", None),
                    t_initial_guess=1.0,
                    graph_group_name=graph_group_name,
                )
            )

        metadata = {
            "bundle_format_version": cls.BUNDLE_FORMAT_VERSION,
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
                    "integrand_expression": (
                        t.integrand_expression.to_canonical_string()
                        if t.integrand_expression is not None
                        else None
                    ),
                    "t_initial_guess": t.t_initial_guess,
                    "graph_group_name": t.graph_group_name,
                }
                for t in terms
            ],
        }
        with open(pjoin(out_dir, cls.METADATA_FILE), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return cls(process, integrand_name, n_loops, terms, loaded_evaluators)

    @classmethod
    def load(cls, process: str, integrand_name: str) -> DYCompiledBundle:
        out_dir = cls._bundle_dir(process, integrand_name)
        metadata_path = pjoin(out_dir, cls.METADATA_FILE)
        if not os.path.isfile(metadata_path):
            raise pygloopException(f"Missing bundle metadata: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        terms: list[DYCompiledTerm] = []
        evaluators: dict[str, PygloopEvaluator] = {}
        bundle_format_version = int(metadata.get("bundle_format_version", 1))

        for t in metadata["terms"]:
            name = t["evaluator_name"]
            evaluators[name] = PygloopEvaluator.load(out_dir, name)
            graph_group_name = t.get("graph_group_name")
            if graph_group_name is None:
                graph_group_name = cls._graph_group_name_from_evaluator_name(name)
            terms.append(
                DYCompiledTerm(
                    evaluator_name=name,
                    e_surface=E(t["e_surface"]),
                    theta_expressions=[E(x) for x in t["theta_expressions"]],
                    integrand_expression=(
                        E(t["integrand_expression"])
                        if bundle_format_version >= 2
                        and t.get("integrand_expression") is not None
                        else None
                    ),
                    t_initial_guess=float(t.get("t_initial_guess", 1.0)),
                    graph_group_name=graph_group_name,
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
        # out = DYCompiledBundle._strip_python_namespace(expr)
        out = expr
        for k, v in values.items():
            out = out.replace(k, E(f"{v:.16e}"))
        return out

    @staticmethod
    def _decimal_from_number(value: float | Decimal | str | int) -> Decimal:
        if isinstance(value, Decimal):
            return value
        if isinstance(value, int):
            return Decimal(value)
        if isinstance(value, float):
            if not math.isfinite(value):
                raise pygloopException(
                    f"Cannot convert non-finite float '{value}' to Decimal."
                )
            return Decimal(repr(value))
        return Decimal(str(value))

    @staticmethod
    def _evaluate_expression_with_prec(
        expr: Expression,
        values: dict[Expression, Decimal],
        decimal_digit_precision: int,
    ) -> Decimal | None:
        string_values = {key: str(value) for key, value in values.items()}
        try:
            value = expr.evaluate_with_prec(string_values, {}, decimal_digit_precision)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return None

        try:
            decimal_value = DYCompiledBundle._decimal_from_number(value)
        except (InvalidOperation, ValueError, pygloopException):
            return None

        if decimal_value.is_nan() or not decimal_value.is_finite():
            return None
        return decimal_value

    def supports_arb(self) -> bool:
        return all(t.integrand_expression is not None for t in self.terms)

    def require_arb_supported(self) -> None:
        if self.supports_arb():
            return
        raise pygloopException(
            f"DY bundle '{self.integrand_name}' does not contain symbolic term expressions "
            "needed for arbitrary-precision fallback. Regenerate the DY bundle with "
            "'--clean --process dy generate'."
        )

    def _build_runtime_values(
        self,
        loop_momenta: list[Vector],
        p1: Vector,
        p2: Vector,
        z: float,
        m_uv: float,
    ) -> tuple[
        dict[Expression, float], tuple[float, float, float, float, float, float]
    ]:
        vals: dict[Expression, float] = {}
        for i, k in enumerate(loop_momenta):
            kx, ky, kz = k.to_list()
            k1, k2, k3 = self._k_keys[i]
            vals[k1] = float(kx)
            vals[k2] = float(ky)
            vals[k3] = float(kz)

        p1x, p1y, p1z = p1.to_list()
        p2x, p2y, p2z = p2.to_list()
        vals[self._p11] = float(p1x)
        vals[self._p12] = float(p1y)
        vals[self._p13] = float(p1z)
        vals[self._p21] = float(p2x)
        vals[self._p22] = float(p2y)
        vals[self._p23] = float(p2z)
        vals[self._z_key] = float(z)
        vals[self._muv_key] = float(m_uv)

        return vals, (p1x, p1y, p1z, p2x, p2y, p2z)

    def _initial_t_guess(
        self,
        term: DYCompiledTerm,
        vals: dict[Expression, float],
        p1x: float,
        p1y: float,
        p1z: float,
    ) -> float:
        valst1 = vals.copy()
        valst1[self._t_key] = 1.0
        p_norm = math.sqrt(p1x**2 + p1y**2 + p1z**2)
        if p_norm == 0.0:
            return 1.0
        try:
            denom = term.e_surface.evaluate(valst1, {}) + 2.0 * p_norm
            if denom == 0.0 or not math.isfinite(denom):
                return 1.0
            guess = abs(2.0 * p_norm / denom)
            if not math.isfinite(guess) or guess <= 0.0:
                return 1.0
            return guess
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return 1.0

    #    @staticmethod
    #    def _set_inputs(pe: PygloopEvaluator, values: dict[str, float]) -> None:
    #        for head in pe.param_builder.order:
    #            raw_key = head[0].to_canonical_string()
    #            key = DYCompiledBundle._normalize_symbol_key(raw_key)
    #
    #            if key not in values:
    #                raise pygloopException(
    #                    f"Missing value for symbol '{raw_key}' (normalized '{key}') in compiled evaluator '{pe.name}'."
    #                )
    #            pe.param_builder.set_parameter_values(head, [values[key]])

    @staticmethod
    def _set_inputs(
        pe: PygloopEvaluator,
        values: dict[Expression, float],
        input_plan: list[tuple[tuple[Expression], Expression]],
    ) -> None:
        for head, value_key in input_plan:
            if value_key not in values:
                raise pygloopException(
                    f"Missing value for symbol '{value_key.to_canonical_string()}' in compiled evaluator '{pe.name}'."
                )
            pe.param_builder.set_parameter_values(head, [values[value_key]])

    @staticmethod
    def _set_inputs_fast(
        pe: PygloopEvaluator,
        values: dict[Expression, float],
        input_index_plan: list[tuple[int, Expression]],
    ) -> None:
        arr = pe.param_builder.np
        for idx, value_key in input_index_plan:
            if value_key not in values:
                raise pygloopException(
                    f"Missing value for symbol '{value_key.to_canonical_string()}' in compiled evaluator '{pe.name}'."
                )
            arr[idx] = values[value_key]

    def solve_t_newton_bisect(
        self,
        term_e_surface: Expression,
        vals: dict[Expression, float],
        t_key: Expression,
        t0: float = 1.0,
        tol_f: float = 1e-12,
        tol_x: float = 1e-12,
        max_iter: int = 32,
        max_bracket_expands: int = 12,
        eval_map: dict[Expression, float] | None = None,
    ) -> float | None:
        """Fast hybrid secant+bisection root solve for term_e_surface(t)=0 using evaluate()."""

        # Reuse one mutable map to avoid allocations in hot loop.
        if eval_map is None:
            eval_map = vals

        def f(t: float) -> float:
            eval_map[t_key] = t
            # try:
            #    y = term_e_surface.evaluate(eval_map, {})
            #    return y if math.isfinite(y) else None
            # except Exception:
            #    return None
            y = term_e_surface.evaluate(eval_map, {})
            return y

        # Initial point
        x0 = float(t0)
        f0 = f(x0)
        if abs(f0) <= tol_f:
            return x0

        # Bracket around x0
        span = max(1.0, abs(x0))
        a = max(0.0, x0 - span)
        b = max(a + 1e-14, x0 + span)
        fa, fb = f(a), f(b)

        for _ in range(max_bracket_expands):
            # if fa * fb <= 0.0:
            #    break
            span *= 2.0
            a = max(0.0, x0 - span)
            b = max(a + 1e-14, x0 + span)
            fa, fb = f(a), f(b)

        # else:
        #    print("hereeeeeeeee")
        #    return None  # no bracket

        # Secant state (keep points inside bracket)
        x_prev, f_prev = a, fa
        x_curr, f_curr = b, fb

        for _ in range(max_iter):
            if abs(f_curr) <= tol_f or abs(b - a) <= tol_x * max(1.0, abs(x_curr)):
                return x_curr

            # Secant step; fallback to bisection if degenerate or out of bracket.
            if f_curr != f_prev:
                x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            else:
                x_next = 0.5 * (a + b)
            if not (min(a, b) <= x_next <= max(a, b)) or not math.isfinite(x_next):
                x_next = 0.5 * (a + b)

            f_next = f(x_next)
            # if f_next is None:
            #    x_next = 0.5 * (a + b)
            #    f_next = f(x_next)
            #    if f_next is None:
            #        return None

            # Keep bracket valid
            if fa * f_next <= 0.0:
                b, fb = x_next, f_next
            else:
                a, fa = x_next, f_next

            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_next, f_next

        return x_curr  # if math.isfinite(x_curr) else None

    def solve_t_convex_bisect(
        self,
        term_e_surface: Expression,
        vals: dict[Expression, float],
        t_key: Expression,
        t0: float = 1.0,
        tol_f: float = 1e-16,
        tol_x: float = 1e-16,
        max_iter: int = 80,
        max_expand_rounds: int = 24,
        probes_per_round: int = 17,
        eval_map: dict[Expression, float] | None = None,
    ) -> float | None:
        """
        Robust convex-friendly root finder for term_e_surface(t)=0.
        Strategy:
        1) Discover a valid sign-change bracket by sampling an expanding interval.
        2) Refine with pure bisection.
        Returns one root (prefers bracket closest to t0), or None if no bracket found.
        """
        if eval_map is None:
            eval_map = vals

        def f(t: float) -> float | None:
            if not math.isfinite(t):
                return None
            eval_map[t_key] = t
            try:
                y = term_e_surface.evaluate(eval_map, {})
                return y if math.isfinite(y) else None
            except Exception:
                return None

        x0 = float(t0)
        if not math.isfinite(x0):
            x0 = 1.0

        y0 = f(x0)
        if y0 is not None and abs(y0) <= tol_f:
            return x0

        # Bracket discovery by sampled expanding intervals around x0.
        # For convex functions there may be 0/1/2 roots; we choose bracket nearest x0.
        span = max(1.0, abs(x0))
        best_bracket: tuple[float, float, float, float] | None = None
        for r in range(max_expand_rounds):
            left = x0 - span
            right = x0 + span
            step = (right - left) / float(probes_per_round - 1)

            prev_x: float | None = None
            prev_y: float | None = None
            for i in range(probes_per_round):
                x = left + step * i
                y = f(x)
                if y is None:
                    continue
                if abs(y) <= tol_f:
                    return x
                if prev_x is not None and prev_y is not None and prev_y * y <= 0.0:
                    # Pick bracket whose midpoint is closest to x0.
                    mid = 0.5 * (prev_x + x)
                    if best_bracket is None or abs(mid - x0) < abs(
                        0.5 * (best_bracket[0] + best_bracket[1]) - x0
                    ):
                        best_bracket = (prev_x, x, prev_y, y)
                prev_x, prev_y = x, y

            if best_bracket is not None:
                break
            span *= 2.0

        if best_bracket is None:
            return None

        a, b, fa, fb = best_bracket
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        # Pure bisection refinement.
        for _ in range(max_iter):
            m = 0.5 * (a + b)
            fm = f(m)
            if fm is None:
                # If midpoint fails, try quarter points before giving up.
                q1 = 0.25 * a + 0.75 * b
                fq1 = f(q1)
                if fq1 is not None:
                    m, fm = q1, fq1
                else:
                    q2 = 0.75 * a + 0.25 * b
                    fq2 = f(q2)
                    if fq2 is None:
                        return None
                    m, fm = q2, fq2

            if abs(fm) <= tol_f or abs(b - a) <= tol_x * max(1.0, abs(m)):
                return m

            if fa * fm <= 0.0:
                b, fb = m, fm
            else:
                a, fa = m, fm

        return 0.5 * (a + b)

    def solve_t_convex_bisect_prec(
        self,
        term_e_surface: Expression,
        vals: dict[Expression, Decimal],
        t_key: Expression,
        decimal_digit_precision: int,
        t0: float = 1.0,
        max_iter: int = 80,
        max_expand_rounds: int = 24,
        probes_per_round: int = 17,
        eval_map: dict[Expression, Decimal] | None = None,
    ) -> Decimal | None:
        if decimal_digit_precision <= 0:
            raise pygloopException(
                "Arbitrary-precision evaluation requires a positive decimal precision."
            )

        if eval_map is None:
            eval_map = vals

        tol_power = min(max(decimal_digit_precision // 2, 12), 32)
        tol_f = Decimal(10) ** (-tol_power)
        tol_x = Decimal(10) ** (-tol_power)

        def f(t: Decimal) -> Decimal | None:
            if t.is_nan() or t < 0:
                return None
            eval_map[t_key] = t
            return self._evaluate_expression_with_prec(
                term_e_surface, eval_map, decimal_digit_precision
            )

        try:
            x0 = self._decimal_from_number(t0)
        except (InvalidOperation, ValueError, pygloopException):
            x0 = Decimal(1)
        if not x0.is_finite():
            x0 = Decimal(1)
        if x0 < 0:
            x0 = -x0

        y0 = f(x0)
        if y0 is not None and abs(y0) <= tol_f:
            return x0

        one = Decimal(1)
        zero = Decimal(0)
        best_bracket: tuple[Decimal, Decimal, Decimal, Decimal] | None = None
        span = max(one, abs(x0))
        for _ in range(max_expand_rounds):
            left = max(zero, x0 - span)
            right = x0 + span
            if right <= left:
                right = left + one
            step = (right - left) / Decimal(probes_per_round - 1)

            prev_x: Decimal | None = None
            prev_y: Decimal | None = None
            for i in range(probes_per_round):
                x = left + step * Decimal(i)
                y = f(x)
                if y is None:
                    continue
                if abs(y) <= tol_f:
                    return x
                if prev_x is not None and prev_y is not None and prev_y * y <= 0:
                    mid = (prev_x + x) / 2
                    if best_bracket is None or abs(mid - x0) < abs(
                        (best_bracket[0] + best_bracket[1]) / 2 - x0
                    ):
                        best_bracket = (prev_x, x, prev_y, y)
                prev_x, prev_y = x, y

            if best_bracket is not None:
                break
            span *= 2

        if best_bracket is None:
            return None

        a, b, fa, fb = best_bracket
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        for _ in range(max_iter):
            m = (a + b) / 2
            fm = f(m)
            if fm is None:
                q1 = (a + 3 * b) / 4
                fq1 = f(q1)
                if fq1 is not None:
                    m, fm = q1, fq1
                else:
                    q2 = (3 * a + b) / 4
                    fq2 = f(q2)
                    if fq2 is None:
                        return None
                    m, fm = q2, fq2

            if abs(fm) <= tol_f or abs(b - a) <= tol_x * max(one, abs(m)):
                return m

            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm

        return (a + b) / 2

    def evaluate_arb(
        self,
        loop_momenta: list[Vector],
        p1: Vector,
        p2: Vector,
        z: float,
        m_uv: float = 1.0,
        decimal_digit_precision: int = 80,
        theta_tolerance: float = 0.0,
        channel_selector: int | None = None,
    ) -> Decimal:
        self.require_arb_supported()

        vals, (p1x, p1y, p1z, _p2x, _p2y, _p2z) = self._build_runtime_values(
            loop_momenta, p1, p2, z, m_uv
        )
        dec_vals = {
            key: self._decimal_from_number(value) for key, value in vals.items()
        }
        total = Decimal(0)
        theta_tol = self._decimal_from_number(theta_tolerance)

        for term in self.terms_for_channel(channel_selector):
            my_t0 = self._initial_t_guess(term, vals, p1x, p1y, p1z)
            t_sol = self.solve_t_convex_bisect_prec(
                term.e_surface,
                dec_vals,
                self._t_key,
                decimal_digit_precision=decimal_digit_precision,
                t0=my_t0,
                eval_map=dec_vals,
            )
            if t_sol is None:
                raise pygloopException(
                    f"Failed to solve t in arbitrary precision for DY term '{term.evaluator_name}'."
                )

            dec_vals[self._t_key] = t_sol

            theta_passes = True
            for th in term.theta_expressions:
                th_val = self._evaluate_expression_with_prec(
                    th, dec_vals, decimal_digit_precision
                )
                if th_val is None or th_val < -theta_tol:
                    theta_passes = False
                    break
            if not theta_passes:
                continue

            assert term.integrand_expression is not None
            term_value = self._evaluate_expression_with_prec(
                term.integrand_expression, dec_vals, decimal_digit_precision
            )
            if term_value is None:
                raise pygloopException(
                    f"Failed to evaluate DY term '{term.evaluator_name}' in arbitrary precision."
                )
            total += term_value

        return total

    def evaluate(
        self,
        loop_momenta: list[Vector],
        p1: Vector,
        p2: Vector,
        z: float,
        m_uv: float = 1.0,
        mode: str = "compiled",
        decimal_digit_precision: int | None = None,
        theta_tolerance: float = 0.0,
        channel_selector: int | None = None,
    ) -> complex:
        if mode == "arb":
            if decimal_digit_precision is None:
                decimal_digit_precision = 80
            return complex(
                float(
                    self.evaluate_arb(
                        loop_momenta,
                        p1,
                        p2,
                        z,
                        m_uv,
                        decimal_digit_precision=decimal_digit_precision,
                        theta_tolerance=theta_tolerance,
                        channel_selector=channel_selector,
                    )
                ),
                0.0,
            )
        if mode != "compiled":
            raise pygloopException(f"Unsupported DY bundle evaluation mode '{mode}'.")

        vals, (p1x, p1y, p1z, _p2x, _p2y, _p2z) = self._build_runtime_values(
            loop_momenta, p1, p2, z, m_uv
        )

        total = 0.0 + 0.0j
        theta_tol = float(theta_tolerance)

        # Sum over all cut graphs
        for term in self.terms_for_channel(channel_selector):
            my_t0 = self._initial_t_guess(term, vals, p1x, p1y, p1z)

            t_sol = self.solve_t_newton_bisect(
                term.e_surface,
                vals,
                self._t_key,
                t0=my_t0,  # fixed per-term start for benchmark-stable branch
                eval_map=vals,
            )

            # t_sol = self.solve_t_convex_bisect(
            #     term.e_surface,
            #     vals,
            #     self._t_key,
            #     t0=my_t0,  # fixed per-term start for benchmark-stable branch
            #     eval_map=vals,
            # )

            if t_sol is None:
                print("t solving problem")
                print(t_sol)
                print(vals)
                continue

            vals[self._t_key] = t_sol

            # print("---------------")
            theta = 1
            for th in term.theta_expressions:
                th_val = th.evaluate(vals, {})
                # if th_val > 0:
                #    print("-->", 1)
                # else:
                #    print("-->", 0)

                if th_val < -theta_tol:
                    theta = 0
                    break
            if theta == 0:
                continue

            pe = self.evaluators[term.evaluator_name]

            self._set_inputs_fast(
                pe, vals, self._input_index_plans[term.evaluator_name]
            )

            # print("---")
            # print("vals")
            # print(vals)
            # print("thetas")
            # print(theta_exprs)
            # print("t")
            # print(t_sol)
            # print("e_surface")
            # print(term.e_surface)

            total += complex(pe.evaluate(eager=False)[0])

        return total


class compile_integrands:
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
