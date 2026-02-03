from enum import StrEnum
from typing import Any

import numpy
from symbolica import E, Expression, S

from utils.utils import Colour, np_cmplx_one, np_cmplx_zero


class EdgeOrientation(StrEnum):
    DEFAULT = "DEFAULT"
    REVERSED = "REVERSED"
    UNDIRECTED = "UNDIRECTED"

    @staticmethod
    def from_str(label: str) -> "EdgeOrientation":
        if label.upper() == "DEFAULT":
            return EdgeOrientation.DEFAULT
        if label.upper() == "REVERSED":
            return EdgeOrientation.REVERSED
        if label.upper() == "UNDIRECTED":
            return EdgeOrientation.UNDIRECTED
        raise ValueError(f"Unknown EdgeOrientation: {label}")

    def is_reversed(self) -> bool:
        return self == EdgeOrientation.REVERSED

    def is_undirected(self) -> bool:
        return self == EdgeOrientation.UNDIRECTED

    def __str__(self) -> str:
        match self:
            case EdgeOrientation.DEFAULT:
                return f"{Colour.GREEN}+{Colour.END}"
            case EdgeOrientation.REVERSED:
                return f"{Colour.RED}-{Colour.END}"
            case EdgeOrientation.UNDIRECTED:
                return f"{Colour.BLUE}-{Colour.BLUE}"


class CFFTerm(object):
    def __init__(
        self,
        id: int,
        orientation: tuple[EdgeOrientation, ...],
        expression: Expression,
        families: tuple[tuple[bool, ...], ...],
    ):
        self.id = id
        self.orientation = orientation
        self.orientation_signs = numpy.zeros(len(orientation), complex)

        vals = []
        for o in orientation:
            if o.is_reversed():
                vals.append(-np_cmplx_one)
            elif o.is_undirected():
                vals.append(np_cmplx_zero)
            else:
                vals.append(np_cmplx_one)
        self.orientation_signs[:] = vals

        # self.orientation_signs[:] = [
        #    -np_cmplx_one if o.is_reversed() else np_cmplx_one for o in orientation
        # ]
        self.expression = expression
        self.families = families
        self.masks = []
        for family in families:
            family_mask = numpy.zeros(len(families[0]), complex)
            family_mask[:] = [np_cmplx_one if f else np_cmplx_zero for f in family]
            self.masks.append(family_mask)

    def __str__(self, show_families=False):
        res = [
            f"{''.join(str(o) for o in self.orientation)}: {self.expression.format(show_namespaces=False)}"
        ]
        if show_families:
            for cff_family in self.families:
                res.append(
                    f"      {''.join('■' if included else '□' for included in cff_family)}"
                )
        return "\n".join(res)


class ESurface(object):
    def __init__(
        self,
        id: int,
        oses: tuple[int, ...],
        external_shift: tuple[tuple[int, int], ...],
    ):
        self.id = id
        self.oses = oses
        self.external_shift = external_shift
        self.expression: Expression = self.get_expression()

    def get_expression(self) -> Expression:
        e_surf = E("0")
        for e_id in self.oses:
            e_surf += CFFStructure.SB["E"](e_id)
        for e_id, sign in self.external_shift:
            e_surf += E(str(sign)) * CFFStructure.SB["p0"](e_id)
        return e_surf

    def __str__(self):
        return self.expression.format(show_namespaces=False)


class HSurface(object):
    def __init__(
        self,
        id: int,
        positive_oses: tuple[int, ...],
        negative_oses: tuple[int, ...],
    ):
        self.id = id
        self.positive_oses = positive_oses
        self.negative_oses = negative_oses
        self.expression: Expression = self.get_expression()

    def get_expression(self) -> Expression:
        h_surf = E("0")
        for e_id in self.positive_oses:
            h_surf += CFFStructure.SB["E"](e_id)
        for e_id in self.negative_oses:
            h_surf -= CFFStructure.SB["E"](e_id)
        return h_surf

    def __str__(self):
        return self.expression.format(show_namespaces=False)


class CFFStructure(object):
    # fmt: off
    SB = {
        "E": S("pygloop::E"),
        "p0": S("pygloop::E"),
        "eta": S("pygloop::η"),
        "heta": S("pygloop::γ"),
        "x_": S("pygloop::x_"),
        "eta(x_)": E("η(x_)", default_namespace="pygloop"),
        "x__+y__": E("x__ + y__", default_namespace="pygloop")
    }
    # fmt: on

    def __init__(self, cff_dict: dict):
        self.cff_dict = cff_dict
        self.e_surfaces: list[ESurface] = []
        self.h_surfaces: list[HSurface] = []
        self.expressions: list[CFFTerm] = []
        self.build_cff_expression()

    def __str__(self, show_families=False):
        res = [""]
        res.append(f"{Colour.GREEN}{len(self.e_surfaces)}{Colour.END} e-surfaces:")
        for e_surf in self.e_surfaces:
            res.append(f" {Colour.BLUE}#{e_surf.id:-3}{Colour.END} > {str(e_surf)}")
        res.append("")
        res.append(f"{Colour.GREEN}{len(self.h_surfaces)}{Colour.END} h-surfaces:")
        for h_surf in self.h_surfaces:
            res.append(f" {Colour.BLUE}#{e_surf.id:-3}{Colour.END} > {str(h_surf)}")
        res.append("")
        res.append(f"{Colour.GREEN}{len(self.expressions)}{Colour.END} orientations:")
        for cff_expr in self.expressions:
            res.append(
                f" {Colour.BLUE}#{cff_expr.id:-3}{Colour.END} > {cff_expr.__str__(show_families=show_families)}"
            )
        res.append("")
        return "\n".join(res)

    @classmethod
    def expression_from_node(
        cls, node_id: int, nodes_list: list[dict]
    ) -> tuple[Expression, Expression]:
        iteration_list = nodes_list[node_id]["data"]
        if "Hsurface" in iteration_list:
            eta = cls.SB["heta"](nodes_list[node_id]["data"]["Hsurface"])
        if "Esurface" in iteration_list:
            eta = cls.SB["eta"](nodes_list[node_id]["data"]["Esurface"])
        if len(nodes_list[node_id]["children"]) == 0:
            return (eta, 1 / eta)
        children_expression = E("0")
        children_expression_inv = E("0")
        for child in nodes_list[node_id]["children"]:
            child_expression, child_expression_inv = CFFStructure.expression_from_node(
                child, nodes_list
            )
            children_expression += child_expression
            children_expression_inv += child_expression_inv
        return (eta * children_expression, (1 / eta) * children_expression_inv)

    def build_cff_expression(self):
        for e_id, e_surf in enumerate(self.cff_dict["surfaces"]["esurface_cache"]):
            self.e_surfaces.append(
                ESurface(
                    id=e_id,
                    oses=e_surf["energies"],
                    external_shift=e_surf["external_shift"],
                )
            )

        for e_id, h_surf in enumerate(self.cff_dict["surfaces"]["hsurface_cache"]):
            self.h_surfaces.append(
                HSurface(
                    id=e_id,
                    positive_oses=h_surf["positive_energies"],
                    negative_oses=h_surf["negative_energies"],
                )
            )

        print(self.cff_dict["orientations"])
        print("now inside loop")
        for o_id, o_info in enumerate(self.cff_dict["orientations"]):
            nodes = o_info["expression"]["nodes"]
            print(o_info)
            o_expression, o_expression_inv = CFFStructure.expression_from_node(0, nodes)

            o_families = []
            expanded_cff_expression = o_expression.expand()
            cff_terms = []
            if not expanded_cff_expression.matches(self.SB["x__+y__"]):
                cff_terms.append(expanded_cff_expression)
            else:
                cff_terms = [t for t in expanded_cff_expression]
            for cff_family in cff_terms:
                eta_ids = [
                    int(str(m[self.SB["x_"]]))
                    for m in cff_family.match(self.SB["eta(x_)"])
                ]
                mask = tuple(
                    eta_id in eta_ids for eta_id in range(len(self.e_surfaces))
                )
                o_families.append(mask)

            self.expressions.append(
                CFFTerm(
                    id=o_id,
                    orientation=tuple(
                        EdgeOrientation.from_str(d)
                        for d in o_info["data"]["orientation"]
                    ),
                    expression=o_expression_inv,
                    families=tuple(o_families),
                )
            )

    def get(self, key: str, default: Any = None) -> Any:
        return self.cff_dict.get(key, default)
