from __future__ import annotations

import ast
import contextlib
import io
import logging
import math
import os
import sys
from copy import deepcopy
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from processes.dy.dy import DY  # noqa: E402
from processes.dy.dy_classes import (  # noqa: E402
    DYDotGraphs,
    VacuumDotGraph,
    canonicalise_vacuum_graph,
)
from processes.dy.dy_evaluators import DYCompiledBundle, evaluate_integrand  # noqa: E402
from processes.dy.dy_integrand import (  # noqa: E402
    LoopIntegrandConstructor,
    routed_cut_graph,
)
from utils.vectors import LorentzVector, Vector  # noqa: E402


class DYGraphCutSumReferenceProcess(DY):
    name = "DY_GRAPH_CUT_SUM_REFERENCE_TEST"


class TTGraphChannelReferenceProcess(DY):
    name = "TT_GRAPH_CHANNEL_REFERENCE_TEST"


class TTTwoLoopChannelReferenceProcess(DY):
    name = "TT_TWO_LOOP_CHANNEL_REFERENCE_TEST"


class TTTwoLoopQGChannelReferenceProcess(DY):
    name = "TT_TWO_LOOP_QG_CHANNEL_REFERENCE_TEST"


GENERATE_CONFIG = ROOT / "configs" / "DY" / "generate.toml"
RUNTIME_CONFIG = ROOT / "configs" / "DY" / "runtime.toml"
RUN_SLOW_2L_QG = os.environ.get("PYGLOOP_RUN_SLOW_2L_QG") == "1"
POINT_Z = 0.6
POINTS = {
    "symmetric": {
        "ks": [
            math.sqrt(POINT_Z)
            * np.array([0.1 / math.sqrt(3), 0.1 / math.sqrt(3), 0.1 / math.sqrt(3)])
        ],
        "p1": [0.0, 0.0, 1.0],
        "p2": [0.0, 0.0, -1.0],
        "z": POINT_Z,
    },
    "anisotropic": {
        "ks": [
            math.sqrt(POINT_Z)
            * np.array([0.1 / math.sqrt(3), 0.1 / math.sqrt(3), 1.0 / math.sqrt(3)])
        ],
        "p1": [0.0, 0.0, 1.0],
        "p2": [0.0, 0.0, -1.0],
        "z": POINT_Z,
    },
}
OBSERVABLE_PARAMS = {
    "zmin": 0.0,
    "zmax": 1.0,
    "Lambdasq": 2.0,
    "mUV": 1.0,
}
CHANNEL_COMMANDS = {
    "ddx": {
        "base_name": "DY_GRAPH_CUT_SUM_REFERENCE_TEST_DDX",
        "graphs_name": "DY_GRAPH_CUT_SUM_REFERENCE_TEST_DDX_generated_graphs",
        "particle_filter": ["a"],
        "command": (
            "generate xs d d~ > a | d d~ g a QED^2==2 [{{1}} QCD=1] "
            "--only-diagrams --numerator-grouping only_detect_zeroes "
            "-p DY_GRAPH_CUT_SUM_REFERENCE_TEST_DDX "
            "-i DY_GRAPH_CUT_SUM_REFERENCE_TEST_DDX_generated_graphs "
            "--max-multiplicity-for-fast-cut-filter 99"
        ),
    },
    "dg": {
        "base_name": "DY_GRAPH_CUT_SUM_REFERENCE_TEST_DG",
        "graphs_name": "DY_GRAPH_CUT_SUM_REFERENCE_TEST_DG_generated_graphs",
        "particle_filter": ["a"],
        "command": (
            "generate xs d g > a | d d~ g a QED^2==2 [{{1}} QCD=1] "
            "--only-diagrams --numerator-grouping only_detect_zeroes "
            "-p DY_GRAPH_CUT_SUM_REFERENCE_TEST_DG "
            "-i DY_GRAPH_CUT_SUM_REFERENCE_TEST_DG_generated_graphs "
            "--max-multiplicity-for-fast-cut-filter 99"
        ),
    },
}
GGTT_COMMAND = {
    "base_name": "TT_GRAPH_CHANNEL_REFERENCE_TEST",
    "graphs_name": "TT_GRAPH_CHANNEL_REFERENCE_TEST_generated_graphs",
    "command": (
        "generate xs g g > t t~ | d d~ g t t~ [{{1}} QCD=1] "
        "--only-diagrams --numerator-grouping "
        "group_identical_graphs_up_to_scalar_rescaling "
        "--symmetrize-left-right-states true "
        "--symmetrize-initial-states true "
        "-p TT_GRAPH_CHANNEL_REFERENCE_TEST "
        "-i TT_GRAPH_CHANNEL_REFERENCE_TEST_generated_graphs "
        "--max-multiplicity-for-fast-cut-filter 99"
    ),
}
REFERENCE_SUMS: dict[tuple[str, str], dict[str, Decimal]] = {
    ("ddx", "GL1"): {
        "anisotropic": Decimal("11.06485143967556911630848559"),
        "symmetric": Decimal("-1185.267710416734376395246026"),
    },
    ("ddx", "GL2"): {
        "anisotropic": Decimal("11.06485143967556911630848571"),
        "symmetric": Decimal("-1185.267710416734376395246026"),
    },
    ("ddx", "GL3"): {
        "anisotropic": Decimal("-3.426357200058932674774999939"),
        "symmetric": Decimal("6.602166557884883592640098369"),
    },
    ("ddx", "GL4"): {
        "anisotropic": Decimal("-36.82655303339209532432538534"),
        "symmetric": Decimal("-17637.83564765314998972055674"),
    },
    ("ddx", "GL6"): {
        "anisotropic": Decimal("-3.426357200058932674774999910"),
        "symmetric": Decimal("6.602166557884883592640098374"),
    },
    ("ddx", "GL7"): {
        "anisotropic": Decimal("-36.82655303339209532432538534"),
        "symmetric": Decimal("-17637.83564765314998972055673"),
    },
    ("dg", "GL0"): {
        "anisotropic": Decimal("-6.179030686855646589789820527"),
        "symmetric": Decimal("30.53939886477111631220712559"),
    },
    ("dg", "GL1"): {
        "anisotropic": Decimal("-2.790888316442310638561596182"),
        "symmetric": Decimal("3.421606828727281949577332126"),
    },
    ("dg", "GL2"): {
        "anisotropic": Decimal("-2.790888316442310638561596230"),
        "symmetric": Decimal("3.421606828727281949577332132"),
    },
    ("dg", "GL3"): {
        "anisotropic": Decimal("0.5929447457977601929020119492"),
        "symmetric": Decimal("-1.712180072456517031351684803"),
    },
}
GGTT_POINTS = {
    "probe_a": {
        "ks": [Vector(150.0, -40.0, 220.0)],
        "p1": Vector(0.0, 0.0, 500.0),
        "p2": Vector(0.0, 0.0, -500.0),
        "z": POINT_Z,
    },
    "probe_b": {
        "ks": [Vector(250.0, 60.0, -90.0)],
        "p1": Vector(0.0, 0.0, 500.0),
        "p2": Vector(0.0, 0.0, -500.0),
        "z": POINT_Z,
    },
}
GGTT_REFERENCE_SUMS: dict[str, dict[str, Decimal]] = {
    "graph_0": {
        "probe_a": Decimal("4.535837345765991e-12"),
        "probe_b": Decimal("1.5242388435430973e-12"),
    },
    "graph_1": {
        "probe_a": Decimal("-3.6164341239834297e-14"),
        "probe_b": Decimal("-1.6754230169188702e-14"),
    },
    "graph_2": {
        "probe_a": Decimal("1.6432398306166366e-12"),
        "probe_b": Decimal("1.4646399443508783e-12"),
    },
    "graph_3": {
        "probe_a": Decimal("-2.3712594247880328e-12"),
        "probe_b": Decimal("-2.010861742122394e-12"),
    },
}
TT_TWO_LOOP_POINT = {
    "ks": [Vector(130.0, -80.0, 210.0), Vector(-70.0, 120.0, -160.0)],
    "p1": Vector(0.0, 0.0, 500.0),
    "p2": Vector(0.0, 0.0, -500.0),
    "z": POINT_Z,
    "m_uv": 2000.0,
}
TT_TWO_LOOP_QQBAR_PROBE_POINTS = {
    "collinear_anti_collinear": {
        "ks": [Vector(0.01, 0.0025, -65.0), Vector(-70.0, 120.0, -160.0)],
        "p1": Vector(0.0, 0.0, 500.0),
        "p2": Vector(0.0, 0.0, -500.0),
        "z": 0.25,
        "m_uv": 2000.0,
    },
    "soft": {
        "ks": [Vector(0.01, 0.002, 0.003), Vector(-70.0, 120.0, -160.0)],
        "p1": Vector(0.0, 0.0, 500.0),
        "p2": Vector(0.0, 0.0, -500.0),
        "z": POINT_Z,
        "m_uv": 2000.0,
    },
}
TT_TWO_LOOP_QG_PROBE_POINTS = {
    "collinear": {
        "ks": [Vector(0.01, 0.0025, 185.0), Vector(-70.0, 120.0, -160.0)],
        "p1": Vector(0.0, 0.0, 500.0),
        "p2": Vector(0.0, 0.0, -500.0),
        "z": POINT_Z,
        "m_uv": 2000.0,
    },
    "anti_collinear": {
        "ks": [Vector(0.01, 0.0025, -65.0), Vector(-70.0, 120.0, -160.0)],
        "p1": Vector(0.0, 0.0, 500.0),
        "p2": Vector(0.0, 0.0, -500.0),
        "z": 0.25,
        "m_uv": 2000.0,
    },
}
TT_TWO_LOOP_CHANNEL_NAMES = tuple(f"graph_{index}" for index in range(16))
TT_TWO_LOOP_QG_CHANNEL_NAMES = tuple(f"graph_{index}" for index in range(11))
TwoLoopStableTermKey = tuple[str, str, tuple[tuple[str, ...], ...]]
TT_TWO_LOOP_REFERENCE_TERMS: dict[str, Decimal] = {
    "graph_0_cut_3_term_0_integrand": Decimal("2.564934270982978362558987076623E-22"),
    "graph_0_cut_3_term_1_integrand": Decimal("-1.663339937854731049099220197308E-22"),
    "graph_0_cut_3_term_2_integrand": Decimal("-3.211818454038580202478615000159E-24"),
    "graph_0_cut_3_term_3_integrand": Decimal("-1.525525757370483854323889844303E-22"),
    "graph_1_cut_1_term_0_integrand": Decimal("3.005649832397274938443490561310E-21"),
    "graph_2_cut_0_term_0_integrand": Decimal("0"),
    "graph_2_cut_2_term_0_integrand": Decimal("-3.038825330624469579374345245539E-21"),
    "graph_3_cut_0_term_0_integrand": Decimal("-2.246598835620985116738360358854E-20"),
    "graph_3_cut_3_term_0_integrand": Decimal("0"),
    "graph_4_cut_1_term_0_integrand": Decimal("0"),
    "graph_4_cut_4_term_0_integrand": Decimal("-3.563233685485752405164429907474E-21"),
    "graph_4_cut_4_term_1_integrand": Decimal("4.934358776777936739552483039113E-22"),
    "graph_4_cut_4_term_2_integrand": Decimal("3.329362655651698603590808121084E-23"),
    "graph_4_cut_4_term_3_integrand": Decimal("4.884584595917869163451666013929E-22"),
    "graph_4_cut_10_term_0_integrand": Decimal("0"),
    "graph_5_cut_3_term_0_integrand": Decimal("-3.654984385592332288387655741350E-20"),
    "graph_5_cut_3_term_1_integrand": Decimal("3.442020125877740895747976445814E-20"),
    "graph_5_cut_3_term_2_integrand": Decimal("-5.312825197190258050623963110653E-22"),
    "graph_5_cut_3_term_3_integrand": Decimal("-7.457315340715731118170756226007E-22"),
    "graph_6_cut_1_term_0_integrand": Decimal("0"),
    "graph_6_cut_2_term_0_integrand": Decimal("-1.089377818897777092597997970613E-20"),
    "graph_6_cut_5_term_0_integrand": Decimal("0"),
    "graph_7_cut_0_term_0_integrand": Decimal("5.525196313866040811599398990203E-21"),
    "graph_7_cut_2_term_0_integrand": Decimal("0"),
    "graph_7_cut_4_term_0_integrand": Decimal("0"),
    "graph_7_cut_6_term_0_integrand": Decimal("0"),
    "graph_7_cut_6_term_1_integrand": Decimal("0"),
    "graph_7_cut_6_term_2_integrand": Decimal("0"),
    "graph_8_cut_0_term_0_integrand": Decimal("0"),
    "graph_8_cut_2_term_0_integrand": Decimal("-4.366014594668096655144365910294E-22"),
    "graph_8_cut_4_term_0_integrand": Decimal("0"),
    "graph_9_cut_0_term_0_integrand": Decimal("0"),
    "graph_9_cut_1_term_0_integrand": Decimal("0"),
    "graph_9_cut_2_term_0_integrand": Decimal("0"),
    "graph_9_cut_2_term_1_integrand": Decimal("0"),
    "graph_9_cut_2_term_2_integrand": Decimal("0"),
    "graph_9_cut_6_term_0_integrand": Decimal("6.608178683235815854427477140313E-22"),
    "graph_9_cut_6_term_1_integrand": Decimal("-1.516154645692777678835814949209E-21"),
    "graph_9_cut_6_term_2_integrand": Decimal("-3.851990052356234492219057607268E-24"),
    "graph_9_cut_6_term_3_integrand": Decimal("-4.573761247209048215782700342945E-23"),
    "graph_9_cut_8_term_0_integrand": Decimal("-1.320946037728080752156242061225E-21"),
    "graph_9_cut_8_term_1_integrand": Decimal("1.375339753513663681966068215298E-21"),
    "graph_9_cut_8_term_2_integrand": Decimal("-3.184567265424894901331183371942E-24"),
    "graph_9_cut_13_term_0_integrand": Decimal("0"),
    "graph_9_cut_14_term_0_integrand": Decimal("0"),
    "graph_9_cut_15_term_0_integrand": Decimal("0"),
    "graph_9_cut_15_term_1_integrand": Decimal("0"),
    "graph_9_cut_15_term_2_integrand": Decimal("0"),
    "graph_10_cut_1_term_0_integrand": Decimal("8.717966039232527296676493224216E-21"),
    "graph_10_cut_1_term_1_integrand": Decimal("-5.653533212903497492135157459039E-21"),
    "graph_10_cut_1_term_2_integrand": Decimal("-5.148941299851106949762318679853E-22"),
    "graph_10_cut_1_term_3_integrand": Decimal("-2.419744185225200642016213919358E-22"),
    "graph_10_cut_3_term_0_integrand": Decimal("-5.016857409659023892861433948565E-21"),
    "graph_10_cut_3_term_1_integrand": Decimal("1.050269863882578175602206283722E-20"),
    "graph_10_cut_3_term_2_integrand": Decimal("-5.879753724216092717881308646393E-22"),
    "graph_10_cut_3_term_3_integrand": Decimal("-3.087280396105554915100382199013E-22"),
    "graph_11_cut_3_term_0_integrand": Decimal("-3.976955998381152751975123094783E-20"),
    "graph_11_cut_3_term_1_integrand": Decimal("-3.883713797912208321015756660918E-21"),
    "graph_11_cut_3_term_2_integrand": Decimal("3.078603767202858722146739418186E-23"),
    "graph_11_cut_3_term_3_integrand": Decimal("4.884584595917869143398958792963E-22"),
    "graph_11_cut_5_term_0_integrand": Decimal("3.461081735850811052716981414106E-21"),
    "graph_11_cut_7_term_0_integrand": Decimal("3.461081735850811052716981414106E-21"),
    "graph_12_cut_1_term_0_integrand": Decimal("0"),
    "graph_12_cut_4_term_0_integrand": Decimal("0"),
    "graph_12_cut_6_term_0_integrand": Decimal("-2.586239688450960417395514877166E-21"),
    "graph_12_cut_6_term_1_integrand": Decimal("3.224333870065831183111888172671E-21"),
    "graph_12_cut_8_term_0_integrand": Decimal("-2.911338346177787027247370930424E-22"),
    "graph_12_cut_10_term_0_integrand": Decimal("-2.911338346177786886560965260468E-22"),
    "graph_12_cut_14_term_0_integrand": Decimal("0"),
    "graph_12_cut_18_term_0_integrand": Decimal("0"),
    "graph_13_cut_1_term_0_integrand": Decimal("-4.328319169893077468462637994803E-21"),
    "graph_13_cut_1_term_1_integrand": Decimal("1.056391346669312302942911602693E-22"),
    "graph_13_cut_1_term_2_integrand": Decimal("-5.909784687694777883980791684832E-24"),
    "graph_13_cut_3_term_0_integrand": Decimal("-4.328319169893077468462637994803E-21"),
    "graph_13_cut_3_term_1_integrand": Decimal("1.056391346669312302942911602693E-22"),
    "graph_13_cut_3_term_2_integrand": Decimal("-5.909784687694777883980791684832E-24"),
    "graph_13_cut_5_term_0_integrand": Decimal("1.292993801994151788374504549905E-22"),
    "graph_14_cut_1_term_0_integrand": Decimal("-4.940628291603824928140195459890E-23"),
    "graph_14_cut_1_term_1_integrand": Decimal("-1.988609259890100765850310771204E-24"),
    "graph_14_cut_1_term_2_integrand": Decimal("-1.458881843915004055878351425373E-23"),
    "graph_14_cut_3_term_0_integrand": Decimal("1.715756429772403219253368646098E-23"),
    "graph_14_cut_5_term_0_integrand": Decimal("1.715756429772403219253368646098E-23"),
    "graph_14_cut_7_term_0_integrand": Decimal("6.437945967166907432650994320421E-23"),
    "graph_14_cut_7_term_1_integrand": Decimal("-1.773903662940787761261662247963E-24"),
    "graph_14_cut_7_term_2_integrand": Decimal("-1.143440311802262980384281142739E-23"),
    "graph_15_cut_1_term_0_integrand": Decimal("-3.935537632112864889947192462565E-21"),
    "graph_15_cut_1_term_1_integrand": Decimal("6.749024294852110407333636509173E-21"),
    "graph_15_cut_1_term_2_integrand": Decimal("-2.945865006928652452822869151834E-22"),
    "graph_15_cut_1_term_3_integrand": Decimal("-1.535430353022803590799785362526E-22"),
    "graph_15_cut_3_term_0_integrand": Decimal("3.327558724891431883492064721818E-21"),
    "graph_15_cut_3_term_1_integrand": Decimal("-1.789667561998715631174156557694E-21"),
    "graph_15_cut_3_term_2_integrand": Decimal("-2.578616417989364555986530104986E-22"),
    "graph_15_cut_3_term_3_integrand": Decimal("-1.203437392091734511207876958003E-22"),
}
TT_TWO_LOOP_QG_REFERENCE_TERMS: dict[str, Decimal] = {
    "graph_0_cut_0_term_0_integrand": Decimal("0"),
    "graph_0_cut_3_term_0_integrand": Decimal("2.100812738071582716485520423878E-21"),
    "graph_0_cut_4_term_0_integrand": Decimal("0"),
    "graph_1_cut_0_term_0_integrand": Decimal("4.662691409151571164247618699652E-20"),
    "graph_1_cut_2_term_0_integrand": Decimal("0"),
    "graph_1_cut_5_term_0_integrand": Decimal("0"),
    "graph_2_cut_1_term_0_integrand": Decimal("2.940048368371780227582345491742E-20"),
    "graph_2_cut_3_term_0_integrand": Decimal("0"),
    "graph_2_cut_5_term_0_integrand": Decimal("0"),
    "graph_3_cut_0_term_0_integrand": Decimal("1.001760427199295921951054906678E-21"),
    "graph_3_cut_3_term_0_integrand": Decimal("0"),
    "graph_4_cut_0_term_0_integrand": Decimal("0"),
    "graph_4_cut_2_term_0_integrand": Decimal("-8.313682635992704964386512547388E-19"),
    "graph_5_cut_0_term_0_integrand": Decimal("3.158685653791957004261844563418E-20"),
    "graph_6_cut_1_term_0_integrand": Decimal("0"),
    "graph_6_cut_3_term_0_integrand": Decimal("0"),
    "graph_6_cut_4_term_0_integrand": Decimal("3.462804212262270012745607139956E-18"),
    "graph_7_cut_0_term_0_integrand": Decimal("0"),
    "graph_7_cut_3_term_0_integrand": Decimal("0"),
    "graph_7_cut_4_term_0_integrand": Decimal("-7.532678340029632283088453669329E-19"),
    "graph_7_cut_6_term_0_integrand": Decimal("0"),
    "graph_7_cut_9_term_0_integrand": Decimal("0"),
    "graph_7_cut_10_term_0_integrand": Decimal("-7.532678340029634043984843715030E-19"),
    "graph_8_cut_1_term_0_integrand": Decimal("0"),
    "graph_8_cut_3_term_0_integrand": Decimal("0"),
    "graph_8_cut_4_term_0_integrand": Decimal("-5.254546225448249414936714009729E-18"),
    "graph_8_cut_6_term_0_integrand": Decimal("0"),
    "graph_8_cut_8_term_0_integrand": Decimal("0"),
    "graph_8_cut_11_term_0_integrand": Decimal("-5.254546225448262401853832462430E-18"),
    "graph_9_cut_0_term_0_integrand": Decimal("0"),
    "graph_9_cut_2_term_0_integrand": Decimal("0"),
    "graph_9_cut_5_term_0_integrand": Decimal("1.473613188205021654828868072981E-20"),
    "graph_9_cut_6_term_0_integrand": Decimal("1.473613188205021984302355170436E-20"),
    "graph_10_cut_1_term_0_integrand": Decimal("0"),
    "graph_10_cut_2_term_0_integrand": Decimal("2.230405092540014639811067394671E-19"),
    "graph_10_cut_5_term_0_integrand": Decimal("0"),
}
TT_TWO_LOOP_QQBAR_SINGLE_CUT_REFERENCE_TERMS: dict[TwoLoopStableTermKey, Decimal] = {
    ("GL00", "PM", (("0",), ("1",))): Decimal(
        "2.564934270982978362558987077E-22"
    ),
    ("GL00", "threshold", (("0",), ("1",))): Decimal(
        "-1.663339937854731049099220197E-22"
    ),
    ("GL00", "uv", (("0",), ("1",))): Decimal(
        "-3.211818454038580202478615000E-24"
    ),
    ("GL00", "uv_int", (("0",), ("1",))): Decimal(
        "-1.525525757370483854323889844E-22"
    ),
    ("GL01", "PM", (("0",), ("1",))): Decimal(
        "3.005649832397274938443490561E-21"
    ),
    ("GL02", "PM", (("0",), ("1",))): Decimal(
        "-3.038825330624469579374345246E-21"
    ),
    ("GL02", "anti-collinear", (("0",), ("6", "8"))): Decimal("0"),
    ("GL03", "PM", (("0",), ("1",))): Decimal(
        "-2.246598835620985116738360359E-20"
    ),
    ("GL03", "collinear", (("1",), ("6", "8"))): Decimal("0"),
    ("GL04", "PM", (("0",), ("1",))): Decimal(
        "-3.563233685485752405164429907E-21"
    ),
    ("GL04", "anti-collinear", (("0",), ("7", "8"))): Decimal("0"),
    ("GL04", "collinear", (("1",), ("6", "8"))): Decimal("0"),
    ("GL04", "threshold", (("0",), ("1",))): Decimal(
        "4.934358776777936739552483039E-22"
    ),
    ("GL04", "uv", (("0",), ("1",))): Decimal(
        "3.329362655651698603590808121E-23"
    ),
    ("GL04", "uv_int", (("0",), ("1",))): Decimal(
        "4.884584595917869163451666014E-22"
    ),
    ("GL05", "PM", (("0",), ("1",))): Decimal(
        "-3.654984385592332288387655741E-20"
    ),
    ("GL05", "threshold", (("0",), ("1",))): Decimal(
        "3.442020125877740895747976446E-20"
    ),
    ("GL05", "uv", (("0",), ("1",))): Decimal(
        "-5.312825197190258050623963111E-22"
    ),
    ("GL05", "uv_int", (("0",), ("1",))): Decimal(
        "-7.457315340715731118170756226E-22"
    ),
    ("GL06", "PM", (("0",), ("1",))): Decimal(
        "-1.089377818897777092597997971E-20"
    ),
    ("GL06", "anti-collinear", (("0",), ("6", "7"))): Decimal("0"),
    ("GL06", "anti-collinear", (("0",), ("7", "8"))): Decimal("0"),
    ("GL07", "PM", (("0",), ("1",))): Decimal(
        "5.525196313866040811599398990E-21"
    ),
    ("GL07", "anti-collinear", (("0",), ("7", "8"))): Decimal("0"),
    ("GL07", "collinear", (("1",), ("6", "7"))): Decimal("0"),
    ("GL07", "soft", (("6", "7"), ("7", "8"))): Decimal("0"),
    ("GL07", "soft-anti-collinear", (("6", "7"), ("7", "8"))): Decimal("0"),
    ("GL07", "soft-collinear", (("6", "7"), ("7", "8"))): Decimal("0"),
    ("GL08", "PM", (("0",), ("1",))): Decimal(
        "-4.366014594668096655144365910E-22"
    ),
    ("GL08", "collinear", (("1",), ("6", "7"))): Decimal("0"),
    ("GL08", "collinear", (("1",), ("7", "8"))): Decimal("0"),
    ("GL09", "PM", (("0",), ("1",))): Decimal(
        "6.608178683235815854427477140E-22"
    ),
    ("GL09", "PM", (("6",), ("7",))): Decimal(
        "-1.320946037728080752156242061E-21"
    ),
    ("GL09", "anti-collinear", (("0",), ("7", "8"))): Decimal("0"),
    ("GL09", "anti-collinear", (("1", "8"), ("6",))): Decimal("0"),
    ("GL09", "collinear", (("0", "8"), ("7",))): Decimal("0"),
    ("GL09", "collinear", (("1",), ("6", "8"))): Decimal("0"),
    ("GL09", "soft", (("0", "8"), ("7", "8"))): Decimal("0"),
    ("GL09", "soft", (("1", "8"), ("6", "8"))): Decimal("0"),
    ("GL09", "soft-anti-collinear", (("0", "8"), ("7", "8"))): Decimal("0"),
    ("GL09", "soft-anti-collinear", (("1", "8"), ("6", "8"))): Decimal("0"),
    ("GL09", "soft-collinear", (("0", "8"), ("7", "8"))): Decimal("0"),
    ("GL09", "soft-collinear", (("1", "8"), ("6", "8"))): Decimal("0"),
    ("GL09", "threshold", (("0",), ("1",))): Decimal(
        "-1.516154645692777678835814949E-21"
    ),
    ("GL09", "threshold", (("6",), ("7",))): Decimal(
        "1.375339753513663681966068215E-21"
    ),
    ("GL09", "uv", (("0",), ("1",))): Decimal(
        "-3.851990052356234492219057607E-24"
    ),
    ("GL09", "uv", (("6",), ("7",))): Decimal(
        "-3.184567265424894901331183372E-24"
    ),
    ("GL09", "uv_int", (("0",), ("1",))): Decimal(
        "-4.573761247209048215782700343E-23"
    ),
    ("GL10", "PM", (("0",), ("1",))): Decimal(
        "-5.016857409659023892861433949E-21"
    ),
    ("GL10", "PM", (("6",), ("7",))): Decimal(
        "8.717966039232527296676493224E-21"
    ),
    ("GL10", "threshold", (("0",), ("1",))): Decimal(
        "1.050269863882578175602206284E-20"
    ),
    ("GL10", "threshold", (("6",), ("7",))): Decimal(
        "-5.653533212903497492135157459E-21"
    ),
    ("GL10", "uv", (("0",), ("1",))): Decimal(
        "-5.879753724216092717881308646E-22"
    ),
    ("GL10", "uv", (("6",), ("7",))): Decimal(
        "-5.148941299851106949762318680E-22"
    ),
    ("GL10", "uv_int", (("0",), ("1",))): Decimal(
        "-3.087280396105554915100382199E-22"
    ),
    ("GL10", "uv_int", (("6",), ("7",))): Decimal(
        "-2.419744185225200642016213919E-22"
    ),
    ("GL11", "PM", (("0",), ("1",))): Decimal(
        "-3.284739651210990541431726813E-20"
    ),
    ("GL11", "threshold", (("0",), ("1",))): Decimal(
        "-3.883713797912208321015756661E-21"
    ),
    ("GL11", "uv", (("0",), ("1",))): Decimal(
        "3.078603767202858722146739418E-23"
    ),
    ("GL11", "uv_int", (("0",), ("1",))): Decimal(
        "4.884584595917869143398958793E-22"
    ),
    ("GL13", "PM", (("0",), ("1",))): Decimal(
        "-3.168507357686517808776348496E-21"
    ),
    ("GL13", "anti-collinear", (("0",), ("6", "8"))): Decimal("0"),
    ("GL13", "collinear", (("1",), ("7", "8"))): Decimal("0"),
    ("GL13", "threshold", (("0",), ("1",))): Decimal(
        "3.224333870065831183111888173E-21"
    ),
    ("GL15", "PM", (("0",), ("1",))): Decimal(
        "-8.527338959586739758087825535E-21"
    ),
    ("GL15", "uv", (("0",), ("1",))): Decimal(
        "2.112782693338624605885823206E-22"
    ),
    ("GL15", "uv_int", (("0",), ("1",))): Decimal(
        "-1.181956937538955576796158337E-23"
    ),
    ("GL17", "PM", (("0",), ("1",))): Decimal(
        "4.928830535107888943017536152E-23"
    ),
    ("GL17", "uv", (("0",), ("1",))): Decimal(
        "-3.762512922830888527111973019E-24"
    ),
    ("GL17", "uv_int", (("0",), ("1",))): Decimal(
        "-2.602322155717267036262632568E-23"
    ),
    ("GL18", "PM", (("0",), ("1",))): Decimal(
        "-6.079789072214330064551277412E-22"
    ),
    ("GL18", "threshold", (("0",), ("1",))): Decimal(
        "4.959356732853394776159479951E-21"
    ),
    ("GL18", "uv", (("0",), ("1",))): Decimal(
        "-5.524481424918017008809399257E-22"
    ),
    ("GL18", "uv_int", (("0",), ("1",))): Decimal(
        "-2.738867745114538102007662321E-22"
    ),
}
TT_TWO_LOOP_QQBAR_COLLINEAR_REFERENCE_TERMS: dict[TwoLoopStableTermKey, Decimal] = {
    ("GL04", "collinear", (("1",), ("6", "8"))): Decimal(
        "7.628105502159485966771295864E-12"
    ),
    ("GL07", "collinear", (("1",), ("6", "7"))): Decimal(
        "-5.188756155721537388198338262E-12"
    ),
    ("GL08", "collinear", (("1",), ("6", "7"))): Decimal(
        "1.042382186581516850904888815E-12"
    ),
    ("GL08", "collinear", (("1",), ("7", "8"))): Decimal(
        "1.042382186581516850904889692E-12"
    ),
    ("GL09", "collinear", (("1",), ("6", "8"))): Decimal(
        "2.097346550277158898464725974E-12"
    ),
    ("GL13", "collinear", (("1",), ("7", "8"))): Decimal(
        "7.554890778756194532480068586E-12"
    ),
}
TT_TWO_LOOP_QQBAR_ANTI_COLLINEAR_REFERENCE_TERMS: dict[
    TwoLoopStableTermKey, Decimal
] = {
    ("GL02", "anti-collinear", (("0",), ("6", "8"))): Decimal(
        "1.263969016808372857623376992E-11"
    ),
    ("GL04", "anti-collinear", (("0",), ("7", "8"))): Decimal(
        "7.628105502159485966771295864E-12"
    ),
    ("GL09", "anti-collinear", (("0",), ("7", "8"))): Decimal(
        "2.097346550277158898464725974E-12"
    ),
    ("GL13", "anti-collinear", (("0",), ("6", "8"))): Decimal(
        "7.554890778756195479545998151E-12"
    ),
}
TT_TWO_LOOP_QQBAR_SOFT_REFERENCE_TERMS: dict[TwoLoopStableTermKey, Decimal] = {
    ("GL07", "soft", (("6", "7"), ("7", "8"))): Decimal(
        "-1.889442430165611184266423204E-8"
    ),
    ("GL09", "soft", (("0", "8"), ("7", "8"))): Decimal(
        "1.889442430165611184266423204E-8"
    ),
    ("GL09", "soft", (("1", "8"), ("6", "8"))): Decimal(
        "1.889442430165611184266423204E-8"
    ),
}
TT_TWO_LOOP_QQBAR_SOFT_ANTI_COLLINEAR_REFERENCE_TERMS: dict[
    TwoLoopStableTermKey, Decimal
] = {
    ("GL07", "soft-anti-collinear", (("6", "7"), ("7", "8"))): Decimal(
        "6.695016179142717319873222485E-8"
    ),
    ("GL09", "soft-anti-collinear", (("1", "8"), ("6", "8"))): Decimal(
        "-6.695016179142717319873222485E-8"
    ),
}
TT_TWO_LOOP_QQBAR_SOFT_COLLINEAR_REFERENCE_TERMS: dict[
    TwoLoopStableTermKey, Decimal
] = {
    ("GL09", "soft-collinear", (("0", "8"), ("7", "8"))): Decimal(
        "-6.695016179142717319873222485E-8"
    ),
}
TT_TWO_LOOP_QG_COLLINEAR_REFERENCE_TERMS: dict[TwoLoopStableTermKey, Decimal] = {
    ("GL01", "collinear", (("1",), ("6", "7"))): Decimal(
        "-2.417805977402003581681108060E-11"
    ),
    ("GL02", "collinear", (("1",), ("6", "7"))): Decimal(
        "-5.822151515639001639740207557E-12"
    ),
    ("GL02", "collinear", (("1",), ("7", "8"))): Decimal(
        "-5.822151515639001639737656550E-12"
    ),
    ("GL04", "collinear", (("1",), ("6", "8"))): Decimal(
        "4.388531129978856262186577392E-11"
    ),
    ("GL06", "collinear", (("1",), ("3", "7"))): Decimal(
        "-9.089253434562675967356486797E-11"
    ),
    ("GL08", "collinear", (("0", "3"), ("5",))): Decimal(
        "7.061221102857132658099698176E-11"
    ),
    ("GL08", "collinear", (("1",), ("3", "8"))): Decimal(
        "7.061221102857133742920880353E-11"
    ),
    ("GL10", "collinear", (("0", "3"), ("5",))): Decimal(
        "1.279638059666248667205162782E-10"
    ),
    ("GL10", "collinear", (("0", "7"), ("5",))): Decimal(
        "1.279638059666248667205147677E-10"
    ),
    ("GL12", "collinear", (("0", "3"), ("5",))): Decimal(
        "-1.063111863696687186947233226E-11"
    ),
    ("GL12", "collinear", (("1",), ("3", "7"))): Decimal(
        "-1.063111863696687265985814000E-11"
    ),
    ("GL14", "collinear", (("1",), ("3", "8"))): Decimal(
        "1.313408775677294365276743504E-12"
    ),
}
TT_TWO_LOOP_QG_ANTI_COLLINEAR_REFERENCE_TERMS: dict[
    TwoLoopStableTermKey, Decimal
] = {
    ("GL00", "anti-collinear", (("0",), ("6", "7"))): Decimal(
        "-2.330708725018462936142540531E-12"
    ),
    ("GL00", "anti-collinear", (("0",), ("7", "8"))): Decimal(
        "-2.330708725018462936142540531E-12"
    ),
}


def _default_ps_point() -> list[LorentzVector]:
    return [
        LorentzVector(500.0, 0.0, 0.0, 500.0),
        LorentzVector(500.0, 0.0, 0.0, -500.0),
        LorentzVector(
            438.5555662246945,
            155.3322001835378,
            348.0160396513587,
            -177.3773615718412,
        ),
        LorentzVector(
            356.3696374921922,
            -16.80238900851100,
            -318.7291102436005,
            97.48719163688098,
        ),
        LorentzVector(
            205.0747962831133,
            -138.5298111750267,
            -29.28692940775817,
            79.89016993496030,
        ),
    ]


def _graph_name(graph) -> str:
    return str(graph.dot.get_name()).strip('"')


def _build_process() -> DYGraphCutSumReferenceProcess:
    with contextlib.redirect_stdout(io.StringIO()):
        return DYGraphCutSumReferenceProcess(
            m_top=173.0,
            m_higgs=125.0,
            ps_point=_default_ps_point(),
            n_loops=1,
            clean=True,
            logger_level=logging.CRITICAL,
            skip_ps_validation=True,
            toml_config_path=str(GENERATE_CONFIG),
            runtime_toml_config_path=str(RUNTIME_CONFIG),
        )


def _generate_channel_graphs(
    process: DYGraphCutSumReferenceProcess, channel: str
) -> DYDotGraphs:
    spec = CHANNEL_COMMANDS[channel]
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        process.gl_worker.run(spec["command"])
        process.gl_worker.run("save state -o")

    amplitudes, cross_sections = process.gl_worker.list_outputs()
    if (
        spec["graphs_name"] not in amplitudes
        and spec["graphs_name"] not in cross_sections
    ):
        raise AssertionError(f"Generated output '{spec['graphs_name']}' was not found.")
    process_id = (
        amplitudes[spec["graphs_name"]]
        if spec["graphs_name"] in amplitudes
        else cross_sections[spec["graphs_name"]]
    )

    dot_str = process.gl_worker.get_dot_files(
        process_id=process_id,
        integrand_name=spec["graphs_name"],
    )
    return DYDotGraphs(dot_str=dot_str)


def _filtered_graphs(graphs: DYDotGraphs, channel: str) -> DYDotGraphs:
    particle_filter = CHANNEL_COMMANDS[channel]["particle_filter"]
    filtered = DYDotGraphs()
    filtered.extend(deepcopy(graphs.filter_particle_definition(particle_filter)))
    return filtered


def _routed_integrands_for_graph(graph, loop_processor) -> list:
    vac_g = canonicalise_vacuum_graph(deepcopy(graph))
    vacuum_g = VacuumDotGraph(deepcopy(vac_g.dot))
    routed_graphs = vacuum_g.cut_graphs_with_routing_leading_virtuality([], ["a"])

    routed_integrands = []
    for gg in routed_graphs:
        cut_graph = deepcopy(routed_cut_graph(gg[3], gg[0], gg[1], gg[2]))
        routed_integrands.extend(loop_processor.get_integrand(deepcopy(cut_graph)))
    return routed_integrands


def _evaluate_cut_sum(routed_integrands: list, point: dict[str, object]) -> Decimal:
    evaluators = [
        evaluate_integrand(
            1,
            "DY",
            deepcopy(routed_integrand),
            n_hornerscheme_iterations=1000,
            n_cpe_iterations=10000,
            observable_params=OBSERVABLE_PARAMS,
        )
        for routed_integrand in routed_integrands
    ]

    total = Decimal(0)
    for evaluator in evaluators:
        total += evaluator.eval(
            point["ks"],
            point["p1"],
            point["p2"],
            point["z"],
            mode="arb",
            decimal_digit_precision=64,
        )
    return total


@lru_cache(maxsize=1)
def _current_graph_results() -> dict[tuple[str, str], dict[str, Decimal]]:
    results: dict[tuple[str, str], dict[str, Decimal]] = {}
    process = _build_process()

    for channel in CHANNEL_COMMANDS:
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            graphs = _generate_channel_graphs(process, channel)
            filtered = _filtered_graphs(graphs, channel)
            loop_processor = LoopIntegrandConstructor([], process.process_name, 1)

            for graph in filtered:
                routed_integrands = _routed_integrands_for_graph(graph, loop_processor)
                graph_key = (channel, _graph_name(graph))
                results[graph_key] = {
                    point_name: _evaluate_cut_sum(routed_integrands, point)
                    for point_name, point in POINTS.items()
                }

    return results


def _assert_decimal_close(actual: Decimal, expected: Decimal) -> None:
    scale = max(Decimal(1), abs(expected))
    tolerance = Decimal("1e-12") * scale
    if abs(abs(actual) - abs(expected)) > tolerance:
        raise AssertionError(
            f"Unexpected cut sum: actual={actual}, expected={expected}, tolerance={tolerance}"
        )


def _build_ggtt_process() -> TTGraphChannelReferenceProcess:
    with contextlib.redirect_stdout(io.StringIO()):
        return TTGraphChannelReferenceProcess(
            m_top=173.0,
            m_higgs=125.0,
            ps_point=_default_ps_point(),
            n_loops=1,
            clean=True,
            logger_level=logging.CRITICAL,
            skip_ps_validation=True,
            toml_config_path=str(GENERATE_CONFIG),
            runtime_toml_config_path=str(RUNTIME_CONFIG),
            final_state=["t", "t"],
            process_name="tt~",
        )


def _generate_ggtt_graphs(process: TTGraphChannelReferenceProcess) -> DYDotGraphs:
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        process.gl_worker.run(GGTT_COMMAND["command"])
        process.gl_worker.run("save state -o")

    amplitudes, cross_sections = process.gl_worker.list_outputs()
    if (
        GGTT_COMMAND["graphs_name"] not in amplitudes
        and GGTT_COMMAND["graphs_name"] not in cross_sections
    ):
        raise AssertionError(
            f"Generated output '{GGTT_COMMAND['graphs_name']}' was not found."
        )
    process_id = (
        amplitudes[GGTT_COMMAND["graphs_name"]]
        if GGTT_COMMAND["graphs_name"] in amplitudes
        else cross_sections[GGTT_COMMAND["graphs_name"]]
    )

    dot_str = process.gl_worker.get_dot_files(
        process_id=process_id,
        integrand_name=GGTT_COMMAND["graphs_name"],
    )
    return DYDotGraphs(dot_str=dot_str)


@lru_cache(maxsize=1)
def _current_ggtt_graph_results() -> dict[str, dict[str, Decimal]]:
    process = _build_ggtt_process()
    with contextlib.redirect_stdout(io.StringIO()):
        process.process_1L_generated_graphs(_generate_ggtt_graphs(process))

    bundle = DYCompiledBundle.load("tt~", process.get_integrand_name())
    results: dict[str, dict[str, Decimal]] = {}
    for channel_index, channel_name in enumerate(bundle.graph_channel_names()):
        point_values: dict[str, Decimal] = {}
        for point_name, point in GGTT_POINTS.items():
            value = bundle.evaluate(
                point["ks"],
                point["p1"],
                point["p2"],
                point["z"],
                mode="compiled",
                channel_selector=channel_index,
            )
            if not math.isclose(value.imag, 0.0, abs_tol=1.0e-30):
                raise AssertionError(
                    f"Expected a real ggtt graph value for '{channel_name}', got {value}."
                )
            point_values[point_name] = Decimal(repr(value.real))
        results[channel_name] = point_values

    return results


def _build_2l_ttbar_process() -> TTTwoLoopChannelReferenceProcess:
    with contextlib.redirect_stdout(io.StringIO()):
        return TTTwoLoopChannelReferenceProcess(
            m_top=173.0,
            m_higgs=125.0,
            ps_point=_default_ps_point(),
            n_loops=2,
            clean=True,
            logger_level=logging.CRITICAL,
            skip_ps_validation=True,
            toml_config_path=str(GENERATE_CONFIG),
            runtime_toml_config_path=str(RUNTIME_CONFIG),
            final_state=["t", "t"],
            process_name="tt~",
            dy_channel=(1, -1),
            dy_fallback_precision=32,
            external_gluon_polarisation=True,
            disable_integrated_uv_cts=False,
            dy_parallel_graphs=16,
            load_compiled_bundle=False,
        )


def _build_2l_ttbar_qg_process() -> TTTwoLoopQGChannelReferenceProcess:
    with contextlib.redirect_stdout(io.StringIO()):
        return TTTwoLoopQGChannelReferenceProcess(
            m_top=173.0,
            m_higgs=125.0,
            ps_point=_default_ps_point(),
            n_loops=2,
            clean=True,
            logger_level=logging.CRITICAL,
            skip_ps_validation=True,
            toml_config_path=str(GENERATE_CONFIG),
            runtime_toml_config_path=str(RUNTIME_CONFIG),
            final_state=["t", "t"],
            process_name="tt~",
            dy_channel=(1, 0),
            dy_fallback_precision=32,
            external_gluon_polarisation=True,
            disable_integrated_uv_cts=False,
            dy_parallel_graphs=11,
            load_compiled_bundle=False,
        )


@lru_cache(maxsize=1)
def _current_2l_ttbar_bundle() -> DYCompiledBundle:
    process = _build_2l_ttbar_process()
    with contextlib.redirect_stdout(io.StringIO()):
        process.generate_graphs()

    return DYCompiledBundle.load("tt~", process.get_integrand_name())


def _evaluate_2l_ttbar_cut_terms(
    point: dict[str, object],
) -> tuple[list[str], dict[str, Decimal], dict[TwoLoopStableTermKey, Decimal]]:
    bundle = _current_2l_ttbar_bundle()
    total, terms = bundle.evaluate_arb_terms(
        point["ks"],
        point["p1"],
        point["p2"],
        point["z"],
        m_uv=point["m_uv"],
        decimal_digit_precision=32,
    )
    assert total == sum(value for _name, value in terms)
    stable_terms = _stable_two_loop_terms(bundle, terms)
    return bundle.graph_channel_names(), dict(terms), stable_terms


@lru_cache(maxsize=1)
def _current_2l_ttbar_cut_data() -> (
    tuple[list[str], dict[str, Decimal], dict[TwoLoopStableTermKey, Decimal]]
):
    return _evaluate_2l_ttbar_cut_terms(TT_TWO_LOOP_POINT)


@lru_cache(maxsize=1)
def _current_2l_ttbar_cut_terms() -> tuple[list[str], dict[str, Decimal]]:
    channel_names, terms, _stable_terms = _current_2l_ttbar_cut_data()
    return channel_names, terms


@lru_cache(maxsize=1)
def _current_2l_ttbar_single_cut_terms() -> (
    tuple[list[str], dict[TwoLoopStableTermKey, Decimal]]
):
    channel_names, _terms, stable_terms = _current_2l_ttbar_cut_data()
    return channel_names, stable_terms


@lru_cache(maxsize=None)
def _current_2l_ttbar_probe_cut_terms(
    point_name: str,
) -> tuple[list[str], dict[TwoLoopStableTermKey, Decimal]]:
    channel_names, _terms, stable_terms = _evaluate_2l_ttbar_cut_terms(
        TT_TWO_LOOP_QQBAR_PROBE_POINTS[point_name]
    )
    return channel_names, stable_terms


@lru_cache(maxsize=1)
def _current_2l_ttbar_qg_bundle() -> DYCompiledBundle:
    process = _build_2l_ttbar_qg_process()
    with contextlib.redirect_stdout(io.StringIO()):
        process.generate_graphs()

    return DYCompiledBundle.load("tt~", process.get_integrand_name())


def _evaluate_2l_ttbar_qg_cut_terms(
    point: dict[str, object],
) -> tuple[list[str], dict[str, Decimal], dict[TwoLoopStableTermKey, Decimal]]:
    bundle = _current_2l_ttbar_qg_bundle()
    total, terms = bundle.evaluate_arb_terms(
        point["ks"],
        point["p1"],
        point["p2"],
        point["z"],
        m_uv=point["m_uv"],
        decimal_digit_precision=32,
    )
    assert total == sum(value for _name, value in terms)
    stable_terms = _stable_two_loop_terms(bundle, terms)
    return bundle.graph_channel_names(), dict(terms), stable_terms


@lru_cache(maxsize=1)
def _current_2l_ttbar_qg_cut_terms() -> tuple[list[str], dict[str, Decimal]]:
    channel_names, terms, _stable_terms = _evaluate_2l_ttbar_qg_cut_terms(
        TT_TWO_LOOP_POINT
    )
    return channel_names, terms


@lru_cache(maxsize=None)
def _current_2l_ttbar_qg_probe_cut_terms(
    point_name: str,
) -> tuple[list[str], dict[TwoLoopStableTermKey, Decimal]]:
    channel_names, _terms, stable_terms = _evaluate_2l_ttbar_qg_cut_terms(
        TT_TWO_LOOP_QG_PROBE_POINTS[point_name]
    )
    return channel_names, stable_terms


def _assert_small_decimal_close(actual: Decimal, expected: Decimal) -> None:
    tolerance = max(Decimal("1e-18"), Decimal("1e-9") * abs(expected))
    if abs(actual - expected) > tolerance:
        raise AssertionError(
            f"Unexpected ggtt graph value: actual={actual}, expected={expected}, tolerance={tolerance}"
        )


def _assert_tiny_decimal_close(actual: Decimal, expected: Decimal) -> None:
    tolerance = max(Decimal("1e-30"), Decimal("1e-9") * abs(expected))
    if abs(actual - expected) > tolerance:
        raise AssertionError(
            "Unexpected two-loop ttbar cut term: "
            f"actual={actual}, expected={expected}, tolerance={tolerance}"
        )


def _two_loop_graph_name(term_name: str) -> str:
    graph_name, separator, _rest = term_name.partition("_cut_")
    if separator == "":
        raise AssertionError(f"Unexpected two-loop ttbar term name: {term_name}")
    return graph_name


def _two_loop_terms_by_graph(terms: dict[str, Decimal]) -> dict[str, list[Decimal]]:
    terms_by_graph: dict[str, list[Decimal]] = {}
    for term_name, value in terms.items():
        terms_by_graph.setdefault(_two_loop_graph_name(term_name), []).append(value)
    return terms_by_graph


def _stable_two_loop_partition(
    routed_graph_name: str,
) -> tuple[tuple[str, ...], ...]:
    marker = "_partition_"
    if marker not in routed_graph_name:
        raise AssertionError(
            f"Missing partition in routed graph name: {routed_graph_name}"
        )
    partition_text = routed_graph_name.split(marker, 1)[1]
    left_raw, separator, right_raw = partition_text.partition("_")
    if separator == "":
        raise AssertionError(
            f"Unexpected routed graph partition name: {routed_graph_name}"
        )

    sides = []
    for raw_side in (left_raw, right_raw):
        side = ast.literal_eval(raw_side)
        if not isinstance(side, list):
            raise AssertionError(
                f"Unexpected routed graph partition side: {routed_graph_name}"
            )
        sides.append(tuple(sorted(str(edge_id) for edge_id in side)))
    return tuple(sorted(sides))


def _stable_two_loop_term_key(
    bundle: DYCompiledBundle, term
) -> TwoLoopStableTermKey:
    evaluator = bundle.evaluators[term.evaluator_name]
    source_graph_name = evaluator.additional_data.get("source_graph_name")
    routed_graph_name = evaluator.additional_data.get("routed_graph_name")
    approximation_type = term.approximation_type
    if source_graph_name is None or routed_graph_name is None:
        raise AssertionError(
            f"Missing stable metadata for term '{term.evaluator_name}'."
        )
    if approximation_type is None:
        raise AssertionError(
            f"Missing approximation type for term '{term.evaluator_name}'."
        )
    return (
        str(source_graph_name),
        str(approximation_type),
        _stable_two_loop_partition(str(routed_graph_name)),
    )


def _stable_two_loop_terms(
    bundle: DYCompiledBundle, terms: list[tuple[str, Decimal]]
) -> dict[TwoLoopStableTermKey, Decimal]:
    term_by_name = {term.evaluator_name: term for term in bundle.terms}
    stable_terms: dict[TwoLoopStableTermKey, Decimal] = {}
    for term_name, value in terms:
        key = _stable_two_loop_term_key(bundle, term_by_name[term_name])
        stable_terms[key] = stable_terms.get(key, Decimal(0)) + value
    return stable_terms


def _assert_two_loop_graph_terms_close(
    graph_name: str, actual_values: list[Decimal], expected_values: list[Decimal]
) -> None:
    assert len(actual_values) == len(expected_values), (
        f"Unexpected number of two-loop ttbar terms for {graph_name}: "
        f"actual={len(actual_values)}, expected={len(expected_values)}"
    )
    for actual_value, expected_value in zip(
        sorted(actual_values), sorted(expected_values), strict=True
    ):
        _assert_tiny_decimal_close(actual_value, expected_value)


def _assert_selected_two_loop_terms_close(
    actual_terms: dict[TwoLoopStableTermKey, Decimal],
    expected_terms: dict[TwoLoopStableTermKey, Decimal],
) -> None:
    assert expected_terms
    for term_key, expected_value in expected_terms.items():
        assert expected_value != 0
        assert term_key in actual_terms
        _assert_tiny_decimal_close(actual_terms[term_key], expected_value)


def _assert_all_two_loop_terms_close(
    actual_terms: dict[TwoLoopStableTermKey, Decimal],
    expected_terms: dict[TwoLoopStableTermKey, Decimal],
) -> None:
    assert set(actual_terms) == set(expected_terms)
    for term_key, expected_value in expected_terms.items():
        _assert_tiny_decimal_close(actual_terms[term_key], expected_value)


@pytest.mark.slow
def test_dy_graph_cut_sums_match_references():
    current = _current_graph_results()
    assert set(current) == set(REFERENCE_SUMS)

    for graph_key, expected_points in REFERENCE_SUMS.items():
        actual_points = current[graph_key]
        assert set(actual_points) == set(expected_points)
        for point_name, expected_value in expected_points.items():
            _assert_decimal_close(actual_points[point_name], expected_value)


@pytest.mark.slow
def test_ggtt_graph_channels_match_references():
    current = _current_ggtt_graph_results()
    assert set(current) == set(GGTT_REFERENCE_SUMS)

    for graph_name, expected_points in GGTT_REFERENCE_SUMS.items():
        actual_points = current[graph_name]
        assert set(actual_points) == set(expected_points)
        for point_name, expected_value in expected_points.items():
            _assert_small_decimal_close(actual_points[point_name], expected_value)


@pytest.mark.slow
def test_2l_ttbar_qqbar_cut_terms_match_references():
    channel_names, actual_terms = _current_2l_ttbar_cut_terms()
    assert tuple(channel_names) == TT_TWO_LOOP_CHANNEL_NAMES

    actual_terms_by_graph = _two_loop_terms_by_graph(actual_terms)
    expected_terms_by_graph = _two_loop_terms_by_graph(TT_TWO_LOOP_REFERENCE_TERMS)
    assert set(actual_terms_by_graph) == set(TT_TWO_LOOP_CHANNEL_NAMES)
    assert set(expected_terms_by_graph) == set(TT_TWO_LOOP_CHANNEL_NAMES)

    for graph_name in TT_TWO_LOOP_CHANNEL_NAMES:
        _assert_two_loop_graph_terms_close(
            graph_name,
            actual_terms_by_graph[graph_name],
            expected_terms_by_graph[graph_name],
        )

    single_cut_channel_names, single_cut_terms = _current_2l_ttbar_single_cut_terms()
    assert tuple(single_cut_channel_names) == TT_TWO_LOOP_CHANNEL_NAMES
    _assert_all_two_loop_terms_close(
        single_cut_terms,
        TT_TWO_LOOP_QQBAR_SINGLE_CUT_REFERENCE_TERMS,
    )

    probe_checks = [
        (
            "collinear_anti_collinear",
            TT_TWO_LOOP_QQBAR_COLLINEAR_REFERENCE_TERMS,
        ),
        (
            "collinear_anti_collinear",
            TT_TWO_LOOP_QQBAR_ANTI_COLLINEAR_REFERENCE_TERMS,
        ),
        (
            "soft",
            TT_TWO_LOOP_QQBAR_SOFT_REFERENCE_TERMS,
        ),
        (
            "soft",
            TT_TWO_LOOP_QQBAR_SOFT_COLLINEAR_REFERENCE_TERMS,
        ),
        (
            "soft",
            TT_TWO_LOOP_QQBAR_SOFT_ANTI_COLLINEAR_REFERENCE_TERMS,
        ),
    ]
    for point_name, expected_terms in probe_checks:
        probe_channel_names, probe_terms = _current_2l_ttbar_probe_cut_terms(
            point_name
        )
        assert tuple(probe_channel_names) == TT_TWO_LOOP_CHANNEL_NAMES
        _assert_selected_two_loop_terms_close(probe_terms, expected_terms)


@pytest.mark.slow
@pytest.mark.skipif(
    not RUN_SLOW_2L_QG,
    reason="Set PYGLOOP_RUN_SLOW_2L_QG=1 to run the slow two-loop ttbar (1,0) regression.",
)
def test_2l_ttbar_qg_cut_terms_match_references():
    channel_names, actual_terms = _current_2l_ttbar_qg_cut_terms()
    assert tuple(channel_names) == TT_TWO_LOOP_QG_CHANNEL_NAMES

    actual_terms_by_graph = _two_loop_terms_by_graph(actual_terms)
    expected_terms_by_graph = _two_loop_terms_by_graph(TT_TWO_LOOP_QG_REFERENCE_TERMS)
    assert set(actual_terms_by_graph) == set(TT_TWO_LOOP_QG_CHANNEL_NAMES)
    assert set(expected_terms_by_graph) == set(TT_TWO_LOOP_QG_CHANNEL_NAMES)

    for graph_name in TT_TWO_LOOP_QG_CHANNEL_NAMES:
        _assert_two_loop_graph_terms_close(
            graph_name,
            actual_terms_by_graph[graph_name],
            expected_terms_by_graph[graph_name],
        )

    probe_checks = [
        (
            "collinear",
            TT_TWO_LOOP_QG_COLLINEAR_REFERENCE_TERMS,
        ),
        (
            "anti_collinear",
            TT_TWO_LOOP_QG_ANTI_COLLINEAR_REFERENCE_TERMS,
        ),
    ]
    for point_name, expected_terms in probe_checks:
        probe_channel_names, probe_terms = _current_2l_ttbar_qg_probe_cut_terms(
            point_name
        )
        assert tuple(probe_channel_names) == TT_TWO_LOOP_QG_CHANNEL_NAMES
        _assert_selected_two_loop_terms_close(
            probe_terms,
            expected_terms,
        )
