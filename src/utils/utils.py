import json
import logging
import math
import os
import pickle
import statistics
import timeit
from enum import StrEnum
from functools import wraps
from pprint import pprint  # noqa: F401
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import numpy
import pydot
from numpy.typing import NDArray
from pydot import Edge, Node  # noqa: F401
from symbolica import (
    CompiledComplexEvaluator,
    CompiledRealEvaluator,
    E,
    Evaluator,
    Expression,
    Replacement,
    Sample,
)

from utils.vectors import LorentzVector, Vector  # noqa: F401

PYGLOOP_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PYGLOOP_FOLDER, "src")

OUTPUTS_FOLDER = os.path.join(PYGLOOP_FOLDER, "outputs")
DOTS_FOLDER = os.path.join(OUTPUTS_FOLDER, "dot_files")
RESOURCES_FOLDER = os.path.join(PYGLOOP_FOLDER, "resources")
INTEGRATION_WORKSPACE_FOLDER = os.path.join(OUTPUTS_FOLDER, "integration_workspaces")
EVALUATORS_FOLDER = os.path.join(OUTPUTS_FOLDER, "evaluators")
GAMMALOOP_STATES_FOLDER = os.path.join(OUTPUTS_FOLDER, "gammaloop_states")
CONFIGS_FOLDER = os.path.join(PYGLOOP_FOLDER, "configs")

np_cmplx_one = numpy.complex128(1.0, 0.0)
np_cmplx_zero = numpy.complex128(0.0, 0.0)

try:
    import symjit  # noqa: F401

    SYMJIT_AVAILABLE = True
except ImportError:
    SYMJIT_AVAILABLE = False


def setup_logging():
    logging.basicConfig(
        format=f"{Colour.GREEN}%(levelname)s{Colour.END} {Colour.BLUE}%(funcName)s l.%(lineno)d{Colour.END} {Colour.CYAN}t=%(asctime)s.%(msecs)03d{Colour.END} > %(message)s",  # fmt: off
        datefmt="%Y-%m-%d,%H:%M:%S",
    )


logger = logging.getLogger("pygloop")


class pygloopException(Exception):
    pass


def set_gammaloop_level(enter_level: int, exit_level: int):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self.set_log_level(enter_level)
            try:
                return func(self, *args, **kwargs)
            finally:
                self.set_log_level(exit_level)

        return wrapper

    return decorator


def set_tmp_logger_level(level: int):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            previous_level = logger.level
            logger.setLevel(level)
            try:
                return func(*args, **kwargs)
            finally:
                logger.setLevel(previous_level)

        return wrapper

    return decorator


class Colour(StrEnum):
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


GAMMA_EPS_EXPANSIONS = [
    Replacement(E("dim"), E("4 - 2*ε")),
    Replacement(E("𝚪(1-ε)"), E("1 + γₑ*ε + (1/12)*( 6*γₑ^2 + 𝜋^2)*ε^2 + ε^3*O(Gamma,eps^3)")),
    Replacement(E("𝚪(1+ε)"), E("1 - γₑ*ε + (1/12)*( 6*γₑ^2 + 𝜋^2)*ε^2 + ε^3*O(Gamma,eps^3)")),
    Replacement(E("𝚪(1-b_*ε)"), E("1 + γₑ*b_*ε + (1/12)*( 6*γₑ^2 + 𝜋^2)*b_^2*ε^2 + ε^3*O(Gamma,eps^3)")),
    Replacement(E("𝚪(1+b_*ε)"), E("1 - γₑ*b_*ε + (1/12)*( 6*γₑ^2 + 𝜋^2)*b_^2*ε^2 + ε^3*O(Gamma,eps^3)")),
    Replacement(E("𝚪(ε)"), E("1/ε - γₑ + (1/12)*( 6*γₑ^2 + 𝜋^2)*ε + ε^2*O(Gamma,eps^2)")),
    Replacement(E("𝚪(b_*ε)"), E("1/(b_*ε) - γₑ + (1/12)*( 6*γₑ^2 + 𝜋^2)*b_*ε + ε^2*O(Gamma,eps^2)")),
    Replacement(E("𝚪(2-ε)"), E("1 + (γₑ-1)*ε + (1/12)*( -12*γₑ + 6 * γₑ^2 + 𝜋^2)*ε^2 + ε^3*O(Gamma,eps^3)")),
    Replacement(E("𝚪(2+ε)"), E("1 + (1-γₑ)*ε + (1/12)*( -12*γₑ + 6 * γₑ^2 + 𝜋^2)*ε^2 + ε^3*O(Gamma,eps^3)")),
    Replacement(E("𝚪(2-b_*ε)"), E("1 + (γₑ-1)*b_*ε + (1/12)*( -12*γₑ + 6 * γₑ^2 + 𝜋^2)*b_^2*ε^2 + ε^3*O(Gamma,eps^3)")),
    Replacement(E("𝚪(2+b_*ε)"), E("1 + (1-γₑ)*b_*ε + (1/12)*( -12*γₑ + 6 * γₑ^2 + 𝜋^2)*b_^2*ε^2 + ε^3*O(Gamma,eps^3)")),
]


def eps_expansion_finite(expr: Expression, coeff_index: int = -1) -> Expression:
    expansion = expr.replace_multiple(GAMMA_EPS_EXPANSIONS).series(E("ε"), 0, 0, depth_is_absolute=True).to_expression().coefficient_list(E("ε"))
    if coeff_index is None:
        return expansion
    else:
        return expansion[coeff_index][-1]


def expr_to_string(expr: Expression) -> str:
    """Convert a symbolica expression to string."""
    # return expr.to_canonical_string()
    return expr.format_plain()


# Work around expressions given as strings containing the wrapping quotes
def Es(expr: str) -> Expression:
    return E(expr.replace('"', ""), default_namespace="gammalooprs")


def chunks(a_list: list[Any], n: int) -> Iterator[list[Any]]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(a_list), n):
        yield a_list[i : i + n]


class ParamBuilder(list):
    def __init__(self, cache: Any = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.positions: dict[tuple[Expression, ...], tuple[int, int]] = {}
        self.order: list[tuple[Expression, ...]] = []
        self.np: NDArray[numpy.complex128] = numpy.zeros(0, dtype=numpy.complex128)
        self.real_valued_inputs: list[int] = []
        self.purely_imaginary_valued_inputs: list[int] = []
        self.forced_complex_valued_inputs: list[int] = []
        if cache is None:
            self.cache = {}
        else:
            self.cache = cache

    def add_parameter_list(self, head: tuple[Expression, ...], length: int):
        if head in self.positions:
            raise pygloopException(f"Parameter {head} already exists")

        self.positions[head] = (len(self.np), len(self.np) + length)
        self.order.append(head)
        self.np = numpy.resize(self.np, len(self.np) + length)

    def freeze_all_current_parameters_phase(self, parameters: list[tuple[Expression, ...]] | None = None):
        if parameters is None:
            parameters = list(self.positions.keys())
        for param in parameters:
            if param not in self.positions:
                raise pygloopException(f"Could not find parameter {param} in param builder.")
            min, max = self.positions[param]
            for i in range(min, max):
                if self.np[i].imag == 0.0:
                    if i not in self.real_valued_inputs and i not in self.forced_complex_valued_inputs:
                        self.real_valued_inputs.append(i)
                elif self.np[i].real == 0.0:
                    if i not in self.purely_imaginary_valued_inputs and i not in self.forced_complex_valued_inputs:
                        self.purely_imaginary_valued_inputs.append(i)

    def force_parameters_to_real(self, parameters: list[tuple[Expression, ...]]):
        for param in parameters:
            if param not in self.positions:
                raise pygloopException(f"Could not find parameter {param} in param builder.")
            min, max = self.positions[param]
            for i in range(min, max):
                if self.np[i].imag != 0.0:
                    raise pygloopException(f"Cannot set parameter {param} to real-valued input; it has non-zero imaginary part {self.np[i].imag}.")
                self.real_valued_inputs.append(i)

    def force_parameters_to_imaginary(self, parameters: list[tuple[Expression, ...]]):
        for param in parameters:
            if param not in self.positions:
                raise pygloopException(f"Could not find parameter {param} in param builder.")
            min, max = self.positions[param]
            for i in range(min, max):
                if self.np[i].real != 0.0:
                    raise pygloopException(
                        f"Cannot set parameter {param} to imaginary-valued input; it has non-zero imaginary part {self.np[i].real}."
                    )
                self.purely_imaginary_valued_inputs.append(i)

    def force_parameters_to_complex(self, parameters: list[tuple[Expression, ...]]):
        for param in parameters:
            if param not in self.positions:
                raise pygloopException(f"Could not find parameter {param} in param builder.")
            min, max = self.positions[param]
            for i in range(min, max):
                if i in self.real_valued_inputs:
                    self.real_valued_inputs.remove(i)
                if i in self.purely_imaginary_valued_inputs:
                    self.purely_imaginary_valued_inputs.remove(i)
                if i not in self.forced_complex_valued_inputs:
                    self.forced_complex_valued_inputs.append(i)

    def add_parameter(self, param: tuple[Expression, ...]):
        return self.add_parameter_list(param, 1)

    def get_real_components(self) -> list[int]:
        return self.real_valued_inputs

    def get_components_phase(self) -> list[int | None]:
        # print("TOTAL PARAM =", len(self.np))
        # print("TOTAL PURE REAL =", len(self.real_valued_inputs))
        # print("TOTAL PURE IMAG =", len(self.purely_imaginary_valued_inputs))
        # print("TOTAL FULL COMPLEX =", len(self.np) - len(self.real_valued_inputs) - len(self.purely_imaginary_valued_inputs))
        # return [0 for _ in range(len(self.np))]

        phase_components: list[int | None] = []
        for i in range(len(self.np)):
            if i in self.real_valued_inputs:
                phase_components.append(0)
            elif i in self.purely_imaginary_valued_inputs:
                phase_components.append(1)
            else:
                phase_components.append(None)
        return phase_components

    def set_parameter_values(self, head: tuple[Expression, ...], values: list[complex] | NDArray[Any], check_phase_flag_consistency=True):
        if head not in self.positions:
            raise pygloopException(f"Could not find parameter {head} in param builder.")

        min, max = self.positions[head]
        if (max - min) != len(values):
            raise pygloopException(f"Length of parameters {head} declared as {max - min}, but {len(values)} values are provided.")
        if check_phase_flag_consistency:
            for i, v in enumerate(values):
                idx = min + i
                if idx in self.real_valued_inputs and v.imag != 0.0:
                    raise pygloopException(f"Cannot set parameter {head} at index {idx} to complex value {v}; it is marked as real-valued input.")
                if idx in self.purely_imaginary_valued_inputs and v.real != 0.0:
                    raise pygloopException(
                        f"Cannot set parameter {head} at index {idx} to non-imaginary value {v}; it is marked as purely-imaginary input."
                    )
        self.np[min:max] = values

    def set_parameter_values_within_range(self, min: int, max: int, values: list[complex] | NDArray[Any]):
        if (max - min) != len(values):
            raise pygloopException(f"Range declared of ({min},{max}) of different length that the number of values ({len(values)}) provided.")
        self.np[min:max] = values

    def set_parameter(self, param: tuple[Expression, ...], value: complex, check_phase_flag_consistency=True):
        return self.set_parameter_values(param, [value,], check_phase_flag_consistency)  # fmt: off

    def check_phase_flag_consistency(self):
        real_idx = numpy.asarray(self.real_valued_inputs, dtype=int)
        imag_parts = self.np[real_idx].imag
        bad = numpy.nonzero(imag_parts != 0.0)[0]
        if bad.size:
            idx = int(real_idx[bad[0]])
            raise pygloopException(f"Parameter at index {idx} is marked as real-valued input, but has non-zero imaginary part {self.np[idx].imag}.")
        imag_idx = numpy.asarray(self.purely_imaginary_valued_inputs, dtype=int)
        real_parts = self.np[imag_idx].real
        bad = numpy.nonzero(real_parts != 0.0)[0]
        if bad.size:
            idx = int(imag_idx[bad[0]])
            raise pygloopException(f"Parameter at index {idx} is marked as purely-imaginary input, but has non-zero real part {self.np[idx].real}.")

    def get_parameters(self):
        params = []
        for p in self.order:
            min, max = self.positions[p]
            if max - min == 1:
                if len(p[1:]) == 0:
                    params.append(p[0])
                else:
                    params.append(p[0](*p[1:]))
            else:
                params.extend(p[0](*p[1:], i) for i in range(max - min))

        return params

    def get_complex_values(self) -> NDArray[numpy.complex128]:
        return self.np

    def get_values(self, complexified_evaluator=False) -> NDArray[numpy.complex128] | NDArray[numpy.double]:
        if not complexified_evaluator:
            return self.np
        else:
            real_parts, imag_parts = self.np.real, self.np.imag
            zero_imag = numpy.float64(0.0)
            if self.real_valued_inputs:
                numpy.put(imag_parts, self.real_valued_inputs, zero_imag)

            values = numpy.zeros(2 * len(self.np), dtype=numpy.double)
            values[0::2] = real_parts
            values[1::2] = imag_parts

            return values


class PygloopEvaluator(object):
    @staticmethod
    def load(dir: str, name: str) -> "PygloopEvaluator":
        param_builder_path = os.path.join(dir, f"{name}_param_builder.json")
        lib_path = os.path.join(dir, f"{name}.so")
        symjit_path = os.path.join(dir, f"{name}.sjb")

        if not os.path.isfile(param_builder_path):
            raise pygloopException(f"Could not find parameter builder file '{param_builder_path}'.")
        if not os.path.isfile(lib_path):
            raise pygloopException(f"Could not find compiled evaluator library '{lib_path}'.")

        with open(param_builder_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        stored_name = data.get("name")
        parameters_data = data.get("parameters")
        values_data = data.get("values")
        real_valued_inputs = data.get("real_valued_inputs", [])
        purely_imaginary_valued_inputs = data.get("purely_imaginary_valued_inputs", [])
        forced_complex_valued_inputs = data.get("forced_complex_valued_inputs", [])
        output_length = data.get("output_length")
        if parameters_data is None or values_data is None:
            raise pygloopException(f"Malformed parameter builder file '{param_builder_path}'.")
        if stored_name is None or output_length is None:
            raise pygloopException(f"Malformed parameter builder file '{param_builder_path}'.")
        if stored_name != name:
            raise pygloopException(f"Loaded evaluator name '{stored_name}' does not match expected '{name}'.")

        param_builder = ParamBuilder()
        for param in parameters_data:
            head_data = param.get("head")
            range_data = param.get("range")
            if head_data is None or range_data is None or len(range_data) != 2:
                raise pygloopException(f"Malformed parameter entry in '{param_builder_path}'.")
            head = tuple(E(expr) for expr in head_data)
            min_idx, max_idx = int(range_data[0]), int(range_data[1])
            param_builder.positions[head] = (min_idx, max_idx)
            param_builder.order.append(head)

        values: list[complex] = []
        for value in values_data:
            if isinstance(value, (int, float)):
                values.append(complex(value))
            elif isinstance(value, list) and len(value) == 2:
                values.append(complex(value[0], value[1]))
            else:
                raise pygloopException(f"Malformed parameter value entry in '{param_builder_path}'.")

        param_builder.np = numpy.array(values, dtype=numpy.complex128)

        param_builder.real_valued_inputs = real_valued_inputs
        param_builder.purely_imaginary_valued_inputs = purely_imaginary_valued_inputs
        param_builder.forced_complex_valued_inputs = forced_complex_valued_inputs

        max_index = max((rng[1] for rng in param_builder.positions.values()), default=0)
        if max_index != len(param_builder.np):
            raise pygloopException(f"Parameter value array length ({len(param_builder.np)}) does not match declared ranges (max index {max_index}).")

        param_builder.check_phase_flag_consistency()

        additional_data_path = os.path.join(dir, f"{name}_additional_data.pkl")
        additional_data: dict[str, Any] = {}
        if os.path.isfile(additional_data_path):
            try:
                with open(additional_data_path, "rb") as handle:
                    loaded_additional = pickle.load(handle)
                if isinstance(loaded_additional, dict):
                    additional_data = loaded_additional
                else:
                    raise pygloopException(f"Additional data in '{additional_data_path}' is not a dictionary.")
            except Exception as e:
                raise pygloopException(f"Error loading additional data from '{additional_data_path}': {e}") from e

        loaded_evaluator = PygloopEvaluator(None, param_builder, stored_name, int(output_length), additional_data=additional_data)
        loaded_evaluator.complexified = data.get("complexified", False)
        n_inputs = len(loaded_evaluator.param_builder.np) * (2 if loaded_evaluator.complexified else 1)
        n_outputs = loaded_evaluator.output_length * (2 if loaded_evaluator.complexified else 1)
        evaluator_class = CompiledRealEvaluator if loaded_evaluator.complexified else CompiledComplexEvaluator
        try:
            loaded_evaluator.compiled_evaluator = evaluator_class.load(
                lib_path,
                stored_name,
                n_inputs,
                n_outputs,
            )
        except Exception as e:
            raise pygloopException(f"Error loading compiled evaluator from '{lib_path}': {e}") from e
        if os.path.exists(symjit_path):
            if not SYMJIT_AVAILABLE:
                raise pygloopException(f"symjit is not available but symjit evaluator file '{symjit_path}' exists.")
            try:
                symjit_evaluator = symjit.load_func(symjit_path)
                # // 2 because symjit counts real/imag pairs for complex inputs
                if symjit_evaluator.count_params // 2 != n_inputs:
                    raise pygloopException(f"Symjit evaluator input count mismatch: expected {n_inputs}, got {symjit_evaluator.count_params}.")
                if type(symjit_evaluator) is not symjit.SymbolicaFunc:
                    raise pygloopException("Loaded symjit evaluator is not of type SymbolicaFunc.")
                loaded_evaluator.symjit_evaluator = symjit_evaluator
            except Exception as e:
                raise pygloopException(f"Error loading symjit evaluator from '{symjit_path}': {e}") from e
        loaded_evaluator.eager_evaluator = None
        return loaded_evaluator

    def save(self, dir: str):
        lib_path = os.path.join(dir, f"{self.name}.so")
        if not os.path.isfile(lib_path):
            raise pygloopException(f"Compiled evaluator library '{lib_path}' not found. Please compile the evaluator before saving.")

        param_builder_path = os.path.join(dir, f"{self.name}_param_builder.json")

        parameters = []
        for head in self.param_builder.order:
            min_idx, max_idx = self.param_builder.positions[head]
            parameters.append(
                {
                    "head": [expr.to_canonical_string() for expr in head],
                    "range": [int(min_idx), int(max_idx)],
                }
            )

        values = [[float(complex(v).real), float(complex(v).imag)] for v in self.param_builder.np.tolist()]

        data = {
            "complexified": self.complexified,
            "name": self.name,
            "output_length": self.output_length,
            "parameters": parameters,
            "values": values,
            "real_valued_inputs": self.param_builder.real_valued_inputs,
            "purely_imaginary_valued_inputs": self.param_builder.purely_imaginary_valued_inputs,
            "forced_complex_valued_inputs": self.param_builder.forced_complex_valued_inputs,
        }

        with open(param_builder_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

        if self.additional_data:
            additional_data_path = os.path.join(dir, f"{self.name}_additional_data.pkl")
            with open(additional_data_path, "wb") as handle:
                pickle.dump(self.additional_data, handle)

    def __init__(
        self,
        evaluator: Evaluator | None,
        param_builder: ParamBuilder,
        name: str,
        output_length: int = 1,
        additional_data: dict[str, Any] | None = None,
        complexified: bool = False,
    ):
        self.eager_evaluator: Evaluator | None = evaluator
        self.compiled_evaluator: CompiledRealEvaluator | CompiledComplexEvaluator | None = None
        self.symjit_evaluator: symjit.SymbolicaFunc | None = None
        self.param_builder = param_builder
        self.name = name
        self.additional_data = additional_data or {}
        self.output_length = output_length
        self.complexified = complexified

    @staticmethod
    def _pairwise_to_complex(res: NDArray[Any], expected_output_len: int) -> NDArray[numpy.complex128]:
        """
        Convert an array shaped (batch, 2*n) or (2*n,) containing real/imag pairs
        into complex numbers shaped (batch, n) / (n,). If `res` is already complex,
        it is returned as-is (with dtype coerced to complex128).
        """
        res_arr = numpy.asarray(res)

        # Already complex -> just ensure dtype
        if numpy.iscomplexobj(res_arr):
            return res_arr.astype(numpy.complex128, copy=False)

        # Add batch dim when a single flat vector is returned
        if res_arr.ndim == 1:
            res_arr = res_arr.reshape(1, -1)

        if res_arr.shape[-1] % 2 != 0:
            raise pygloopException(f"Expected even number of entries (real/imag pairs), got {res_arr.shape[-1]}.")

        if expected_output_len > 0 and res_arr.shape[-1] != expected_output_len * 2:
            raise pygloopException(f"Output length mismatch: expected {expected_output_len * 2} real/imag entries, got {res_arr.shape[-1]}.")

        paired = res_arr.reshape(res_arr.shape[0], -1, 2)
        complex_res = paired[..., 0] + 1j * paired[..., 1]
        return complex_res

    def freeze_input_phases(self, verbose: bool = False):
        if self.eager_evaluator is None:
            raise pygloopException(f"Eager evaluator for '{self.name}' not available to set input phases for.")
        self.eager_evaluator.set_real_params(
            self.param_builder.get_real_components(),
            sqrt_real=True,
            log_real=True,
            powf_real=True,
            verbose=verbose,
        )

    def complexify(self):
        if self.eager_evaluator is None:
            raise pygloopException(f"Eager evaluator for '{self.name}' not available to complexify.")
        raise pygloopException("Complexification of evaluators is not supported anymore.")
        # self.eager_evaluator.complexify(
        #     real_components=self.param_builder.get_real_components(),
        # )

        self.complexified = True

    def get_eager_evaluator(self) -> Evaluator:
        if self.eager_evaluator is None:
            raise pygloopException(f"Eager evaluator for '{self.name}' not available.")
        return self.eager_evaluator

    def get_compiled_evaluator(self) -> CompiledComplexEvaluator | CompiledRealEvaluator:
        if self.compiled_evaluator is None:
            raise pygloopException(f"Compiled evaluator for '{self.name}' not available.")
        return self.compiled_evaluator

    def compile(self, out_dir: str, integrand_evaluator_compiler: str = "symbolica_only", **opts):
        if integrand_evaluator_compiler not in ["symbolica_only", "symjit"]:
            raise pygloopException(f"Unsupported integrand evaluator compiler '{integrand_evaluator_compiler}' for '{self.name}'.")

        eager_evaluator = self.get_eager_evaluator()
        if integrand_evaluator_compiler == "symjit":
            # For now only support symjit compilation for single-output expressions
            # Also we do not support complexified evaluators with symjit yet (or ever to be frank...)
            if self.output_length == 1 and not self.complexified:
                if not SYMJIT_AVAILABLE:
                    raise pygloopException(
                        f"Aymjit is not available to compile evaluator for '{self.name}'. Please install symjit or use 'symbolica_only' compiler."
                    )
                self.symjit_evaluator = symjit.compile_evaluator(eager_evaluator, dtype="complex128", use_threads=False, use_simd=True)
                logger.info(f"Compiling symjit evaluator for '{self.name}' to '{out_dir}'.")
                self.symjit_evaluator.save(os.path.join(out_dir, f"{self.name}.sjb"))

        pygloop_default_options = {
            "inline_asm": "default",
            "optimization_level": 3,
            "native": True,
        }
        pygloop_default_options.update(opts)

        logger.info(f"Compiling symbolica evaluator for '{self.name}' to '{out_dir}'.")
        self.compiled_evaluator = eager_evaluator.compile(
            self.name,
            os.path.join(out_dir, f"{self.name}.cpp"),
            os.path.join(out_dir, f"{self.name}.so"),
            "real" if self.complexified else "complex",
            **pygloop_default_options,
        )

    def evaluate(self, eager: bool | None = None, prefer_symjit=False, check_phase_flag_consistency=False) -> NDArray[numpy.complex128]:
        inputs = self.param_builder.get_values(self.complexified)[None, :]

        if check_phase_flag_consistency:
            self.param_builder.check_phase_flag_consistency()

        if eager is True:
            if self.complexified:
                res = self.get_eager_evaluator().evaluate(inputs)
            else:
                res = self.get_eager_evaluator().evaluate_complex(inputs)
            res = self._pairwise_to_complex(res, self.output_length)
            return res[0]
        elif eager is False:
            if prefer_symjit and self.symjit_evaluator is not None:
                res = self.symjit_evaluator.evaluate_complex(inputs)
            else:
                res = self.get_compiled_evaluator().evaluate(inputs)
            res = self._pairwise_to_complex(res, self.output_length)
            return res[0]
        else:
            if self.compiled_evaluator is not None or (prefer_symjit and self.symjit_evaluator is not None):
                if prefer_symjit and self.symjit_evaluator is not None:
                    res = self.symjit_evaluator.evaluate_complex(inputs)
                else:
                    res = self.compiled_evaluator.evaluate(inputs)  # type: ignore
                res = self._pairwise_to_complex(res, self.output_length)
                return res[0]
            elif self.eager_evaluator is not None:
                if self.complexified:
                    res = self.eager_evaluator.evaluate(inputs)
                else:
                    res = self.eager_evaluator.evaluate_complex(inputs)
                res = self._pairwise_to_complex(res, self.output_length)
                return res[0]
            else:
                raise pygloopException(f"No evaluator available for '{self.name}'.")


class SymbolicaSample(object):
    def __init__(self, sample: Sample):
        self.c: list[float] = sample.c
        self.d: list[int] = sample.d


class IntegrationResult(object):
    def __init__(
        self,
        central_value: float,
        error: float,
        n_samples: int = 0,
        elapsed_time: float = 0.0,
        max_wgt: float | None = None,
        max_wgt_point: list[float] | None = None,
    ):
        self.n_samples = n_samples
        self.central_value = central_value
        self.error = error
        self.max_wgt = max_wgt
        self.max_wgt_point = max_wgt_point
        self.elapsed_time = elapsed_time

    def combine_with(self, other):
        """Combine self statistics with all those of another IntegrationResult object."""
        self.n_samples += other.n_samples
        self.elapsed_time += other.elapsed_time
        self.central_value += other.central_value
        self.error += other.error
        if other.max_wgt is not None:
            if self.max_wgt is None or abs(self.max_wgt) > abs(other.max_wgt):
                self.max_wgt = other.max_wgt
                self.max_wgt_point = other.max_wgt_point

    def normalize(self):
        """Normalize the statistics."""
        self.central_value /= self.n_samples
        self.error = math.sqrt(abs(self.error / self.n_samples - self.central_value**2) / self.n_samples)

    def str_report(self, target: float | None = None) -> str:
        if self.central_value == 0.0 or self.n_samples == 0:
            return "No integration result available yet"

        # First printout sample and timing statitics
        report = [
            f"Integration result after {Colour.GREEN}{self.n_samples}{Colour.END} evaluations in {Colour.GREEN}{self.elapsed_time:.2f} CPU-s{Colour.END}"  # fmt: off
        ]
        if self.elapsed_time > 0.0:
            report[-1] += f" {Colour.BLUE}({1.0e6 * self.elapsed_time / self.n_samples:.1f} µs / eval){Colour.END}"

        # Also indicate max weight encountered if provided
        if self.max_wgt is not None and self.max_wgt_point is not None:
            report.append(f"Max weight encountered = {self.max_wgt:.5e} at xs = [{' '.join(f'{x:.16e}' for x in self.max_wgt_point)}]")  # fmt: off

        # Finally return information about current best estimate of the central value
        report.append(f"{Colour.GREEN}Central value{Colour.END} : {self.central_value:<+25.16e} +/- {self.error:<12.2e}")  # fmt: off

        err_perc = abs(self.error / self.central_value) * 100
        if err_perc < 1.0:
            report[-1] += f" ({Colour.GREEN}{err_perc:.3f}%{Colour.END})"
        else:
            report[-1] += f" ({Colour.RED}{err_perc:.3f}%{Colour.END})"

        # Also indicate distance to target if specified
        if target is not None and target != 0.0:
            report.append(f"    vs target : {target:<+25.16e} Δ = {self.central_value - target:<+12.2e}")  # fmt: off
            diff_perc = (self.central_value - target) / target * 100
            if abs(diff_perc) < 1.0:
                report[-1] += f" ({Colour.GREEN}{diff_perc:.3f}%{Colour.END}"
            else:
                report[-1] += f" ({Colour.RED}{diff_perc:.3f}%{Colour.END}"
            if abs(diff_perc / err_perc) < 3.0:
                report[-1] += f" {Colour.GREEN} = {abs(diff_perc / err_perc):.2f}σ{Colour.END})"  # fmt: off
            else:
                report[-1] += f" {Colour.RED} = {abs(diff_perc / err_perc):.2f}σ{Colour.END})"  # fmt: off

        # Join all lines and return
        return "\n".join(f"| > {line}" for line in report)


def write_text_with_dirs(path: str, content: str, mode: str = "w", encoding: str = "utf-8") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode, encoding=encoding) as handle:
        handle.write(content)


class DotGraph(object):
    def __init__(self, dot_graph: pydot.Dot):
        self.dot = dot_graph

    def get_attributes(self) -> dict:
        return self.dot.get_attributes()

    def get_numerator(self, include_overall_factor=False) -> Expression:
        num = E("1")
        for node in self.dot.get_nodes():
            if node.get_name() not in ["edge", "node"]:
                n_num = node.get("num")
                if n_num:
                    num *= Es(n_num)
        for edge in self.dot.get_edges():
            e_num = edge.get("num")
            if e_num:
                num *= Es(e_num)

        g_attrs = self.dot.get_attributes()
        if "num" in g_attrs:
            num *= Es(g_attrs["num"])
        if include_overall_factor and "overall_factor_evaluated" in g_attrs:
            num *= Es(g_attrs["overall_factor_evaluated"])

        return num

    def set_local_numerators_to_one(self):
        for node in self.dot.get_nodes():
            if node.get_name() not in ["edge", "node"]:
                node.set("num", "1")
        for edge in self.dot.get_edges():
            edge.set("num", "1")
        self.dot.set_edge_defaults(num='"1"')
        self.dot.set_node_defaults(num='"1"')

    def get_internal_edges(self) -> list[Edge]:
        internal_nodes = [
            n.get_name() for n in self.dot.get_nodes() if not any(marker in n.get_name() for marker in ["graph", "ext", "edge", "node"])
        ]
        external_edges = []
        for edge in self.dot.get_edges():
            source = edge.get_source().split(":")[0]  # type: ignore
            destination = edge.get_destination().split(":")[0]  # type: ignore
            if source in internal_nodes and destination in internal_nodes:
                external_edges.append(edge)

        return external_edges

    def get_external_edges(self) -> list[Edge]:
        internal_nodes = [
            n.get_name() for n in self.dot.get_nodes() if not any(marker in n.get_name() for marker in ["graph", "ext", "edge", "node"])
        ]
        external_edges = []
        for edge in self.dot.get_edges():
            source = edge.get_source().split(":")[0]  # type: ignore
            destination = edge.get_destination().split(":")[0]  # type: ignore
            if not (source in internal_nodes and destination in internal_nodes):
                external_edges.append(edge)

        return external_edges

    def get_propagator_denominators(self) -> Expression:
        den = E("1")
        for edge in self.get_internal_edges():
            attrs = edge.get_attributes()
            a_den = E(f"gammalooprs::Q({edge.get('id')},spenso::cind(0))^2")
            a_den -= E(f"gammalooprs::Q({edge.get('id')},spenso::cind(1))^2")
            a_den -= E(f"gammalooprs::Q({edge.get('id')},spenso::cind(2))^2")
            a_den -= E(f"gammalooprs::Q({edge.get('id')},spenso::cind(3))^2")
            if "mass" in attrs:
                a_den -= Es(f"{attrs.get('mass')}^2")
            den *= a_den
        return den

    def get_projector(self) -> Expression:
        g_attrs = self.dot.get_attributes()
        projector = None
        if "projector" in g_attrs:
            projector = Es(g_attrs["projector"])
        else:
            projector = Es(self.dot.get_graph_defaults()[0]["projector"])

        # TMPVH temporary fix to current issue in gammaloop when building external proectors
        # projector = projector.replace(E("gammalooprs::u(2,x__)"),E("gammalooprs::vbar(2,x__)"),repeat=True)

        return projector

    def get_emr_replacements(self, head="gammalooprs::Q") -> list[tuple[Expression, Expression]]:
        replacements = []
        for edge in self.dot.get_edges():
            replacements.append((E(f"{head}({edge.get('id')},gammalooprs::a___)"), Es(edge.get("lmb_rep"))))
        return replacements

    def to_string(self) -> str:
        return self.dot.to_string()


class DotGraphs(list):
    def __init__(self, dot_str: str | None = None, dot_path: str | None = None):
        if dot_str is None and dot_path is None:
            return
        if dot_path is not None and dot_str is not None:
            raise pygloopException("Only one of dot_str or dot_path should be provided.")

        if dot_path:
            dot_graphs = pydot.graph_from_dot_file(dot_path)
            if dot_graphs is None:
                raise ValueError(f"No graphs found in DOT file: {dot_path}")
            self.extend([DotGraph(g) for g in dot_graphs])
        elif dot_str:
            dot_graphs = pydot.graph_from_dot_data(dot_str)
            if dot_graphs is None:
                raise ValueError("No graphs found in DOT data string.")
            self.extend([DotGraph(g) for g in dot_graphs])

    def get_graph_names(self) -> list[str]:
        return [g.dot.get_name() for g in self]

    def select_graphs_by_names(self, graph_names: list[str]):
        self[:] = (g for g in self if g.dot.get_name() in graph_names)

    def __str__(self) -> str:
        return "\n\n".join([g.to_string() for g in self])

    def get_graph(self, graph_name) -> DotGraph:
        for g in self:
            if g.dot.get_name() == graph_name:
                return g
        raise KeyError(f"Graph with name {graph_name} not found.")

    def save_to_file(self, file_path: str):
        write_text_with_dirs(file_path, "\n\n".join([g.to_string() for g in self]))


def time_function(
    func: Callable[..., Any],
    *args: Any,
    repeats: int = 7,
    target_time: float = 0.2,  # desired seconds per repeat
    warmup_evals: int = 5,
    number: Optional[int] = None,  # override auto-chosen call count
    timer: Callable[[], float] = timeit.default_timer,
    **kwargs: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Benchmark `func(*args, **kwargs)` using timeit.repeat in-code.

    - If `number` is None, choose it from a timing estimate derived from the warmup evals,
      aiming for ~`target_time` seconds per repeat.
    - Returns: (result, stats_dict)
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if warmup_evals < 1:
        raise ValueError("warmup_evals must be >= 1")
    if target_time <= 0:
        raise ValueError("target_time must be > 0")

    def stmt() -> Any:
        return func(*args, **kwargs)

    # Warmup + estimate time per call (using warmup evals)
    # Measure loop overhead and subtract it to reduce bias.
    t0 = timer()
    last = None
    for _ in range(warmup_evals):
        last = stmt()
    t1 = timer()

    o0 = timer()
    for _ in range(warmup_evals):
        pass
    o1 = timer()

    warmup_total = t1 - t0
    loop_overhead = o1 - o0
    adjusted = warmup_total - loop_overhead
    if adjusted <= 0:
        # fallback if subtraction overshoots (very fast funcs)
        per_call_est = warmup_total / warmup_evals
    else:
        per_call_est = adjusted / warmup_evals

    if number is None:
        if per_call_est <= 0:
            n = 1
        else:
            n = int(target_time / per_call_est)
            if n < 1:
                n = 1
            # avoid pathological huge counts if estimate is extremely small
            if n > 10_000_000:
                n = 10_000_000
    else:
        n = int(number)
        if n < 1:
            raise ValueError("number must be >= 1")

    # Final measurements
    times = timeit.repeat(stmt, repeat=repeats, number=n, timer=timer)
    per_call = [t / n for t in times]

    stats = {
        "number": n,
        "repeats": repeats,
        "target_time_s": target_time,
        "warmup_evals": warmup_evals,
        "warmup_total_s": warmup_total,
        "loop_overhead_s": loop_overhead,
        "per_call_est_s": per_call_est,
        "min_s": min(per_call),
        "median_s": statistics.median(per_call),
        "mean_s": statistics.mean(per_call),
        "stdev_s": statistics.pstdev(per_call),
        "all_s": per_call,
    }
    return last, stats
