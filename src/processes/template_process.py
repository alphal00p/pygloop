from __future__ import annotations

import logging
from typing import Any

from utils.utils import Colour, IntegrationResult, logger, pygloopException
from utils.vectors import LorentzVector, Vector


class TemplateProcess(object):
    """Placeholder process mirroring the GGHHH public API used by pygloop.py."""

    name = "template_process"

    def __init__(
        self,
        m_top: float,
        m_higgs: float,
        ps_point: list[LorentzVector],
        helicities: list[int] | None = None,
        n_loops: int = 1,
        toml_config_path: str | None = None,
        runtime_toml_config_path: str | None = None,
        clean: bool = True,
        logger_level: int | None = None,
        **opts,
    ):
        self.m_top = m_top
        self.m_higgs = m_higgs
        self.ps_point = ps_point
        self.helicities = helicities or []
        self.n_loops = n_loops
        self.toml_config_path = toml_config_path
        self.runtime_toml_config_path = runtime_toml_config_path
        self.clean = clean
        self.set_log_level(logger_level)
        logger.info(f"{Colour.YELLOW}TemplateProcess placeholder initialised; no physics implemented yet.{Colour.END}")

    def __deepcopy__(self, _memo: dict[int, Any]) -> "TemplateProcess":
        return TemplateProcess(
            self.m_top,
            self.m_higgs,
            self.ps_point,
            list(self.helicities),
            self.n_loops,
            self.toml_config_path,
            self.runtime_toml_config_path,
            clean=self.clean,
            logger_level=logging.CRITICAL,
        )

    def set_log_level(self, level: int | None) -> None:
        if level is not None:
            logger.setLevel(level)

    def builder_inputs(self) -> tuple:
        return (
            self.m_top,
            self.m_higgs,
            self.ps_point,
            self.helicities,
            self.n_loops,
            self.toml_config_path,
            self.runtime_toml_config_path,
        )

    def _placeholder(self, feature: str) -> None:
        logger.warning(f"{Colour.YELLOW}TemplateProcess placeholder invoked for '{feature}'. No implementation is available yet.{Colour.END}")
        raise pygloopException(f"TemplateProcess placeholder does not implement '{feature}' yet.")

    def generate_graphs(self, *args, **opts) -> None:  # type: ignore
        self._placeholder("generate_graphs")

    def generate_gammaloop_code(self, *args, **opts) -> None:  # type: ignore
        self._placeholder("generate_gammaloop_code")

    def generate_spenso_code(self, *args, **opts) -> None:  # type: ignore
        self._placeholder("generate_spenso_code")

    def parameterize(self, xs: list[float], parameterisation: str, origin: Vector | None = None) -> tuple[Vector, float]:  # type: ignore
        self._placeholder("parameterize")

    def integrand_xspace(
        self,
        xs: list[float],
        parameterization: str,
        integrand_implementation: dict[str, Any],
        phase: str | bool | None = None,
        multi_channeling: bool | int = True,
    ) -> float:  # type: ignore
        self._placeholder("integrand_xspace")

    def integrand(self, loop_momenta: list[Vector], integrand_implementation: dict[str, Any]) -> complex:  # type: ignore
        self._placeholder("integrand")

    def integrate(
        self,
        integrator: str,
        parameterisation: str,
        integrand_implementation: dict[str, Any],
        target: float | complex | None = None,
        toml_config_path: str | None = None,
        **opts: Any,
    ) -> IntegrationResult:  # type: ignore
        self._placeholder("integrate")

    def plot(self, **opts: Any) -> None:  # type: ignore
        self._placeholder("plot")
