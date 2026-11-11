"""Configuration shared by DY cards, CLI and the paired-Sobol integrator."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PairedSobolSettings:
    replicas: int = 16
    learning_points: tuple[int, ...] = (1024, 2048, 4096)
    production_points: int = 131072
    max_points: int = 1048576
    audit_block_size: int = 512
    target_relative_error: float = 0.01
    auxiliary_error_fraction: float = 0.1
    output_directory: str | None = None
    z_histogram_edges: tuple[float, ...] = ()

    def validate(self):
        if self.z_histogram_edges:
            from processes.dy.dy_histogram import validate_edges
            validate_edges(self.z_histogram_edges)
        if self.replicas < 2:
            raise ValueError("paired_sobol.replicas must be at least two")
        for name, points in (
            *[("learning_points", n) for n in self.learning_points],
            ("production_points", self.production_points),
            ("max_points", self.max_points),
            ("audit_block_size", self.audit_block_size),
        ):
            if points < 2 or points & (points - 1) or points > 2**29:
                raise ValueError(f"paired_sobol.{name} must be a power of two in [2, 2^29]")
        if not self.learning_points:
            raise ValueError("paired_sobol.learning_points cannot be empty")
        if self.max_points < self.production_points:
            raise ValueError("paired_sobol.max_points must be >= production_points")
        for name in ("target_relative_error", "auxiliary_error_fraction"):
            if not math.isfinite(getattr(self, name)) or not 0 < getattr(self, name) < 1:
                raise ValueError(f"paired_sobol.{name} must be between zero and one")
        return self

    @classmethod
    def from_arguments(cls, arguments):
        values = {name: getattr(arguments, "dy_paired_" + name, field.default)
                  for name, field in cls.__dataclass_fields__.items()}
        values["learning_points"] = tuple(values["learning_points"])
        values["z_histogram_edges"] = tuple(values["z_histogram_edges"])
        return cls(**values).validate()
