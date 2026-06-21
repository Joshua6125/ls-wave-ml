'''
Configs for integration objects.
'''
# Author: Joshua van Rooij
# University: UvA
# Email: joshuavanrooij@gmail.com

from dataclasses import dataclass
from typing import Literal, TypeAlias


@dataclass(frozen=True)
class IntegrationConfigBase:
    """
    Common configuration for integration over a space-time hyperrectangle

        [t_min, t_max] x [x_min, x_max]^d,

    where `d = spatial_dim`.
    """

    integration_method: str = "base"

    # Temporal domain bounds.
    t_min: float = 0.0
    t_max: float = 1.0

    # Number of spatial dimensions.
    spatial_dim: int = 1

    # Spatial domain bounds (applied identically to every spatial axis).
    x_min: float = 0.0
    x_max: float = 1.0

    def validate_domain(self) -> None:
        """Validate basic domain geometry."""
        if self.spatial_dim <= 0:
            raise ValueError("dim must be strictly positive")

        if self.x_min >= self.x_max:
            raise ValueError("x_min must be < x_max")

        if self.t_min >= self.t_max:
            raise ValueError("t_min must be < t_max")


@dataclass(frozen=True)
class QuadratureConfig(IntegrationConfigBase):
    """
    Configuration for tensor-product Gauss-Legendre quadrature.

    The 1D quadrature rule of degree `degree` is applied on each of
    `grid_size` subintervals per axis before forming a tensor-product
    rule over the full domain.
    """

    integration_method: Literal["quadrature"] = "quadrature"

    # Degree of the Gauss-Legendre rule on each segment.
    degree: int = 5

    # Number of segments used to partition each coordinate axis.
    grid_size: int = 1000

    # Placeholder for future adaptive refinement support.
    adaptive_integration: bool = False

    def validate(self) -> None:
        """Validate quadrature-specific parameters."""
        self.validate_domain()

        if self.degree <= 0:
            raise ValueError("degree must be strictly positive.")

        if self.grid_size <= 0:
            raise ValueError("grid_size must be strictly positive.")


@dataclass(frozen=True)
class MonteCarloConfig(IntegrationConfigBase):
    """
    Configuration for Monte Carlo integration.
    """

    integration_method: Literal["monte_carlo"] = "monte_carlo"

    # Number of samples drawn from the interior of the domain.
    interior_samples: int = 10_000

    # Number of samples drawn on each boundary face.
    boundary_samples: int = 1_000

    def validate(self) -> None:
        """Validate Monte Carlo sampling parameters."""
        self.validate_domain()

        if self.interior_samples <= 0:
            raise ValueError("interior_samples must be strictly positive.")

        if self.boundary_samples <= 0:
            raise ValueError("boundary_samples must be strictly positive.")


# Any supported integration configuration.
AnyIntegrationConfig: TypeAlias = QuadratureConfig | MonteCarloConfig
