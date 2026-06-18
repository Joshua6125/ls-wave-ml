from dataclasses import replace
from src.loss_functions.pinn import PINNConfig

import pytest


pytestmark = pytest.mark.pinn


# AI-Generated
def test_config_weights_are_stored(pinn_config_default):
    config = replace(
        pinn_config_default,
        ic_weight=3.0,
        bc_weight=4.0,
    )

    assert config.ic_weight == 3.0
    assert config.bc_weight == 4.0
