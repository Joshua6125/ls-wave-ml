# Author: Joshua van Rooij
# University: UvA
# Email: joshuavanrooij@gmail.com

from dataclasses import replace

import pytest


pytestmark = pytest.mark.gpinn


# AI-Generated
def test_config_weights_are_stored(gpinn_config_default):
    config = replace(
        gpinn_config_default,
        ic_weight=3.0,
        bc_weight=4.0,
        residual_grad_weight=5.0,
    )

    assert config.ic_weight == 3.0
    assert config.bc_weight == 4.0
    assert config.residual_grad_weight == 5.0
