# Author: Joshua van Rooij
# University: UvA
# Email: joshuavanrooij@gmail.com

from src.loss_functions.fosls import FOSLSConfig

import pytest


pytestmark = pytest.mark.fosls


# AI-Generated
def test_config_stores_weight():
    config = FOSLSConfig(
        ic_weight=7.0,
    )

    assert config.ic_weight == 7.0

