# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""apply_overrides is how campaigns sample the terrain envelope; a silent no-op
would record one envelope point while running another."""

import pytest

from regolith_terrain_gen.config import TerrainConfig, apply_overrides


def test_empty_spec_is_identity():
    cfg = TerrainConfig(seed=5)
    assert apply_overrides(cfg, "") == cfg
    assert apply_overrides(cfg, None) == cfg


def test_values_are_coerced_to_field_types():
    cfg = apply_overrides(TerrainConfig(seed=5),
                          "crater_count=150, rock_count=200,regional_slope_deg=2.5")
    assert cfg.crater_count == 150 and isinstance(cfg.crater_count, int)
    assert cfg.rock_count == 200
    assert cfg.regional_slope_deg == 2.5 and isinstance(cfg.regional_slope_deg, float)
    assert cfg.seed == 5


def test_override_reaches_the_generator():
    # Asserted on what the generator produces, not on the dataclass: the failure
    # this guards against is an override that sets a field nothing reads.
    import numpy as np
    from regolith_terrain_gen.craters import place_craters

    base = TerrainConfig(seed=5)
    few = apply_overrides(base, "crater_count=40")
    assert len(place_craters(base, np.random.default_rng(5))) != len(
        place_craters(few, np.random.default_rng(5)))


@pytest.mark.parametrize("spec", ["crater_cout=150", "seed=3", "rock_count", "rock_count=",
                                  "sky_enabled=maybe", "spawn_zone_center=1"])
def test_bad_specs_raise(spec):
    with pytest.raises(ValueError):
        apply_overrides(TerrainConfig(seed=5), spec)
