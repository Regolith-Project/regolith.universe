# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""generate_world_shared may only skip generation for exactly the world the
generator would produce now; anything else must regenerate."""

import dataclasses

from regolith_terrain_gen import generate
from regolith_terrain_gen.config import TerrainConfig


def _fake_generate(calls):
    def fake(cfg, output_dir, start_paused=True):
        calls.append(cfg)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "world.sdf").write_text("<sdf/>")
        return output_dir / "world.sdf"
    return fake


def test_second_identical_call_is_skipped(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(generate, "generate_world", _fake_generate(calls))
    cfg = TerrainConfig(seed=11)
    lock = tmp_path / "lock"
    generate.generate_world_shared(cfg, tmp_path / "w", lock_path=lock)
    generate.generate_world_shared(cfg, tmp_path / "w", lock_path=lock)
    assert len(calls) == 1


def test_any_config_difference_regenerates(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(generate, "generate_world", _fake_generate(calls))
    lock = tmp_path / "lock"
    cfg = TerrainConfig(seed=11)
    generate.generate_world_shared(cfg, tmp_path / "w", lock_path=lock)
    generate.generate_world_shared(dataclasses.replace(cfg, rock_count=99), tmp_path / "w", lock_path=lock)
    generate.generate_world_shared(cfg, tmp_path / "w", start_paused=True, lock_path=lock)
    assert len(calls) == 3


def test_generator_source_change_regenerates(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(generate, "generate_world", _fake_generate(calls))
    lock = tmp_path / "lock"
    cfg = TerrainConfig(seed=11)
    generate.generate_world_shared(cfg, tmp_path / "w", lock_path=lock)
    monkeypatch.setattr(generate, "_generator_source_hash", lambda: "changed")
    generate.generate_world_shared(cfg, tmp_path / "w", lock_path=lock)
    assert len(calls) == 2


def test_interrupted_generation_is_not_trusted(tmp_path, monkeypatch):
    # A stamp is written only after generate_world returns, so a crash mid-way
    # leaves no stamp and the next call regenerates.
    calls = []

    def crashing(cfg, output_dir, start_paused=True):
        calls.append(cfg)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "world.sdf").write_text("<partial")
        raise RuntimeError("killed mid-generation")

    monkeypatch.setattr(generate, "generate_world", crashing)
    cfg = TerrainConfig(seed=11)
    try:
        generate.generate_world_shared(cfg, tmp_path / "w", lock_path=tmp_path / "lock")
    except RuntimeError:
        pass
    monkeypatch.setattr(generate, "generate_world", _fake_generate(calls))
    generate.generate_world_shared(cfg, tmp_path / "w", lock_path=tmp_path / "lock")
    assert len(calls) == 2
