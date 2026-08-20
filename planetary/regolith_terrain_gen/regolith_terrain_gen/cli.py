# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""CLI entry point: ros2 run regolith_terrain_gen generate_terrain --seed 42 [--output-dir DIR]."""

import argparse
from pathlib import Path

from regolith_terrain_gen.config import TerrainConfig
from regolith_terrain_gen.generate import generate_world


def default_output_dir(seed: int) -> Path:
    return Path.home() / ".cache" / "regolith" / "worlds" / f"seed_{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a procedural lunar terrain world for Regolith."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = TerrainConfig(seed=args.seed)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.seed)

    world_sdf_path = generate_world(cfg, output_dir)
    print(str(world_sdf_path))


if __name__ == "__main__":
    main()
