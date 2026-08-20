# regolith_terrain_gen

Procedural lunar terrain generator. Deterministic from a `--seed`: composes a
fractal-Brownian-motion base, a power-law crater field (bowl + raised rim),
and a gentle regional slope into a 16-bit heightmap PNG, generates matching
PBR surface textures (albedo/normal/roughness) and a scatter of low-poly
procedural rock meshes, and assembles it all into a Gazebo world SDF plus a
`manifest.json` describing every crater and rock (consumed later by
`regolith_costmap`).

No external noise/mesh libraries are required - just numpy, scipy, and
Pillow. See `noise.py` for the value-noise fBm implementation and `rocks.py`
for the icosphere-based rock generator.

## Usage

```bash
ros2 run regolith_terrain_gen generate_terrain --seed 42 [--output-dir DIR]
```

Prints the path to the generated `world.sdf`. Defaults to
`~/.cache/regolith/worlds/seed_<N>/`. Normally invoked indirectly via
`ros2 launch regolith_bringup terrain_only.launch.py seed:=42`.

## Known issues

- A faint bright streak sometimes appears on the terrain in certain sun/camera
  configurations - a shadow-mapping precision artefact on the heightmap mesh,
  not a data or material bug (verified: crater profiles are genuinely
  depressions, not bumps - see `craters.py`). Cosmetic only.
