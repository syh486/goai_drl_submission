"""Check SRU height samples against the native MuJoCo hfield surface."""

from __future__ import annotations

import mujoco
import numpy as np

from training.terrains import SruAtlasConfig, build_sru_mujoco_model, generate_sru_atlas
from training.terrains.constants import VERTICAL_SCALE


def main() -> None:
    atlas = generate_sru_atlas(
        SruAtlasConfig(
            seed=42,
            num_rows=1,
            num_cols=4,
            grass_fraction=0.0,
            gravel_fraction=0.0,
        )
    )
    model = build_sru_mujoco_model(atlas)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert floor_id >= 0 and model.geom_group[floor_id] == 5
    assert model.geom_contype[floor_id] == model.geom_conaffinity[floor_id] == 0
    terrain_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "main_body")
    assert terrain_body > 0

    group = np.asarray((1, 0, 0, 0, 0, 0), dtype=np.uint8)
    direction = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
    # Include corners, center, and asymmetric points to catch transpose errors.
    samples = ((0, 0), (1, 1), (50, 73), (149, 149), (250, 211), (300, 300))
    for tile in atlas.tiles:
        center = tile.origin[:2]
        for x_index, y_index in samples:
            world = np.asarray(
                (
                    center[0] + x_index * 0.1 - 15.0,
                    center[1] + y_index * 0.1 - 15.0,
                    10.0,
                ),
                dtype=np.float64,
            )
            geom_id = np.asarray((-1,), dtype=np.int32)
            distance = mujoco.mj_ray(
                model, data, world, direction, group, 1, -1, geom_id, None
            )
            assert distance >= 0.0
            actual = world[2] - distance
            expected = float(tile.height_full[x_index, y_index] * VERTICAL_SCALE)
            np.testing.assert_allclose(actual, expected, atol=2.0e-5)
            assert model.geom_bodyid[geom_id[0]] == terrain_body
    print("SRU_MUJOCO_HFIELD_CONTRACT_OK", model.nhfield, model.ngeom)


if __name__ == "__main__":
    main()
