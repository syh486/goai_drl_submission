"""Named terrain profiles for the maintained SRU training protocols.

The profile is deliberately separate from the geometry generator.  A profile
controls which terrain families are sampled and how difficult their geometry
is, while ``sru_generator`` remains responsible for producing one tile.  The
legacy profile is retained only for deterministic regression tests; maintained
training configs must select one of the named stage profiles.
"""

from __future__ import annotations

from dataclasses import dataclass


TERRAIN_FAMILY_NAMES = ("maze", "non_maze", "stairs", "pits")


@dataclass(frozen=True)
class TerrainCurriculumProfile:
    """Terrain distribution and geometry controls for one training stage."""

    name: str
    proportions: tuple[float, float, float, float]
    difficulty_range: tuple[float, float]
    obstacle_density_scale: float = 1.0
    stair_obstacle_density_scale: float = 1.0
    stair_height_scale: float = 1.0
    include_pits: bool = False
    flat_only: bool = False
    pit_probability: float = 0.0

    def validate(self) -> None:
        if len(self.proportions) != len(TERRAIN_FAMILY_NAMES):
            raise ValueError(
                f"{self.name}: proportions must have {len(TERRAIN_FAMILY_NAMES)} values"
            )
        if any(value < 0.0 for value in self.proportions):
            raise ValueError(f"{self.name}: terrain proportions must be non-negative")
        if sum(self.proportions) <= 0.0:
            raise ValueError(f"{self.name}: terrain proportions must not be all zero")
        low, high = self.difficulty_range
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError(f"{self.name}: difficulty_range must be within [0, 1]")
        if self.obstacle_density_scale < 0.0:
            raise ValueError(f"{self.name}: obstacle_density_scale must be non-negative")
        if self.stair_obstacle_density_scale < 0.0:
            raise ValueError(
                f"{self.name}: stair_obstacle_density_scale must be non-negative"
            )
        if not 0.0 < self.stair_height_scale:
            raise ValueError(f"{self.name}: stair_height_scale must be positive")
        if not 0.0 <= self.pit_probability <= 1.0:
            raise ValueError(f"{self.name}: pit_probability must be within [0, 1]")
        if not self.include_pits and self.proportions[3] != 0.0:
            raise ValueError(f"{self.name}: pits proportion must be zero when pits are disabled")
        if not self.include_pits and self.pit_probability != 0.0:
            raise ValueError(f"{self.name}: pit_probability must be zero when pits are disabled")
        if self.flat_only and self.proportions[:3] != (0.0, 1.0, 0.0):
            raise ValueError(
                f"{self.name}: flat_only profiles use the non_maze family as a flat tile"
            )


# Kept for generator and fixed-seed regression tests.  It is not used by the
# Maintained training YAMLs omit pits because they are outside the target terrain set.
LEGACY_FULL = TerrainCurriculumProfile(
    name="legacy_full",
    proportions=(0.3, 0.2, 0.3, 0.2),
    difficulty_range=(0.5, 1.0),
    include_pits=True,
    pit_probability=0.15,
)


STAGE1_FLAT = TerrainCurriculumProfile(
    name="stage1_flat",
    # The non-maze family is used as the stable flat-tile bucket.
    proportions=(0.0, 1.0, 0.0, 0.0),
    difficulty_range=(0.0, 0.0),
    obstacle_density_scale=0.0,
    include_pits=False,
    flat_only=True,
)


STAGE2_LOW_DENSITY_OBSTACLES = TerrainCurriculumProfile(
    name="stage2_low_density_obstacles",
    proportions=(0.5, 0.5, 0.0, 0.0),
    difficulty_range=(0.25, 0.65),
    obstacle_density_scale=0.5,
    include_pits=False,
)


STAGE3_REDUCED_HEIGHT = TerrainCurriculumProfile(
    name="stage3_reduced_height",
    proportions=(0.3, 0.2, 0.3, 0.0),
    difficulty_range=(0.5, 1.0),
    obstacle_density_scale=1.0,
    stair_height_scale=0.6,
    include_pits=False,
)


STAGE4_FULL_NO_PITS = TerrainCurriculumProfile(
    name="stage4_full_no_pits",
    proportions=(0.3, 0.2, 0.3, 0.0),
    difficulty_range=(0.5, 1.0),
    obstacle_density_scale=1.0,
    stair_height_scale=1.0,
    include_pits=False,
)


STAGE5_LOWER_DENSITY_STAIRS = TerrainCurriculumProfile(
    name="stage5_lower_density_stairs",
    # Normalized stairs share becomes 50% (Stage 4 is 37.5%).
    proportions=(0.25, 0.15, 0.40, 0.0),
    difficulty_range=(0.5, 1.0),
    obstacle_density_scale=1.0,
    stair_obstacle_density_scale=0.6,
    stair_height_scale=0.75,
    include_pits=False,
)


PROFILES = {
    profile.name: profile
    for profile in (
        LEGACY_FULL,
        STAGE1_FLAT,
        STAGE2_LOW_DENSITY_OBSTACLES,
        STAGE3_REDUCED_HEIGHT,
        STAGE4_FULL_NO_PITS,
        STAGE5_LOWER_DENSITY_STAIRS,
    )
}

for _profile in PROFILES.values():
    _profile.validate()


def get_terrain_profile(name: str) -> TerrainCurriculumProfile:
    """Return a validated named profile."""

    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown terrain profile {name!r}; choose one of {sorted(PROFILES)}"
        ) from exc


__all__ = [
    "LEGACY_FULL",
    "PROFILES",
    "STAGE1_FLAT",
    "STAGE2_LOW_DENSITY_OBSTACLES",
    "STAGE3_REDUCED_HEIGHT",
    "STAGE4_FULL_NO_PITS",
    "STAGE5_LOWER_DENSITY_STAIRS",
    "TERRAIN_FAMILY_NAMES",
    "TerrainCurriculumProfile",
    "get_terrain_profile",
]
