"""Regression checks for ordered-route forward progression costs."""

from __future__ import annotations

import numpy as np

from deployment.localization.continuous_map_localization import (
    ContinuousLocalizationConfig,
    LocalizationHypothesis,
    _hypothesis_selection_key,
    _multisession_route_cost,
    _multisession_support_is_reliable,
    _registration_rejection_reasons,
    _route_prior_penalties,
)


def main() -> None:
    config = ContinuousLocalizationConfig()
    stale_step, stale_lag = _route_prior_penalties(config, 160, 160, 161)
    next_step, next_lag = _route_prior_penalties(config, 160, 161, 161)
    assert np.isclose(stale_step, 0.0)
    assert np.isclose(stale_lag, 0.05)
    assert np.isclose(next_step, 0.015)
    assert np.isclose(next_lag, 0.0)
    assert stale_step + stale_lag > next_step + next_lag

    # A wildly advanced odometry estimate is capped by the normal tracking
    # route-step limit instead of forcing an arbitrary long jump.
    _, capped_lag = _route_prior_penalties(config, 20, 20, 80)
    assert np.isclose(capped_lag, 2 * config.route_lag_cost)

    stale_prediction = _registration_rejection_reasons(
        config,
        fitness=0.9,
        rmse_m=0.1,
        descriptor_distance=0.2,
        translation_innovation_m=20.0,
        yaw_innovation_deg=35.0,
        continuity_candidate=False,
        recovering_without_metric_prior=False,
    )
    assert stale_prediction == ["translation_innovation", "yaw_innovation"]
    recovery = _registration_rejection_reasons(
        config,
        fitness=0.9,
        rmse_m=0.1,
        descriptor_distance=0.2,
        translation_innovation_m=20.0,
        yaw_innovation_deg=35.0,
        continuity_candidate=False,
        recovering_without_metric_prior=True,
    )
    assert recovery == []
    bad_geometry = _registration_rejection_reasons(
        config,
        fitness=0.1,
        rmse_m=0.8,
        descriptor_distance=1.2,
        translation_innovation_m=20.0,
        yaw_innovation_deg=35.0,
        continuity_candidate=False,
        recovering_without_metric_prior=True,
    )
    assert bad_geometry == ["fitness", "rmse", "descriptor"]

    canonical = {
        "route_index": 12,
        "observation": np.eye(4),
        "fitness": 0.90,
        "rmse_m": 0.15,
        "descriptor_distance": 0.50,
        "translation_innovation_m": 0.12,
        "yaw_innovation_deg": 0.5,
    }
    support_pose = np.eye(4)
    support_pose[0, 3] = 0.08
    reliable_support = {
        **canonical,
        "observation": support_pose,
        "rmse_m": 0.14,
        "translation_innovation_m": 0.10,
    }
    assert _multisession_support_is_reliable(
        config, canonical, reliable_support
    )
    unreliable_support = {
        **reliable_support,
        "translation_innovation_m": 0.20,
    }
    assert not _multisession_support_is_reliable(
        config, canonical, unreliable_support
    )
    repeated_cost = _multisession_route_cost(
        config,
        canonical,
        hypothesis_route_index=11,
        odometry_route_index=12,
        reliable_support_count=1,
    )
    single_cost = _multisession_route_cost(
        config,
        canonical,
        hypothesis_route_index=11,
        odometry_route_index=12,
        reliable_support_count=0,
    )
    assert np.isclose(
        single_cost - repeated_cost,
        config.multisession_repeatability_bonus,
    )

    coast = LocalizationHypothesis(
        cost=0.09,
        map_from_odom=np.eye(4),
        coast_updates=1,
        source="coast",
        route_index=0,
    )
    tracking = LocalizationHypothesis(
        cost=0.31,
        map_from_odom=np.eye(4),
        coast_updates=0,
        source="submap:1",
        route_index=1,
        trusted_tracking_candidate=True,
    )
    assert min((coast, tracking), key=_hypothesis_selection_key) is tracking

    print("CONTINUOUS_MAP_PROGRESSION_OK")


if __name__ == "__main__":
    main()
