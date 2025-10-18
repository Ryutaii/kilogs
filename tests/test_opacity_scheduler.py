from __future__ import annotations

import math
import unittest

from distill.lego_response_distill import LossConfig, OpacityTargetScheduler


def _make_scheduler(
    *,
    start_weight: float,
    target_weight: float,
    schedule: str = "linear",
    duration: int = 200,
    hysteresis: bool = True,
) -> OpacityTargetScheduler:
    cfg = LossConfig(
        color_weight=1.0,
        opacity_weight=1.0,
        opacity_target_weight=target_weight,
        opacity_target_start_weight=start_weight,
        opacity_target_schedule=schedule,
        opacity_target_schedule_duration=duration,
        opacity_target_warmup_steps=0,
        opacity_target_hysteresis=hysteresis,
    )
    return OpacityTargetScheduler(cfg)


class OpacitySchedulerTests(unittest.TestCase):
    def test_monotonic_increasing(self) -> None:
        scheduler = _make_scheduler(start_weight=0.05, target_weight=0.4)
        weights = [scheduler.compute(step) for step in range(0, 401, 10)]
        for idx in range(len(weights) - 1):
            self.assertLessEqual(weights[idx], weights[idx + 1] + 1e-6)

    def test_monotonic_decreasing(self) -> None:
        scheduler = _make_scheduler(start_weight=0.45, target_weight=0.1)
        weights = [scheduler.compute(step) for step in range(0, 401, 10)]
        for idx in range(len(weights) - 1):
            self.assertGreaterEqual(weights[idx], weights[idx + 1] - 1e-6)

    def test_hysteresis_tracks_maximum(self) -> None:
        scheduler = _make_scheduler(start_weight=0.05, target_weight=0.5, duration=100)
        max_history: list[float] = []
        for step in range(0, 201, 5):
            weight = scheduler.compute(step)
            self.assertGreaterEqual(weight, 0.0)
            if scheduler.max_weight is not None:
                max_history.append(scheduler.max_weight)
        self.assertTrue(max_history)
        self.assertEqual(max_history, sorted(max_history))
        self.assertTrue(math.isclose(max_history[-1], scheduler.last_weight, rel_tol=1e-6))

    def test_state_roundtrip_preserves_progress(self) -> None:
        scheduler = _make_scheduler(start_weight=0.05, target_weight=0.4)
        for step in range(0, 150, 5):
            scheduler.compute(step)
        snapshot = scheduler.state_dict()

        restored = _make_scheduler(start_weight=0.05, target_weight=0.4)
        restored.load_state_dict(snapshot)
        self.assertTrue(
            math.isclose(restored.last_weight, scheduler.last_weight, rel_tol=1e-9)
        )
        self.assertTrue(
            math.isclose(restored.max_weight or 0.0, scheduler.max_weight or 0.0, rel_tol=1e-9)
        )
        restored_values = [restored.compute(step) for step in range(150, 251, 5)]
        for idx in range(len(restored_values) - 1):
            self.assertLessEqual(restored_values[idx], restored_values[idx + 1] + 1e-6)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
