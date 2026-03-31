"""
Retraining policy definitions for benchmark experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PolicyDecision:
    retrain: bool
    reason: str
    train_on_window: bool = False
    train_on_all_labeled: bool = False
    use_reference_mix: bool = True
    schedule_hit: bool = False


@dataclass
class PolicyState:
    current_batch_index: int
    drift_detected: bool
    drift_share: float
    batch_metrics: Dict[str, float]
    degraded: bool
    last_retrain_batch: int
    labeled_buffer_size: int
    consecutive_drift_batches: int
    recent_f1_scores: List[float] = field(default_factory=list)
    baseline_post_drift_f1: float = 0.0
    expected_recovery_gain: float = 0.0


class BasePolicy:
    name = "base"

    def decide(self, state: PolicyState) -> PolicyDecision:
        raise NotImplementedError


class NoRetrainPolicy(BasePolicy):
    name = "no_retrain"

    def decide(self, state: PolicyState) -> PolicyDecision:
        return PolicyDecision(retrain=False, reason="baseline_no_retraining")


class ScheduledRetrainPolicy(BasePolicy):
    name = "scheduled"

    def __init__(self, interval_batches: int = 2):
        self.interval_batches = interval_batches

    def decide(self, state: PolicyState) -> PolicyDecision:
        schedule_hit = (state.current_batch_index + 1) % self.interval_batches == 0
        return PolicyDecision(
            retrain=schedule_hit and state.labeled_buffer_size > 0,
            reason="scheduled_interval" if schedule_hit else "waiting_for_schedule",
            train_on_window=True,
            schedule_hit=schedule_hit,
        )


class ThresholdRetrainPolicy(BasePolicy):
    name = "threshold"

    def __init__(self, drift_threshold: float = 0.30, min_labeled_rows: int = 48):
        self.drift_threshold = drift_threshold
        self.min_labeled_rows = min_labeled_rows

    def decide(self, state: PolicyState) -> PolicyDecision:
        drift_trigger = state.drift_detected and state.drift_share >= self.drift_threshold
        performance_trigger = state.degraded
        should_retrain = (drift_trigger or performance_trigger) and state.labeled_buffer_size >= self.min_labeled_rows
        if drift_trigger:
            reason = "drift_threshold_exceeded"
        elif performance_trigger:
            reason = "performance_drop_detected"
        else:
            reason = "monitoring_only"
        return PolicyDecision(
            retrain=should_retrain,
            reason=reason,
            train_on_all_labeled=True,
        )


class AdaptiveWindowPolicy(BasePolicy):
    name = "adaptive_window"

    def __init__(
        self,
        drift_threshold: float = 0.24,
        min_labeled_rows: int = 96,
        consecutive_drift_trigger: int = 2,
    ):
        self.drift_threshold = drift_threshold
        self.min_labeled_rows = min_labeled_rows
        self.consecutive_drift_trigger = consecutive_drift_trigger

    def decide(self, state: PolicyState) -> PolicyDecision:
        sustained_drift = state.consecutive_drift_batches >= self.consecutive_drift_trigger
        quality_drop = state.degraded or (
            len(state.recent_f1_scores) >= 2 and state.recent_f1_scores[-1] < state.recent_f1_scores[-2] - 0.03
        )
        should_retrain = (
            state.labeled_buffer_size >= self.min_labeled_rows
            and (state.drift_share >= self.drift_threshold and sustained_drift or quality_drop)
        )
        if quality_drop:
            reason = "adaptive_quality_guard"
        elif sustained_drift:
            reason = "adaptive_sustained_drift"
        else:
            reason = "adaptive_hold"
        return PolicyDecision(
            retrain=should_retrain,
            reason=reason,
            train_on_window=True,
            use_reference_mix=False,
        )


class ChallengerPolicy(BasePolicy):
    name = "challenger"

    def __init__(
        self,
        drift_threshold: float = 0.20,
        min_labeled_rows: int = 84,
        expected_gain_threshold: float = 0.035,
        cooldown_batches: int = 2,
    ):
        self.drift_threshold = drift_threshold
        self.min_labeled_rows = min_labeled_rows
        self.expected_gain_threshold = expected_gain_threshold
        self.cooldown_batches = cooldown_batches

    def decide(self, state: PolicyState) -> PolicyDecision:
        in_cooldown = (
            state.last_retrain_batch >= 0
            and (state.current_batch_index - state.last_retrain_batch) < self.cooldown_batches
        )
        enough_signal = (
            state.drift_share >= self.drift_threshold
            or state.degraded
            or state.consecutive_drift_batches >= 2
        )
        enough_gain = state.expected_recovery_gain >= self.expected_gain_threshold
        should_retrain = (
            not in_cooldown
            and state.labeled_buffer_size >= self.min_labeled_rows
            and enough_signal
            and enough_gain
        )
        if in_cooldown:
            reason = "challenger_cooldown"
        elif not enough_signal:
            reason = "challenger_waiting_for_signal"
        elif not enough_gain:
            reason = "challenger_gain_not_worth_cost"
        else:
            reason = "challenger_promotion"
        return PolicyDecision(
            retrain=should_retrain,
            reason=reason,
            train_on_window=True,
            use_reference_mix=False,
        )


POLICY_REGISTRY = {
    NoRetrainPolicy.name: NoRetrainPolicy,
    ScheduledRetrainPolicy.name: ScheduledRetrainPolicy,
    ThresholdRetrainPolicy.name: ThresholdRetrainPolicy,
    AdaptiveWindowPolicy.name: AdaptiveWindowPolicy,
    ChallengerPolicy.name: ChallengerPolicy,
}


def build_policies(policy_specs: Optional[List[Dict[str, object]]] = None) -> List[BasePolicy]:
    """Build policies from config or use the standard policy comparison set."""
    if not policy_specs:
        return [
            NoRetrainPolicy(),
            ScheduledRetrainPolicy(interval_batches=2),
            ThresholdRetrainPolicy(),
            AdaptiveWindowPolicy(),
            ChallengerPolicy(),
        ]

    policies: List[BasePolicy] = []
    for spec in policy_specs:
        spec = dict(spec)
        policy_name = spec.pop("name")
        policy_class = POLICY_REGISTRY.get(policy_name)
        if policy_class is None:
            raise ValueError(f"Unknown policy name: {policy_name}")
        policies.append(policy_class(**spec))
    return policies
