# Drift Response Benchmark

## Overall Leaderboard

| Policy | Avg Rank | Avg Composite | Avg Final F1 | Avg Post-Drift F1 | Avg Train Secs | Avg Promoted Retrains | Avg Inference ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| challenger | 2.29 | 0.4828 | 0.7522 | 0.7126 | 0.3582 | 2.71 | 16.3630 |
| no_retrain | 2.43 | 0.4584 | 0.9508 | 0.5994 | 0.0000 | 0.00 | 16.6053 |
| scheduled | 3.00 | 0.4493 | 0.8985 | 0.6493 | 0.2005 | 2.14 | 16.3419 |
| threshold | 3.57 | 0.4245 | 0.9226 | 0.7136 | 0.6043 | 5.43 | 16.3199 |
| adaptive_window | 3.71 | 0.4495 | 0.7825 | 0.7162 | 0.4834 | 4.29 | 16.3730 |

## Scenario Winners

- `adaptive_recovery_window`: `scheduled` won with composite `0.3813`, final F1 `0.6952`, and post-drift F1 `0.5589`
- `concept_drift`: `no_retrain` won with composite `0.4045`, final F1 `0.9650`, and post-drift F1 `0.4966`
- `hybrid_regime_shift`: `challenger` won with composite `0.4391`, final F1 `0.8188`, and post-drift F1 `0.6026`
- `label_shift`: `no_retrain` won with composite `0.6714`, final F1 `0.9505`, and post-drift F1 `0.9787`
- `late_breaking_regime_shift`: `no_retrain` won with composite `0.4455`, final F1 `0.9648`, and post-drift F1 `0.6177`
- `mild_covariate`: `threshold` won with composite `0.6660`, final F1 `0.9718`, and post-drift F1 `0.9511`
- `severe_covariate`: `challenger` won with composite `0.5944`, final F1 `0.8670`, and post-drift F1 `0.8066`

## Detailed Results

- `mild_covariate` / `threshold`: rank 1, avg batch F1 `0.9610`, final F1 `0.9718`, post-drift F1 `0.9511`, stability `0.9781`, retrains `8`, train secs `0.5064`
- `mild_covariate` / `scheduled`: rank 2, avg batch F1 `0.9433`, final F1 `0.9718`, post-drift F1 `0.9444`, stability `0.9717`, retrains `2`, train secs `0.1383`
- `mild_covariate` / `no_retrain`: rank 3, avg batch F1 `0.8843`, final F1 `0.9369`, post-drift F1 `0.8762`, stability `0.9307`, retrains `0`, train secs `0.0000`
- `mild_covariate` / `challenger`: rank 4, avg batch F1 `0.9305`, final F1 `0.9208`, post-drift F1 `0.9433`, stability `0.9663`, retrains `5`, train secs `0.2767`
- `mild_covariate` / `adaptive_window`: rank 5, avg batch F1 `0.9305`, final F1 `0.9208`, post-drift F1 `0.9433`, stability `0.9663`, retrains `7`, train secs `0.4009`
- `severe_covariate` / `challenger`: rank 1, avg batch F1 `0.8053`, final F1 `0.8670`, post-drift F1 `0.8066`, stability `0.7750`, retrains `6`, train secs `0.3535`
- `severe_covariate` / `adaptive_window`: rank 2, avg batch F1 `0.8074`, final F1 `0.8670`, post-drift F1 `0.8066`, stability `0.7739`, retrains `8`, train secs `0.4742`
- `severe_covariate` / `threshold`: rank 3, avg batch F1 `0.7309`, final F1 `0.9860`, post-drift F1 `0.7561`, stability `0.7474`, retrains `9`, train secs `0.5909`
- `severe_covariate` / `no_retrain`: rank 4, avg batch F1 `0.6820`, final F1 `0.9510`, post-drift F1 `0.3853`, stability `0.7465`, retrains `0`, train secs `0.0000`
- `severe_covariate` / `scheduled`: rank 5, avg batch F1 `0.6820`, final F1 `0.9510`, post-drift F1 `0.3853`, stability `0.7465`, retrains `3`, train secs `0.1824`
- `label_shift` / `no_retrain`: rank 1, avg batch F1 `0.9377`, final F1 `0.9505`, post-drift F1 `0.9787`, stability `0.9596`, retrains `0`, train secs `0.0000`
- `label_shift` / `challenger`: rank 2, avg batch F1 `0.9518`, final F1 `0.9646`, post-drift F1 `0.9856`, stability `0.9586`, retrains `6`, train secs `0.3589`
- `label_shift` / `adaptive_window`: rank 3, avg batch F1 `0.9518`, final F1 `0.9646`, post-drift F1 `0.9856`, stability `0.9586`, retrains `7`, train secs `0.4159`
- `label_shift` / `threshold`: rank 4, avg batch F1 `0.9611`, final F1 `0.9577`, post-drift F1 `0.9787`, stability `0.9688`, retrains `8`, train secs `0.5241`
- `label_shift` / `scheduled`: rank 5, avg batch F1 `0.9352`, final F1 `0.9505`, post-drift F1 `0.9787`, stability `0.9285`, retrains `3`, train secs `0.1808`
- `concept_drift` / `no_retrain`: rank 1, avg batch F1 `0.6706`, final F1 `0.9650`, post-drift F1 `0.4966`, stability `0.8192`, retrains `0`, train secs `0.0000`
- `concept_drift` / `challenger`: rank 2, avg batch F1 `0.6802`, final F1 `0.6347`, post-drift F1 `0.5467`, stability `0.8283`, retrains `6`, train secs `0.3691`
- `concept_drift` / `adaptive_window`: rank 3, avg batch F1 `0.6802`, final F1 `0.6347`, post-drift F1 `0.5467`, stability `0.8283`, retrains `7`, train secs `0.4297`
- `concept_drift` / `scheduled`: rank 4, avg batch F1 `0.6746`, final F1 `0.8328`, post-drift F1 `0.5065`, stability `0.8253`, retrains `3`, train secs `0.1960`
- `concept_drift` / `threshold`: rank 5, avg batch F1 `0.6644`, final F1 `0.8547`, post-drift F1 `0.5310`, stability `0.8198`, retrains `8`, train secs `0.5140`
- `hybrid_regime_shift` / `challenger`: rank 1, avg batch F1 `0.5925`, final F1 `0.8188`, post-drift F1 `0.6026`, stability `0.7966`, retrains `5`, train secs `0.3150`
- `hybrid_regime_shift` / `scheduled`: rank 2, avg batch F1 `0.6163`, final F1 `0.9510`, post-drift F1 `0.5616`, stability `0.8200`, retrains `3`, train secs `0.1808`
- `hybrid_regime_shift` / `adaptive_window`: rank 3, avg batch F1 `0.5696`, final F1 `0.8201`, post-drift F1 `0.5480`, stability `0.7858`, retrains `7`, train secs `0.4352`
- `hybrid_regime_shift` / `threshold`: rank 4, avg batch F1 `0.6312`, final F1 `0.9930`, post-drift F1 `0.5413`, stability `0.8189`, retrains `8`, train secs `0.5366`
- `hybrid_regime_shift` / `no_retrain`: rank 5, avg batch F1 `0.5825`, final F1 `0.9582`, post-drift F1 `0.3476`, stability `0.7924`, retrains `0`, train secs `0.0000`
- `adaptive_recovery_window` / `scheduled`: rank 1, avg batch F1 `0.5958`, final F1 `0.6952`, post-drift F1 `0.5589`, stability `0.8276`, retrains `4`, train secs `0.2475`
- `adaptive_recovery_window` / `no_retrain`: rank 2, avg batch F1 `0.5805`, final F1 `0.9295`, post-drift F1 `0.4939`, stability `0.8241`, retrains `0`, train secs `0.0000`
- `adaptive_recovery_window` / `challenger`: rank 3, avg batch F1 `0.6483`, final F1 `0.5318`, post-drift F1 `0.5756`, stability `0.8576`, retrains `6`, train secs `0.3913`
- `adaptive_recovery_window` / `threshold`: rank 4, avg batch F1 `0.6329`, final F1 `0.7229`, post-drift F1 `0.6149`, stability `0.8523`, retrains `11`, train secs `0.7809`
- `adaptive_recovery_window` / `adaptive_window`: rank 5, avg batch F1 `0.6543`, final F1 `0.5318`, post-drift F1 `0.5756`, stability `0.8591`, retrains `10`, train secs `0.6079`
- `late_breaking_regime_shift` / `no_retrain`: rank 1, avg batch F1 `0.6618`, final F1 `0.9648`, post-drift F1 `0.6177`, stability `0.7776`, retrains `0`, train secs `0.0000`
- `late_breaking_regime_shift` / `scheduled`: rank 2, avg batch F1 `0.6751`, final F1 `0.9369`, post-drift F1 `0.6095`, stability `0.7929`, retrains `4`, train secs `0.2780`
- `late_breaking_regime_shift` / `challenger`: rank 3, avg batch F1 `0.6831`, final F1 `0.5276`, post-drift F1 `0.5278`, stability `0.8023`, retrains `7`, train secs `0.4428`
- `late_breaking_regime_shift` / `threshold`: rank 4, avg batch F1 `0.6897`, final F1 `0.9719`, post-drift F1 `0.6222`, stability `0.8067`, retrains `11`, train secs `0.7775`
- `late_breaking_regime_shift` / `adaptive_window`: rank 5, avg batch F1 `0.6620`, final F1 `0.7385`, post-drift F1 `0.6076`, stability `0.7902`, retrains `10`, train secs `0.6200`
