# Drift Response Benchmark

## Overall Leaderboard

| Policy | Avg Rank | Avg Composite | Avg Final F1 | Avg Post-Drift F1 | Avg Train Secs | Avg Promoted Retrains | Avg Inference ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_retrain | 1.00 | 0.5067 | 0.9663 | 0.6521 | 0.0000 | 0.00 | 16.3697 |
| adaptive_window | 2.86 | 0.4715 | 0.9350 | 0.6180 | 0.5491 | 0.29 | 16.2419 |
| scheduled | 3.00 | 0.4603 | 0.9222 | 0.6166 | 0.2175 | 0.71 | 16.7220 |
| challenger | 3.29 | 0.4716 | 0.9350 | 0.6180 | 0.5428 | 0.29 | 16.4233 |
| threshold | 4.86 | 0.3811 | 0.9289 | 0.6327 | 0.6603 | 4.71 | 17.1244 |

## Scenario Winners

- `adaptive_recovery_window`: `no_retrain` won with composite `0.3355`, final F1 `0.9551`, and post-drift F1 `0.3679`
- `concept_drift`: `no_retrain` won with composite `0.3854`, final F1 `0.9622`, and post-drift F1 `0.4095`
- `hybrid_regime_shift`: `no_retrain` won with composite `0.4290`, final F1 `0.9732`, and post-drift F1 `0.5744`
- `label_shift`: `no_retrain` won with composite `0.7038`, final F1 `0.9755`, and post-drift F1 `0.9889`
- `late_breaking_regime_shift`: `no_retrain` won with composite `0.4283`, final F1 `0.9645`, and post-drift F1 `0.5493`
- `mild_covariate`: `no_retrain` won with composite `0.6995`, final F1 `0.9707`, and post-drift F1 `0.9620`
- `severe_covariate`: `no_retrain` won with composite `0.5654`, final F1 `0.9626`, and post-drift F1 `0.7128`

## Detailed Results

- `mild_covariate` / `no_retrain`: rank 1, avg batch F1 `0.9712`, final F1 `0.9707`, post-drift F1 `0.9620`, stability `0.9852`, retrains `0`, train secs `0.0000`
- `mild_covariate` / `scheduled`: rank 2, avg batch F1 `0.9712`, final F1 `0.9707`, post-drift F1 `0.9620`, stability `0.9852`, retrains `2`, train secs `0.1391`
- `mild_covariate` / `challenger`: rank 3, avg batch F1 `0.9712`, final F1 `0.9707`, post-drift F1 `0.9620`, stability `0.9852`, retrains `7`, train secs `0.4822`
- `mild_covariate` / `adaptive_window`: rank 4, avg batch F1 `0.9712`, final F1 `0.9707`, post-drift F1 `0.9620`, stability `0.9852`, retrains `7`, train secs `0.4948`
- `mild_covariate` / `threshold`: rank 5, avg batch F1 `0.8961`, final F1 `0.9776`, post-drift F1 `0.9667`, stability `0.9059`, retrains `8`, train secs `0.5573`
- `severe_covariate` / `no_retrain`: rank 1, avg batch F1 `0.8807`, final F1 `0.9626`, post-drift F1 `0.7128`, stability `0.8814`, retrains `0`, train secs `0.0000`
- `severe_covariate` / `scheduled`: rank 2, avg batch F1 `0.8807`, final F1 `0.9626`, post-drift F1 `0.7128`, stability `0.8814`, retrains `3`, train secs `0.2105`
- `severe_covariate` / `adaptive_window`: rank 3, avg batch F1 `0.8807`, final F1 `0.8287`, post-drift F1 `0.5053`, stability `0.8814`, retrains `8`, train secs `0.5384`
- `severe_covariate` / `challenger`: rank 4, avg batch F1 `0.8807`, final F1 `0.8287`, post-drift F1 `0.5053`, stability `0.8814`, retrains `8`, train secs `0.5558`
- `severe_covariate` / `threshold`: rank 5, avg batch F1 `0.8703`, final F1 `0.9550`, post-drift F1 `0.6603`, stability `0.9155`, retrains `9`, train secs `0.6552`
- `label_shift` / `no_retrain`: rank 1, avg batch F1 `0.9777`, final F1 `0.9755`, post-drift F1 `0.9889`, stability `0.9782`, retrains `0`, train secs `0.0000`
- `label_shift` / `adaptive_window`: rank 2, avg batch F1 `0.9777`, final F1 `0.9755`, post-drift F1 `0.9889`, stability `0.9782`, retrains `7`, train secs `0.4793`
- `label_shift` / `challenger`: rank 3, avg batch F1 `0.9777`, final F1 `0.9755`, post-drift F1 `0.9889`, stability `0.9782`, retrains `7`, train secs `0.4849`
- `label_shift` / `scheduled`: rank 4, avg batch F1 `0.9319`, final F1 `0.9330`, post-drift F1 `0.9420`, stability `0.9479`, retrains `3`, train secs `0.2094`
- `label_shift` / `threshold`: rank 5, avg batch F1 `0.9403`, final F1 `0.9778`, post-drift F1 `0.9824`, stability `0.9506`, retrains `8`, train secs `0.5489`
- `concept_drift` / `no_retrain`: rank 1, avg batch F1 `0.6321`, final F1 `0.9622`, post-drift F1 `0.4095`, stability `0.8020`, retrains `0`, train secs `0.0000`
- `concept_drift` / `challenger`: rank 2, avg batch F1 `0.6183`, final F1 `0.8774`, post-drift F1 `0.3780`, stability `0.7917`, retrains `6`, train secs `0.4038`
- `concept_drift` / `adaptive_window`: rank 3, avg batch F1 `0.6183`, final F1 `0.8774`, post-drift F1 `0.3780`, stability `0.7917`, retrains `7`, train secs `0.4822`
- `concept_drift` / `scheduled`: rank 4, avg batch F1 `0.5806`, final F1 `0.8215`, post-drift F1 `0.3639`, stability `0.7884`, retrains `3`, train secs `0.2070`
- `concept_drift` / `threshold`: rank 5, avg batch F1 `0.5885`, final F1 `0.8932`, post-drift F1 `0.3895`, stability `0.7916`, retrains `8`, train secs `0.5576`
- `hybrid_regime_shift` / `no_retrain`: rank 1, avg batch F1 `0.6140`, final F1 `0.9732`, post-drift F1 `0.5744`, stability `0.8117`, retrains `0`, train secs `0.0000`
- `hybrid_regime_shift` / `adaptive_window`: rank 2, avg batch F1 `0.6140`, final F1 `0.9732`, post-drift F1 `0.5744`, stability `0.8117`, retrains `7`, train secs `0.4820`
- `hybrid_regime_shift` / `challenger`: rank 3, avg batch F1 `0.6140`, final F1 `0.9732`, post-drift F1 `0.5744`, stability `0.8117`, retrains `7`, train secs `0.4860`
- `hybrid_regime_shift` / `threshold`: rank 4, avg batch F1 `0.6020`, final F1 `0.9599`, post-drift F1 `0.5502`, stability `0.8016`, retrains `8`, train secs `0.5895`
- `hybrid_regime_shift` / `scheduled`: rank 5, avg batch F1 `0.5766`, final F1 `0.8482`, post-drift F1 `0.4186`, stability `0.7864`, retrains `3`, train secs `0.1993`
- `adaptive_recovery_window` / `no_retrain`: rank 1, avg batch F1 `0.5137`, final F1 `0.9551`, post-drift F1 `0.3679`, stability `0.7872`, retrains `0`, train secs `0.0000`
- `adaptive_recovery_window` / `scheduled`: rank 2, avg batch F1 `0.5137`, final F1 `0.9551`, post-drift F1 `0.3679`, stability `0.7872`, retrains `4`, train secs `0.2793`
- `adaptive_recovery_window` / `adaptive_window`: rank 3, avg batch F1 `0.5137`, final F1 `0.9551`, post-drift F1 `0.3679`, stability `0.7872`, retrains `10`, train secs `0.6923`
- `adaptive_recovery_window` / `challenger`: rank 4, avg batch F1 `0.5137`, final F1 `0.9551`, post-drift F1 `0.3679`, stability `0.7872`, retrains `10`, train secs `0.6942`
- `adaptive_recovery_window` / `threshold`: rank 5, avg batch F1 `0.5077`, final F1 `0.7653`, post-drift F1 `0.3286`, stability `0.7809`, retrains `11`, train secs `0.8587`
- `late_breaking_regime_shift` / `no_retrain`: rank 1, avg batch F1 `0.6517`, final F1 `0.9645`, post-drift F1 `0.5493`, stability `0.7650`, retrains `0`, train secs `0.0000`
- `late_breaking_regime_shift` / `scheduled`: rank 2, avg batch F1 `0.6517`, final F1 `0.9645`, post-drift F1 `0.5493`, stability `0.7650`, retrains `4`, train secs `0.2778`
- `late_breaking_regime_shift` / `adaptive_window`: rank 3, avg batch F1 `0.6517`, final F1 `0.9645`, post-drift F1 `0.5493`, stability `0.7650`, retrains `10`, train secs `0.6748`
- `late_breaking_regime_shift` / `challenger`: rank 4, avg batch F1 `0.6517`, final F1 `0.9645`, post-drift F1 `0.5493`, stability `0.7650`, retrains `10`, train secs `0.6925`
- `late_breaking_regime_shift` / `threshold`: rank 5, avg batch F1 `0.6628`, final F1 `0.9735`, post-drift F1 `0.5515`, stability `0.7714`, retrains `11`, train secs `0.8550`
