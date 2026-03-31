# Drift Response Benchmark

## Overall Leaderboard

| Policy | Avg Rank | Avg Composite | Avg Final F1 | Avg Post-Drift F1 | Avg Train Secs | Avg Promoted Retrains | Avg Inference ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_retrain | 1.25 | 0.3650 | 0.9677 | 0.4666 | 0.0000 | 0.00 | 16.5785 |
| scheduled | 2.50 | 0.3376 | 0.8822 | 0.4403 | 0.2513 | 0.50 | 17.5045 |
| threshold | 2.75 | 0.3290 | 0.9070 | 0.4418 | 1.0767 | 2.00 | 23.7045 |
| challenger | 4.25 | 0.3153 | 0.8433 | 0.4263 | 0.8442 | 1.50 | 22.1323 |
| adaptive_window | 4.25 | 0.3152 | 0.8433 | 0.4263 | 0.8901 | 1.50 | 22.0168 |

## Scenario Winners

- `digits_adaptive_window`: `no_retrain` won with composite `0.3624`, final F1 `0.9684`, and post-drift F1 `0.4534`
- `digits_concept_shift`: `no_retrain` won with composite `0.3603`, final F1 `0.9733`, and post-drift F1 `0.4360`
- `digits_hybrid_shift`: `no_retrain` won with composite `0.3664`, final F1 `0.9621`, and post-drift F1 `0.5261`
- `digits_late_break_shift`: `threshold` won with composite `0.3876`, final F1 `0.9757`, and post-drift F1 `0.4863`

## Detailed Results

- `digits_concept_shift` / `no_retrain`: rank 1, avg batch F1 `0.5735`, final F1 `0.9733`, post-drift F1 `0.4360`, stability `0.7928`, retrains `0`, train secs `0.0000`
- `digits_concept_shift` / `scheduled`: rank 2, avg batch F1 `0.5735`, final F1 `0.9733`, post-drift F1 `0.4360`, stability `0.7928`, retrains `2`, train secs `0.2048`
- `digits_concept_shift` / `threshold`: rank 3, avg batch F1 `0.5661`, final F1 `0.9552`, post-drift F1 `0.4371`, stability `0.7890`, retrains `8`, train secs `0.9020`
- `digits_concept_shift` / `adaptive_window`: rank 4, avg batch F1 `0.5645`, final F1 `0.9475`, post-drift F1 `0.4289`, stability `0.7884`, retrains `8`, train secs `0.7601`
- `digits_concept_shift` / `challenger`: rank 5, avg batch F1 `0.5645`, final F1 `0.9475`, post-drift F1 `0.4289`, stability `0.7884`, retrains `8`, train secs `0.7627`
- `digits_hybrid_shift` / `no_retrain`: rank 1, avg batch F1 `0.5971`, final F1 `0.9621`, post-drift F1 `0.5261`, stability `0.8265`, retrains `0`, train secs `0.0000`
- `digits_hybrid_shift` / `scheduled`: rank 2, avg batch F1 `0.5971`, final F1 `0.9621`, post-drift F1 `0.5261`, stability `0.8265`, retrains `2`, train secs `0.1949`
- `digits_hybrid_shift` / `challenger`: rank 3, avg batch F1 `0.6016`, final F1 `0.9031`, post-drift F1 `0.4972`, stability `0.8330`, retrains `8`, train secs `0.7688`
- `digits_hybrid_shift` / `adaptive_window`: rank 4, avg batch F1 `0.6016`, final F1 `0.9031`, post-drift F1 `0.4972`, stability `0.8330`, retrains `9`, train secs `0.8637`
- `digits_hybrid_shift` / `threshold`: rank 5, avg batch F1 `0.6098`, final F1 `0.9539`, post-drift F1 `0.4715`, stability `0.8367`, retrains `9`, train secs `1.0080`
- `digits_late_break_shift` / `threshold`: rank 1, avg batch F1 `0.7359`, final F1 `0.9757`, post-drift F1 `0.4863`, stability `0.7848`, retrains `11`, train secs `1.2910`
- `digits_late_break_shift` / `no_retrain`: rank 2, avg batch F1 `0.7308`, final F1 `0.9669`, post-drift F1 `0.4510`, stability `0.7846`, retrains `0`, train secs `0.0000`
- `digits_late_break_shift` / `scheduled`: rank 3, avg batch F1 `0.7308`, final F1 `0.9669`, post-drift F1 `0.4510`, stability `0.7846`, retrains `3`, train secs `0.3051`
- `digits_late_break_shift` / `adaptive_window`: rank 4, avg batch F1 `0.7308`, final F1 `0.9669`, post-drift F1 `0.4510`, stability `0.7846`, retrains `11`, train secs `1.0493`
- `digits_late_break_shift` / `challenger`: rank 5, avg batch F1 `0.7308`, final F1 `0.9669`, post-drift F1 `0.4510`, stability `0.7846`, retrains `11`, train secs `1.0862`
- `digits_adaptive_window` / `no_retrain`: rank 1, avg batch F1 `0.5613`, final F1 `0.9684`, post-drift F1 `0.4534`, stability `0.7699`, retrains `0`, train secs `0.0000`
- `digits_adaptive_window` / `threshold`: rank 2, avg batch F1 `0.5418`, final F1 `0.7431`, post-drift F1 `0.3723`, stability `0.7565`, retrains `9`, train secs `1.1058`
- `digits_adaptive_window` / `scheduled`: rank 3, avg batch F1 `0.5203`, final F1 `0.6265`, post-drift F1 `0.3479`, stability `0.7395`, retrains `3`, train secs `0.3004`
- `digits_adaptive_window` / `challenger`: rank 4, avg batch F1 `0.4825`, final F1 `0.5559`, post-drift F1 `0.3282`, stability `0.7109`, retrains `8`, train secs `0.7590`
- `digits_adaptive_window` / `adaptive_window`: rank 5, avg batch F1 `0.4825`, final F1 `0.5559`, post-drift F1 `0.3282`, stability `0.7109`, retrains `9`, train secs `0.8874`
