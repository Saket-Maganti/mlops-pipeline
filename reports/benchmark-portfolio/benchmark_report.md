# Drift Response Benchmark

## Overall Leaderboard

| Policy | Avg Rank | Avg Composite | Avg Final F1 | Avg Post-Drift F1 | Avg Train Secs | Avg Promoted Retrains | Avg Inference ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| challenger | 2.20 | 0.3916 | 0.6774 | 0.6148 | 0.4460 | 3.80 | 16.1448 |
| scheduled | 2.40 | 0.3922 | 0.8707 | 0.5910 | 0.2158 | 2.40 | 16.3612 |
| no_retrain | 3.20 | 0.3683 | 0.9608 | 0.5163 | 0.0000 | 0.00 | 16.3350 |
| threshold | 3.40 | 0.3976 | 0.9411 | 0.6699 | 0.7218 | 6.20 | 16.6094 |
| adaptive_window | 3.80 | 0.3584 | 0.7160 | 0.6118 | 0.6580 | 6.20 | 16.4772 |

## Scenario Winners

- `portfolio_concept_shift`: `challenger` won with composite `0.4386`, final F1 `0.3846`, and post-drift F1 `0.7317`
- `portfolio_hybrid_shift`: `no_retrain` won with composite `0.3587`, final F1 `0.9860`, and post-drift F1 `0.5681`
- `portfolio_late_regime_shift`: `scheduled` won with composite `0.4873`, final F1 `0.9356`, and post-drift F1 `0.6661`
- `portfolio_mild_covariate`: `scheduled` won with composite `0.5791`, final F1 `0.9356`, and post-drift F1 `0.9172`
- `portfolio_severe_covariate`: `threshold` won with composite `0.4865`, final F1 `0.9930`, and post-drift F1 `0.6955`

## Detailed Results

- `portfolio_mild_covariate` / `scheduled`: rank 1, avg batch F1 `0.9649`, final F1 `0.9356`, post-drift F1 `0.9172`, stability `0.9815`, retrains `2`, train secs `0.1445`
- `portfolio_mild_covariate` / `challenger`: rank 2, avg batch F1 `0.9555`, final F1 `0.9575`, post-drift F1 `0.9516`, stability `0.9715`, retrains `5`, train secs `0.3599`
- `portfolio_mild_covariate` / `threshold`: rank 3, avg batch F1 `0.9670`, final F1 `0.9646`, post-drift F1 `0.9653`, stability `0.9840`, retrains `8`, train secs `0.5893`
- `portfolio_mild_covariate` / `no_retrain`: rank 4, avg batch F1 `0.9592`, final F1 `0.9648`, post-drift F1 `0.8556`, stability `0.9869`, retrains `0`, train secs `0.0000`
- `portfolio_mild_covariate` / `adaptive_window`: rank 5, avg batch F1 `0.9611`, final F1 `0.9575`, post-drift F1 `0.9516`, stability `0.9798`, retrains `8`, train secs `0.5797`
- `portfolio_severe_covariate` / `threshold`: rank 1, avg batch F1 `0.8566`, final F1 `0.9930`, post-drift F1 `0.6955`, stability `0.8684`, retrains `9`, train secs `0.7412`
- `portfolio_severe_covariate` / `no_retrain`: rank 2, avg batch F1 `0.7653`, final F1 `0.9443`, post-drift F1 `0.3616`, stability `0.7994`, retrains `0`, train secs `0.0000`
- `portfolio_severe_covariate` / `scheduled`: rank 3, avg batch F1 `0.8241`, final F1 `0.9575`, post-drift F1 `0.2004`, stability `0.8303`, retrains `3`, train secs `0.2183`
- `portfolio_severe_covariate` / `challenger`: rank 4, avg batch F1 `0.8067`, final F1 `0.9502`, post-drift F1 `0.2004`, stability `0.8179`, retrains `5`, train secs `0.3600`
- `portfolio_severe_covariate` / `adaptive_window`: rank 5, avg batch F1 `0.8071`, final F1 `0.9502`, post-drift F1 `0.2004`, stability `0.8260`, retrains `9`, train secs `0.6361`
- `portfolio_concept_shift` / `challenger`: rank 1, avg batch F1 `0.6453`, final F1 `0.3846`, post-drift F1 `0.7317`, stability `0.8402`, retrains `6`, train secs `0.4282`
- `portfolio_concept_shift` / `scheduled`: rank 2, avg batch F1 `0.6331`, final F1 `0.5746`, post-drift F1 `0.6851`, stability `0.8432`, retrains `3`, train secs `0.2199`
- `portfolio_concept_shift` / `adaptive_window`: rank 3, avg batch F1 `0.6436`, final F1 `0.3846`, post-drift F1 `0.7317`, stability `0.8448`, retrains `9`, train secs `0.6465`
- `portfolio_concept_shift` / `no_retrain`: rank 4, avg batch F1 `0.6282`, final F1 `0.9369`, post-drift F1 `0.5261`, stability `0.8317`, retrains `0`, train secs `0.0000`
- `portfolio_concept_shift` / `threshold`: rank 5, avg batch F1 `0.6436`, final F1 `0.7902`, post-drift F1 `0.5764`, stability `0.8433`, retrains `9`, train secs `0.6936`
- `portfolio_hybrid_shift` / `no_retrain`: rank 1, avg batch F1 `0.6297`, final F1 `0.9860`, post-drift F1 `0.5681`, stability `0.7996`, retrains `0`, train secs `0.0000`
- `portfolio_hybrid_shift` / `challenger`: rank 2, avg batch F1 `0.6521`, final F1 `0.5998`, post-drift F1 `0.5661`, stability `0.8147`, retrains `7`, train secs `0.4983`
- `portfolio_hybrid_shift` / `adaptive_window`: rank 3, avg batch F1 `0.6524`, final F1 `0.5998`, post-drift F1 `0.5661`, stability `0.8152`, retrains `9`, train secs `0.6458`
- `portfolio_hybrid_shift` / `threshold`: rank 4, avg batch F1 `0.6436`, final F1 `0.9719`, post-drift F1 `0.5836`, stability `0.8172`, retrains `9`, train secs `0.6933`
- `portfolio_hybrid_shift` / `scheduled`: rank 5, avg batch F1 `0.6227`, final F1 `0.9502`, post-drift F1 `0.4862`, stability `0.7902`, retrains `3`, train secs `0.2131`
- `portfolio_late_regime_shift` / `scheduled`: rank 1, avg batch F1 `0.6735`, final F1 `0.9356`, post-drift F1 `0.6661`, stability `0.7737`, retrains `4`, train secs `0.2834`
- `portfolio_late_regime_shift` / `challenger`: rank 2, avg batch F1 `0.7008`, final F1 `0.4948`, post-drift F1 `0.6243`, stability `0.7997`, retrains `8`, train secs `0.5838`
- `portfolio_late_regime_shift` / `adaptive_window`: rank 3, avg batch F1 `0.6871`, final F1 `0.6881`, post-drift F1 `0.6091`, stability `0.7855`, retrains `11`, train secs `0.7818`
- `portfolio_late_regime_shift` / `threshold`: rank 4, avg batch F1 `0.6613`, final F1 `0.9860`, post-drift F1 `0.5287`, stability `0.7552`, retrains `11`, train secs `0.8915`
- `portfolio_late_regime_shift` / `no_retrain`: rank 5, avg batch F1 `0.6566`, final F1 `0.9718`, post-drift F1 `0.2702`, stability `0.7405`, retrains `0`, train secs `0.0000`
