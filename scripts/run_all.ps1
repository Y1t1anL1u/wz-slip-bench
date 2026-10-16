# Reproduce every number in the paper from task_c_slip_nc.parquet + entities_*.parquet.
# Set $env:WZDX_DIR to the folder that contains out\ and cache\ (default C:\wzdx).
$env:WZDX_DIR = if ($env:WZDX_DIR) { $env:WZDX_DIR } else { "C:/wzdx" }
Set-Location $PSScriptRoot\..\bench

python mark_censored.py                                    # §3 label validation
python group_stats.py                                      # §5 slip structure

1..10 | % { python runner.py --template knn   --stratify --no-noslip --seed $_ }                      # free
1..10 | % { python runner.py --template warm3 --stratify --no-noslip --seed $_ }                      # haiku  ~$0.35
1..10 | % { python runner.py --template warm3 --stratify --no-noslip --model claude-sonnet-4-6 --seed $_ }  # ~$1.1

python compare_runs.py --a "$env:WZDX_DIR/out/pilot_claude-sonnet-4-6_c_warm_v3_k8_hard_wzonly_seed*.parquet" --b "$env:WZDX_DIR/out/pilot_nomodel_knn30_v1_hard_wzonly_seed*.parquet" --label-a sonnet --label-b knn
python compare_runs.py --a "$env:WZDX_DIR/out/pilot_claude-haiku-4-5_c_warm_v3_k8_hard_wzonly_seed*.parquet"  --b "$env:WZDX_DIR/out/pilot_nomodel_knn30_v1_hard_wzonly_seed*.parquet" --label-a haiku  --label-b knn

foreach ($t in 7,14,30) { python conformal.py --source pool --tau $t --alpha 0.1 }                  # §7
python conformal.py --source pilot --files "$env:WZDX_DIR/out/pilot_claude-sonnet-4-6_c_warm_v3_k8_hard_wzonly_seed*.parquet" --tau 7
foreach ($f in 0.1,0.2,0.3,0.5) { python conformal.py --source pool --tau 7 --alpha 0.1 --cal-frac $f --splits 50 }  # appendix

Set-Location ..
python make_figures.py
