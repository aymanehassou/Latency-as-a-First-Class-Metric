@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem ---------------------------------------------------------------------------
rem Full reproduction, from nothing to the LaTeX tables.
rem
rem Every model is trained in its OWN process. That is not cosmetic: the Torch
rem thread pool cannot be reconfigured twice inside one interpreter on this
rem Windows build without the process hanging or dying outright, and a fresh
rem process per model also means one crash cannot take the rest down with it.
rem
rem Each stage writes its own log under logs\. Re-running the script re-runs
rem everything; to redo a single model, just run that one train_*.py.
rem ---------------------------------------------------------------------------

mkdir logs 2>nul

echo ================= BEHAVIOUR TRACK (WISDM) =================
for %%M in (rf mlp cnn1d tcn lstm bigru transformer) do (
  echo [%%M] %date% %time%
  python -u train_%%M.py > logs\train_%%M.log 2>&1
  echo     exit=!errorlevel!
)

echo ================= STATE TRACK (SKAB + SMD) =================
for %%M in (pca iforest dense_ae conv_ae lstm_ae transformer_ae) do (
  echo [%%M] %date% %time%
  python -u train_%%M.py > logs\train_%%M.log 2>&1
  echo     exit=!errorlevel!
)

echo ================= DEGENERATE CONTROLS =================
rem always-alarm and random-score. Not models, but analyze.py needs them and
rem the state-detection table is not readable without them.
python -u train_controls.py > logs\train_controls.log 2>&1
echo     exit=!errorlevel!

echo ================= COLLECT =================
python -u collect.py > logs\collect.log 2>&1
echo     exit=!errorlevel!
type logs\collect.log

echo ================= DATASET FACTS =================
python -u make_datasets_table.py > logs\datasets.log 2>&1
echo     exit=!errorlevel!

echo ================= LATENCY (quiescent, CPU, 5 passes) =================
rem One process per pass, for the same reason as above. Nothing else should be
rem running on the machine while this stage is in progress.
for %%i in (1 2 3 4 5) do (
  echo [latency pass %%i] %date% %time%
  python -u bench_pass.py results\lat_pass%%i.json > logs\lat_pass%%i.log 2>&1
  echo     exit=!errorlevel!
)
python -u merge_latency.py > logs\lat_merge.log 2>&1
echo     exit=!errorlevel!

echo ================= ANALYSIS =================
python -u analyze.py            > logs\analyze.log 2>&1
echo     analyze  exit=!errorlevel!
python -u verify.py             > logs\verify.log 2>&1
echo     verify   exit=!errorlevel!

echo ================= DONE %date% %time% =================
type logs\verify.log
