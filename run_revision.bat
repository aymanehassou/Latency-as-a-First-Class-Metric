@echo off
rem Supplementary experiments and the whole-decision latency measurement.
rem Run train_all.bat first. Keep the machine on mains power and idle during the
rem five latency passes: anything else running distorts the timings.
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
echo [%time%] environment record
python rev_env.py
echo [%time%] SKAB sensitivity to q and Nc
python rev_sensitivity_skab.py || goto :err
echo [%time%] SMD, three seeds on six entities
python rev_seeds_smd.py || goto :err
for %%k in (1 2 3 4 5) do (
  echo latency pass %%k of 5
  python rev_latency_real.py --pass-id %%k || goto :err
)
python rev_summarize.py || goto :err
rem make the whole-decision measurement the one the tables report
python rev_apply_latency.py || goto :err
python rev_summarize.py || goto :err
python analyze.py || goto :err
python verify.py || goto :err
echo.
echo ALL DONE. Tables and figures are in paper_out\.
goto :eof
:err
echo.
echo *** A step failed. See the messages above. ***
exit /b 1
