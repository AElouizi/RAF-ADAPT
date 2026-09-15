@echo off
REM Dashboard officiel RAF-ADAPT (strategies C1-C5 · NSGA-III)
REM NE PAS lancer bvc_recommender\app.py (ancienne version TFT + 3 portefeuilles)

cd /d "%~dp0"
echo.
echo  RAF-ADAPT — application web officielle
echo  Strategies : meta-selection (meilleur algo/titre) + NSGA champion
echo  URL        : http://localhost:8501
echo.
py -m streamlit run experiments/factorial_hybrid_adapt/web/app_c4.py --server.port 8501
pause
