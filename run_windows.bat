@echo off
echo =================================================================
echo UAV AI Context-Switching Mission Control Console
echo =================================================================
cd /d "%~dp0"
python -m streamlit run app/gui.py
pause
