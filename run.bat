@echo off
echo ==================================================
echo Installing Dependencies...
echo ==================================================
py -3.11 -m pip install -r requirements.txt
echo.
echo ==================================================
echo Starting Argus Bid AI...
echo ==================================================
py -3.11 -m streamlit run tender_audit_platform.py
pause
