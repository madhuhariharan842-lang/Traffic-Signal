@echo off
REM Activate your virtual environment first if you use one.
REM Example: call venv\Scripts\activate

REM 1) Start the adaptive processing (runs detection & writes state.json)
start python adaptive_signal.py

REM 2) Start Streamlit dashboard
start streamlit run app.py

pause
