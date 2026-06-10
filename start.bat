@echo off
chcp 65001 >nul
echo Starting CSRC RAG Demo Server...
echo Retrieval mode: hybrid (BM25 + Dense)
echo.
python scripts\run_demo_server.py
pause
