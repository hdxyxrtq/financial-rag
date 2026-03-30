@echo off
echo Starting Financial RAG API...
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
pause
