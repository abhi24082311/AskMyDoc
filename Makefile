.PHONY: install ingest serve eval eval-candidates test clean help

PYTHON := .venv\Scripts\python.exe
PIP    := .venv\Scripts\pip.exe

help:
	@echo.
	@echo   Ask My Docs RAG - Developer Commands
	@echo   ======================================
	@echo   make install    Install all dependencies
	@echo   make ingest     Run document ingestion
	@echo   make serve      Start the FastAPI server
	@echo   make eval       Run the evaluation and quality gate
	@echo   make eval-candidates Draft Q&A pairs from indexed documents
	@echo   make test       Run all unit tests
	@echo   make clean      Remove __pycache__ dirs
	@echo.

install:
	$(PIP) install -r requirements.txt

ingest:
	$(PYTHON) ingest_cli.py

serve:
	$(PYTHON) -m app.main

eval:
	$(PYTHON) -m eval.run
	$(PYTHON) -m eval.quality_gate

eval-candidates:
	$(PYTHON) -m eval.generate_candidates

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	for /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	for /r . %%f in (*.pyc) do @if exist "%%f" del /q "%%f"
