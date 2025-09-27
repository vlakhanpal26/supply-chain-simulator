.PHONY: install test report app lint

install:
	pip install -r requirements.txt

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest

report:
	python run_phase2.py

app:
	streamlit run app/app.py

lint:
	black engine_core.py engine.py engine2.py run_phase2.py scm_cli.py app/ tests/
