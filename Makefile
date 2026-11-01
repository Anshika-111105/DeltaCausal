.PHONY: test smoke-test pipeline run-api run-dashboard clean

smoke-test:
	python scripts/run_smoke_test.py

test:
	pytest -v

pipeline:
	python scripts/run_pipeline.py --sample-size 25000

run-api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard:
	streamlit run app/dashboard.py

