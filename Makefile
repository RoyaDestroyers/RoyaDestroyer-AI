PYTHON ?= python

.PHONY: verify unify verify-images dedup report split train evaluate export serve

verify:
	$(PYTHON) scripts/verify_environment.py

unify:
	$(PYTHON) scripts/unify_datasets.py

verify-images:
	$(PYTHON) scripts/verify_images.py

dedup:
	$(PYTHON) scripts/dedup.py

report:
	$(PYTHON) scripts/report_distribution.py

split:
	$(PYTHON) scripts/split.py

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate.py

export:
	$(PYTHON) scripts/export_model.py

serve:
	uvicorn service.app.main:app --host 0.0.0.0 --port 8000
