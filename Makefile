.PHONY: .venv

run-test:
ifdef dst
	PYTHONPATH=.:src python -m pytest $(dst) -v
else
	PYTHONPATH=.:src python -m pytest -v
endif

clear-cache:
	find . -type d -name '__pycache__' -exec rm -r {} + \
	&& find . -type f -name '*.pyc' -delete

run-test-cov:
	@PYTHONPATH=.:app:$PYTHONPATH pytest --cov=app --cov-report=xml

precommit:
	pre-commit run --all-files
