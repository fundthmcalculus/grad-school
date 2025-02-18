.PHONY: format deploy

format:
	black ./
	flake8 ./
	mypy ./
