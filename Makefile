SHELL:=/bin/bash

init-venv:
	python3 -m venv venv && source ./venv/bin/activate

clean-venv:
	source ./venv/bin/activate && \
	pip freeze > make_venv_to_uninstall.txt && \
	pip uninstall -r make_venv_to_uninstall.txt && \
	rm make_venv_to_uninstall.txt

install:
	python3 -m pip install .

install-dev:
	python3 -m pip install '.[dev]'

install-cubit:
	python3 -m pip install '.[cubit]'