
clean-venv:
	pip freeze > make_venv_to_uninstall.txt && \
	pip uninstall -r make_venv_to_uninstall.txt && \
	rm make_venv_to_uninstall.txt

install:
	python3 -m pip install .