MYPY=$(VENV)/bin/mypy

$(MYPY):
	git clone https://github.com/JukkaL/mypy.git
	. $(VENV)/bin/activate && \
	cd mypy && $(PYTHON) setup.py install
	rm -rf mypy

types: $(MYPY)
	@echo "\nChecking types ...\n"
	. $(VENV)/bin/activate && \
	for FILE in ./lib/*.py; do mypy $$FILE; done

check: flakes types
