VENV=../.venv-mmpl
PYTHON=python3.4
IPYTHON=ipython3
PIP=pip3.4
SYSTEM_PYTHON=$(shell which $(PYTHON))
SOURCE=./lib

virtual-env:
	-$(SYSTEM_PYTHON) -m venv $(VENV)

base-deps:
	. $(VENV)/bin/activate && \
	$(PIP) install -r requirements/base.txt

project-setup: virtual-env deps

run:
	. $(VENV)/bin/activate && \
	$(IPYTHON) notebook $(NOTEBOOK)

clean:
	rm -rf $(VENV)

repl:
	. $(VENV)/bin/activate && $(IPYTHON)

flakes:
	@echo "\nChecking for flakes ...\n"
	flake8 $(SOURCE)

update:
	git submodule foreach git pull origin master

