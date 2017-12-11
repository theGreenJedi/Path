pygraphviz:
	@git clone https://github.com/pygraphviz/pygraphviz
	@. $(VENV)/bin/activate && \
	cd pygraphviz && \
	git checkout 6c0876c9bb158452f1193d562531d258e9193f2e && \
	git apply ../patches/graphviz-includes.diff && \
	python setup.py install
	@rm -rf pygraphviz
