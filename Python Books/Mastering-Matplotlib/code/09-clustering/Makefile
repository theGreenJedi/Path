INCLUDE_REPO=https://github.com/masteringmatplotlib/includes.git
INCLUDE_DIR=include
NAME=clustering
NOTEBOOK=notebooks/mmpl-$(NAME).ipynb

-include include/common.mk

# The following target is intended for use by project-creators only. When
# creating a new notebook project, add a copy of this Makefile and run this
# target to get the includes set up:
#
# $ make setup-submodule
setup-submodule:
	git submodule add $(INCLUDE_REPO) $(INCLUDE_DIR)

# The 'setup' target needs to be run before the 'project-deps' target,
# so that the includes are present (done by 'make project-setup').
#
# Note that this repo overrides the standard project deps to make
# installation in Docker easier.
deps: base-deps
	. $(VENV)/bin/activate && \
	pip3.4 install -U pip pyzmq
	. $(VENV)/bin/activate && \
	pip3.4 install -r requirements/part1.txt
	. $(VENV)/bin/activate && \
	pip3.4 install -r requirements/part2.txt
	. $(VENV)/bin/activate && \
	pip3.4 install -r requirements/part3.txt

setup:
	@git submodule init
	@git submodule update
	@make project-setup

disco-python: disco
	. $(VENV)/bin/activate && \
	cd disco && \
	cd lib && python3 setup.py install

disco-cli: disco
	. $(VENV)/bin/activate && \
	cd disco/bin && \
	cp * $(VIRTUAL_ENV)/bin

disco-erlang:
	. $(VENV)/bin/activate && \
	cd disco && \
	make

disco:
	git clone https://github.com/discoproject/disco && \
	cd disco && \
	git checkout tags/0.5.4


docker-setup:
	@git submodule init
	@git submodule update
	pip3 install -r requirements/part2.txt
	pip3 install -r requirements/part3.txt

clean-docker:
	-docker rm $(shell docker ps -a -q)
	docker rmi $(shell docker images -q --filter 'dangling=true')

docker-images:
	docker build -t masteringmatplotlib/erlang ./docker/erlang && \
	docker build -t masteringmatplotlib/disco ./docker/disco

run-manager:
	docker run -i -p 8989:8989 -t masteringmatplotlib/disco

run-worker:
	docker run -i -p 8989:8989 -t masteringmatplotlib/disco

.DEFAULT_GOAL :=
default: setup
	make run
