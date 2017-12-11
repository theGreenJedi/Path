FROM masteringmatplotlib/python
MAINTAINER Duncan McGreggor <oubiwann@gmail.com>
RUN apt-get update
RUN apt-get -y upgrade
RUN echo -e '#!/bin/sh\nexit 0' > /usr/sbin/policy-rc.d
RUN apt-get install -y -q wget
RUN wget https://packages.erlang-solutions.com/erlang-solutions_1.0_all.deb && \
    dpkg -i erlang-solutions_1.0_all.deb && \
    rm erlang-solutions_1.0_all.deb
RUN apt-get -y install erlang erlang-base-hipe erlang-manpages erlang-doc \
    erlang-mode erlang-dev erlang-src erlang-nox
RUN git clone https://github.com/rebar/rebar.git && \
    cd rebar && ./bootstrap && mv rebar /usr/local/bin/ && \
    cd ../ && rm -rf rebar
CMD ["erl"]
