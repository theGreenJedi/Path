FROM ubuntu:vivid
MAINTAINER Duncan McGreggor <oubiwann@gmail.com>
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y -q apt-utils
RUN apt-get install -y -q ca-certificates git build-essential
RUN apt-get install -y -q libssl-dev libcurl4-openssl-dev
RUN apt-get install -y -q curl
RUN apt-get install -y -q cython3 libpython3.4-dev python3.4-dev \
    python3-setuptools python3-pip
CMD python3
