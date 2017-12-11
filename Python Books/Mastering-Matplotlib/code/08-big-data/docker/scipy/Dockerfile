FROM masteringmatplotlib/python
MAINTAINER Duncan McGreggor <oubiwann@gmail.com>
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get install -y -q libatlas3-base libblas-dev libblas3 \
    libatlas-base-dev libatlas-dev liblapack-dev gfortran
ENV LAPACK /usr/lib/liblapack.so
ENV ATLAS /usr/lib/libatlas.so
ENV BLAS /usr/lib/libblas.so
RUN apt-get install -y -q python3-six python3-flake8 python3-dateutil \
    python3-pyparsing python3-numpy python3-scipy python3-matplotlib \
    python3-pandas
RUN pip3 install seaborn
CMD python3
