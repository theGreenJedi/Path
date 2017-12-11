FROM masteringmatplotlib/scipy
MAINTAINER Py3 Hacker <you@py.hacker>
ENV HOME /root
ENV REPO cloud-deploy
RUN cd $HOME && \
    git clone https://github.com/masteringmatplotlib/${REPO}.git
RUN cd $HOME/$REPO && \
    make docker-setup
CMD PYTHONPATH=${HOME}/${REPO}/lib:$PYTHONPATH python3
