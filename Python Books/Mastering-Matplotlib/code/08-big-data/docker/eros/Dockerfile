FROM masteringmatplotlib/scipy
MAINTAINER Py3 Hacker <you@py.hacker>
ENV HOME /root
ENV REPO cloud-deploy
RUN cd $HOME && \
    git clone https://github.com/masteringmatplotlib/${REPO}.git
RUN cd $HOME/$REPO && \
    make docker-setup
CMD DOCKER_CONTAINER=true \
    PYTHONPATH=${HOME}/${REPO}/lib:$PYTHONPATH \
    python3 -c "import ec2s3eros;ec2s3eros.s3_generate_image();"
