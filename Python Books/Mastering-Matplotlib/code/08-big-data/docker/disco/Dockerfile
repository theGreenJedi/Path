FROM masteringmatplotlib/erlang
MAINTAINER Duncan McGreggor <oubiwann@gmail.com>
ENV ORG discoproject
ENV REPO disco
ENV DISCO_USER disco
ENV HOME /home/$DISCO_USER
ENV DISCO_HOME $HOME/$REPO
ENV RELEASE 0.5.4
ENV PYTHONPATH /usr/lib/python3.4/site-packages:$PYTHONPATH
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y -q openssh-server
RUN ln -s `which python3` `dirname $(which python3)`/python

# setup SSH for root user
RUN ssh-keygen -N '' -f /root/.ssh/id_dsa
RUN cat /root/.ssh/id_dsa.pub >> /root/.ssh/authorized_keys
RUN echo -n "localhost " > /root/.ssh/known_hosts
RUN cat /etc/ssh/ssh_host_rsa_key.pub >> /root/.ssh/known_hosts

# setup SSH for disco user
RUN adduser --system $DISCO_USER --shell /bin/sh
RUN mkdir /home/$DISCO_USER/.ssh
RUN ssh-keygen -N '' -f /home/$DISCO_USER/.ssh/id_dsa && \
    cat /home/$DISCO_USER/.ssh/id_dsa.pub >> \
        /home/$DISCO_USER/.ssh/authorized_keys && \
    echo -n "localhost " > /home/$DISCO_USER/.ssh/known_hosts && \
    cat /etc/ssh/ssh_host_rsa_key.pub >> \
        /home/$DISCO_USER/.ssh/known_hosts && \
    chown $DISCO_USER -R /home/$DISCO_USER/.ssh
RUN mkdir -p /usr/var/disco/data && \
    chown -R disco /usr/var/disco

# install disco from git clone
RUN cd $HOME && \
    git clone https://github.com/$ORG/$REPO.git && \
    cd $REPO && \
    git checkout tags/$RELEASE && \
    make && make install
RUN echo "defaultcookievalue" > $HOME/.erlang.cookie && \
    chmod 400 $HOME/.erlang.cookie
RUN chown -R $DISCO_USER $HOME

RUN echo '#!/bin/sh' > $HOME/start.sh
RUN echo "/etc/init.d/ssh start" >> $HOME/start.sh
RUN echo "su disco -c '$DISCO_HOME/bin/disco nodaemon'" >> $HOME/start.sh
RUN chmod 755 $HOME/start.sh

EXPOSE 22/tcp 8989/tcp 8990/tcp 999/tcp

CMD $HOME/start.sh
