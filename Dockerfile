# syntax=docker/dockerfile:1
FROM python:3.11.8-bookworm

LABEL bittensor.image.authors="bittensor.com" \
	bittensor.image.vendor="Bittensor" \
	bittensor.image.title="bittensor/bittensor" \
	bittensor.image.description="Bittensor: Incentivized Peer to Peer Neural Networks" \
	bittensor.image.source="https://github.com/opentensor/bittensor.git" \
	bittensor.image.revision="${VCS_REF}" \
	bittensor.image.created="${BUILD_DATE}" \
	bittensor.image.documentation="https://app.gitbook.com/@opentensor/s/bittensor/"
ARG DEBIAN_FRONTEND=noninteractive

# Update the base image
RUN apt update && apt upgrade -y
# Install bittensor
## Install dependencies
RUN apt install -y curl sudo nano git htop netcat-openbsd wget unzip tmux apt-utils cmake build-essential
## Upgrade pip
RUN pip3 install --upgrade pip

# Install nvm and pm2
RUN curl -o install_nvm.sh https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh && \
	echo 'fabc489b39a5e9c999c7cab4d281cdbbcbad10ec2f8b9a7f7144ad701b6bfdc7 install_nvm.sh' | sha256sum --check && \
	bash install_nvm.sh

RUN bash -c "source $HOME/.nvm/nvm.sh && \
    # use node 16
    nvm install 16 && \
    # install pm2
    npm install --location=global pm2"

RUN mkdir -p /root/.bittensor/bittensor
COPY . /root/.bittensor/bittensor
RUN cd /root/.bittensor/bittensor && python3 -m pip install .

# Increase ulimit to 1,000,000
RUN prlimit --pid=$PPID --nofile=1000000

EXPOSE 8091
