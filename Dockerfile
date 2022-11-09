# syntax=docker/dockerfile:1
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

LABEL bittensor.image.authors="bittensor.com" \
	bittensor.image.vendor="Bittensor" \
	bittensor.image.title="bittensor/bittensor" \
	bittensor.image.description="Bittensor: Incentivized Peer to Peer Neural Networks" \
	bittensor.image.source="https://github.com/opentensor/bittensor.git" \
	bittensor.image.revision="${VCS_REF}" \
	bittensor.image.created="${BUILD_DATE}" \
	bittensor.image.documentation="https://app.gitbook.com/@opentensor/s/bittensor/"
ARG DEBIAN_FRONTEND=noninteractive

#nvidia key migration
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub
# Update the base image
RUN apt update && apt upgrade -y
# Install bittensor
## Install dependencies
RUN apt install -y curl sudo nano git htop netcat wget unzip python3-dev python3-pip tmux apt-utils cmake build-essential
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
RUN cd ~/.bittensor/bittensor && \
    python3 -m pip install bittensor

# Increase ulimit to 1,000,000
RUN prlimit --pid=$PPID --nofile=1000000

EXPOSE 8091
