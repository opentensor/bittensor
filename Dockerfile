FROM nvidia/cuda:11.2.1-base

LABEL bittensor.image.authors="bittensor.com" \
	bittensor.image.vendor="Bittensor" \
	bittensor.image.title="bittensor/bittensor" \
	bittensor.image.description="Bittensor: Incentivized Peer to Peer Neural Networks" \
	bittensor.image.source="https://github.com/opentensor/bittensor.git" \
	bittensor.image.revision="${VCS_REF}" \
	bittensor.image.created="${BUILD_DATE}" \
	bittensor.image.documentation="https://opentensor.bittensor.io"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y apt-utils curl git cmake build-essential unzip python3-pip  wget iproute2 software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install python3.7 python3.7-dev -y
RUN python3.7 -m pip install --upgrade pip

RUN rm /usr/bin/python3
RUN ln -s //usr/bin/python3.7 /usr/bin/python3

# add Bittensor code to docker image
RUN mkdir /bittensor
RUN mkdir /home/.bittensor
COPY . /bittensor

WORKDIR /bittensor
RUN pip install --upgrade numpy pandas setuptools "tqdm>=4.27,<4.50.0" wheel
RUN pip install -r requirements.txt
RUN pip install -e .

EXPOSE 8091
