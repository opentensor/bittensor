FROM nvidia/cuda:10.2-base

LABEL bittensor.image.authors="bittensor.com" \
	bittensor.image.vendor="Bittensor" \
	bittensor.image.title="bittensor/bittensor" \
	bittensor.image.description="Bittensor: Incentivized Peer to Peer Neural Networks" \
	bittensor.image.source="https://github.com/opentensor/bittensor.git" \
	bittensor.image.revision="${VCS_REF}" \
	bittensor.image.created="${BUILD_DATE}" \
	bittensor.image.documentation="https://opentensor.bittensor.io"

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y apt-utils curl git cmake build-essential unzip python3.7 python3-pip python3.7-dev wget
RUN python3.7 -m pip install --upgrade pip

# add Bittensor code to docker image
RUN mkdir /bittensor
RUN mkdir /home/.bittensor
RUN mkdir -p /subtensor/v1.0.1
RUN mkdir -p /subtensor/v1.1.0
COPY . /bittensor

WORKDIR /bittensor
RUN pip install --upgrade numpy pandas setuptools "tqdm>=4.27,<4.50.0" wheel
RUN pip install -r requirements.txt
RUN pip install -e .

WORKDIR /subtensor/v1.0.1
RUN wget https://github.com/opentensor/subtensor/releases/download/v1.0.1/subtensor-v1.0.1-linux-unknown-gnu-x86_64.tar.gz
RUN tar -xzf subtensor-v1.0.1-linux-unknown-gnu-x86_64.tar.gz

EXPOSE 8091
