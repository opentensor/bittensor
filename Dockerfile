FROM nvidia/cuda:10.2-base

LABEL io.parity.image.authors="bittensor.com" \
	io.parity.image.vendor="Bittensor" \
	io.parity.image.title="bittensor/bittensor" \
	io.parity.image.description="Bittensor: Incentivized Peer to Peer Neural Networks" \
	io.parity.image.source="https://github.com/opentensor/bittensor.git" \
	io.parity.image.revision="${VCS_REF}" \
	io.parity.image.created="${BUILD_DATE}" \
	io.parity.image.documentation="https://opentensor.bittensor.io"

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip


# add Bittensor code to docker image
RUN mkdir /bittensor
COPY . /bittensor
