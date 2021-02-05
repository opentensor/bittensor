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
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y apt-utils curl git cmake build-essential unzip python3.7 python3-pip python3.7-dev
RUN python3.7 -m pip install --upgrade pip

# add Bittensor code to docker image
RUN mkdir /bittensor
COPY . /bittensor

# Arrow installation prep
RUN apt-get install -y -V ca-certificates lsb-release wget
RUN wget https://apache.bintray.com/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-archive-keyring-latest-$(lsb_release --codename --short).deb
RUN apt-get install -y -V ./apache-arrow-archive-keyring-latest-$(lsb_release --codename --short).deb
RUN apt-get update
RUN apt-get install -y -V libarrow-dev 
RUN apt-get install -y -V libarrow-glib-dev 
RUN apt-get install -y -V libarrow-dataset-dev 
RUN apt-get install -y -V libarrow-flight-dev

WORKDIR /bittensor
RUN pip install --upgrade numpy pandas setuptools "tqdm>=4.27,<4.50.0"
RUN pip install -r requirements.txt
RUN pip install -e . 
RUN ./scripts/create_wallet.sh

EXPOSE 8091
VOLUME ["/bittensor"]
