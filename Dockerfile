FROM nvidia/cuda:11.2.1-base


LABEL bittensor.image.authors="bittensor.com" \
	bittensor.image.vendor="Bittensor" \
	bittensor.image.title="bittensor/bittensor" \
	bittensor.image.description="Bittensor: Incentivized Peer to Peer Neural Networks" \
	bittensor.image.source="https://github.com/opentensor/bittensor.git" \
	bittensor.image.revision="${VCS_REF}" \
	bittensor.image.created="${BUILD_DATE}" \
	bittensor.image.documentation="https://app.gitbook.com/@opentensor/s/bittensor/"

ARG DEBIAN_FRONTEND=noninteractive

# Nvidia Key Rotation
## https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
## Remove old Nvidia GPG key and source list
## Prevents apt from failing to find the key
RUN rm /etc/apt/sources.list.d/cuda.list \
	&& rm /etc/apt/sources.list.d/nvidia-ml.list 
RUN apt-key del 7fa2af80

# apt update and Install dependencies
RUN apt-get update && \
	apt-get install --no-install-recommends --no-install-suggests -y apt-utils curl git \
	cmake build-essential unzip python3-pip wget iproute2 software-properties-common

# Nvidia Key Rotation
## https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
## Add new Nvidia GPG key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get upgrade -y  
RUN apt-get install python3 python3-dev -y
RUN python3 -m pip install --upgrade pip

# add Bittensor code to docker image
RUN mkdir /bittensor
RUN mkdir /home/.bittensor
COPY . /bittensor

WORKDIR /bittensor
RUN pip install --upgrade numpy pandas setuptools "tqdm>=4.27,<4.50.0" wheel
RUN pip install -r requirements.txt
RUN pip install -e .

EXPOSE 8091
