# BitTensor: Mnist Example

## Run locally
1. Install python

```
# Install python dependencies
pip install -r requirments.txt

# Install protocol.
pip install -e .

# Run server 1.
python examples/mnist/main.py --port=8120

# Run server 2.
python examples/mnist/main.py --bootstrap='0.0.0.0:8120'
```

## Run remote on Digital Ocean
1. Install Docker

1. Install Docker Machine

```
# Get API-token on Digital Ocean as $TOKEN
$ TOKEN=(your api token)

# Create a remote instance.
$ docker-machine create --driver digitalocean --digitalocean-size s-4vcpu-8gb --digitalocean-access-token ${TOKEN} opentensor

# Run install 1.
$ eval $(docker-machine env opentensor)

# Build the image.
$ docker build -t opentensor .

# Run the neuron.
$ docker run -it -p 8888:8888 opentensor examples/mnist/main.py --port = 8765 --remoteip = ipaddress1



