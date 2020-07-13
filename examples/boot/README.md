# Bootstrap node

## Run remote on Digital Ocean
1. Install Docker
1. Install Docker Machine

```
# Get API-token on Digital Ocean as $TOKEN
$ TOKEN=(your api token)

# Create a remote instance
$ docker-machine create --driver digitalocean --digitalocean-size s-4vcpu-8gb --digitalocean-access-token ${TOKEN} bootstrap

# Switch to instance context
$ eval $(docker-machine env bootstrap)

# Build the docker image.
$ docker build -t opentensor .

# Run the server
$ docker run -it -p 8888:8888 opentensor examples/bootstrap/main.py

# Get instance ip address
$ boot_address=$(eval docker-machine ip bootstrap)
```




