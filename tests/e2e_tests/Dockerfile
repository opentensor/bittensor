FROM ubuntu:latest

# Install Git
RUN apt-get update && \
    apt-get install -y git

# Install Python and dependencies
RUN apt-get install -y python3 python3-pip python3-venv curl clang libssl-dev llvm libudev-dev protobuf-compiler

# Install Rustup non-interactively
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path

# Set environment variables
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CARGO_TERM_COLOR always
ENV RUST_BACKTRACE full

# Install Rust toolchain and components
RUN /root/.cargo/bin/rustup toolchain install nightly-2024-03-05 && \
    /root/.cargo/bin/rustup target add wasm32-unknown-unknown --toolchain nightly-2024-03-05 && \
    /root/.cargo/bin/rustup component add rust-src --toolchain nightly-2024-03-05

# Create a virtual environment for Python packages
RUN python3 -m venv /opt/venv

# Activate virtual environment and install pytest
COPY ../requirements/prod.txt /app/prod.txt
COPY ../requirements/dev.txt /app/dev.txt
RUN /opt/venv/bin/pip install --upgrade pip
RUN pip3 install -r /app/prod.txt
RUN pip3 install -r /app/dev.txt

# Set environment variables to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Set up your repository and clone subtensor repo
WORKDIR /app
COPY . /app
RUN git clone https://github.com/opentensor/subtensor.git

# Add any additional setup steps here

# Set the entry point for running tests
ENTRYPOINT ["pytest", "."]
