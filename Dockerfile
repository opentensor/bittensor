from pytorch/pytorch

RUN apt-get update && apt-get install make

COPY . .
RUN pip install -e .


