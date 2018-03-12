FROM ubuntu:16.04

MAINTAINER Binoy Das <binoyd@amazon.com>

ARG PYTHON_VERSION=3.6
ARG CONDA_PYTHON_VERSION=3
ARG CONDA_VERSION=4.4.10
ARG CONDA_DIR=/opt/conda

RUN apt-get update && \
  apt-get install -y --no-install-recommends apt-utils git wget nginx ca-certificates curl vim

RUN apt-get update && \
  apt-get install -y --no-install-recommends python3-dev python3-pip

RUN apt-get update && \
  apt-get install -y --no-install-recommends libfreetype6-dev libpng12-dev libhdf5-dev  && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install setuptools
RUN pip3 install tensorflow
RUN pip3 install numpy scipy pandas sklearn seaborn pyyaml h5py
RUN pip3 install keras --no-deps
RUN pip3 install flask gevent gunicorn

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY lstm /opt/program
WORKDIR /opt/program
