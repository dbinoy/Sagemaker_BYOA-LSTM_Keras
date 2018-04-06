#FROM ubuntu:16.04
FROM nvidia/cuda:9.0-base-ubuntu16.04
MAINTAINER Binoy Das <binoyd@amazon.com>

ARG PYTHON_VERSION=3.6
ARG CONDA_PYTHON_VERSION=3
ARG CONDA_VERSION=4.4.10
ARG CONDA_DIR=/opt/conda

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-9-0 \
        cuda-cublas-9-0 \
        cuda-cufft-9-0 \
        cuda-curand-9-0 \
        cuda-cusolver-9-0 \
        cuda-cusparse-9-0 \
        curl \
        libcudnn7=7.0.5.15-1+cuda9.0 \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        libhdf5-dev \
        pkg-config \
        python3-dev \
        python3-pip \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
  apt-get install -y --no-install-recommends python3-dev python3-pip

RUN apt-get update && \
  apt-get install -y --no-install-recommends libfreetype6-dev libpng12-dev libhdf5-dev  && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
      apt-utils \
      wget \
      nginx \
      ca-certificates \
      && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 --no-cache-dir install \
        setuptools \
        tensorflow \
        numpy \
        scipy \
        pandas \
        sklearn \
        seaborn \
        pyyaml \
        h5py
RUN pip3 --no-cache-dir install keras --no-deps
RUN pip3 --no-cache-dir install flask gevent gunicorn

RUN pip3 --no-cache-dir install \
    http://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.0.0-cp27-none-linux_x86_64.whl

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

COPY lstm /opt/program
WORKDIR /opt/program
