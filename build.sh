#!/usr/bin/env bash
# Install system dependencies for dlib
apt-get update && apt-get install -y \
  build-essential \
  cmake \
  libboost-all-dev \
  libopenblas-dev \
  liblapack-dev \
  libx11-dev \
  libgtk-3-dev \
  libblas-dev \
  libsm6 \
  libxext6 \
  libxrender-dev
