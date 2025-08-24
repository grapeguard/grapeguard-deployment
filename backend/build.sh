#!/bin/bash
# Install system dependencies for OpenCV and other libraries
apt-get update
apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 libgl1

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt