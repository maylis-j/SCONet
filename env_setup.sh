#!/bin/bash

conda create --name sconet
conda activate sconet

# install most packages with pip
pip install -r requirements.txt

# install torch-scatter separately to avoid conflicts
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
