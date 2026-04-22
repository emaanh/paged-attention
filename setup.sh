#!/bin/bash
set -e

module load cuda cmake
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH