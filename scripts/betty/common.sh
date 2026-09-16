#!/usr/bin/env bash

# Shared Betty/Pyxis runtime. Override these variables before sbatch when the
# project is moved to a different allocation or directory.
export ZOO_BETTY_STORAGE_ROOT="${ZOO_BETTY_STORAGE_ROOT:-/vast/projects/ajw/wharton/jpipping}"
export ZOO_BETTY_PROJECT_HOST="${ZOO_BETTY_PROJECT_HOST:-${ZOO_BETTY_STORAGE_ROOT}/zoo}"
export ZOO_BETTY_MOUNT="${ZOO_BETTY_MOUNT:-/workspace}"
export ZOO_BETTY_PROJECT="${ZOO_BETTY_PROJECT:-${ZOO_BETTY_MOUNT}/zoo}"
export ZOO_BETTY_OVERLAY="${ZOO_BETTY_OVERLAY:-${ZOO_BETTY_MOUNT}/envs/zoo-ngc25.02-overlay}"
export ZOO_BETTY_IMAGE="${ZOO_BETTY_IMAGE:-docker://nvcr.io/nvidia/tensorflow:25.02-tf2-py3}"

export PYTHONPATH="${ZOO_BETTY_OVERLAY}:${ZOO_BETTY_PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-20260817}"
export TF_DETERMINISTIC_OPS="${TF_DETERMINISTIC_OPS:-1}"
# cuBLAS may select a different internal workspace/implementation across
# streams unless CUDA receives one of its documented reproducible workspace
# layouts before the first library handle is created.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
# NGC sets legacy tf_keras by default. This project targets Keras 3, and the
# legacy initializer is incompatible with the image's Python 3.12 runtime.
export TF_USE_LEGACY_KERAS=0
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/zoo-matplotlib}"
