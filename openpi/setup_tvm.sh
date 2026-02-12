#!/bin/bash
export TVM_HOME=/workspace/external/tvm
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
export LD_LIBRARY_PATH=$TVM_HOME/build:$TVM_HOME/build/lib:$TVM_HOME/build/3rdparty/cutlass_fpA_intB_gemm/cutlass_kernels:$TVM_HOME/build/3rdparty/libflash_attn/src:$LD_LIBRARY_PATH
echo 'TVM environment configured'
