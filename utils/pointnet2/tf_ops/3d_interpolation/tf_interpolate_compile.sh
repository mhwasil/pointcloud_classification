#!/bin/bash

# Get TF variables
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

echo $TF_INC
echo $TF_LIB
# TF1.2
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.14.1 build from source
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so_bitbots.so -shared -fPIC -I $TF_INC -I /usr/local/cuda-9.2/include -I $TF_INC/external/nsync/public -lcudart -L /usr/local/cuda-9.2/lib64/ -L $TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=1