#/bin/bash
/usr/local/cuda-9.2/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.2
#g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /usr/local/lib/python3.5/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.14.1 using tf built from source
g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so_hk.so -shared -fPIC -I /home/mwasil2s/anaconda3/envs/tf-source/lib/python3.6/site-packages/tensorflow/include -I /usr/local/cuda-9.2/include -I /home/mwasil2s/anaconda3/envs/tf-source/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.2/lib64 -L /home/mwasil2s/anaconda3/envs/tf-source/lib/python3.6/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
