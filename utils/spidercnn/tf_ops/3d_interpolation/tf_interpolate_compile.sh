# TF1.2
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.14.1 build from source
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so_hk.so -shared -fPIC -I /home/mwasil2s/anaconda3/envs/tf-source/lib/python3.6/site-packages/tensorflow/include -I /usr/local/cuda-9.2/include -I /home/mwasil2s/anaconda3/envs/tf-source/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.2/lib64/ -L /home/mwasil2s/anaconda3/envs/tf-source/lib/python3.6/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=1
