# after activate, install cuda and cudnn
# conda install -c conda-forge cudatoolkit=11.3 cudnn=8.2

# install jax and jaxlib
pip install --upgrade "jax[cuda12]"
# pip install --upgrade jax==0.4.7 jaxlib==0.4.7+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# install requirements.txt
pip install -r requirements.txt
pip install pybullet h5py termcolor 
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

export http_proxy=http://192.168.32.11:18000
export https_proxy=http://192.168.32.11:18000
pip install git+https://github.com/nakamotoo/D4RL.git
export http_proxy=
export https_proxy=
pip install Cython==0.29.21

mv /opt/conda/envs/cal-ql/lib/libstdc++.so.6 /opt/conda/envs/cal-ql/lib/libstdc++.so.6.old
ln -s  /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/envs/cal-ql/lib/libstdc++.so.6