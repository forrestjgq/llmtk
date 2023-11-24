# Used by docker in building

set -ex
llmtk=$(dirname $(dirname $(realpath $0)))
trtllm=$llmtk/tensorrt_llm

if [ ! -z $http_proxy ]; then
   rm -f ~/.wgetrc
   echo "http_proxy=$http_proxy" >  /root/.wgetrc
   echo "https_proxy=$http_proxy" >>  /root/.wgetrc
   echo "use_proxy=yes" >>  /root/.wgetrc
fi

# install dependency of trtllm
pushd $trtllm/docker/common/
bash install_base.sh
bash install_cmake.sh 
bash install_tensorrt.sh 
bash install_polygraphy.sh 
bash install_pytorch.sh pypi
popd


# install ammo
if [ ! -e $llmtk/nvidia_ammo-0.3.0 ]; then
   if [ ! -e $llmtk/nvidia_ammo-0.3.0.tar.gz ]; then
      # Download and install the AMMO package/home/gqjiang/src/backend/ci from the DevZone.
      wget https://developer.nvidia.com/downloads/assets/cuda/files/nvidia-ammo/nvidia_ammo-0.3.0.tar.gz -O $llmtk/nvidia_ammo-0.3.0.tar.gz
      tar -xzf nvidia_ammo-0.3.0.tar.gz
   fi
fi

# Obtain the cuda version from the system. Assuming nvcc is available in path.
cuda_version=$(nvcc --version | grep 'release' | awk '{print $6}' | awk -F'[V.]' '{print $2$3}')
# Obtain the python version from the system.
python_version=$(python3 --version 2>&1 | awk '{print $2}' | awk -F. '{print $1$2}')
pip install nvidia_ammo-0.3.0/nvidia_ammo-0.3.0+cu$cuda_version-cp$python_version-cp$python_version-linux_x86_64.whl
rm -rf nvidia_ammo-0.3.0
# Install the additional requirements
pushd $trtllm/examples/quantization
pip install cython # extra dependancy
pip install -r requirements.txt
popd

# for mistral model running
pip install --upgrade flash-attn
# for mistral model building
pip install pynvml

rm -f ~/.wgetrc

