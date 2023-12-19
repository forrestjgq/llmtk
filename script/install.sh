# Used by docker in building

set -ex
llmtk=$(dirname $(dirname $(realpath $0)))
trtllm=$llmtk/tensorrt_llm
if [ $# < 1 ]; then
   echo "expect a root path"
   exit 1
fi
root=$1

if [ ! -z $http_proxy ]; then
   rm -f /root/.wgetrc
   echo "http_proxy=$http_proxy" >  /root/.wgetrc
   echo "https_proxy=$http_proxy" >>  /root/.wgetrc
   echo "use_proxy=yes" >>  /root/.wgetrc
fi

pushd $root

# install dependency of trtllm
pushd $root/trtllm
bash install_tensorrt.sh 
bash install_polygraphy.sh 
bash install_cmake.sh 
bash install_mpi4py.sh 
# backend does not require this
#bash install_base.sh
bash install_pytorch.sh pypi
popd



# Obtain the python version from the system.
python_version=$(python3 --version 2>&1 | awk '{print $2}' | awk -F. '{print $1$2}')
# Download and install the AMMO package from the DevZone.
wget https://developer.nvidia.com/downloads/assets/cuda/files/nvidia-ammo/nvidia_ammo-0.5.0.tar.gz
tar -xzf nvidia_ammo-0.5.0.tar.gz
pip install nvidia_ammo-0.5.0/nvidia_ammo-0.5.0-cp$python_version-cp$python_version-linux_x86_64.whl
rm -rf $llmtk/nvidia_ammo*

# Install the additional requirements
pushd $root/quantization
pip install cython # extra dependancy
pip install -r requirements.txt
popd

# for mistral model running
pip install --upgrade flash-attn
# for mistral model building
pip install pynvml

rm -f ~/.wgetrc

popd