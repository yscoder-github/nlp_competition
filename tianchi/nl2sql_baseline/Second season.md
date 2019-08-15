#### 第二赛季开发环境配置

1. 安装nvidia-docker  [参考](https://github.com/NVIDIA/nvidia-docker)

``` shell 
# Add the package repositories
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docker
``` 
测试是否安装成功
``` shell 
#### Test nvidia-smi with the latest official CUDA image
$ docker run --gpus all nvidia/cuda:9.0-base nvidia-smi

# Start a GPU enabled container on two GPUs
$ docker run --gpus 2 nvidia/cuda:9.0-base nvidia-smi

# Starting a GPU enabled container on specific GPUs
$ docker run --gpus '"device=1,2"' nvidia/cuda:9.0-base nvidia-smi
$ docker run --gpus '"device=UUID-ABCDEF,1'" nvidia/cuda:9.0-base nvidia-smi

# Specifying a capability (graphics, compute, ...) for my container
# Note this is rarely if ever used this way
$ docker run --gpus all,capabilities=utility nvidia/cuda:9.0-base nvidia-smi
``` 

2. 下载官方镜像  [地址](https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586973.0.0.4e7340bfbAe15C&postId=67720)

    这里选择其中的 registry.cn-shanghai.aliyuncs.com/tcc-public/keras:latest-cuda10.0-py3    
    执行命令如下: 
       
    ``` shell 
    sudo docker pull registry.cn-shanghai.aliyuncs.com/tcc-public/keras:latest-cuda10.0-py3  
    ``` 

3. 进入并执行镜像

``` shell 
sudo docker run --gpus all  -i -t  registry.cn-shanghai.aliyuncs.com/tcc-public/keras:latest-cuda10.0-py3 /bin/bash 
``` 
当前镜像的各个核心组件的版本如下: 
  - tensorflow  1.13.1
  - keras 2.2.4
  - python3.5 

