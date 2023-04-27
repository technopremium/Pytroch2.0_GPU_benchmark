# Unleashing the Power of PyTorch 2.0: A Revolution in Deep Learning Performance

Welcome to our GPU performance benchmark project! In this repository, we aim to showcase the remarkable improvements in training time, train samples per second, and accuracy using PyTorch 2.0. This project is inspired by Phil Schmid's [developer blog post](https://www.philschmid.de/getting-started-pytorch-2-0-transformers) and the official [PyTorch 2.0 announcement](https://pytorch.org/get-started/pytorch-2.0/#overview).

To get started, follow these simple steps:

## 1. Clone the repository and set up the environment
```bash
git clone https://github.com/technopremium/Pytroch2.0_GPU_benchmark.git
cd Pytroch2.0_GPU_benchmark
conda env create -f pytorch2.0_environment.yml
conda activate pytorch2.0
```
## 2. Run Jupyter Lab for an interactive experience
```bash
jupyter lab
```
## 3. Explore the trainer notebook
Once inside Jupyter Lab, open the trainer.ipynb notebook and start running the cells to witness the power of PyTorch 2.0 in action!


# Docker installation 
```
# To build the container
docker build -t pytorch2-gpu-benchmark .
# To launch the container
docker run --gpus all -it -it --shm-size=1g --ulimit memlock=-1  --ulimit stack=67108864 --rm pytorch2-gpu-benchmark
```

# Important note

This code only runs in single GPUs, if your system has more than 1 gpu then set the container to run on a specific gpu, example: docker run --gpus '"device=0"' -it --rm pytorch2-gpu-benchmark. 

For more information and a detailed walkthrough, check out [our comprehensive blog post](https://bizonbizon.notion.site/Unleash-the-Power-of-PyTorch-2-0-A-Revolution-in-Deep-Learning-Performance-e0740febe9364abab8899dbe3a6021a6).

Bizon 2023 - Ruben Roberto Copyright

