# Paged Attention: Efficient Memory Management for LLMs
CUDA implementation of PagedAttention ([Kwon et al., SOSP 2023](https://arxiv.org/abs/2309.06180)) benchmarked against naive attention.

## how to run
### 1. ssh into cluster
connect to vpn.usc.edu
```bash
ssh <username>@discovery.usc.edu
```
### 2. request gpu node
small request:
```bash
srun --account=snazaria_1817 --partition=gpu --gpus=1 --cpus-per-task=2 --mem=8G --pty bash
```
large request:
``` bash
srun --account=snazaria_1817 --partition=gpu --gpus=a40:1 --cpus-per-task=8 --mem=32G --pty bash
```
### 3. clone this
```bash
git clone https://github.com/emaanh/paged-attention.git
cd paged-attention
```

### 4. load modules & build
```bash
bash setup.sh
bash easy_build.sh
```

### 4.5 rebuilding 
if you have moved files or created new ones, update ```CMakeLists.txt```.
```bash
mkdir -p build
cd build
cmake ..
make
```
if you have modified files
```bash
cd build
make
```

### 5. run
```bash 
./build/test
```