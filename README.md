# Paged Attention: Efficient Memory Management for LLMs
CUDA implementation of PagedAttention ([Kwon et al., SOSP 2023](https://arxiv.org/abs/2309.06180)) benchmarked against naive attention.

## how to run
### 1. ssh into cluster
connect to vpn.usc.edu
```bash
ssh <username>@discovery.usc.edu
```
### 2. request gpu node
Smaller Request:
```bash
srun --account=snazaria_1817 --partition=gpu --gpus=1 --cpus-per-task=2 --mem=8G --pty bash
```
Larger Request:
``` bash
srun --account=snazaria_1817 --partition=gpu --gpus=a40:1 --cpus-per-task=8 --mem=32G --pty bash
```
#### 3. clone this
```bash
git clone https://github.com/emaanh/paged-attention.git
cd paged-attention
git switch emaan/testing_cuda
```

### 4. load modules & build
```bash
bash setup.sh
bash build.sh
```

### 5. run
```bash 
./build/test
```