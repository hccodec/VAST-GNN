# 20240903 环境配置 备忘

conda create -n epilearn python=3.10 -y
ca epilearn
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install torch_geometric
# Linux
# pip install https://data.pyg.org/whl/torch-2.4.0%2Bcu124/torch_scatter-2.1.2%2Bpt24cu124-cp310-cp310-linux_x86_64.whl
# Windows
https://data.pyg.org/whl/torch-2.4.0%2Bcu124/torch_scatter-2.1.2%2Bpt24cu124-cp310-cp310-win_amd64.whl

pip install epilearn
add_kernel epilearn
