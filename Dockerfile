# 使用 PyTorch 官方镜像（CUDA 12.1）
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 复制 Conda 环境文件
COPY environment.yml .

# 1. 更新 Conda 基础环境（避免重复安装 PyTorch/CUDA）
# 2. 仅安装 pip 部分的依赖（因为 Conda 已提供 PyTorch）
RUN conda env update -n base -f environment.yml && \
    conda clean -afy && \
    pip install --no-cache-dir -r <(grep -E "^    - " environment.yml | sed 's/    - //g')

# 验证 GPU 是否可用
RUN python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"

# 设置非 root 用户（可选）
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# 默认命令（按需修改）
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]