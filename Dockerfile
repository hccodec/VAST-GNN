# 使用 Miniconda 作为基础镜像，更轻量
FROM continuumio/miniconda3:latest

# 设置构建参数
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000
ARG USER_PASSWORD=userpass # 强烈建议在生产环境中使用更安全的方式管理密码
ARG MY_ENV_NAME=lp         # Conda 环境的名称
ARG PYTHON_VERSION=3.8     # Python 版本

# 1. 安装必要的系统依赖 (以 root 用户执行)
# 包括 sudo 和一些基础工具，以及用于 Conda 包下载的 ca-certificates
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        bzip2 \
        sudo \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# 2. 复制 requirements.txt 文件到临时位置
# 假设 requirements.txt 在 Dockerfile 同级目录
COPY requirements.txt /tmp/requirements.txt

# 3. 创建 Conda 环境并安装 PyTorch/CUDA 和 Pip 依赖 (以 root 用户执行)
# 所有 Conda 和 Pip 安装步骤都在一个 RUN 命令中，减少 Docker 层数并利用缓存
RUN /opt/conda/bin/conda create -n $MY_ENV_NAME python=$PYTHON_VERSION -y && \
    /opt/conda/bin/conda run -n $MY_ENV_NAME conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y && \
    /opt/conda/bin/conda run -n $MY_ENV_NAME pip install -r /tmp/requirements.txt && \
    /opt/conda/bin/conda clean --all -y && \
    rm /tmp/requirements.txt

# 4. 创建一个普通用户并设置sudo权限和密码 (以 root 用户执行)
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME:$USER_PASSWORD" | chpasswd \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# 切换到普通用户，后续操作以该用户身份进行
USER $USERNAME
WORKDIR /home/$USERNAME

# 5. 配置用户的 .bashrc，使其默认激活 Conda 环境
# 使用 /opt/conda/bin/conda init bash 来初始化 bash
# 然后在 .bashrc 中添加激活命令
RUN /opt/conda/bin/conda init bash && \
    echo "conda activate $MY_ENV_NAME" >> ~/.bashrc

# 6. 设置默认的 Shell 和 CMD
CMD ["bash", "-l"]