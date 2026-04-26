#!/bin/bash
#SBATCH --job-name=sd_ink_style          # 任务名称
#SBATCH --output=logs/%j_sd_train.out
#SBATCH --error=logs/%j_sd_train.err
#SBATCH --partition=3080ti-nipeng       # 重要：请修改为您服务器真实的GPU分区名
#SBATCH --nodes=1                       # 申请节点数
#SBATCH --ntasks-per-node=1             # 每个节点的任务数
#SBATCH --cpus-per-task=26              # 每个任务分配的 CPU 核心数
#SBATCH --gres=gpu:3080ti:1             # 申请 1 块 GPU

source ~/.bashrc

# 1. 初始化 Conda 环境
# 这里的路径通常是默认的，如果您的 conda 安装在非标准路径请修改
source /home/yzy0322/anaconda3/etc/profile.d/conda.sh
conda activate case9

# 2. 进入 CycleGAN 代码目录
# 脚本假设从 case9 根目录提交
cd /home/yzy0322/case9/stable-diffusion

# --- 新增：查看 GPU 状态 ---
echo "========= GPU 资源检查 ========="
# 1. 查看 SLURM 分配给该任务的 GPU ID
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
# 2. 查看当前环境可见的 GPU 索引 (通常由 SLURM 自动设置)
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# 3. 输出完整的 GPU 硬件状态信息
nvidia-smi
echo "================================"

# 4. 执行训练命令
python -u train_style_diffusion.py