#!/bin/bash
#SBATCH --job-name=cyclegan_ink          # 任务名称
#SBATCH --output=logs/%j_train.out      # 标准输出日志 (%j 会替换为任务ID)
#SBATCH --error=logs/%j_train.err       # 错误日志
#SBATCH --partition=3080ti-nipeng       # 重要：请修改为您服务器真实的GPU分区名
#SBATCH --nodes=1                       # 申请节点数
#SBATCH --ntasks-per-node=1             # 每个节点的任务数
#SBATCH --cpus-per-task=78               # 每个任务分配的 CPU 核心数
#SBATCH --gres=gpu:3080ti:1              # 申请 1 块 GPU

export PATH=/home/yzy0322/anaconda3/bin:/home/yzy0322/anaconda3/condabin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/opt/app/spack/bin:/opt/app/spack/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/yzy0322/.local/bin:/home/yzy0322/bin

# 1. 初始化 Conda 环境
# 这里的路径通常是默认的，如果您的 conda 安装在非标准路径请修改
source /home/yzy0322/anaconda3/etc/profile.d/conda.sh
conda activate case9

# 2. 进入 CycleGAN 代码目录
# 脚本假设从 case9 根目录提交
cd /home/yzy0322/case9/CycleGAN

# 3. 自动创建日志目录
mkdir -p logs

# 4. 执行训练命令
# 注意：此时已在 CycleGAN 目录内，因此 dataset 路径使用 ../dataset
python train.py \
  --dataroot /home/yzy0322/case9/dataset \
  --name landscape_to_ink_gan_v4 \
  --model cycle_gan \
  --netG unet_256 \
  --norm batch \
  --batch_size 4 \
  --n_epochs 25 \
  --n_epochs_decay 25