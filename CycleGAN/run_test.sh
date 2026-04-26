#!/bin/bash
#SBATCH --job-name=gan_inference
#SBATCH --output=logs/%j_gan_test.out
#SBATCH --error=logs/%j_gan_test.err
#SBATCH --partition=3080ti-nipeng
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=26

source ~/.bashrc

# 1. 环境准备
source /home/yzy0322/anaconda3/etc/profile.d/conda.sh
conda activate case9

# 2. 【已彻底修正】进入真实的 CycleGAN 目录
cd /home/yzy0322/case9/CycleGAN/

# ==========================================
# 3. 【已彻底修正】运行测试命令
# ==========================================
# --dataroot: 指向外层的 datasets 文件夹 (脚本会自动进去找 testA)
# --name: 精准匹配您的权重文件夹名称 landscape_to_ink_gan
python test.py \
    --dataroot ../datasets \
    --name landscape_to_ink_gan \
    --model test \
    --no_dropout \
    --results_dir results \
    --num_test 50 \
    --load_size 256 \
    --crop_size 256