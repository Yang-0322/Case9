import os
import sys

# ================= 0. 物理环境隔离与探针 =================
print("===== 探针1：Python 进程启动，强制断网 =====", flush=True)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

print("===== 探针2：开始导入 PyTorch 基础库 =====", flush=True)
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from transformers import Adafactor

# 引入官方混合精度工具 (新版 API)
from torch.amp import autocast, GradScaler

print("===== 探针3：开始导入 Stable Diffusion 框架库 =====", flush=True)
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

print("===== 探针4：所有库导入完毕！ =====", flush=True)

# ================= 1. 基础配置 =================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
# 神经风格迁移的标准操作：把 Style 权重拉到巨大！
STYLE_WEIGHT = 10000.0   # 放大一万倍，甚至可以试着加到 100000.0
CONTENT_WEIGHT = 10.0    # 同样等比例放大一点
NOISE_WEIGHT = 1.0


def gram_matrix(input_tensor):
    a, b, c, d = input_tensor.size()
    features = input_tensor.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


# ================= 2. 数据集加载器 =================
class LandscapeInkDataset(Dataset):
    def __init__(self, content_dir, style_dir, size=256):
        self.content_paths = [os.path.join(content_dir, f) for f in os.listdir(content_dir) if
                              f.endswith(('.png', '.jpg'))]
        self.style_paths = [os.path.join(style_dir, f) for f in os.listdir(style_dir) if f.endswith(('.png', '.jpg'))]
        self.transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return min(len(self.content_paths), len(self.style_paths))

    def __getitem__(self, idx):
        content_img = self.transform(Image.open(self.content_paths[idx]).convert("RGB"))
        style_img = self.transform(
            Image.open(self.style_paths[torch.randint(0, len(self.style_paths), (1,)).item()]).convert("RGB"))
        return content_img, style_img


# ================= 3. 核心训练逻辑 =================
def main():
    print("===== 探针5：进入 main，准备加载 VGG 特征提取器 =====", flush=True)
    vgg_full = models.vgg16(weights=None)
    vgg_full.load_state_dict(torch.load('/home/yzy0322/case9/vgg16-397923af.pth', weights_only=False))
    vgg = vgg_full.features.to(DEVICE).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    print("===== 探针6：准备初始化数据集 =====", flush=True)
    dataset = LandscapeInkDataset("../dataset/trainA", "../dataset/trainB")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print("===== 探针7：正在实例化 Stable Diffusion 结构 =====", flush=True)
    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")

    if hasattr(config.model.params, 'unet_config'):
        config.model.params.unet_config.params.use_checkpoint = False

    model = instantiate_from_config(config.model)

    print("===== 探针8：准备读取 4GB 预训练权重 =====", flush=True)
    model.load_state_dict(torch.load("models/ldm/stable-diffusion-v1/model.ckpt", weights_only=False)["state_dict"],
                          strict=False)

    print("===== 探针9：权重加载成功，推入显卡 =====", flush=True)
    model.to(DEVICE)
    model.first_stage_model.eval()
    model.cond_stage_model.eval()

    print("===== 探针9.5：预计算文本条件并卸载 CLIP =====", flush=True)
    with torch.no_grad():
        c_base = model.get_learned_conditioning([""]).detach()
    model.cond_stage_model.to("cpu")
    torch.cuda.empty_cache()

    for name, module in model.named_modules():
        if hasattr(module, 'use_checkpoint'):
            module.use_checkpoint = False
        if hasattr(module, 'checkpoint'):
            module.checkpoint = False

    # 【注意】：这里已经彻底去掉了 model.half() 的硬转换，回归官方优雅模式！

    if torch.cuda.device_count() > 1:
        print(f"===== 探针9.8：检测到 {torch.cuda.device_count()} 张显卡，启动 DataParallel 并行训练！ =====", flush=True)
        model.model.diffusion_model = torch.nn.DataParallel(model.model.diffusion_model)

    optimizer = Adafactor(
        model.model.diffusion_model.parameters(),
        lr=1e-5,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False
    )

    # 【恢复防 NaN 神器】：加入新版 GradScaler
    scaler = GradScaler('cuda')

    print("===== 探针10：🚀 正式开始训练！ 🚀 =====", flush=True)
    for epoch in range(EPOCHS):
        for step, (content_img, style_img) in enumerate(dataloader):
            content_img, style_img = content_img.to(DEVICE), style_img.to(DEVICE)

            torch.cuda.empty_cache()
            optimizer.zero_grad()

            with torch.no_grad():
                z_content = model.get_first_stage_encoding(model.encode_first_stage(content_img))
                c = c_base.repeat(content_img.shape[0], 1, 1)

                # 提前算出 target 特征，并转为 float 防 NaN
                content_features_target = vgg(content_img.float())
                style_features_target = vgg(style_img.float())

            t = torch.randint(0, model.num_timesteps, (z_content.shape[0],), device=DEVICE).long()
            noise = torch.randn_like(z_content)

            # 使用官方 autocast，仅在需要的地方自动降精度
            with autocast('cuda'):
                z_noisy = model.q_sample(x_start=z_content, t=t, noise=noise)
                # 传入正常变量，让 autocast 自动决定精度
                noise_pred = model.apply_model(z_noisy, t, c)

                loss_noise = F.mse_loss(noise_pred, noise)

                pred_z0 = (z_noisy - (1 - model.alphas_cumprod[t].view(-1, 1, 1, 1)).sqrt() * noise_pred) / \
                          model.alphas_cumprod[t].view(-1, 1, 1, 1).sqrt()
                pred_x0 = model.decode_first_stage(pred_z0)

            # 【核心保护】：出了 autocast 后，强行转回 float32 再送给 VGG
            # VGG 在 FP16 下必出 NaN，必须这样保护！
            pred_features = vgg(pred_x0.float())

            # 损失计算全在 float32 下进行
            loss_content = F.mse_loss(pred_features, content_features_target.float())
            loss_style = F.mse_loss(gram_matrix(pred_features), gram_matrix(style_features_target.float()))

            total_loss = NOISE_WEIGHT * loss_noise + CONTENT_WEIGHT * loss_content + STYLE_WEIGHT * loss_style

            print_loss_n = loss_noise.item()
            print_loss_s = loss_style.item()

            # 使用 scaler 安全反向传播
            scaler.scale(total_loss).backward()

            del z_content, c, noise, z_noisy, noise_pred, pred_z0, pred_x0, pred_features
            del content_features_target, style_features_target
            del loss_noise, loss_content, loss_style, total_loss
            torch.cuda.empty_cache()

            # 安全步进并更新缩放器
            scaler.step(optimizer)
            scaler.update()

            if step % 10 == 0:
                print(
                    f"Epoch: {epoch + 1}/{EPOCHS} | Step: {step} | Loss_Noise: {print_loss_n:.4f} | Loss_Style: {print_loss_s:.4e}",
                    flush=True)

        unet_to_save = model.model.diffusion_model.module if hasattr(model.model.diffusion_model,
                                                                     'module') else model.model.diffusion_model
        torch.save(unet_to_save.state_dict(), f"unet_epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    main()