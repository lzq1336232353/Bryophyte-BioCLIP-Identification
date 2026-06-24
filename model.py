import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 论文 3.2 节：CPE 卷积分块嵌入模块
class ConvolutionalPatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=4)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

# 2. 论文 3.4 节：DCEA 空洞卷积增强注意力块
class DCEABlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

        self.k_conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, dilation=1, groups=dim)
        self.k_conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim)
        
        self.v_conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim)
        self.v_conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, dilation=1, groups=dim)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        
        x_flat = x.permute(0, 2, 3, 1).reshape(B, N, C)
        
        # 此时 head_dim 能够完美整除，绝不会再报张量形状错误
        q = self.q_proj(x_flat).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        k_spatial = self.k_proj(x_flat).reshape(B, H, W, C).permute(0, 3, 1, 2)
        v_spatial = self.v_proj(x_flat).reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        k_feat = self.k_conv1(k_spatial) + self.k_conv2(k_spatial)
        v_feat = self.v_conv1(v_spatial) + self.v_conv2(v_spatial)
        
        k = k_feat.permute(0, 2, 3, 1).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v_feat.permute(0, 2, 3, 1).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out

# 3. 论文 3.3 节：DBMS 双分支多尺度结构模块
class DBMSModule(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
        self.attn_branch = DCEABlock(dim, num_heads)
        
        self.r1 = nn.Parameter(torch.ones(1) * 0.5)
        self.r2 = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        return x + self.r1 * self.attn_branch(x) + self.r2 * self.conv_branch(x)

# 4. MOSSNet 全流程主干金字塔结构（加入 Dropout 阻断过拟合）
class MOSSNet(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.cpe = ConvolutionalPatchEmbedding(in_channels=3, out_channels=64)
        
        # Stage 1: dim=64, num_heads=4 -> head_dim = 16
        self.stage1 = nn.Sequential(DBMSModule(64, num_heads=4))
        self.merge1 = nn.Conv2d(64, 128, kernel_size=2, stride=2)
        
        # Stage 2: dim=128, num_heads=8 -> head_dim = 16
        self.stage2 = nn.Sequential(*[DBMSModule(128, num_heads=8) for _ in range(2)])
        self.merge2 = nn.Conv2d(128, 288, kernel_size=2, stride=2) # 256微调为288
        
        # Stage 3: dim=288, num_heads=12 -> head_dim = 24
        self.stage3 = nn.Sequential(*[DBMSModule(288, num_heads=12) for _ in range(8)])
        self.merge3 = nn.Conv2d(288, 512, kernel_size=2, stride=2)
        
        # Stage 4: dim=512, num_heads=16 -> head_dim = 32
        self.stage4 = nn.Sequential(*[DBMSModule(512, num_heads=16) for _ in range(2)])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 🌟 核心改进：注入 Dropout(0.3)，在训练时随机切断 30% 特征，强迫网络协同举一反三
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.Flatten(),
            nn.Dropout(p=0.3),  
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.cpe(x)
        x = self.stage1(x)
        x = self.merge1(x)
        x = self.stage2(x)
        x = self.merge2(x)
        x = self.stage3(x)
        x = self.merge3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        return self.classifier(x)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在使用硬件设备: {device}")
    
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    model = MOSSNet(num_classes=7).to(device)  # 本地测试对应你的7个苔藓物种
    
    output = model(dummy_input)
    print("✨ 模型前向传播成功！")
    print("🌟 矩阵最终输出形状为:", output.shape)