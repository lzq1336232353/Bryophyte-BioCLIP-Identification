import torch
import torch.nn as nn
import torch.optim as optim
from model import MOSSNet  # 导入我们通关的骨架网络
from dataset import get_moss_dataloaders  # 导入刚刚写好的数据器

def train_model():
    # 1. 自动锁定硬件设备 (优先调用你本地配置好的英伟达 CUDA 独显)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 正式开启本地苔藓分类训练流！当前硬件设备: {device}")
    
    # 2. 核心参数配置 (针对你目前的7个物种测试集)
    NUM_CLASSES = 7 
    BATCH_SIZE = 16
    EPOCHS = 15  # 本地初步测试先跑15轮查看收敛态势
    
    # 3. 加载数据流与初始化模型（修复了缩进问题）
    train_loader, val_loader, test_loader = get_moss_dataloaders(batch_size=BATCH_SIZE)
    model = MOSSNet(num_classes=NUM_CLASSES).to(device)
    
    # 4. 定义学术通用的交叉熵损失函数和优化器
    # 开启 0.1 的标签平滑
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # 🌟 安全升级 1：针对小规模数据集，将学习率降低一个数量级（1e-3 -> 1e-4）进行平稳降温
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    # 🌟 新增：余弦退火调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    
    best_acc = 0.0
    
    # 5. 核心训练与验证大循环
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # 训练子循环
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 🌟 安全升级 2：强行卡死反向传播的梯度上限（1.0），从数学根源上彻底封印梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total_train
        train_acc = 100.0 * correct_train / total_train
        
        # 验证子循环 (不计算梯度，全力测试精度)
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
        
        val_acc = 100.0 * correct_val / total_val
        
        # 打印当前轮次的宏观学术战报
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] -> Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # 6. 如果当前精度的 Val Acc 创下新高，自动将大脑神经网络权重保存到物理硬盘
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_moss_model.pth")
            print(f"    ✨ 监测到精度提升！已将当前最优模型保存至: best_moss_model.pth")
            
        # 🌟 完美嵌入：在每一个 epoch 结束的最后一行，让学习率调度器向前走一步
        scheduler.step()
            
    print(f"\n🎉 训练全盘圆满结束！在 4050 独显加速下达到的最高验证准确率为: {best_acc:.2f}%")

if __name__ == "__main__":
    train_model()