import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models  # 🌟 新增：导入 PyTorch 官方模型大仓库
from dataset import get_moss_dataloaders  

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 预训练微调流正式开启！当前硬件设备: {device}")
    
    NUM_CLASSES = 7 
    BATCH_SIZE = 16
    EPOCHS = 15  
    
    # 加载你已经人工裁剪好的 test1.2 数据集
    train_loader, val_loader, test_loader = get_moss_dataloaders(batch_size=BATCH_SIZE)
    
    # 🌟 核心改动：直接从官方仓库召唤自带千万级记忆的预训练 ResNet50
    # weights=models.ResNet50_Weights.DEFAULT 会自动为你下载最新最顶配的官方 ImageNet 权重
    print("🧠 正在下载/加载 ResNet50 官方预训练权重大脑...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # 🌟 核心改动：把 ResNet50 原本认 1000 种物体的最后一层分类输出（fc层），
    # 暴力替换为符合你本地 7 个物种的全新全连接层
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),  # 依然保留我们的防过拟合铠甲
        nn.Linear(in_features, NUM_CLASSES)
    )
    
    model = model.to(device)
    
    # 开启标签平滑
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 🌟 核心改动：由于模型已经极其聪明，我们将初始学习率降得更低（5e-5），以最优雅细致的步伐进行微调
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total_train
        train_acc = 100.0 * correct_train / total_train
        
        # 验证子循环
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
        
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] -> Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_resnet50.pth")
            print(f"    ✨ 监测到精度跨越！已将当前最优预训练模型保存至: best_resnet50.pth")
            
        scheduler.step()
            
    print(f"\n🎉 预训练微调圆满结束！本地最高验证准确率成功飙升至: {best_acc:.2f}%")

if __name__ == "__main__":
    train_model()