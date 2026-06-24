import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_moss_dataloaders(data_dir="C:/mossnet/test1.2", batch_size=16):
    # 1. 训练集数据增强流水线（在这里装配 RandomErasing）
    train_transform = transforms.Compose([
        transforms.Resize(256),                  
        transforms.CenterCrop(224),              
        transforms.RandomRotation(degrees=(-5, 10)), 
        transforms.ColorJitter(                  
            brightness=(0.7, 1.3), 
            contrast=(0.7, 1.3)
        ),
        transforms.ToTensor(),                   
        transforms.Normalize(                    
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
        # 🌟 正确位置：仅在训练集最后加入随机擦除，彻底斩断过拟合！
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), value='random')
    ])
    
    # 2. 验证集和测试集流水线（必须保持干净，绝对不能做任何随机遮挡干扰）
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 读取原始文件夹
    full_dataset = datasets.ImageFolder(root=data_dir)
    total_samples = len(full_dataset)
    
    # 严格按照论文 4.1 节的 6:2:2 比例精准划分样本数量
    train_size = int(total_samples * 0.6)
    val_size = int(total_samples * 0.2)
    test_size = total_samples - train_size - val_size  
    
    # 使用固定随机种子进行三向切分
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices, test_indices = random_split(
        range(total_samples), 
        [train_size, val_size, test_size], 
        generator=generator
    )
    
    # 动态覆写 Subset 的 transform 属性
    class DatasetFromSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
        def __len__(self):
            return len(self.subset)
            
    raw_train = torch.utils.data.Subset(full_dataset, train_indices)
    raw_val = torch.utils.data.Subset(full_dataset, val_indices)
    raw_test = torch.utils.data.Subset(full_dataset, test_indices)
    
    train_dataset = DatasetFromSubset(raw_train, transform=train_transform)
    val_dataset = DatasetFromSubset(raw_val, transform=val_test_transform)
    test_dataset = DatasetFromSubset(raw_test, transform=val_test_transform)
    
    # 构建三个独立的 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"数据就绪（严格遵循论文 6:2:2 三向切分比例）:")
    print(f"训练集(Train 60%): {len(train_dataset)} | 验证集(Val 20%): {len(val_dataset)} | 测试集(Test 20%): {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_moss_dataloaders()