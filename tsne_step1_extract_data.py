import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.manifold import TSNE
import open_clip

# 🔗 导入你的本地 MOSSNet 架构
try:
    from model import MOSSNet 
except ImportError:
    raise ImportError("未能在当前目录下找到 model.py，请检查路径。")

def compute_and_save_tsne_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🌌 [Step 1] 启动潜空间特征提取与降维... 当前计算硬件: {device}")
    
    NUM_CLASSES = 7
    BATCH_SIZE = 4  
    DATA_DIR = "test1.2" 
    
    weights_checking = {
        "ResNet50": "best_resnet50.pth",
        "MOSSNet": "best_mossnet_model.pth",
        "BioCLIP 2.5": "best_bioclip_custom.pth"
    }
    
    transform_standard = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_bioclip = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    # --- 初始化网络空壳 ---
    resnet_full = models.resnet50()
    resnet_full.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(resnet_full.fc.in_features, NUM_CLASSES))
    
    mossnet_full = MOSSNet(num_classes=NUM_CLASSES)
    class MOSSNetFeatureExtractor(nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.orig = original_model
        def forward(self, x):
            x = self.orig.cpe(x)
            x = self.orig.stage1(x)
            x = self.orig.merge1(x)
            x = self.orig.stage2(x)
            x = self.orig.merge2(x)
            x = self.orig.stage3(x)
            x = self.orig.merge3(x)
            x = self.orig.stage4(x)
            x = self.orig.avgpool(x)  
            return torch.flatten(x, 1) 
    mossnet_features = MOSSNetFeatureExtractor(mossnet_full)

    bioclip_base, _, _ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2.5-vith14')
    bioclip_features = bioclip_base.visual

    models_dict = {
        "MOSSNet": {"model_obj": mossnet_features, "weight_path": weights_checking["MOSSNet"], "transform": transform_standard},
        "ResNet50": {"model_obj": resnet_full, "weight_path": weights_checking["ResNet50"], "transform": transform_standard},
        "BioCLIP 2.5": {"model_obj": bioclip_features, "weight_path": weights_checking["BioCLIP 2.5"], "transform": transform_bioclip}
    }

    # 用于缓存最终作图所需所有数据的字典
    tsne_cache_data = {}

    for model_name, config in models_dict.items():
        print(f"\n⚡ 当前处理模型: {model_name}")
        
        current_dataset = datasets.ImageFolder(root=DATA_DIR, transform=config["transform"])
        dataloader = DataLoader(current_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        display_names = [name.replace('_test', '').replace('1.2', '').strip() for name in current_dataset.classes]

        model = config["model_obj"]
        if model_name == "ResNet50":
            model.load_state_dict(torch.load(config["weight_path"], map_location=device))
            model.fc = nn.Identity() 
        elif model_name == "MOSSNet":
            model.orig.load_state_dict(torch.load(config["weight_path"], map_location=device))
        elif model_name == "BioCLIP 2.5":
            model.load_state_dict(torch.load(config["weight_path"], map_location=device), strict=False)
        
        model = model.to(device)
        model.eval()
        
        all_features, all_labels = [], []
        print(f"  🧬 正在抽取潜空间向量...")
        with torch.no_grad():
            for images, labels in dataloader:
                features = model(images.to(device)) 
                if len(features.shape) > 2:
                    features = features.view(features.size(0), -1)
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        print(f"  🌌 执行 t-SNE 流形降维...")
        tsne = TSNE(n_components=2, perplexity=25, random_state=42, init='pca', max_iter=1000)
        features_2d = tsne.fit_transform(all_features)
        
        # 将降维结果打包存入字典
        tsne_cache_data[model_name] = {
            "features_2d": features_2d,
            "labels": all_labels,
            "display_names": display_names
        }

    # 将庞大的结果安全保存到本地硬盘
    cache_file = "tsne_cache_data.npy"
    np.save(cache_file, tsne_cache_data, allow_pickle=True)
    print(f"\n✅ 所有数据已成功缓存至本地文件: {cache_file}")

if __name__ == "__main__":
    compute_and_save_tsne_data()