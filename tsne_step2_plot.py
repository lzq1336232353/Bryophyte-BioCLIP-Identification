import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def render_tsne_plot():
    cache_file = "tsne_cache_data.npy"
    if not os.path.exists(cache_file):
        print(f"❌ 找不到缓存文件 {cache_file}！请先运行 step1_extract_tsne_data.py")
        return
        
    print(f"🎨 [Step 2] 瞬间读取缓存数据，启动图表渲染流...")
    
    # 瞬间读取硬盘里的数据
    tsne_data = np.load(cache_file, allow_pickle=True).item()
    
    NUM_CLASSES = 7
    # 🌟 统一定义全局高对比度调色板
    global_palette = sns.color_palette("husl", NUM_CLASSES)
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=300)
    plt.subplots_adjust(wspace=0.15, bottom=0.22) 

    global_handles = None
    global_labels = None

    # 我们依然按照字典的键（MOSSNet, ResNet50, BioCLIP 2.5）顺序遍历
    for idx, (model_name, data) in enumerate(tsne_data.items()):
        
        features_2d = data["features_2d"]
        all_labels = data["labels"]
        display_names = data["display_names"]
        
        ax = axes[idx]
        sns.set_theme(style="ticks")
        
        # 使用读取到的特征和统一的调色板画图
        sns.scatterplot(
            ax=ax,
            x=features_2d[:, 0], y=features_2d[:, 1],
            hue=[display_names[label] for label in all_labels],
            hue_order=display_names, 
            palette=global_palette, # 强制三张图颜色绝对一致
            s=70, alpha=0.85, edgecolor='w', linewidth=0.3 
        )
        
        sns.despine(ax=ax)
        
        ax.set_title(model_name, fontsize=20, pad=18, weight='bold') 
        ax.set_xlabel("t-SNE Dimension 1", fontsize=16, labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        if idx == 0:
            ax.set_ylabel("t-SNE Dimension 2", fontsize=16, labelpad=10)
        else:
            ax.set_ylabel("") 
            
        # 提取第一个图的图注，清理每个子图内部的图注
        if idx == 0:
            global_handles, global_labels = ax.get_legend_handles_labels()
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # 在底部生成超宽的 SCI 级别全局图注
    fig.legend(global_handles, global_labels, loc='lower center', ncol=7, 
               bbox_to_anchor=(0.5, 0.02), fontsize=13, title="Bryophyte Taxa", 
               title_fontsize=15, frameon=False, columnspacing=1.5)

    output_img_path = "comparative_tsne_3models_optimized.png"
    plt.savefig(output_img_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"🎉 渲染完成！高规格排版的对比图已秒速生成: {output_img_path}")

if __name__ == "__main__":
    render_tsne_plot()