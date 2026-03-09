import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# 创建画布和坐标轴
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制特征选择模块
# 输入层
input_box = patches.FancyBboxPatch((0.1, 0.8), 0.2, 0.15, boxstyle="round,pad=0.1",
                                   edgecolor='blue', facecolor='lightblue', linewidth=2,
                                   label='输入: 密集候选区域 (YOLOX)')
ax.add_patch(input_box)

# 分类置信度排序
sort_box = patches.FancyBboxPatch((0.4, 0.8), 0.2, 0.15, boxstyle="round,pad=0.1",
                                  edgecolor='green', facecolor='lightgreen', linewidth=2,
                                  label='处理: 分类置信度排序')
ax.add_patch(sort_box)

# NMS操作
nms_box = patches.FancyBboxPatch((0.7, 0.8), 0.2, 0.15, boxstyle="round,pad=0.1",
                                 edgecolor='red', facecolor='lightcoral', linewidth=2,
                                 label='处理: NMS去冗余')
ax.add_patch(nms_box)

# 输出层
output_box = patches.FancyBboxPatch((0.95, 0.8), 0.2, 0.15, boxstyle="round,pad=0.1",
                                    edgecolor='purple', facecolor='plum', linewidth=2,
                                    label='输出: 高质量候选区域')
ax.add_patch(output_box)

# 绘制特征聚合模块
# 输入层（多帧特征）
multi_frame_input = patches.FancyBboxPatch((0.1, 0.3), 0.2, 0.15, boxstyle="round,pad=0.1",
                                           edgecolor='orange', facecolor='lightsalmon', linewidth=2,
                                           label='输入: 当前帧 & 参考帧特征')
ax.add_patch(multi_frame_input)

# 特征投影层
projection_box = patches.FancyBboxPatch((0.4, 0.3), 0.2, 0.15, boxstyle="round,pad=0.1",
                                        edgecolor='brown', facecolor='bisque', linewidth=2,
                                        label='处理: 特征投影 (线性层)')
ax.add_patch(projection_box)

# 自注意力机制
attention_box = patches.FancyBboxPatch((0.7, 0.3), 0.2, 0.15, boxstyle="round,pad=0.1",
                                       edgecolor='gray', facecolor='lightgray', linewidth=2,
                                       label='处理: 自注意力机制')
ax.add_patch(attention_box)

# 残差连接
residual_box = patches.FancyBboxPatch((0.95, 0.3), 0.2, 0.15, boxstyle="round,pad=0.1",
                                      edgecolor='black', facecolor='white', linewidth=2,
                                      label='处理: 残差连接')
ax.add_patch(residual_box)

# 平均池化
avg_pool_box = patches.FancyBboxPatch((0.4, 0.1), 0.2, 0.15, boxstyle="round,pad=0.1",
                                      edgecolor='cyan', facecolor='lightcyan', linewidth=2,
                                      label='处理: 平均池化过滤')
ax.add_patch(avg_pool_box)

# 绘制箭头连接各个模块
arrow_props = dict(arrowstyle='->', color='black', lw=1.5)

# 特征选择模块连接
ax.annotate('', xy=(0.3, 0.875), xytext=(0.3, 0.875), arrowprops=arrow_props)
ax.annotate('', xy=(0.6, 0.875), xytext=(0.5, 0.875), arrowprops=arrow_props)
ax.annotate('', xy=(0.85, 0.875), xytext=(0.8, 0.875), arrowprops=arrow_props)

# 特征聚合模块连接
ax.annotate('', xy=(0.3, 0.375), xytext=(0.3, 0.375), arrowprops=arrow_props)
ax.annotate('', xy=(0.6, 0.375), xytext=(0.5, 0.375), arrowprops=arrow_props)
ax.annotate('', xy=(0.85, 0.375), xytext=(0.8, 0.375), arrowprops=arrow_props)
ax.annotate('', xy=(0.5, 0.25), xytext=(0.5, 0.25), arrowprops=arrow_props)

# 添加图例
legend_elements = [
    Line2D([0], [0], color='w', label='特征选择模块', marker='s', markersize=10, markerfacecolor='lightblue',
           markeredgecolor='blue'),
    Line2D([0], [0], color='w', label='特征聚合模块', marker='s', markersize=10, markerfacecolor='lightsalmon',
           markeredgecolor='orange')]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

# 设置坐标轴范围和隐藏坐标轴
ax.set_xlim(0, 1.2)
ax.set_ylim(0, 1)
ax.axis('off')

# 添加标题
plt.title('YOLOV算法特征选择与聚合模块结构图', fontsize=14, pad=20)

# 保存图像
plt.savefig('yolov_feature_modules.png', bbox_inches='tight', dpi=300)
plt.close()