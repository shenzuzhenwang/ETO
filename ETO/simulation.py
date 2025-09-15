import os
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# 全局字体设置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体也匹配


def plot_single_bar(
        x_ticks: List,
        series: Dict[str, List[float]],
        subtitle: str,
        xlabel: str,
        ylabel: str,
        yticks: List[float],
        save_path: str,
        bar_scale: float = 0.6,  # 每组柱子的总宽比例（越小越细）
):
    """
    单图分组柱状图（非子图）：
      - 颜色 + 条纹
      - 下方小标题 (a)/(b)
      - 指定 y 轴刻度值
      - 图例固定左上；图向右移避免截断；柱宽可调更细
    """
    names = list(series.keys())
    n = len(x_ticks)
    values = np.array([series[nm] for nm in names])
    num_series = len(names)

    idx = np.arange(n)
    width = bar_scale / num_series
    offsets = (idx[:, None] + (np.arange(num_series) - (num_series - 1) / 2) * width)

    # 样式
    hatches = ['//', '..', 'xx', '\\\\', 'oo', '**']
    colors = cm.Set3.colors  # 浅色系

    fig, ax = plt.subplots(figsize=(5, 3), dpi=150, constrained_layout=True)
    for j, nm in enumerate(names):
        ax.bar(offsets[:, j], values[j],
               width=width,
               label=nm,
               hatch=hatches[j % len(hatches)],
               edgecolor='black',
               color=colors[j % len(colors)],
               alpha=0.8)

    ax.set_xticks(idx)
    ax.set_xticklabels([str(x) for x in x_ticks])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, labelpad=8)

    # 指定纵坐标刻度
    ax.set_ylim(min(yticks), max(yticks))
    if ylabel == "Fallback Rate (%)":
        yticks = yticks[:-1]
    ax.set_yticks(yticks)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.02)

    # 图例左上
    ax.legend(frameon=False, ncol=2,  # 默认：True（显示边框） 默认：1（单列）
              loc="upper left",  # 默认：'best'（自动选择最佳位置）
              labelspacing=0.2,  # 图例项之间的垂直间距 0.5
              columnspacing=0.6,  # 列与列之间的间距（多列时） 2.0
              handlelength=1.5,  # 图例句柄（矩形条）的长度 2.0
              handletextpad=0.3,  # 句柄与文本之间的水平间距 0.8
              borderpad=0,  # 图例边框与内容之间的内边距 0.4
              fontsize=15)  # 更小的字体

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# =================== 数据 ===================
out_dir = "/mnt/d/work/论文/moe"

Experts_ls = [32, 64, 128, 256]
Experts_ft = [32, 64, 128, 256]
stds = [0.0, 0.3, 0.6, 0.9]

# fig10_leaf_spine = {'ETO': [0.125, 0.125, 0.125, 0.125], 'DSA': [0.238, 0.467, 0.71, 0.608], 'ATP': [0.238, 0.467, 0.721, 0.641], 'PS-only': [0.25, 0.5, 0.75, 1.0]}
# fig10_fat_tree = {'ETO': [0.125, 0.125, 0.125, 0.125], 'DSA': [0.24, 0.438, 0.677, 0.799], 'ATP': [0.24, 0.447, 0.677, 0.819], 'PS-only': [0.25, 0.5, 0.75, 1.0]}
# fig11_leaf_spine = {'ETO': [0.125, 0.375, 0.625, 0.875], 'DSA': [0.012, 0.033, 0.029, 0.392], 'ATP': [0.012, 0.033, 0.029, 0.358], 'PS-only': [0.0, 0.0, 0.0, 0.0]}
# fig11_fat_tree = {'ETO': [0.125, 0.375, 0.625, 0.875], 'DSA': [0.01, 0.062, 0.073, 0.201], 'ATP': [0.01, 0.053, 0.073, 0.181], 'PS-only': [0.0, 0.0, 0.0, 0.0]}
# fig12_leaf_spine = {'ETO': [0.913, 1.604, 2.396, 3.014], 'DSA': [0.727, 1.435, 2.192, 2.502], 'ATP': [0.727, 1.435, 2.192, 2.552], 'PS-only': [0.75, 1.5, 2.25, 3.0]}
# fig12_fat_tree = {'ETO': [1.49, 2.612, 3.895, 5.061], 'DSA': [0.976, 1.868, 2.841, 3.573], 'ATP': [0.976, 1.887, 2.841, 3.612], 'PS-only': [1.0, 2.0, 3.0, 4.0]}
# fig13_leaf_spine = {'ETO': [0.013, 0.13, 1.3, 1.3], 'DSA': [86.812, 98.088, 95.975, 99.3], 'ATP': [86.812, 98.088, 99.975, 99.337], 'PS-only': [100.0, 100.0, 100.0, 100.0]}
# fig13_fat_tree = {'ETO': [0.013, 0.13, 1.3, 1.3], 'DSA': [75.538, 87.312, 87.138, 95.475], 'ATP': [75.538, 89.225, 87.138, 94.463], 'PS-only': [100.0, 100.0, 100.0, 100.0]}

fig10_leaf_spine = {'ETO': [0.125, 0.125, 0.125, 0.125], 'DSA': [0.238, 0.467, 0.721, 0.776],
                    'ATP': [0.238, 0.467, 0.721, 0.841], 'PS-only': [0.25, 0.5, 0.75, 1.0]}
fig10_fat_tree = {'ETO': [0.125, 0.125, 0.125, 0.125], 'DSA': [0.24, 0.437, 0.677, 0.806],
                  'ATP': [0.24, 0.447, 0.677, 0.819], 'PS-only': [0.25, 0.5, 0.75, 1.0]}
fig11a_leaf_spine = {'ETO': [0.25, 0.5, 0.75, 1.0], 'DSA': [0.095, 0.187, 0.175, 0.839],
                     'ATP': [0.095, 0.187, 0.175, 0.611], 'PS-only': [0.0, 0.0, 0.0, 0.0]}
fig11a_fat_tree = {'ETO': [0.25, 0.5, 0.75, 1.0], 'DSA': [0.126, 0.298, 0.54, 0.888],
                   'ATP': [0.126, 0.265, 0.54, 0.853], 'PS-only': [0.0, 0.0, 0.0, 0.0]}
fig11b_leaf_spine = {'ETO': [0.125, 0.375, 0.625, 0.875], 'DSA': [0.012, 0.033, 0.029, 0.526],
                     'ATP': [0.012, 0.033, 0.029, 0.358], 'PS-only': [0.0, 0.0, 0.0, 0.0]}
fig11b_fat_tree = {'ETO': [0.125, 0.375, 0.625, 0.875], 'DSA': [0.01, 0.063, 0.073, 0.195],
                   'ATP': [0.01, 0.053, 0.073, 0.181], 'PS-only': [0.0, 0.0, 0.0, 0.0]}
fig12_leaf_spine = {'ETO': [0.533, 1.045, 1.539, 2.042], 'DSA': [0.655, 1.313, 2.075, 2.372],
                    'ATP': [0.727, 1.435, 2.192, 2.552], 'PS-only': [0.75, 1.5, 2.25, 3.0]}
fig12_fat_tree = {'ETO': [0.707, 1.441, 2.199, 2.93], 'DSA': [0.874, 1.75, 2.478, 3.184],
                  'ATP': [0.976, 1.887, 2.841, 3.612], 'PS-only': [1.0, 2.0, 3.0, 4.0]}
fig13_leaf_spine = {'ETO': [0.013, 0.013, 0.013, 0.013], 'DSA': [37.438, 14.763, 19.675, 3.513],
                    'ATP': [37.438, 14.763, 19.675, 22.488], 'PS-only': [100.0, 100.0, 100.0, 100.0]}
fig13_fat_tree = {'ETO': [0.013, 0.013, 0.013, 0.013], 'DSA': [23.387, 3.662, 0.05, 0.013],
                  'ATP': [23.387, 9.25, 0.05, 0.188], 'PS-only': [100.0, 100.0, 100.0, 100.0]}

plot_single_bar(Experts_ls, fig10_leaf_spine, "(a) Leaf-Spine",
                "No. of Experts", "PS Agg. Vol. (GB)",
                yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                save_path=os.path.join(out_dir, "fig10_leaf_spine.png"))
plot_single_bar(Experts_ft, fig10_fat_tree, "(b) Fat-Tree",
                "No. of Experts", "PS Agg. Vol. (GB)",
                yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                save_path=os.path.join(out_dir, "fig10_fat_tree.png"))

plot_single_bar(Experts_ls, fig11a_leaf_spine, "(a) Leaf-Spine",
                "No. of Experts", "In-net Tx Vol. (GB)",
                yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                save_path=os.path.join(out_dir, "fig11a_leaf_spine.png"))

plot_single_bar(Experts_ft, fig11a_fat_tree, "(b) Fat-Tree",
                "No. of Experts", "In-net Tx Vol. (GB)",
                yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                save_path=os.path.join(out_dir, "fig11a_fat_tree.png"))

plot_single_bar(Experts_ls, fig11b_leaf_spine, "(a) Leaf-Spine",
                "No. of Experts", "In-net Agg. Vol. (GB)",
                yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                save_path=os.path.join(out_dir, "fig11b_leaf_spine.png"))

plot_single_bar(Experts_ft, fig11b_fat_tree, "(b) Fat-Tree",
                "No. of Experts", "In-net Agg. Vol. (GB)",
                yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                save_path=os.path.join(out_dir, "fig11b_fat_tree.png"))

plot_single_bar(Experts_ls, fig12_leaf_spine, "(a) Leaf-Spine",
                "No. of Experts", "Total Comm. Vol. (GB)",
                yticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                save_path=os.path.join(out_dir, "fig12_leaf_spine.png"))
plot_single_bar(Experts_ft, fig12_fat_tree, "(b) Fat-Tree",
                "No. of Experts", "Total Comm. Vol. (GB)",
                yticks=[0, 1.0, 2.0, 3.0, 4.0],
                save_path=os.path.join(out_dir, "fig12_fat_tree.png"))

plot_single_bar(Experts_ls, fig13_leaf_spine, "(a) Leaf-Spine",
                "No. of Experts", "Fallback Rate (%)",
                yticks=[0, 20, 40, 60, 80, 100, 135],
                save_path=os.path.join(out_dir, "fig13_leaf_spine.png"))
plot_single_bar(Experts_ft, fig13_fat_tree, "(b) Fat-Tree",
                "No. of Experts", "Fallback Rate (%)",
                yticks=[0, 20, 40, 60, 80, 100, 135],
                save_path=os.path.join(out_dir, "fig13_fat_tree.png"))

# fig10_leaf_spine = {"ETO": [0.188, 0.188, 0.188, 0.188], "DSA": [0.152, 0.189, 0.39, 0.632], "ATP": [0.216, 0.456, 0.682, 0.891],
#                     "PS-only": [0.375, 0.75, 1.125, 1.5]}
# fig10_fat_tree = {"ETO": [0.188, 0.188, 0.188, 0.188], "DSA": [0.152, 0.189, 0.39, 0.632], "ATP": [0.224, 0.441, 0.65, 0.936],
#                   "PS-only": [0.375, 0.75, 1.125, 1.5]}
# fig11_leaf_spine = {"ETO": [0.375, 0.75, 1.125, 1.5], "DSA": [0.323, 0.561, 0.735, 0.868], "ATP": [0.159, 0.294, 0.443, 0.609],
#                     "PS-only": [0, 0, 0, 0]}
# fig11_fat_tree = {"ETO": [0.375, 0.75, 1.125, 1.5], "DSA": [0.323, 0.561, 0.735, 0.868], "ATP": [0.151, 0.309, 0.475, 0.564],
#                   "PS-only": [0, 0, 0, 0]}
# fig12_leaf_spine = {"ETO": [0.837, 1.923, 2.864, 3.0], "DSA": [0.803, 2.108, 3.081, 3.778], "ATP": [1.02, 2.14, 3.204, 4.001],
#                     "PS-only": [1.125, 2.25, 3.375, 4.5]}
# fig12_fat_tree = {"ETO": [1.158, 2.426, 3.654, 4.878], "DSA": [1.387, 2.772, 4.233, 5.56], "ATP": [1.405, 2.833, 4.44, 6.0],
#                   "PS-only": [1.5, 3.0, 4.5, 6.0]}
# # fig13_leaf_spine = {"ETO": [10, 20, 30, 50], "DSA": [25, 65, 85, 90], "ATP": [75, 95, 98, 99],
# #                     "PS-only": [100, 100, 100, 100]}
# # fig13_fat_tree = {"ETO": [12, 23, 35, 55], "DSA": [25, 60, 89, 90], "ATP": [70, 90, 95, 99],
# #                   "PS-only": [100, 100, 100, 100]}
# fig13_leaf_spine = {"ETO": [3.0, 3.0, 3.0, 3.0], "DSA": [3.78, 3.781, 4.117, 4.44], "ATP": [3.982, 4.148, 4.264, 4.309],
#                     "PS-only": [4.5, 4.5, 4.5, 4.5]}
# fig13_fat_tree = {"ETO": [4.8, 4.8, 4.8, 4.8], "DSA": [5.5, 5.6, 5.8, 5.9], "ATP": [5.9, 5.9, 5.9, 5.9],
#                   "PS-only": [6.0, 6.0, 6.0, 6.0]}
# =================== 绘制全部 ===================
# plot_single_bar(Experts_ls, fig10_leaf_spine, "(a) Leaf-Spine",
#                 "No. of Experts", "PS Agg. Vol. (GB)",
#                 yticks=[0, 0.3, 0.6, 0.9, 1.2, 1.5],
#                 save_path=os.path.join(out_dir, "fig10a_leaf_spine.png"))
# plot_single_bar(Experts_ft, fig10_fat_tree, "(b) Fat-Tree",
#                 "No. of Experts", "PS Agg. Vol. (GB)",
#                 yticks=[0, 0.4, 0.8, 1.2, 1.6],
#                 save_path=os.path.join(out_dir, "fig10b_fat_tree.png"))
#
# plot_single_bar(Experts_ls, fig11_leaf_spine, "(a) Leaf-Spine",
#                 "No. of Experts", "In-net Agg. Vol. (GB)",
#                 yticks=[0, 0.4, 0.8, 1.2, 1.6],
#                 save_path=os.path.join(out_dir, "fig11a_leaf_spine.png"))
# plot_single_bar(Experts_ft, fig11_fat_tree, "(b) Fat-Tree",
#                 "No. of Experts", "In-net Agg. Vol. (GB)",
#                 yticks=[0, 0.5, 1.0, 1.5, 2.0],
#                 save_path=os.path.join(out_dir, "fig11b_fat_tree.png"))
#
# plot_single_bar(Experts_ls, fig12_leaf_spine, "(a) Leaf-Spine",
#                 "No. of Experts", "Total Comm. Vol. (GB)",
#                 yticks=[0, 1.0, 2.0, 3.0, 4.0, 5.0],
#                 save_path=os.path.join(out_dir, "fig12a_leaf_spine.png"))
# plot_single_bar(Experts_ft, fig12_fat_tree, "(b) Fat-Tree",
#                 "No. of Experts", "Total Comm. Vol. (GB)",
#                 yticks=[0, 1.5, 3.0, 4.5, 6.0, 7.5],
#                 save_path=os.path.join(out_dir, "fig12b_fat_tree.png"))
#
# plot_single_bar(stds, fig13_leaf_spine, "(a) Leaf-Spine",
#                 "Degree of Asynchrony (σ)", "Total Comm. Vol. (GB)",
#                 yticks=[0, 1.5, 3.0, 4.5, 6.0, 7.5],
#                 save_path=os.path.join(out_dir, "fig13a_leaf_spine.png"))
# plot_single_bar(stds, fig13_fat_tree, "(b) Fat-Tree",
#                 "Degree of Asynchrony (σ)", "Total Comm. Vol. (GB)",
#                 yticks=[0, 2.0, 4.0, 6.0, 8.0],
#                 save_path=os.path.join(out_dir, "fig13b_fat_tree.png"))

# plot_single_bar(Experts_ls, fig13_leaf_spine, "(a) Leaf-Spine",
#                 "No. of Experts", "Fallback Rate (%)",
#                 yticks=[0, 20, 40, 60, 80, 100, 135],
#                 save_path=os.path.join(out_dir, "fig13a_leaf_spine.png"))
# plot_single_bar(Experts_ft, fig13_fat_tree, "(b) Fat-Tree",
#                 "No. of Experts", "Fallback Rate (%)",
#                 yticks=[0, 20, 40, 60, 80, 100, 135],
#                 save_path=os.path.join(out_dir, "fig13b_fat_tree.png"))
