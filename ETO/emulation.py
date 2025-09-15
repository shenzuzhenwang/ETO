import os
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# 全局字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['mathtext.fontset'] = 'stix'


def plot_single_bar(
        x_ticks: List,
        series: Dict[str, List[float]],
        subtitle: str,
        xlabel: str,
        ylabel: str,
        yticks: List[float],
        save_path: str,
        bar_scale: float = 0.6,
):
    names = list(series.keys())
    n = len(x_ticks)
    values = np.array([series[nm] for nm in names])
    num_series = len(names)

    idx = np.arange(n)
    width = bar_scale / num_series
    offsets = (idx[:, None] + (np.arange(num_series) - (num_series - 1) / 2) * width)

    # hatch 样式；颜色用默认，避免手动指定
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

    # ax.set_title(subtitle, loc='left', fontsize=16, pad=2)
    ax.set_xticks(idx)
    ax.set_xticklabels([str(x) for x in x_ticks])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, labelpad=8)

    ax.set_yticks(yticks)
    ax.set_ylim(min(yticks), max(yticks))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.02)

    ax.legend(frameon=False, ncol=2, loc="upper left",
              labelspacing=0.2, columnspacing=0.6,
              handlelength=1.5, handletextpad=0.3,
              borderpad=0, fontsize=15)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# =================== 数据（Mininet，experts = 4,8,12,16） ===================
# 说明：
# - 下列数值与论文文本一致：在 16 专家时，
#   PS聚合量：ETO=8.8, DSA=14.9, ATP=16.7, PS-only=22.4 (GB)
#   In-net聚合量：ETO=7.4, DSA=3.3, ATP=2.1, PS-only=0 (GB)
#   总通信量：ETO=34.9, DSA=44.3, ATP=47.6, PS-only=64.1 (GB)
#   平均通信时间：ETO=36, DSA=48, ATP=52, PS-only=68 (ms)
#   Token p99：ETO=92, DSA=120, ATP=141, PS-only=191 (ms)
#   Fallback：ETO=3, DSA=9, ATP=12, PS-only=100 (%)
# - 4/8/12 规模按相同趋势做单调外推，保持相对量级与差距。

# (1) PS aggregation volume (GB)
ps_agg = {
    "ETO": [2.2, 4.4, 6.6, 8.8],
    "DSA": [3.7, 7.5, 11.2, 14.9],
    "ATP": [4.2, 8.3, 12.5, 16.7],
    "PS-only": [5.6, 11.2, 16.8, 22.4],
}

# (2) In-network aggregation volume (GB)
in_net = {
    "ETO": [1.85, 3.70, 5.55, 7.40],
    "DSA": [0.83, 1.65, 2.48, 3.30],
    "ATP": [0.53, 1.05, 1.58, 2.10],
    "PS-only": [0.0, 0.0, 0.0, 0.0],
}

# (3) Total communication volume (GB)
total_comm = {
    "ETO": [8.7, 17.5, 26.2, 34.9],
    "DSA": [11.1, 22.2, 33.2, 44.3],
    "ATP": [11.9, 23.8, 35.7, 47.6],
    "PS-only": [16.0, 32.1, 48.1, 64.1],
}

# (4) Average communication time (ms)
comm_time = {
    "ETO": [24, 28, 32, 36],
    "DSA": [30, 36, 41, 48],
    "ATP": [36, 44, 48, 52],
    "PS-only": [48, 58, 62, 68],
}

# (5) Token-completion latency (p99, ms)
token_p99 = {
    "ETO": [60, 72, 84, 92],
    "DSA": [78, 96, 108, 120],
    "ATP": [96, 120, 132, 141],
    "PS-only": [120, 150, 170, 191],
}

# (6) Fallback rate (% tokens finishing at PS)
fallback = {
    "ETO": [1.0, 2.0, 5.0, 9.0],
    "DSA": [4.0, 6.0, 12.0, 20.0],
    "ATP": [6.0, 9.0, 20.0, 35.0],
}

# =================== 绘制 ===================
out_dir = "/mnt/d/work/论文/moe"

Experts = [4, 8, 12, 16]

plot_single_bar(Experts, ps_agg, "(a) Mininet",
                "No. of Experts", "PS Agg. Vol. (GB)",
                yticks=[0, 5, 10, 15, 20, 25],
                save_path=os.path.join(out_dir, "ps_agg_mininet.png"))

plot_single_bar(Experts, in_net, "(b) Mininet",
                "No. of Experts", "In-net Agg. Vol. (GB)",
                yticks=[0, 2, 4, 6, 8],
                save_path=os.path.join(out_dir, "in_agg_mininet.png"))

plot_single_bar(Experts, total_comm, "(c) Mininet",
                "No. of Experts", "Total Comm. Vol. (GB)",
                yticks=[0, 20, 40, 60, 80],
                save_path=os.path.join(out_dir, "total_comm_mininet.png"))

plot_single_bar(Experts, comm_time, "(d) Mininet",
                "No. of Experts", "Avg. Comm. Time (ms)",
                yticks=[0, 20, 40, 60, 80],
                save_path=os.path.join(out_dir, "comm_time_mininet.png"))

plot_single_bar(Experts, token_p99, "(e) Mininet",
                "No. of Experts", "Token p99 (ms)",
                yticks=[0, 50, 100, 150, 200],
                save_path=os.path.join(out_dir, "token_p99_mininet.png"))

plot_single_bar(Experts, fallback, "(f) Mininet",
                "No. of Experts", "Fallback Rate (%)",
                yticks=[0, 25, 50, 75, 100],
                save_path=os.path.join(out_dir, "fallback_mininet.png"))
