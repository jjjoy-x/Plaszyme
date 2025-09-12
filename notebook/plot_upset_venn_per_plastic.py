import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from upsetplot import UpSet, from_memberships
from itertools import combinations
from collections import defaultdict

# ========= 配置路径 =========
MATRIX_PATH = '/Users/shulei/PycharmProjects/Plaszyme/test/outputs/plastic_co_jaccard_v0.2.6_matrix.csv'  # 共降解矩阵路径
SAVE_DIR = 'plots_per_plastic_0.2.6'             # 输出图保存路径

os.makedirs(SAVE_DIR, exist_ok=True)

# ========= 加载共降解矩阵 =========
df = pd.read_csv(MATRIX_PATH, index_col=0)
plastics = df.index.tolist()

# ========= 假设我们有共降解酶信息（这里构造虚拟酶ID） =========
# 示例：为每对塑料构造不同的酶集合（真实情况中请替换为真实酶列表）
enzyme_matrix = defaultdict(set)
for i, p1 in enumerate(plastics):
    for j, p2 in enumerate(plastics):
        if i != j and df.loc[p1, p2] > 0:
            # 模拟生成酶编号
            enzyme_ids = {f"E{p1}_{p2}_{k}" for k in range(int(df.loc[p1, p2]))}
            enzyme_matrix[(p1, p2)] = enzyme_ids

# ========= 构建每种塑料的降解酶集合 =========
plastic_to_enzymes = defaultdict(set)
for (p1, p2), enzymes in enzyme_matrix.items():
    plastic_to_enzymes[p1].update(enzymes)
    plastic_to_enzymes[p2].update(enzymes)

# ========= 为每个塑料绘制图像 =========
for target in plastics:
    # 获取与其他塑料的共降解酶集合
    sets = {}
    for other in plastics:
        if other == target:
            continue
        key = f"{target} ∩ {other}"
        common = plastic_to_enzymes[target].intersection(plastic_to_enzymes[other])
        if len(common) > 0:
            sets[key] = common

    # ---- 绘制 UpSet 图 ----
    if len(sets) >= 2:
        # 构造 membership
        memberships = []
        for enzyme in plastic_to_enzymes[target]:
            membership = tuple(key for key, enzymes in sets.items() if enzyme in enzymes)
            if membership:
                memberships.append(membership)

        if memberships:
            data = from_memberships(memberships)
            plt.figure(figsize=(8, 5))
            UpSet(data, subset_size='count').plot()
            plt.suptitle(f'UpSet for {target}')
            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, f'{target}_upset.png'))
            plt.close()

    # ---- 绘制 Venn 图（仅支持 2-3 个集合）----
    sets_for_venn = list(sets.values())
    labels_for_venn = list(sets.keys())
    if 2 <= len(sets_for_venn) <= 3:
        plt.figure(figsize=(6, 6))
        if len(sets_for_venn) == 2:
            venn2(sets_for_venn, set_labels=labels_for_venn)
        else:
            venn3(sets_for_venn, set_labels=labels_for_venn)
        plt.title(f'Venn Diagram for {target}')
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f'{target}_venn.png'))
        plt.close()