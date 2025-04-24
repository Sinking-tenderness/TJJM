import pandas as pd
import numpy as np
import random
import math

# 加载数据
df_bed = pd.read_excel('2.xlsx')  # 养老床位数据
df_distance = pd.read_excel('完整城市间距离.xlsx')  # 城市间距离数据

# 创建距离矩阵
distance_matrix = df_distance.pivot_table(index='省会A名称', columns='省会B名称', values='省会间距离（公里）')
distance_matrix = distance_matrix.fillna(0)  # 填充缺失值为0
distance_matrix = distance_matrix + distance_matrix.T  # 确保矩阵对称
np.fill_diagonal(distance_matrix.values, 0)  # 对角线设为0

# 省份到省会映射
province_to_capital = {
    "北京市": "北京市", "天津市": "天津市", "河北省": "石家庄市", "山西省": "太原市",
    "内蒙古自治区": "呼和浩特市", "辽宁省": "沈阳市", "吉林省": "长春市", "黑龙江省": "哈尔滨市",
    "上海市": "上海市", "江苏省": "南京市", "浙江省": "杭州市", "安徽省": "合肥市",
    "福建省": "福州市", "江西省": "南昌市", "山东省": "济南市", "河南省": "郑州市",
    "湖北省": "武汉市", "湖南省": "长沙市", "广东省": "广州市", "广西壮族自治区": "南宁市",
    "海南省": "海口市", "重庆市": "重庆市", "四川省": "成都市", "贵州省": "贵阳市",
    "云南省": "昆明市", "西藏自治区": "拉萨市", "陕西省": "西安市", "甘肃省": "兰州市",
    "青海省": "西宁市", "宁夏回族自治区": "银川市", "新疆维吾尔自治区": "乌鲁木齐市"
}

# 设置索引并提取数据
df_bed.set_index('省份', inplace=True)
provinces = df_bed.index.tolist()  # 省份列表
initial_bed = df_bed['养老床位数'].values  # 初始养老床位数
scores = df_bed['床位评分'].values  # 床位评分
categories = df_bed['类'].values  # 省份类别（1,2,3）
initial_standardized = df_bed['标准化床位数'].values  # 初始标准化床位数

# 反推老年人口数（单位：万人）
elderly_population = [initial_bed[i] / initial_standardized[i] if initial_standardized[i] > 0 else 0 for i in range(len(provinces))]

# 检查老年人口数合理性
for i, pop in enumerate(elderly_population):
    if pop <= 0 or pop > 10000:
        print(f"警告: {provinces[i]} 的老年人口计算结果不合理: {pop} 万人")
        elderly_population[i] = initial_bed[i] / 50  # 使用默认值

# 定义类别调整因子
# factors: 类别调整因子，用于调整不同类别的评分权重
# - 类别3（高需求）: 权重2，放大高需求地区的评分影响
# - 类别1: 权重1，中等需求地区保持标准权重
# - 类别2（低需求）: 权重0.5，降低低需求地区的评分影响
factors = {3: 2, 1: 1, 2: 0.5}
adjusted_scores = [scores[i] * factors[categories[i]] for i in range(len(provinces))]

# 规范化得分和距离
max_adjusted_score = max(adjusted_scores)
normalized_scores = [score / max_adjusted_score for score in adjusted_scores]
max_distance = distance_matrix.max().max()

# 模拟退火参数
T_initial = 20000  # 初始温度：决定算法开始时探索解空间的广度，值越高越容易跳出局部最优
alpha = 0.997      # 冷却率：控制温度下降速度，接近1时冷却慢，收敛更彻底
max_iter = 400000  # 最大迭代次数：算法运行总步数，越大越可能找到全局最优
lambda_cost = 10.0 # 距离成本权重：控制距离对总成本的影响，越大越倾向近距离调度
max_transfer_fraction = 0.15  # 最大调出床位比例：限制每个省份最多调出比例（0-1）
min_bed_fraction = 0.65       # 最小保留床位比例：保证调度后最低床位保留比例
min_transfer_size = 2000      # 最小调度床位数：忽略低于此值的调度量
max_transfer_size = 20000     # 最大调度床位数：单次调度上限
distance_threshold = 1200     # 最大允许调度距离（公里）：限制调度范围
num_runs = 10                # 模拟退火运行次数：重复运行取最优解

# 类别调出因子
# category_factors_out: 调出因子，影响省份调出床位的意愿
# - 类别2（低需求）: 2.5，表示高调出意愿，低需求地区更倾向于调出床位
# - 类别1（中等需求）: 1.2，表示中等调出意愿，保持适度调出倾向
# - 类别3（高需求）: 0.5，表示低调出意愿，高需求地区尽量保留床位
category_factors_out = {2: 2.5, 1: 1.2, 3: 0.5}

# 类别调入因子
# category_factors_in: 调入因子，影响省份调入床位的意愿
# - 类别3（高需求）: 3.0，表示高调入意愿，高需求地区优先接收床位
# - 类别1（中等需求）: 1.2，表示中等调入意愿，保持适度调入倾向
# - 类别2（低需求）: 0.4，表示低调入意愿，低需求地区减少接收床位
category_factors_in = {3: 3.0, 1: 1.2, 2: 0.4}

# 计算标准化床位数
def calculate_standardized_beds(bed, population):
    return [(bed[i] / population[i]) if population[i] > 0 else 0 for i in range(len(bed))]

# 计算调度得分
def calculate_schedule_score(bed, norm_scores, net_changes, in_paths, out_paths):
    score = sum(bed[i] * norm_scores[i] for i in range(len(provinces)))
    non_zero_count = sum(1 for x in net_changes if abs(x) >= min_transfer_size)
    zero_count = sum(1 for x in net_changes if x == 0)
    trivial_count = sum(1 for x in net_changes if 0 < abs(x) < min_transfer_size)
    dual_count = sum(1 for i in range(len(provinces)) if in_paths[i] and out_paths[i])
    return score + non_zero_count * 15000 - zero_count * 25000 - trivial_count * 35000 - dual_count * 60000

# 计算调出指数
def calculate_transfer_out_index(i, norm_scores, std_beds, category, categories_list):
    category_factor = category_factors_out[category]  # 根据类别获取调出因子
    special_factor = 1.0  # 特殊因子，可用于特定省份调整，当前为1.0
    inverse_score = 1 - norm_scores[i]  # 评分越低，调出意愿越高
    transfer_out_index = inverse_score * (std_beds[i] / 100 + 0.5) * category_factor * special_factor
    return max(0.01, transfer_out_index)

# 计算调入指数
def calculate_transfer_in_index(i, norm_scores, std_beds, category, categories_list):
    category_factor = category_factors_in[category]  # 根据类别获取调入因子
    special_factor = 1.0  # 特殊因子，可用于特定省份调整，当前为1.0
    if std_beds[i] > 0:
        transfer_in_index = norm_scores[i] * (10 / (std_beds[i] + 5)) * category_factor * special_factor
    else:
        transfer_in_index = norm_scores[i] * 2 * category_factor * special_factor
    return max(0.01, transfer_in_index)

# 单次模拟退火
def run_simulated_annealing():
    current_bed = initial_bed.copy()
    transferred_out = [0] * len(provinces)
    max_transfer_out = [initial_bed[i] * max_transfer_fraction for i in range(len(provinces))]
    min_bed = [initial_bed[i] * min_bed_fraction for i in range(len(provinces))]
    net_changes = [0] * len(provinces)
    in_paths = [False] * len(provinces)
    out_paths = [False] * len(provinces)
    schedule_log = {}
    T = T_initial

    current_standardized = calculate_standardized_beds(current_bed, elderly_population)
    transfer_out_indices = [calculate_transfer_out_index(i, normalized_scores, current_standardized, categories[i], categories) for i in range(len(provinces))]
    transfer_in_indices = [calculate_transfer_in_index(i, normalized_scores, current_standardized, categories[i], categories) for i in range(len(provinces))]

    print("初始调度指数:")
    for i, province in enumerate(provinces):
        print(f"{province}(类别{categories[i]},评分{scores[i]:.2f},标准化床位{current_standardized[i]:.2f}): "
              f"调出指数={transfer_out_indices[i]:.4f}, 调入指数={transfer_in_indices[i]:.4f}")

    for iter in range(max_iter):
        if iter % 10000 == 0:
            current_standardized = calculate_standardized_beds(current_bed, elderly_population)
            transfer_out_indices = [calculate_transfer_out_index(i, normalized_scores, current_standardized, categories[i], categories) for i in range(len(provinces))]
            transfer_in_indices = [calculate_transfer_in_index(i, normalized_scores, current_standardized, categories[i], categories) for i in range(len(provinces))]

        p_A = [x / sum(transfer_out_indices) for x in transfer_out_indices]
        p_B = [x / sum(transfer_in_indices) for x in transfer_in_indices]

        A = np.random.choice(len(provinces), p=p_A)
        B = np.random.choice(len(provinces), p=p_B)

        attempt = 0
        while A == B and attempt < 5:
            B = np.random.choice(len(provinces), p=p_B)
            attempt += 1
        if A == B:
            continue

        if current_bed[A] - min_bed[A] < min_transfer_size or transferred_out[A] >= max_transfer_out[A]:
            continue

        delta_bed = min(random.randint(min_transfer_size, max_transfer_size), current_bed[A] - min_bed[A], max_transfer_out[A] - transferred_out[A])
        if delta_bed < min_transfer_size:
            continue

        capital_A = province_to_capital[provinces[A]]
        capital_B = province_to_capital[provinces[B]]
        distance = distance_matrix.loc[capital_A, capital_B]
        if pd.isna(distance) or distance > distance_threshold:
            continue

        old_std_A = current_standardized[A]
        old_std_B = current_standardized[B]
        new_bed_A = current_bed[A] - delta_bed
        new_bed_B = current_bed[B] + delta_bed
        new_std_A = new_bed_A / elderly_population[A] if elderly_population[A] > 0 else 0
        new_std_B = new_bed_B / elderly_population[B] if elderly_population[B] > 0 else 0

        # 计算类别收益
        # - 如果从低类别（如1）调度到高类别（如3），说明从需求较低的地区调度到需求较高的地区，通常不利，收益减少
        # - 如果从高类别（如3）调度到低类别（如1），说明从需求较高的地区调度到需求较低的地区，通常有利，收益增加
        # - 如果A的标准化床位数大于B，说明A床位相对充足，B不足，从A调度到B有利，收益增加
        # - 如果A的标准化床位数小于或等于B，说明A可能比B更不足，从A调度到B不利，收益减少
        category_benefit = 0
        if categories[A] < categories[B]:
            category_benefit -= 0.23 * delta_bed  # 从低类别到高类别调度，收益减少
        elif categories[A] > categories[B]:
            category_benefit += 0.33 * delta_bed  # 从高类别到低类别调度，收益增加
        if old_std_A > old_std_B:
            category_benefit += 0.28 * delta_bed * (old_std_A - old_std_B) / max(old_std_A, 1)  # 从床位充足到不足，收益增加
        else:
            category_benefit -= 0.35 * delta_bed * (old_std_B - old_std_A) / max(old_std_B, 1)  # 从床位不足到充足，收益减少

        score_diff = normalized_scores[B] - normalized_scores[A]
        score_benefit = score_diff * delta_bed * 0.4
        distance_cost = lambda_cost * (distance / max_distance) * delta_bed
        delta_E = category_benefit + score_benefit - distance_cost

        accept_prob = math.exp(delta_E / T) if delta_E < 0 else 1.0
        if random.random() < accept_prob:
            current_bed[A] -= delta_bed
            current_bed[B] += delta_bed
            transferred_out[A] += delta_bed
            net_changes[A] -= delta_bed
            net_changes[B] += delta_bed
            out_paths[A] = True
            in_paths[B] = True
            current_standardized[A] = new_std_A
            current_standardized[B] = new_std_B

            key = (provinces[A], provinces[B])
            if key in schedule_log:
                schedule_log[key][2] += delta_bed
            else:
                schedule_log[key] = [provinces[A], provinces[B], delta_bed, distance]

        T *= alpha
        if iter % 50000 == 0:
            print(f"迭代 {iter}/{max_iter}, 温度: {T:.2f}")

    for i in range(len(provinces)):
        if 0 < abs(net_changes[i]) < min_transfer_size:
            net_changes[i] = 0
            current_bed[i] = initial_bed[i]

    return current_bed, schedule_log, net_changes, in_paths, out_paths

# 多次运行选择最优解
best_score = -float('inf')
best_bed = None
best_log = None
best_net_changes = None
best_in_paths = None
best_out_paths = None

print("开始多次模拟退火优化...")
for run in range(num_runs):
    print(f"\n开始第 {run + 1}/{num_runs} 次运行")
    final_bed, schedule_log, net_changes, in_paths, out_paths = run_simulated_annealing()
    score = calculate_schedule_score(final_bed, normalized_scores, net_changes, in_paths, out_paths)

    class3_net = sum(net_changes[i] for i in range(len(provinces)) if categories[i] == 3)
    class1_net = sum(net_changes[i] for i in range(len(provinces)) if categories[i] == 1)
    class2_net = sum(net_changes[i] for i in range(len(provinces)) if categories[i] == 2)

    print(f"运行 {run + 1} 结果: 分数 = {score:.2f}")
    print(f"类别转移情况: 第三类净转移 = {class3_net}, 第一类净转移 = {class1_net}, 第二类净转移 = {class2_net}")
    if score > best_score:
        best_score = score
        best_bed = final_bed.copy()
        best_log = schedule_log.copy()
        best_net_changes = net_changes.copy()
        best_in_paths = in_paths.copy()
        best_out_paths = out_paths.copy()
        print("发现新的最优解!")

# 优化调度路径 - 计算净调度量
print("\n开始优化调度路径...")
net_transfer_log = {}
for (from_prov, to_prov), log in best_log.items():
    reverse_key = (to_prov, from_prov)
    forward_amount = log[2]
    reverse_amount = best_log.get(reverse_key, [None, None, 0, None])[2]
    net_amount = forward_amount - reverse_amount

    # 只保留净调度量大于0且达到最小调度规模的路径
    if net_amount >= min_transfer_size:
        net_transfer_log[(from_prov, to_prov)] = [from_prov, to_prov, net_amount, log[3]]
    elif net_amount < -min_transfer_size:
        net_transfer_log[(to_prov, from_prov)] = [to_prov, from_prov, -net_amount, log[3]]

# 更新净变化
final_net_changes = [0] * len(provinces)
for log in net_transfer_log.values():
    from_idx = provinces.index(log[0])
    to_idx = provinces.index(log[1])
    final_net_changes[from_idx] -= log[2]
    final_net_changes[to_idx] += log[2]

# 计算结果
net_change = [best_bed[i] - initial_bed[i] for i in range(len(provinces))]
final_standardized = [best_bed[i] / elderly_population[i] if elderly_population[i] > 0 else 0 for i in range(len(provinces))]
standardized_change = [final_standardized[i] - initial_standardized[i] for i in range(len(provinces))]
standardized_change_ratio = [standardized_change[i] / initial_standardized[i] * 100 if initial_standardized[i] > 0 else 0 for i in range(len(provinces))]

# 输出结果
print("\n==== 调度结果（按类别分组）====")
for category in [3, 1, 2]:
    print(f"\n=== 第{category}类省份 ===")
    for i, province in enumerate(provinces):
        if categories[i] == category:
            print(f"{province}: 初始床位 {initial_bed[i]}, 最终床位 {best_bed[i]}, 净变化 {net_change[i]} | "
                  f"初始标准化床位数 {initial_standardized[i]:.2f} 张/万老人, 最终标准化床位数 {final_standardized[i]:.2f} 张/万老人, "
                  f"变化 {standardized_change[i]:.2f} 张/万老人 ({standardized_change_ratio[i]:.2f}%)")

# 验证总床位
print(f"\n初始总床位: {sum(initial_bed)}, 最终总床位: {sum(best_bed)}")

# 分析结果
print("\n=== 结果分析 ===")
category_in = {1: 0, 2: 0, 3: 0}
category_out = {1: 0, 2: 0, 3: 0}
for i, change in enumerate(net_change):
    if change > 0:
        category_in[categories[i]] += change
    elif change < 0:
        category_out[categories[i]] -= change

print("\n各类别调度统计:")
for cat in [1, 2, 3]:
    print(f"类别 {cat}: 调入 {category_in[cat]} 张, 调出 {category_out[cat]} 张")

beijing_idx = provinces.index("北京市") if "北京市" in provinces else -1
if beijing_idx >= 0:
    print(f"\n北京市调度情况:")
    print(f"北京市: 调入 {max(0, net_change[beijing_idx])} 张, 调出 {max(0, -net_change[beijing_idx])} 张")

print("\n分类统计分析:")
for category in [1, 2, 3]:
    cat_indices = [i for i, cat in enumerate(categories) if cat == category]
    if len(cat_indices) <= 1:
        continue
    cat_provinces = [provinces[i] for i in cat_indices]
    cat_scores = [scores[i] for i in cat_indices]
    cat_std = [initial_standardized[i] for i in cat_indices]
    cat_changes = [net_change[i] for i in cat_indices]
    transfer_out_index = [(1 - cat_scores[i] / max(cat_scores)) * cat_std[i] for i in range(len(cat_indices))]
    if len(cat_indices) > 1:
        toi_change_corr = np.corrcoef(transfer_out_index, cat_changes)[0, 1]
        print(f"类别 {category} 中调出指数与净变化相关性: {toi_change_corr:.4f}")