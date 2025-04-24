import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
import statsmodels.api as sm
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子以确保可重复性
np.random.seed(42)

# 1. 加载数据
file_path = "各省综合数据.xlsx"
try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"文件 {file_path} 不存在，请检查路径或文件是否正确！")
    exit(1)

# 列名直接使用图片中的标题（确保包含新特征）
df.columns = [
    "统计年度", "地区名称", "人口数", "0-14岁人口数", "15-64岁人口数", "65岁及以上人口数",
    "总抚养比", "少年儿童抚养比", "老年人口抚养比", "老龄化率", "养老床位",
    "经济(亿元)", "养保支出(亿元)"  # 新增特征
]

# 调试：检查原始数据中江苏省的年份
print("原始数据中江苏省的年份：")
print(df[df["地区名称"] == "江苏省"]["统计年度"].unique())

# 2. 数据清洗
# 完整性检查
# 检查缺失值
missing_rate = df.isnull().mean()
print("缺失值比例：\n", missing_rate)

# 可视化缺失值 - 新增
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('数据缺失值可视化图')
plt.tight_layout()
plt.savefig('缺失值可视化图.png', dpi=300)
plt.close()

# 验证一致性：人口数 = 0-14岁人口数 + 15-64岁人口数 + 65岁及以上人口数
df["人口一致性"] = df["人口数"] - (df["0-14岁人口数"] + df["15-64岁人口数"] + df["65岁及以上人口数"])
inconsistent_rows = df[abs(df["人口一致性"]) > 1e-5]  # 允许小误差
if not inconsistent_rows.empty:
    print("人口数据不一致的行：\n", inconsistent_rows)
df = df.drop(columns=["人口一致性"])  # 删除临时列

# 缺失值处理
# 数值列：包括新特征
numeric_cols = [
    "人口数", "0-14岁人口数", "15-64岁人口数", "65岁及以上人口数",
    "总抚养比", "少年儿童抚养比", "老年人口抚养比", "老龄化率", "养老床位",
    "经济(亿元)", "养保支出(亿元)"  # 新增特征
]

# 确保数值列的数据类型为浮点数，防止插值后类型不匹配
for col in numeric_cols:
    df[col] = df[col].astype(float)

# 使用 transform 方法进行插值，确保索引对齐
for col in numeric_cols:
    df[col] = df.groupby("地区名称")[col].transform(lambda x: x.interpolate(method="linear", limit_direction="both"))

# 如果仍有缺失值，用中位数填充
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# 类别列：删除地区名称缺失的行
df = df.dropna(subset=["地区名称"])

# 调试：检查缺失值处理后江苏省的数据
print("缺失值处理后江苏省的数据年份：")
print(df[df["地区名称"] == "江苏省"]["统计年度"].unique())

# 异常值处理
# 过滤负值
for col in numeric_cols:
    df = df[df[col] >= 0]

# 调试：检查负值过滤后江苏省的数据
print("负值过滤后江苏省的数据年份：")
print(df[df["地区名称"] == "江苏省"]["统计年度"].unique())

# 比率校验：老龄化率、抚养比在 [0, 100] 范围内
df = df[(df["老龄化率"] >= 0) & (df["老龄化率"] <= 100)]
df = df[(df["总抚养比"] >= 0) & (df["总抚养比"] <= 100)]
df = df[(df["少年儿童抚养比"] >= 0) & (df["少年儿童抚养比"] <= 100)]
df = df[(df["老年人口抚养比"] >= 0) & (df["老年人口抚养比"] <= 100)]

# 调试：检查比率校验后江苏省的数据
print("比率校验后江苏省的数据年份：")
print(df[df["地区名称"] == "江苏省"]["统计年度"].unique())

# 跳过 IQR 异常值处理（对新特征影响较大，可能导致数据丢失）
print("跳过 IQR 异常值处理后江苏省的数据年份：")
print(df[df["地区名称"] == "江苏省"]["统计年度"].unique())

# 3. 特征工程
# 按地区名称和统计年度排序
df = df.sort_values(by=["地区名称", "统计年度"])

# 新增特征：养老床位_5年均值增长率
df["养老床位_5年均值增长率"] = df.groupby("地区名称")["养老床位"].transform(
    lambda x: x.pct_change().rolling(window=5, min_periods=1).mean()
).fillna(0)

# 新增特征：老龄化率增长率
df["老龄化率增长率"] = df.groupby("地区名称")["老龄化率"].pct_change().fillna(0)

# 计算老龄化率年增长率（与老龄化率增长率相同，这里保留一个，避免冗余）
# df["老龄化率年增长率"] = df.groupby("地区名称")["老龄化率"].pct_change()
# df.loc[df["统计年度"] == 2013, "老龄化率年增长率"] = 0
# 注释掉，因为老龄化率增长率已计算

# 计算床位增长率
df["床位增长率"] = df.groupby("地区名称")["养老床位"].pct_change()
df.loc[df["统计年度"] == 2013, "床位增长率"] = 0

# 计算经济和养老保险支出的增长率
df["经济增长率"] = df.groupby("地区名称")["经济(亿元)"].pct_change()
df.loc[df["统计年度"] == 2013, "经济增长率"] = 0

df["养保支出增长率"] = df.groupby("地区名称")["养保支出(亿元)"].pct_change()
df.loc[df["统计年度"] == 2013, "养保支出增长率"] = 0

# 生成 lag1 特征（包括新特征）
lag1_cols = ["老龄化率", "老年人口抚养比", "养老床位", "65岁及以上人口数", "人口数", "15-64岁人口数", "经济(亿元)",
             "养保支出(亿元)"]
for col in lag1_cols:
    df[f"{col}_lag1"] = df.groupby("地区名称")[col].shift(1)

# 滑动窗口特征：计算 3 年均值和标准差（包括新特征）
window_cols = ["老龄化率", "老年人口抚养比", "养老床位", "65岁及以上人口数", "经济(亿元)", "养保支出(亿元)"]
for col in window_cols:
    df[f"{col}_3年均值"] = df.groupby("地区名称")[col].rolling(window=3, min_periods=1).mean().reset_index(level=0,
                                                                                                           drop=True)
    df[f"{col}_3年标准差"] = df.groupby("地区名称")[col].rolling(window=3, min_periods=1).std().reset_index(level=0,
                                                                                                            drop=True)

# 计算床位覆盖率
df["床位覆盖率"] = df["养老床位"] / df["65岁及以上人口数"]

# 过滤数据：仅保留 2014-2022 年的数据
df = df[df["统计年度"] >= 2014]

# 调试：检查特征工程后江苏省的数据
print("特征工程后（过滤 2014-2022 年）江苏省的数据年份：")
print(df[df["地区名称"] == "江苏省"]["统计年度"].unique())

# 4. 数据标准化
# Z-Score 标准化（包括新特征）
scaler = StandardScaler()
scale_cols = [
                 "人口数", "0-14岁人口数", "15-64岁人口数", "65岁及以上人口数",
                 "总抚养比", "少年儿童抚养比", "老年人口抚养比", "老龄化率", "养老床位",
                 "床位增长率", "床位覆盖率",
                 "经济(亿元)", "养保支出(亿元)", "经济增长率", "养保支出增长率",
                 "老龄化率增长率", "养老床位_5年均值增长率",  # 新增特征
                 # "老龄化率年增长率",  # 已移除，避免冗余
             ] + [f"{col}_lag1" for col in lag1_cols] + \
             [f"{col}_3年均值" for col in window_cols] + [f"{col}_3年标准差" for col in window_cols]

# 确保需要标准化的列为浮点数类型
for col in scale_cols:
    df[col] = df[col].astype(float)

# 保留原始数据副本（仅 2014-2022 年）
df_raw = df.copy()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# 保存 StandardScaler 对象
scaler_path = "standard_scaler.pkl"
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"StandardScaler 已保存至 {scaler_path}，可在预测阶段用于反标准化。")

# 5. 数据探索分析
# 时间趋势：增加新特征的趋势图
plt.figure(figsize=(12, 6))
for col in ["老龄化率", "老年人口抚养比", "养老床位", "经济(亿元)", "养保支出(亿元)", "养老床位_5年均值增长率",
            "老龄化率增长率"]:
    sns.lineplot(data=df_raw, x="统计年度", y=col, hue="地区名称", legend=False)
    plt.title(f"{col} 时间趋势 (2014-2022)")
    plt.savefig(f"{col}_时间趋势_2014_2022.png", dpi=300)
    plt.clf()

# 空间分布：增加新特征的分布图
df_2022 = df_raw[df_raw["统计年度"] == 2022]
for col in ["老龄化率", "床位覆盖率", "经济(亿元)", "养保支出(亿元)", "养老床位_5年均值增长率", "老龄化率增长率"]:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_2022, x="地区名称", y=col)
    plt.title(f"2022年各省{col}分布")
    plt.xticks(rotation=90)
    plt.savefig(f"2022_{col}_分布.png", dpi=300)
    plt.clf()

# 相关性分析：增加新特征
corr_matrix = df_raw[
    ["老龄化率", "老年人口抚养比", "养老床位", "经济(亿元)", "养保支出(亿元)", "养老床位_5年均值增长率",
     "老龄化率增长率"]].corr(method="pearson")
print("相关性矩阵 (2014-2022)：\n", corr_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("老龄化率、老年人口抚养比、养老床位与经济特征的相关性 (2014-2022)")
plt.tight_layout()
plt.savefig("相关性热力图_2014_2022.png", dpi=300)
plt.clf()

# 6. 序列化处理（为 Time-MoE 准备数据）
# 按特征类别分组：人口、老龄化、床位、经济
population_features = [
    "人口数", "0-14岁人口数", "15-64岁人口数", "65岁及以上人口数",
    "人口数_lag1", "15-64岁人口数_lag1", "65岁及以上人口数_lag1",
    "65岁及以上人口数_3年均值", "65岁及以上人口数_3年标准差"
]
aging_features = [
    "老龄化率", "老年人口抚养比", "总抚养比", "少年儿童抚养比",
    "老龄化率_lag1", "老年人口抚养比_lag1", "老龄化率_3年均值",
    "老年人口抚养比_3年均值", "老龄化率_3年标准差", "老年人口抚养比_3年标准差",
    "老龄化率增长率",  # 新增特征
    "养保支出(亿元)", "养保支出增长率", "养保支出(亿元)_lag1",
    "养保支出(亿元)_3年均值", "养保支出(亿元)_3年标准差"  # 将养老保险支出相关特征加入老龄化组
]
bed_features = [
    "养老床位", "床位增长率", "床位覆盖率",
    "养老床位_lag1", "养老床位_3年均值", "养老床位_3年标准差",
    "养老床位_5年均值增长率"  # 新增特征
]
economic_features = [
    "经济(亿元)", "经济增长率", "经济(亿元)_lag1",
    "经济(亿元)_3年均值", "经济(亿元)_3年标准差"  # 新增经济特征组
]

# 为 Time-MoE 构造分组特征
time_moe_data = {
    "population": df[["统计年度", "地区名称"] + population_features],
    "aging": df[["统计年度", "地区名称"] + aging_features],
    "bed": df[["统计年度", "地区名称"] + bed_features],
    "economic": df[["统计年度", "地区名称"] + economic_features]
}

# 保存处理后的数据
df.to_csv("processed_data.csv", index=False)
for key, value in time_moe_data.items():
    value.to_csv(f"time_moe_{key}_data.csv", index=False)

print("数据已分组保存：")
print("- time_moe_population_data.csv（人口相关特征）")
print("- time_moe_aging_data.csv（老龄化相关特征）")
print("- time_moe_bed_data.csv（床位相关特征）")
print("- time_moe_economic_data.csv（经济相关特征）")
print("综合数据表已保存至 processed_data.csv，时间范围为 2014-2022 年。")

# ------------ 新增内容：高级数据可视化与建模分析 ------------

# 7. 高级数据可视化
print("\n开始生成高级数据可视化图表...")

# 7.1 PCA 降维可视化
print("执行 PCA 降维分析...")
features = df_raw[scale_cols].copy()
features = features.fillna(features.mean())  # 确保没有缺失值

pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)

# 可视化 PCA 结果
plt.figure(figsize=(12, 8))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1],
                      c=df_raw['统计年度'].astype('category').cat.codes,
                      cmap='viridis',
                      alpha=0.6,
                      s=50)
plt.colorbar(scatter, label='统计年度')
plt.title('特征空间 PCA 降维可视化')
plt.xlabel(f'主成分 1 (解释方差: {pca.explained_variance_ratio_[0]:.2f})')
plt.ylabel(f'主成分 2 (解释方差: {pca.explained_variance_ratio_[1]:.2f})')

# 添加地区标签
for i, region in enumerate(df_raw['地区名称']):
    if i % 5 == 0:  # 每隔5个点添加一个标签，避免过于拥挤
        plt.annotate(region, (pca_result[i, 0], pca_result[i, 1]),
                     fontsize=8, alpha=0.7)

plt.tight_layout()
plt.savefig('PCA_降维可视化.png', dpi=300)
plt.close()

# 7.2 地区间指标比较雷达图
print("生成地区间指标比较雷达图...")

# 选择几个代表性省份进行比较
key_provinces = ['北京市', '上海市', '广东省', '河南省', '四川省']
key_indicators = ['老龄化率', '床位覆盖率', '经济增长率', '养保支出增长率', '养老床位_5年均值增长率']

# 获取2022年数据
radar_df = df_raw[(df_raw['统计年度'] == 2022) & (df_raw['地区名称'].isin(key_provinces))]

# 准备雷达图数据
radar_data = {}
for province in key_provinces:
    prov_data = radar_df[radar_df['地区名称'] == province]
    if not prov_data.empty:
        data = []
        for indicator in key_indicators:
            # 标准化到0-1之间
            max_val = radar_df[indicator].max()
            min_val = radar_df[indicator].min()
            val = prov_data[indicator].values[0]
            normalized = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
            data.append(normalized)
        radar_data[province] = data

# 绘制雷达图
num_vars = len(key_indicators)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合雷达图

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

for i, (province, values) in enumerate(radar_data.items()):
    values += values[:1]  # 闭合数据
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=province)
    ax.fill(angles, values, alpha=0.1)

# 设置雷达图属性
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(key_indicators)
ax.set_ylim(0, 1)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title('2022年不同地区关键指标比较雷达图', size=15, pad=20)
plt.tight_layout()
plt.savefig('地区指标雷达图.png', dpi=300)
plt.close()

# 7.3 时间序列分解可视化
print("执行时间序列分解可视化...")

# 选择具有代表性的省份和指标进行时间序列分解
key_province = '江苏省'
key_indicator = '老龄化率'

# 准备时间序列数据
ts_data = df_raw[df_raw['地区名称'] == key_province].sort_values('统计年度')
if not ts_data.empty:
    ts = ts_data.set_index('统计年度')[key_indicator]

    # 进行时间序列分解
    decomposition = sm.tsa.seasonal_decompose(ts, model='additive', period=3)

    # 绘制分解结果
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    decomposition.observed.plot(ax=ax1, title='观测值')
    ax1.set_ylabel(key_indicator)

    decomposition.trend.plot(ax=ax2, title='趋势')
    ax2.set_ylabel('趋势')

    decomposition.seasonal.plot(ax=ax3, title='季节性')
    ax3.set_ylabel('季节性')

    decomposition.resid.plot(ax=ax4, title='残差')
    ax4.set_ylabel('残差')

    plt.tight_layout()
    plt.savefig(f'{key_province}_{key_indicator}_时间序列分解.png', dpi=300)
    plt.close()

# 8. 建立和比较多个预测模型
print("\n开始建立和比较多个预测模型...")

# 8.1 准备模型预测数据
# 我们选择预测老龄化率作为目标变量
target_col = '老龄化率'

# 选择部分重要特征进行模型训练
selected_features = [
    '老年人口抚养比', '床位覆盖率',
    '经济(亿元)', '养保支出(亿元)',
    '老龄化率_lag1', '养老床位_5年均值增长率',
    '养老床位_3年均值', '65岁及以上人口数_3年均值'
]

# 准备数据
X = df_raw[selected_features].fillna(0)
y = df_raw[target_col]

# 时间序列分割
tscv = TimeSeriesSplit(n_splits=5)

# 8.2 创建各种模型
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'SVR': SVR(kernel='rbf')
}

# 8.3 训练和评估模型
results = {}
cv_scores = {}

print("开始模型训练和评估...")
for name, model in models.items():
    # 时间序列交叉验证
    mse_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')

    cv_scores[name] = {
        'MSE': -mse_scores.mean(),
        'R2': r2_scores.mean()
    }

    # 分割数据集进行最终评估
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'pred': y_pred,
        'true': y_test
    }

    print(f"{name} 模型评估完成: MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

# 8.4 可视化模型比较
# 绘制不同模型的 MSE 比较
plt.figure(figsize=(12, 6))
model_names = list(results.keys())
mse_values = [results[name]['MSE'] for name in model_names]

sns.barplot(x=model_names, y=mse_values)
plt.title('不同模型的均方误差(MSE)比较')
plt.ylabel('MSE（越低越好）')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('模型MSE比较.png', dpi=300)
plt.close()

# 绘制不同模型的 R2 比较
plt.figure(figsize=(12, 6))
r2_values = [results[name]['R2'] for name in model_names]

sns.barplot(x=model_names, y=r2_values)
plt.title('不同模型的R²分数比较')
plt.ylabel('R²（越高越好）')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('模型R2比较.png', dpi=300)
plt.close()

# 8.5 交叉验证评估可视化
plt.figure(figsize=(14, 6))
mse_cv = [cv_scores[name]['MSE'] for name in model_names]
r2_cv = [cv_scores[name]['R2'] for name in model_names]

width = 0.35
x = np.arange(len(model_names))

plt.bar(x - width / 2, mse_cv, width, label='MSE')
plt.bar(x + width / 2, r2_cv, width, label='R²')

plt.xlabel('模型')
plt.ylabel('评分')
plt.title('模型交叉验证性能比较')
plt.xticks(x, model_names, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('交叉验证比较.png', dpi=300)
plt.close()

# 8.6 详细比较最佳两个模型
# 从结果中选出最佳的两个模型
best_models = sorted(model_names, key=lambda x: results[x]['MSE'])[:2]
print(f"最佳两个模型: {best_models}")

# 绘制预测值与真实值的对比散点图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for i, name in enumerate(best_models):
    y_test = results[name]['true']
    y_pred = results[name]['pred']

    # 散点图
    axes[i].scatter(y_test, y_pred, alpha=0.5)

    # 添加 y=x 线
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')

    # 设置标题和标签
    axes[i].set_title(f'{name} 预测值 vs 真实值')
    axes[i].set_xlabel('真实值')
    axes[i].set_ylabel('预测值')

    # 添加R²信息
    axes[i].text(0.05, 0.95, f"R² = {results[name]['R2']:.4f}", transform=axes[i].transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('最佳模型预测比较.png', dpi=300)
    plt.close()

    # 8.7 特征重要性分析
    print("\n开始进行特征重要性分析...")

    # 对有feature_importances_属性的模型进行特征重要性分析
    for name in ['RandomForest', 'GradientBoosting']:
        if name in models:
            # 重新训练模型确保完整性
            models[name].fit(X, y)

            # 获取特征重要性
            importances = models[name].feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(12, 6))
            plt.title(f'{name} 特征重要性排序')
            plt.bar(range(X.shape[1]), importances[indices], align='center')
            plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
            plt.tight_layout()
            plt.savefig(f'{name}_特征重要性.png', dpi=300)
            plt.close()

            # 打印特征重要性
            print(f"{name} 特征重要性排序:")
            for i in range(X.shape[1]):
                print(f"{X.columns[indices[i]]}: {importances[indices[i]]:.4f}")

    # 8.8 学习曲线分析
    print("\n开始绘制学习曲线...")

    from sklearn.model_selection import learning_curve


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("训练样本数量")
    plt.ylabel("得分")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring="neg_mean_squared_error")

    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="训练集得分")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="验证集得分")

    plt.legend(loc="best")
    return plt


# 为最佳两个模型绘制学习曲线
for name in best_models:
    if name in models:
        plt_curve = plot_learning_curve(
            models[name], f"{name} 学习曲线", X, y,
            cv=TimeSeriesSplit(n_splits=5))
        plt_curve.savefig(f'{name}_学习曲线.png', dpi=300)
        plt.close()

# 8.9 残差分析
print("\n开始进行残差分析...")

for name in best_models:
    # 重新训练模型
    model = models[name]
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred

    # 绘制残差图
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='--')
    plt.title(f'{name} 残差分布')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.grid(True)

    # 添加残差正态性检验结果
    from scipy import stats

    stat, p_value = stats.normaltest(residuals)
    plt.text(0.05, 0.95, f"正态性检验 p值: {p_value:.4f}",
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{name}_残差分析.png', dpi=300)
    plt.close()

    # Q-Q图检验残差正态性
    plt.figure(figsize=(10, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'{name} 残差Q-Q图')
    plt.tight_layout()
    plt.savefig(f'{name}_残差QQ图.png', dpi=300)
    plt.close()

# 9. 高级模型比较与评估
print("\n开始高级模型比较与评估...")

# 9.1 使用随机搜索优化最佳模型的超参数
from sklearn.model_selection import RandomizedSearchCV

# 为最佳模型设置超参数搜索空间
if 'RandomForest' in best_models:
    param_grid_rf = {
        'n_estimators': [50, 100, 200, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rs_rf = RandomizedSearchCV(RandomForestRegressor(random_state=42),
                               param_distributions=param_grid_rf,
                               n_iter=20,
                               cv=TimeSeriesSplit(n_splits=5),
                               scoring='neg_mean_squared_error',
                               random_state=42)
    rs_rf.fit(X, y)

    print("随机森林最佳参数:", rs_rf.best_params_)
    print("随机森林最佳MSE:", -rs_rf.best_score_)

    # 使用最佳参数的模型
    best_rf = rs_rf.best_estimator_

    # 保存优化后的模型
    best_model_path = "best_random_forest_model.pkl"
    with open(best_model_path, "wb") as f:
        pickle.dump(best_rf, f)
    print(f"最佳随机森林模型已保存至 {best_model_path}")

if 'GradientBoosting' in best_models:
    param_grid_gb = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.7, 0.8, 0.9, 1.0]
    }

    rs_gb = RandomizedSearchCV(GradientBoostingRegressor(random_state=42),
                               param_distributions=param_grid_gb,
                               n_iter=20,
                               cv=TimeSeriesSplit(n_splits=5),
                               scoring='neg_mean_squared_error',
                               random_state=42)
    rs_gb.fit(X, y)

    print("梯度提升最佳参数:", rs_gb.best_params_)
    print("梯度提升最佳MSE:", -rs_gb.best_score_)

    # 使用最佳参数的模型
    best_gb = rs_gb.best_estimator_

    # 保存优化后的模型
    best_model_path = "best_gradient_boosting_model.pkl"
    with open(best_model_path, "wb") as f:
        pickle.dump(best_gb, f)
    print(f"最佳梯度提升模型已保存至 {best_model_path}")

# 9.2 绘制最优模型的学习率影响曲线(针对梯度提升模型)
if 'GradientBoosting' in models:
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    train_scores = []
    test_scores = []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for lr in learning_rates:
        model = GradientBoostingRegressor(learning_rate=lr, random_state=42)
        model.fit(X_train, y_train)

        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        train_scores.append(train_score)
        test_scores.append(test_score)

    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, train_scores, marker='o', label='训练集R²')
    plt.plot(learning_rates, test_scores, marker='o', label='测试集R²')
    plt.xscale('log')
    plt.title('学习率对梯度提升模型性能的影响')
    plt.xlabel('学习率 (对数尺度)')
    plt.ylabel('R²得分')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('学习率影响曲线.png', dpi=300)
    plt.close()

# 9.3 使用集成方法组合模型
print("\n开始构建集成模型...")
# 创建包含所有模型的集成
from sklearn.ensemble import VotingRegressor

# 重新定义模型，确保它们都是新的实例
estimators = []
for name, model in models.items():
    if name in best_models:
        estimators.append((name, model.__class__(**model.get_params())))

# 调试：打印 estimators
print("Estimators for VotingRegressor:", estimators)

# 输入验证
if not isinstance(estimators, list) or not all(isinstance(e, tuple) and len(e) == 2 for e in estimators):
    print("错误：estimators 不是 (name, model) 元组的列表")
    exit(1)

# 创建投票回归器
if len(estimators) >= 2:  # 至少需要两个模型
    voting_reg = VotingRegressor(estimators)

    # 在整个数据集上评估集成模型
    tscv = TimeSeriesSplit(n_splits=5)
    mse_scores = cross_val_score(voting_reg, X, y, cv=tscv, scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(voting_reg, X, y, cv=tscv, scoring='r2')

    print(f"集成模型平均MSE: {-mse_scores.mean():.4f}")
    print(f"集成模型平均R²: {r2_scores.mean():.4f}")

    # 训练和测试数据划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练集成模型
    voting_reg.fit(X_train, y_train)
    print("voting_reg.estimators_:", voting_reg.estimators_)  # 调试输出
    y_pred_ensemble = voting_reg.predict(X_test)

    # 评估集成模型
    mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
    r2_ensemble = r2_score(y_test, y_pred_ensemble)

    # 收集单个模型预测结果
    predictions = {}
    metrics = {'MSE': [], 'R²': []}
    model_names = []

    # 使用 named_estimators_ 替代 estimators_
    for name, model in voting_reg.named_estimators_.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics['MSE'].append(mse)
        metrics['R²'].append(r2)
        model_names.append(name)

    # 添加集成模型的指标
    metrics['MSE'].append(mse_ensemble)
    metrics['R²'].append(r2_ensemble)
    model_names.append('Ensemble')

    # 绘制比较图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # MSE 比较
    ax1.bar(model_names, metrics['MSE'])
    ax1.set_title('模型MSE比较 (越低越好)')
    ax1.set_ylabel('MSE')
    ax1.set_ylim(bottom=0)  # 修复可能的语法错误，应为 ax1.set_ylim(bottom=0)

    # R² 比较
    ax2.bar(model_names, metrics['R²'])
    ax2.set_title('模型R²比较 (越高越好)')
    ax2.set_ylabel('R²')

    plt.tight_layout()
    plt.savefig('集成模型性能比较.png', dpi=300)
    plt.close()

    # 保存集成模型
    ensemble_model_path = "ensemble_model.pkl"
    with open(ensemble_model_path, "wb") as f:
        pickle.dump(voting_reg, f)
    print(f"集成模型已保存至 {ensemble_model_path}")
else:
    print("无法创建集成模型：没有足够的模型")

# 10. 区域预测性能比较
print("\n开始区域预测性能比较...")

# 选择几个代表性地区进行预测性能比较
key_regions = ['北京市', '上海市', '广东省', '四川省', '河南省']

# 使用最佳模型进行分区域性能评估
best_model_name = best_models[0]  # 使用表现最好的模型
best_model = models[best_model_name]

# 初始化存储结果的字典
region_metrics = {region: {'MSE': [], 'R2': []} for region in key_regions}

# the best model
for region in key_regions:
    # 筛选该地区的数据
    region_data = df_raw[df_raw['地区名称'] == region]

    if len(region_data) > 5:  # 确保数据量足够
        X_region = region_data[selected_features].fillna(0)
        y_region = region_data[target_col]

        # 使用留一交叉验证(特别适合小样本)
        from sklearn.model_selection import LeaveOneOut

        loo = LeaveOneOut()

        y_true_all = []
        y_pred_all = []

        for train_idx, test_idx in loo.split(X_region):
            X_train, X_test = X_region.iloc[train_idx], X_region.iloc[test_idx]
            y_train, y_test = y_region.iloc[train_idx], y_region.iloc[test_idx]

            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)

            y_true_all.append(y_test.iloc[0])
            y_pred_all.append(y_pred[0])

        # 计算该地区的模型性能
        mse = mean_squared_error(y_true_all, y_pred_all)
        r2 = r2_score(y_true_all, y_pred_all)

        region_metrics[region]['MSE'] = mse
        region_metrics[region]['R2'] = r2

        print(f"{region} 预测性能: MSE={mse:.4f}, R²={r2:.4f}")

# 可视化地区性能对比
mse_values = [region_metrics[region]['MSE'] for region in key_regions if 'MSE' in region_metrics[region]]
r2_values = [region_metrics[region]['R2'] for region in key_regions if 'R2' in region_metrics[region]]
regions = [region for region in key_regions if 'MSE' in region_metrics[region]]

# 创建双Y轴图表
fig, ax1 = plt.subplots(figsize=(12, 6))

x = np.arange(len(regions))
width = 0.35

# MSE 条形图
rects1 = ax1.bar(x - width / 2, mse_values, width, label='MSE', color='crimson')
ax1.set_ylabel('MSE (越低越好)', color='crimson')
ax1.tick_params(axis='y', labelcolor='crimson')

# 添加第二个Y轴
ax2 = ax1.twinx()
rects2 = ax2.bar(x + width / 2, r2_values, width, label='R²', color='royalblue')
ax2.set_ylabel('R² (越高越好)', color='royalblue')
ax2.tick_params(axis='y', labelcolor='royalblue')

# 设置X轴
ax1.set_xticks(x)
ax1.set_xticklabels(regions)
ax1.set_xlabel('地区')

# 添加标题和图例
plt.title(f'{best_model_name} 模型在不同地区的预测性能')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

plt.tight_layout()
plt.savefig('地区预测性能比较.png', dpi=300)
plt.close()

# 11. 年份预测准确度分析
print("\n开始年份预测准确度分析...")

# 选择最佳模型进行年份预测准确度分析
years = sorted(df_raw['统计年度'].unique())
year_metrics = {year: {'MSE': 0, 'R2': 0} for year in years}

for year in years:
    # 分割当年的训练集和测试集
    train_df = df_raw[df_raw['统计年度'] != year]
    test_df = df_raw[df_raw['统计年度'] == year]

    if not test_df.empty:
        X_train = train_df[selected_features].fillna(0)
        y_train = train_df[target_col]

        X_test = test_df[selected_features].fillna(0)
        y_test = test_df[target_col]

        # 使用最佳模型进行训练和预测
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        # 计算指标
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        year_metrics[year]['MSE'] = mse
        year_metrics[year]['R2'] = r2

        print(f"{year}年预测性能: MSE={mse:.4f}, R²={r2:.4f}")

# 可视化年份预测准确度
mse_by_year = [year_metrics[year]['MSE'] for year in years]
r2_by_year = [year_metrics[year]['R2'] for year in years]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# MSE随年份变化
ax1.plot(years, mse_by_year, marker='o', linestyle='-', color='crimson')
ax1.set_title('年份预测MSE变化趋势')
ax1.set_ylabel('MSE (越低越好)')
ax1.grid(True)

# R²随年份变化
ax2.plot(years, r2_by_year, marker='o', linestyle='-', color='royalblue')
ax2.set_title('年份预测R²变化趋势')
ax2.set_xlabel('年份')
ax2.set_ylabel('R² (越高越好)')
ax2.grid(True)

plt.tight_layout()
plt.savefig('年份预测准确度分析.png', dpi=300)
plt.close()

# 12. 特征交互影响分析
print("\n开始特征交互影响分析...")

# 选择两个重要特征进行交互分析
# 假设从RandomForest模型中获取了两个最重要的特征
if 'RandomForest' in models:
    model = models['RandomForest'].fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    if len(indices) >= 2:
        feature1 = X.columns[indices[0]]
        feature2 = X.columns[indices[1]]

        # 创建特征交互网格
        x_min, x_max = X[feature1].min() - 0.1, X[feature1].max() + 0.1
        y_min, y_max = X[feature2].min() - 0.1, X[feature2].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                             np.linspace(y_min, y_max, 50))

        # 创建测试点
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # 创建全0的特征矩阵
        X_grid = np.zeros((grid_points.shape[0], X.shape[1]))

        # 设置特征值
        X_grid[:, X.columns.get_loc(feature1)] = grid_points[:, 0]
        X_grid[:, X.columns.get_loc(feature2)] = grid_points[:, 1]

        # 预测结果
        model = models['RandomForest']
        z = model.predict(X_grid)
        z = z.reshape(xx.shape)

        # 绘制特征交互热力图
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, z, levels=15, cmap='viridis', alpha=0.7)
        plt.colorbar(label=target_col)
        plt.scatter(X[feature1], X[feature2], c=y, cmap='coolwarm',
                    edgecolor='k', s=50, alpha=0.6)
        plt.title(f'{feature1} 和 {feature2} 对 {target_col} 的影响')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('特征交互影响分析.png', dpi=300)
        plt.close()

# 13. 3D 特征影响可视化
print("\n开始3D特征影响可视化...")

# 选择两个最重要特征进行3D可视化
if 'RandomForest' in models and len(indices) >= 2:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制3D散点图
    scatter = ax.scatter(X[feature1], X[feature2], y,
                         c=y, cmap='coolwarm', s=50, alpha=0.6)

    # 创建网格点并预测
    # 为了减少密度，降低点数
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 20),
                         np.linspace(y_min, y_max, 20))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    X_grid = np.zeros((grid_points.shape[0], X.shape[1]))
    X_grid[:, X.columns.get_loc(feature1)] = grid_points[:, 0]
    X_grid[:, X.columns.get_loc(feature2)] = grid_points[:, 1]

    z = model.predict(X_grid)
    z = z.reshape(xx.shape)

    # 绘制预测曲面
    surf = ax.plot_surface(xx, yy, z, cmap='viridis', alpha=0.5, linewidth=0, antialiased=True)

    # 添加标签和图例
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_zlabel(target_col)
    ax.set_title(f'特征 {feature1} 和 {feature2} 对 {target_col} 的3D影响')

    # 添加颜色条
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label=target_col)

    plt.tight_layout()
    plt.savefig('3D特征影响可视化.png', dpi=300)
    plt.close()

# 14. 模型预测误差分布分析
print("\n开始模型预测误差分布分析...")

# 使用最佳模型在全数据集上进行预测和误差分析
best_model.fit(X, y)
y_pred = best_model.predict(X)
errors = y - y_pred

# 14.1 误差直方图
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True, bins=30)
plt.axvline(x=0, color='r', linestyle='--')
plt.title(f'{best_model_name} 模型预测误差分布')
plt.xlabel('预测误差')
plt.ylabel('频率')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('预测误差直方图.png', dpi=300)
plt.close()

# 14.2 误差地区分布
error_by_region = df_raw.copy()
error_by_region['预测值'] = y_pred
error_by_region['实际值'] = y
error_by_region['误差'] = errors
error_by_region['误差绝对值'] = np.abs(errors)

# 计算各地区平均误差
region_error = error_by_region.groupby('地区名称')['误差'].mean().sort_values()
region_abs_error = error_by_region.groupby('地区名称')['误差绝对值'].mean().sort_values()

# 绘制地区误差条形图
plt.figure(figsize=(14, 7))
region_error.plot(kind='bar', color=np.where(region_error > 0, 'indianred', 'steelblue'))
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.title('各地区平均预测误差')
plt.xlabel('地区')
plt.ylabel('平均误差')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('地区平均误差.png', dpi=300)
plt.close()

# 绘制地区绝对误差条形图
plt.figure(figsize=(14, 7))
region_abs_error.plot(kind='bar', color='darkblue')
plt.title('各地区平均绝对预测误差')
plt.xlabel('地区')
plt.ylabel('平均绝对误差')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('地区平均绝对误差.png', dpi=300)
plt.close()

# 14.3 误差与目标值关系
plt.figure(figsize=(10, 6))
plt.scatter(y, errors, alpha=0.6, edgecolor='k')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('预测误差与实际值关系')
plt.xlabel('实际' + target_col)
plt.ylabel('预测误差')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('误差与实际值关系.png', dpi=300)
plt.close()

# 继续之前的代码，从中断处开始
# 15. 经济与老龄化关系分析（热门研究方向）
print("\n继续经济与老龄化关系分析...")

# 绘制经济增长率与老龄化率关系的散点图
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_raw, x="经济增长率", y="老龄化率", hue="地区名称", size="经济(亿元)", sizes=(50, 500),
                alpha=0.7)
plt.title("各地区经济增长率与老龄化率关系 (2014-2022)")
plt.xlabel("经济增长率")
plt.ylabel("老龄化率 (%)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("经济增长率与老龄化率关系.png", dpi=300)
plt.close()

# 15.1 经济与养老保险支出关系
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_raw, x="经济(亿元)", y="养保支出(亿元)", hue="统计年度", palette="viridis", size="老龄化率",
                sizes=(50, 500))
plt.title("经济规模与养老保险支出的关系 (2014-2022)")
plt.xlabel("经济规模 (亿元)")
plt.ylabel("养老保险支出 (亿元)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("经济与养老保险支出关系.png", dpi=300)
plt.close()

# 16. 模型训练过程中的损失曲线
print("\n开始绘制模型训练过程中的损失曲线...")

# 为最佳模型（RandomForest 和 GradientBoosting）绘制训练过程中的损失曲线
best_models = ['RandomForest', 'GradientBoosting']  # 基于之前的分析选择最佳模型
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for name in best_models:
    model = models[name]
    train_losses = []
    test_losses = []
    iterations = range(10, 101, 10) if name == 'RandomForest' else range(10, 101, 10)  # 调整迭代次数

    for n_estimators in iterations:
        if name == 'RandomForest':
            model.set_params(n_estimators=n_estimators, random_state=42)
        else:
            model.set_params(n_estimators=n_estimators, random_state=42)

        model.fit(X_train, y_train)

        # 训练集损失
        y_train_pred = model.predict(X_train)
        train_loss = mean_squared_error(y_train, y_train_pred)
        train_losses.append(train_loss)

        # 测试集损失
        y_test_pred = model.predict(X_test)
        test_loss = mean_squared_error(y_test, y_test_pred)
        test_losses.append(test_loss)

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_losses, label='训练集 MSE', marker='o', color='blue')
    plt.plot(iterations, test_losses, label='测试集 MSE', marker='s', color='red')
    plt.title(f'{name} 模型训练过程中的损失曲线')
    plt.xlabel('树的数量 (n_estimators)')
    plt.ylabel('均方误差 (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{name}_损失曲线.png', dpi=300)
    plt.close()

# 17. 模型预测结果的时间序列可视化
print("\n开始绘制模型预测结果的时间序列...")

# 使用最佳模型对每个地区进行时间序列预测
best_model_name = 'RandomForest'  # 假设 RandomForest 是最佳模型
best_model = models[best_model_name]

for region in key_regions:
    region_data = df_raw[df_raw['地区名称'] == region]
    if not region_data.empty:
        X_region = region_data[selected_features].fillna(0)
        y_region = region_data[target_col]
        y_pred = best_model.predict(X_region)

        plt.figure(figsize=(12, 6))
        plt.plot(region_data['统计年度'], y_region, label='实际值', marker='o', color='blue')
        plt.plot(region_data['统计年度'], y_pred, label='预测值', marker='s', color='red', linestyle='--')
        plt.title(f'{region} {target_col} 预测与实际值对比 (2014-2022)')
        plt.xlabel('年份')
        plt.ylabel(target_col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{region}_时间序列预测.png', dpi=300)
        plt.close()

# 18. 模型预测误差随特征值变化
print("\n开始分析预测误差随特征值变化...")

# 选择两个关键特征，分析误差与特征值的关系
key_features = ['老年人口抚养比', '床位覆盖率']
for feature in key_features:
    plt.figure(figsize=(10, 6))
    errors = y - best_model.predict(X)
    sns.scatterplot(x=X[feature], y=errors, hue=df_raw['统计年度'], palette='viridis', alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title(f'预测误差与 {feature} 的关系')
    plt.xlabel(feature)
    plt.ylabel('预测误差')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'误差与{feature}关系.png', dpi=300)
    plt.close()

# 19. 模型稳定性分析：不同数据子集的性能
print("\n开始模型稳定性分析...")

# 随机抽样不同比例的数据，评估模型性能的稳定性
sample_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
stability_results = {'MSE': [], 'R2': []}

for ratio in sample_ratios:
    sample_idx = np.random.choice(X.index, size=int(len(X) * ratio), replace=False)
    X_sample = X.loc[sample_idx]
    y_sample = y.loc[sample_idx]

    mse_scores = cross_val_score(best_model, X_sample, y_sample, cv=TimeSeriesSplit(n_splits=5),
                                 scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(best_model, X_sample, y_sample, cv=TimeSeriesSplit(n_splits=5), scoring='r2')

    stability_results['MSE'].append(-mse_scores.mean())
    stability_results['R2'].append(r2_scores.mean())

# 绘制稳定性分析图
plt.figure(figsize=(10, 6))
plt.plot(sample_ratios, stability_results['MSE'], label='MSE', marker='o', color='red')
plt.plot(sample_ratios, stability_results['R2'], label='R²', marker='s', color='blue')
plt.title(f'{best_model_name} 模型在不同数据比例下的稳定性')
plt.xlabel('数据比例')
plt.ylabel('性能指标')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('模型稳定性分析.png', dpi=300)
plt.close()

# 20. 特征贡献度分解（SHAP分析）
print("\n开始进行 SHAP 特征贡献度分析...")

try:
    import shap

    # 使用 SHAP 解释 RandomForest 模型
    explainer = shap.TreeExplainer(models['RandomForest'])
    shap_values = explainer.shap_values(X)

    # 绘制 SHAP 总结图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title('特征对老龄化率预测的贡献度 (SHAP)')
    plt.tight_layout()
    plt.savefig('SHAP特征贡献度.png', dpi=300)
    plt.close()

    # 绘制 SHAP 蜂群图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title('特征对老龄化率预测的分布影响 (SHAP)')
    plt.tight_layout()
    plt.savefig('SHAP蜂群图.png', dpi=300)
    plt.close()
except ImportError:
    print("SHAP 库未安装，跳过 SHAP 分析。请安装 shap：pip install shap")

# 21. 模型预测的置信区间分析
print("\n开始模型预测置信区间分析...")

# 使用 GradientBoosting 模型生成预测的置信区间
gb_model = GradientBoostingRegressor(random_state=42, n_estimators=100)
gb_model.fit(X_train, y_train)
y_pred = gb_model.predict(X_test)

# 通过多次采样估计置信区间
n_bootstraps = 100
bootstrap_preds = np.zeros((len(X_test), n_bootstraps))

for i in range(n_bootstraps):
    sample_idx = np.random.choice(X_train.index, size=len(X_train), replace=True)
    X_boot = X_train.loc[sample_idx]
    y_boot = y_train.loc[sample_idx]

    gb_model.fit(X_boot, y_boot)
    bootstrap_preds[:, i] = gb_model.predict(X_test)

# 计算 95% 置信区间
lower_bound = np.percentile(bootstrap_preds, 2.5, axis=1)
upper_bound = np.percentile(bootstrap_preds, 97.5, axis=1)

# 绘制预测值与置信区间
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='实际值', marker='o', color='blue')
plt.plot(y_pred, label='预测值', marker='s', color='red')
plt.fill_between(range(len(y_pred)), lower_bound, upper_bound, color='gray', alpha=0.3, label='95% 置信区间')
plt.title('GradientBoosting 模型预测值与置信区间')
plt.xlabel('测试样本')
plt.ylabel(target_col)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('预测置信区间.png', dpi=300)
plt.close()

# 22. 模型性能随时间窗口变化分析
print("\n开始分析模型性能随时间窗口变化...")

# 测试不同时间窗口（3年、5年、7年）对模型性能的影响
window_sizes = [3, 5, 7]
window_results = {'MSE': [], 'R2': []}

for window in window_sizes:
    # 重新计算滑动窗口特征
    df_temp = df_raw.copy()
    for col in window_cols:
        df_temp[f"{col}_{window}年均值"] = df_temp.groupby("地区名称")[col].rolling(window=window,
                                                                                    min_periods=1).mean().reset_index(
            level=0, drop=True)
        df_temp[f"{col}_{window}年标准差"] = df_temp.groupby("地区名称")[col].rolling(window=window,
                                                                                      min_periods=1).std().reset_index(
            level=0, drop=True)

    # 更新特征列表
    temp_features = [f"{col}_{window}年均值" for col in window_cols] + [f"{col}_{window}年标准差" for col in
                                                                        window_cols]
    temp_features = [f for f in temp_features if f in df_temp.columns] + selected_features
    temp_features = list(set(temp_features).intersection(df_temp.columns))

    X_temp = df_temp[temp_features].fillna(0)
    y_temp = df_temp[target_col]

    mse_scores = cross_val_score(best_model, X_temp, y_temp, cv=TimeSeriesSplit(n_splits=5),
                                 scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(best_model, X_temp, y_temp, cv=TimeSeriesSplit(n_splits=5), scoring='r2')

    window_results['MSE'].append(-mse_scores.mean())
    window_results['R2'].append(r2_scores.mean())

# 绘制时间窗口影响图
plt.figure(figsize=(10, 6))
plt.plot(window_sizes, window_results['MSE'], label='MSE', marker='o', color='red')
plt.plot(window_sizes, window_results['R2'], label='R²', marker='s', color='blue')
plt.title(f'{best_model_name} 模型在不同时间窗口下的性能')
plt.xlabel('时间窗口大小 (年)')
plt.ylabel('性能指标')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('时间窗口性能分析.png', dpi=300)
plt.close()

# 23. 模型对异常数据的鲁棒性分析
print("\n开始分析模型对异常数据的鲁棒性...")

# 向数据中添加不同比例的噪声，测试模型鲁棒性
noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2]
robustness_results = {'MSE': [], 'R2': []}

for noise in noise_levels:
    X_noisy = X.copy()
    noise_matrix = np.random.normal(0, noise * X.std(), X.shape)
    X_noisy += noise_matrix

    mse_scores = cross_val_score(best_model, X_noisy, y, cv=TimeSeriesSplit(n_splits=5),
                                 scoring='neg_mean_squared_error')
    r2_scores = cross_val_score(best_model, X_noisy, y, cv=TimeSeriesSplit(n_splits=5), scoring='r2')

    robustness_results['MSE'].append(-mse_scores.mean())
    robustness_results['R2'].append(r2_scores.mean())

# 绘制鲁棒性分析图
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, robustness_results['MSE'], label='MSE', marker='o', color='red')
plt.plot(noise_levels, robustness_results['R2'], label='R²', marker='s', color='blue')
plt.title(f'{best_model_name} 模型在不同噪声水平下的鲁棒性')
plt.xlabel('噪声水平 (标准差比例)')
plt.ylabel('性能指标')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('模型鲁棒性分析.png', dpi=300)
plt.close()

# 24. 模型预测结果的区域聚类分析
print("\n开始区域聚类分析...")

from sklearn.cluster import KMeans

# 使用预测误差进行区域聚类
error_by_region['误差绝对值'] = np.abs(error_by_region['误差'])
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(error_by_region[['误差', '误差绝对值']])

# 可视化聚类结果
plt.figure(figsize=(12, 8))
plt.scatter(error_by_region['误差'], error_by_region['误差绝对值'], c=clusters, cmap='viridis', s=100)
for i, region in enumerate(error_by_region.index):
    plt.annotate(region, (error_by_region['误差'].iloc[i], error_by_region['误差绝对值'].iloc[i]), fontsize=8)
plt.title('基于预测误差的区域聚类分析')
plt.xlabel('平均误差')
plt.ylabel('平均绝对误差')
plt.grid(True)
plt.tight_layout()
plt.savefig('区域聚类分析.png', dpi=300)
plt.close()

# 25. 模型预测结果的多维标度分析 (MDS)
print("\n开始多维标度分析 (MDS)...")

from sklearn.manifold import MDS

# 使用预测值和实际值的差异进行 MDS
mds = MDS(n_components=2, random_state=42)
mds_result = mds.fit_transform(np.column_stack((y, y_pred)))

plt.figure(figsize=(12, 8))
plt.scatter(mds_result[:, 0], mds_result[:, 1], c=df_raw['统计年度'], cmap='viridis', s=100)
plt.colorbar(label='统计年度')
for i, region in enumerate(df_raw['地区名称']):
    if i % 5 == 0:  # 减少标签拥挤
        plt.annotate(region, (mds_result[i, 0], mds_result[i, 1]), fontsize=8)
plt.title('基于预测与实际值的多维标度分析 (MDS)')
plt.xlabel('MDS 维度 1')
plt.ylabel('MDS 维度 2')
plt.grid(True)
plt.tight_layout()
plt.savefig('MDS分析.png', dpi=300)
plt.close()

# 26. 保存所有可视化结果的描述文件
print("\n生成可视化结果描述文件...")

visualizations = [
    {"filename": "经济增长率与老龄化率关系.png",
     "description": "展示各地区经济增长率与老龄化率的关系，点大小表示经济规模"},
    {"filename": "经济与养老保险支出关系.png", "description": "经济规模与养老保险支出的散点图，点大小表示老龄化率"},
    {"filename": "RandomForest_损失曲线.png", "description": "RandomForest 模型随树数量增加的训练和测试损失曲线"},
    {"filename": "GradientBoosting_损失曲线.png",
     "description": "GradientBoosting 模型随树数量增加的训练和测试损失曲线"},
    {"filename": f"{key_regions[0]}_时间序列预测.png",
     "description": f"{key_regions[0]} 老龄化率的实际值与预测值时间序列对比"},
    {"filename": f"{key_regions[1]}_时间序列预测.png",
     "description": f"{key_regions[1]} 老龄化率的实际值与预测值时间序列对比"},
    {"filename": f"{key_regions[2]}_时间序列预测.png",
     "description": f"{key_regions[2]} 老龄化率的实际值与预测值时间序列对比"},
    {"filename": f"{key_regions[3]}_时间序列预测.png",
     "description": f"{key_regions[3]} 老龄化率的实际值与预测值时间序列对比"},
    {"filename": f"{key_regions[4]}_时间序列预测.png",
     "description": f"{key_regions[4]} 老龄化率的实际值与预测值时间序列对比"},
    {"filename": "误差与老年人口抚养比关系.png", "description": "预测误差与老年人口抚养比的关系散点图"},
    {"filename": "误差与床位覆盖率关系.png", "description": "预测误差与床位覆盖率的关系散点图"},
    {"filename": "模型稳定性分析.png", "description": "RandomForest 模型在不同数据比例下的 MSE 和 R² 稳定性"},
    {"filename": "SHAP特征贡献度.png", "description": "特征对老龄化率预测的平均贡献度 (SHAP 值)"},
    {"filename": "SHAP蜂群图.png", "description": "特征对老龄化率预测的分布影响 (SHAP 蜂群图)"},
    {"filename": "预测置信区间.png", "description": "GradientBoosting 模型预测值的 95% 置信区间"},
    {"filename": "时间窗口性能分析.png", "description": "RandomForest 模型在不同时间窗口大小下的性能"},
    {"filename": "模型鲁棒性分析.png", "description": "RandomForest 模型在不同噪声水平下的鲁棒性"},
    {"filename": "区域聚类分析.png", "description": "基于预测误差的区域聚类分析"},
    {"filename": "MDS分析.png", "description": "基于预测与实际值的多维标度分析"}
]

with open("visualization_descriptions.txt", "w", encoding="utf-8") as f:
    for vis in visualizations:
        f.write(f"文件名: {vis['filename']}\n描述: {vis['description']}\n\n")

print("所有可视化结果的描述已保存至 visualization_descriptions.txt")

# 27. 最终总结输出
print("\n模型分析与可视化完成！")
print("所有可视化图像已保存为 PNG 格式，分辨率为 300 DPI，适合论文使用。")
print("请检查 visualization_descriptions.txt 以获取所有生成图像的描述。")
print("建议在论文中结合这些图像，突出模型性能、特征重要性、区域差异和时间趋势分析。")