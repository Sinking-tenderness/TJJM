import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from pmdarima import auto_arima
from sklearn.decomposition import PCA
from matplotlib import rcParams
import warnings
import os


# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
plt.rcParams['font.sans-serif'] = ['SimHei']  # or another Chinese font you have installed
plt.rcParams['axes.unicode_minus'] = False

# Original functions from the first part
def load_real_data(file_path="消费.xlsx"):
    print(f"正在从{file_path}读取数据...")
    try:
        df = pd.read_excel(file_path)

        column_mapping = {
            '统计年份': '年份',
            '地区名称': '省份',
            '65岁及以上人口数/人': '65岁及以上人口数',
            '城镇单位就业人口/万人': '城镇单位就业人员数',
            '城镇居民消费水平/元': '城镇居民消费水平',
            '城镇人口/万人': '城镇人口数',
            '基本养老保险基金支出/万元': '基本养老保险基金支出',
            '老年人口抚养比/%': '老年人口抚养比'
        }

        df = df.rename(columns=column_mapping)

        required_columns = [
            '年份', '省份', '65岁及以上人口数', '城镇单位就业人员数',
            '城镇居民消费水平', '城镇人口数', '基本养老保险基金支出', '老年人口抚养比'
        ]

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"错误: 缺少以下必需列: {missing_cols}")
            print(f"当前列: {df.columns.tolist()}")
            return None

        df['年份'] = pd.to_numeric(df['年份'], errors='coerce').astype('Int64')
        if df['年份'].isnull().any():
            print("错误: '年份'列包含无效值")
            return None

        for col in required_columns[2:]:
            if df[col].isnull().all():
                print(f"错误: 列 '{col}' 全为空")
                return None
            nan_ratio = df[col].isnull().mean()
            if nan_ratio > 0.5:
                print(f"警告: 列 '{col}' 缺失率 {nan_ratio:.2%}")
            if df[col].eq(0).all():
                print(f"警告: 列 '{col}' 全为零")

        df = df.dropna(subset=['年份', '省份'])
        print(f"成功加载数据，共有{len(df)}条记录，涵盖{df['省份'].nunique()}个省份，"
              f"从{df['年份'].min()}年到{df['年份'].max()}年")
        return df

    except Exception as e:
        print(f"加载数据出错: {e}")
        return None


def preprocess_data(df):
    print("数据预处理...")
    features = [
        '65岁及以上人口数', '城镇单位就业人员数', '城镇居民消费水平',
        '城镇人口数', '基本养老保险基金支出', '老年人口抚养比'
    ]

    for feature in features:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')

    for feature in features:
        if df[feature].isnull().any():
            print(f"警告: '{feature}' 存在缺失值，使用中位数填充")
            df[feature].fillna(df[feature].median(), inplace=True)

    for feature in features:
        if (df[feature] < 0).any():
            print(f"警告: '{feature}' 包含负值，转换为0")
            df[feature] = df[feature].clip(lower=0)

    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features].astype(np.float64))
    return df_scaled, scaler


def kmeans_clustering(df_scaled):
    features = [
        '65岁及以上人口数', '城镇单位就业人员数', '城镇居民消费水平',
        '城镇人口数', '基本养老保险基金支出', '老年人口抚养比'
    ]
    X = df_scaled[features].values

    kmeans = KMeans(n_clusters=3, random_state=42)
    df_scaled['类别'] = kmeans.fit_predict(X)
    cluster_centers = kmeans.cluster_centers_

    center_means = np.mean(cluster_centers, axis=1)
    sorted_indices = np.argsort(center_means)
    category_mapping = {
        sorted_indices[0]: 2,
        sorted_indices[1]: 1,
        sorted_indices[2]: 0
    }

    df_scaled['需求类别'] = df_scaled['类别'].map(category_mapping)
    return df_scaled, cluster_centers, category_mapping


def train_bp_neural_network(df):
    features = [
        '65岁及以上人口数', '城镇单位就业人员数', '城镇居民消费水平',
        '城镇人口数', '基本养老保险基金支出', '老年人口抚养比'
    ]
    X = df[features].values
    y = pd.get_dummies(df['需求类别']).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = Sequential([
        Dense(10, input_dim=6, activation='tanh'),
        Dense(3, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=50, batch_size=16,
                        validation_data=(X_test, y_test), verbose=0)

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"BP神经网络模型精度: {accuracy * 100:.2f}%")

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, np.argmax(y_train, axis=1))
    y_pred_lda = lda.predict(X_test)
    lda_accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred_lda)
    print(f"判别分析模型精度: {lda_accuracy * 100:.2f}%")

    return model, history  # Return history for plotting


def grey_forecasting(data, forecast_steps=4):
    try:
        x0 = np.array(data, dtype=np.float64)
        if len(x0) < 3:
            return np.array([x0[-1] for _ in range(forecast_steps)]), 1.0

        min_val = np.min(x0)
        offset = 0
        if min_val <= 0:
            offset = -min_val + 1
            x0 = x0 + offset

        x1 = np.cumsum(x0)
        n = len(x0)
        B = np.zeros((n - 1, 2))
        Y = np.zeros((n - 1, 1))

        for i in range(n - 1):
            B[i, 0] = -0.5 * (x1[i] + x1[i + 1])
            B[i, 1] = 1
            Y[i, 0] = x0[i + 1]

        params = np.linalg.inv(B.T @ B) @ B.T @ Y
        a, b = params[0, 0], params[1, 0]

        x1_pred = [(x0[0] - b / a) * np.exp(-a * k) + b / a for k in range(n + forecast_steps)]
        x0_pred = np.zeros(n + forecast_steps)
        x0_pred[0] = x0[0]
        for i in range(1, n + forecast_steps):
            x0_pred[i] = x1_pred[i] - x1_pred[i - 1]

        if offset != 0:
            x0_pred -= offset

        e = np.abs(x0 - x0_pred[:n])
        C = np.mean(e) / np.mean(np.abs(x0 - np.mean(x0))) if np.mean(np.abs(x0 - np.mean(x0))) != 0 else 1.0
        return x0_pred[n:], C

    except Exception as e:
        print(f"灰色预测出错: {e}")
        return np.array([np.mean(data) for _ in range(forecast_steps)]), 1.0


def arima_forecasting(data, forecast_steps=4):
    try:
        data = np.array(data, dtype=np.float64)
        if len(data) < 5 or np.any(np.isnan(data)) or np.any(np.isinf(data)) or np.std(data) < 1e-5:
            mean_val = np.mean(data[~np.isnan(data)]) if len(data[~np.isnan(data)]) > 0 else 0
            return np.array([mean_val for _ in range(forecast_steps)]), None

        model = auto_arima(data, start_p=0, start_q=0, max_p=2, max_q=2, max_d=1,
                           seasonal=False, suppress_warnings=True, error_action='ignore',
                           stepwise=True)
        forecast = model.predict(n_periods=forecast_steps)
        if np.any(np.isnan(forecast)) or np.any(np.isinf(forecast)):
            mean_val = np.mean(data)
            return np.array([mean_val for _ in range(forecast_steps)]), None
        return forecast, model.order

    except Exception as e:
        print(f"ARIMA预测出错: {e}")
        mean_val = np.mean(data[~np.isnan(data)]) if len(data[~np.isnan(data)]) > 0 else 0
        return np.array([mean_val for _ in range(forecast_steps)]), None


def predict_2026_indicators(df):
    features = [
        '65岁及以上人口数', '城镇单位就业人员数', '城镇居民消费水平',
        '城镇人口数', '基本养老保险基金支出', '老年人口抚养比'
    ]
    provinces = df['省份'].unique()
    latest_year = df['年份'].max()
    target_year = 2026

    if pd.isna(latest_year):
        print("错误: 无法确定最新年份")
        return pd.DataFrame()

    latest_year = int(latest_year)
    years_to_forecast = target_year - latest_year
    if years_to_forecast <= 0:
        print(f"错误: 数据已包含{target_year}年或更晚的数据")
        return pd.DataFrame()

    print(f"基于{latest_year}年数据预测{target_year}年（预测{years_to_forecast}年）")
    df_2026 = pd.DataFrame(columns=['省份', '年份'] + features)

    for i, province in enumerate(provinces):
        if (i + 1) % 5 == 0 or i + 1 == len(provinces):
            print(f"处理进度: {i + 1}/{len(provinces)}")

        province_data = df[df['省份'] == province].sort_values('年份')
        row_2026 = {'省份': province, '年份': 2026}

        if len(province_data) < 3:
            print(f"警告: {province} 数据不足3年，使用最新值")
            for feature in features:
                latest = province_data[feature].iloc[-1] if not province_data[feature].empty else 0
                row_2026[feature] = max(float(latest), 0)
            df_2026 = pd.concat([df_2026, pd.DataFrame([row_2026])], ignore_index=True)
            continue

        for feature in features:
            historical_data = province_data[feature].values
            historical_data = historical_data[~np.isnan(historical_data)]

            if len(historical_data) == 0:
                print(f"警告: {province} 的 {feature} 数据为空，使用0")
                row_2026[feature] = 0
                continue

            if len(historical_data) < 3:
                print(f"警告: {province} 的 {feature} 数据不足3个，使用最新值")
                row_2026[feature] = max(float(historical_data[-1]), 0)
                continue

            grey_pred, C = grey_forecasting(historical_data, years_to_forecast)
            arima_pred, _ = arima_forecasting(historical_data, years_to_forecast)

            grey_pred = np.array(grey_pred, dtype=np.float64)
            arima_pred = np.array(arima_pred, dtype=np.float64)
            grey_pred[np.isnan(grey_pred) | np.isinf(grey_pred)] = historical_data[-1]
            arima_pred[np.isnan(arima_pred) | np.isinf(arima_pred)] = historical_data[-1]

            final_pred = float(arima_pred[-1] if C > 0.65 else grey_pred[-1])
            final_pred = max(final_pred, 0)

            if feature in ['65岁及以上人口数', '城镇人口数', '城镇单位就业人员数']:
                final_pred = int(final_pred)

            row_2026[feature] = final_pred

        df_2026 = pd.concat([df_2026, pd.DataFrame([row_2026])], ignore_index=True)

    for feature in features:
        invalid = df_2026[feature].isnull() | (df_2026[feature] < 0)
        if invalid.any():
            print(f"警告: {feature} 包含无效值，已修正为0")
            df_2026.loc[invalid, feature] = 0

    return df_2026


def entropy_weight_method(df):
    features = [
        '65岁及以上人口数', '城镇单位就业人员数', '城镇居民消费水平',
        '城镇人口数', '基本养老保险基金支出', '老年人口抚养比'
    ]
    X = df[features].values.astype(np.float64)
    epsilon = 1e-10

    X = np.maximum(X, 0) + epsilon
    X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0) + epsilon)
    X_norm = np.clip(X_norm, epsilon, 1)

    P = X_norm / np.sum(X_norm, axis=0)
    E = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        p = P[:, j]
        p_valid = p[p > epsilon]
        if len(p_valid) == 0:
            E[j] = 1.0
            print(f"警告: 特征 '{features[j]}' 无有效比例，熵设为1")
        else:
            E[j] = -np.sum(p_valid * np.log(p_valid)) / np.log(len(p_valid) + epsilon)

    D = np.maximum(1 - E, 0)
    W = D / np.sum(D) if np.sum(D) > 0 else np.ones_like(D) / len(D)
    S = np.dot(X_norm, W)
    return S, W


# Visualization enhancements
def set_plot_style():
    sns.set_style("whitegrid")
    rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.autolayout': True
    })


def plot_bp_training_enhanced(history, output_dir="figures"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history.history['loss'], label='Training Loss', color='royalblue', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='coral', linewidth=2)
    ax1.set_title('Model Loss Over Epochs', pad=10)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(frameon=False)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(history.history['accuracy'], label='Training Accuracy', color='royalblue', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='coral', linewidth=2)
    ax2.set_title('Model Accuracy Over Epochs', pad=10)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend(frameon=False)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bp_neural_network_training_enhanced.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'bp_neural_network_training_enhanced.pdf'), bbox_inches='tight')
    plt.close()


def plot_kmeans_clusters(df_scaled, cluster_centers, output_dir="figures"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    set_plot_style()
    features = [
        '65岁及以上人口数', '城镇单位就业人员数', '城镇居民消费水平',
        '城镇人口数', '基本养老保险基金支出', '老年人口抚养比'
    ]
    X = df_scaled[features].values

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    centers_pca = pca.transform(cluster_centers)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df_scaled['需求类别'],
                          cmap='viridis', s=50, alpha=0.7, label='Data Points')
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='x',
                s=200, linewidths=3, label='Cluster Centers')

    plt.title('K-means Clustering Results (PCA Projection)', pad=15)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} Variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} Variance)')
    plt.legend(frameon=False)
    plt.colorbar(scatter, label='Demand Category')

    plt.savefig(os.path.join(output_dir, 'kmeans_clusters_pca.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'kmeans_clusters_pca.pdf'), bbox_inches='tight')
    plt.close()


def plot_entropy_weights(weights, features, output_dir="figures"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    set_plot_style()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=weights, y=features, palette='Blues_r')

    plt.title('Entropy-Based Feature Weights', pad=15)
    plt.xlabel('Weight')
    plt.ylabel('Feature')

    plt.savefig(os.path.join(output_dir, 'entropy_weights.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'entropy_weights.pdf'), bbox_inches='tight')
    plt.close()


def plot_2026_predictions(df_result, output_dir="figures"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    set_plot_style()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_result, x='省份', y='评分', hue='需求描述', dodge=True)

    plt.title('2026 Bed Demand Scores by Province', pad=15)
    plt.xlabel('Province')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Demand Category', frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2026_predictions_bar.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, '2026_predictions_bar.pdf'), bbox_inches='tight')
    plt.close()


def plot_forecast_trends(df, df_2026, feature, output_dir="figures"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    set_plot_style()
    provinces = df['省份'].unique()[:5]
    plt.figure(figsize=(10, 6))

    for province in provinces:
        historical = df[df['省份'] == province][['年份', feature]].sort_values('年份')
        forecast = df_2026[df_2026['省份'] == province][['年份', feature]]
        combined = pd.concat([historical, forecast], ignore_index=True)

        plt.plot(combined['年份'], combined[feature], marker='o', label=province)

    plt.title(f'Historical and Forecasted {feature} (2026)', pad=15)
    plt.xlabel('Year')
    plt.ylabel(feature)
    plt.legend(frameon=False)

    plt.savefig(os.path.join(output_dir, f'forecast_trends_{feature}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'forecast_trends_{feature}.pdf'), bbox_inches='tight')
    plt.close()


def run_analysis(excel_file="消费.xlsx"):
    df = load_real_data(excel_file)
    if df is None:
        print("数据加载失败")
        return None

    print("数据预处理...")
    df_scaled, scaler = preprocess_data(df)

    print("执行K均值聚类...")
    df_with_clusters, cluster_centers, category_mapping = kmeans_clustering(df_scaled)

    print("生成K均值聚类可视化...")
    plot_kmeans_clusters(df_with_clusters, cluster_centers)

    print("训练BP神经网络...")
    bp_model, history = train_bp_neural_network(df_with_clusters)

    print("生成BP神经网络训练过程可视化...")
    plot_bp_training_enhanced(history)

    print("预测2026年指标值...")
    df_2026 = predict_2026_indicators(df)

    if df_2026.empty:
        print("预测失败，2026年数据为空")
        return None

    features = [
        '65岁及以上人口数', '城镇单位就业人员数', '城镇居民消费水平',
        '城镇人口数', '基本养老保险基金支出', '老年人口抚养比'
    ]
    df_2026_scaled = df_2026.copy()
    df_2026_scaled[features] = scaler.transform(df_2026[features].astype(np.float64))

    print("使用BP神经网络分类2026年数据...")
    X_2026 = df_2026_scaled[features].values
    y_2026_pred = bp_model.predict(X_2026, verbose=0)
    df_2026['预测类别'] = np.argmax(y_2026_pred, axis=1)

    reverse_mapping = {v: k for k, v in category_mapping.items()}
    df_2026['需求类别'] = df_2026['预测类别'].map(reverse_mapping)
    df_2026['需求描述'] = df_2026['需求类别'].map({
        0: '床位需求最多的地区',
        1: '床位需求适中的地区',
        2: '床位需求最少的地区'
    })

    print("使用熵权法评分...")
    df_2026['原始评分'], weights = entropy_weight_method(df_2026)
    df_result = df_2026.sort_values(by=['需求类别', '原始评分'], ascending=[True, False])

    # Calculate '评分' before plotting
    for category in df_result['需求类别'].unique():
        mask = df_result['需求类别'] == category
        df_result.loc[mask, '评分'] = df_result.loc[mask, '原始评分'] * 1e6

    print("生成熵权法权重可视化...")
    plot_entropy_weights(weights, features)

    print("生成2026年预测结果可视化...")
    plot_2026_predictions(df_result)

    print("生成关键指标预测趋势可视化...")
    plot_forecast_trends(df, df_2026, '65岁及以上人口数')

    columns_to_show = ['省份', '需求描述', '评分'] + features
    print("\n===== 2026年养老床位需求预测结果 =====")
    for category in sorted(df_result['需求类别'].unique()):
        desc = {0: '床位需求最多的地区', 1: '床位需求适中的地区', 2: '床位需求最少的地区'}[category]
        print(f"\n----- {desc} -----")
        category_df = df_result[df_result['需求类别'] == category][columns_to_show].copy()
        category_df['评分'] = category_df['评分'].round(2)
        print(category_df[['省份', '评分']].to_string(index=False))

    output_file = '养老床位需求预测2026.csv'
    df_result[columns_to_show].to_csv(output_file, index=False)
    print(f"\n详细结果已保存到 '{output_file}'")
    print(f"所有可视化图表已保存到 'figures' 文件夹")
    return df_result


if __name__ == "__main__":
    set_plot_style()
    result = run_analysis("消费.xlsx")