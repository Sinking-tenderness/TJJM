import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import os
from pathlib import Path
import shap
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.font_manager as fm
from scipy import stats
import matplotlib.cm as cm
import networkx as nx
from matplotlib.patches import Patch
import matplotlib.transforms as mtransforms

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Check available fonts
available_fonts = [f.name for f in fm.fontManager.ttflist]
if "Microsoft YaHei" in available_fonts:
    print("Microsoft YaHei 字体可用")
else:
    print("Microsoft YaHei 字体不可用，请检查字体安装")

# Set high-quality visualization style
sns.set_style("whitegrid", {"axes.facecolor": ".95"})
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 10, # Adjusted font size
    'axes.labelsize': 12, # Adjusted axes labelsize
    'axes.titlesize': 14, # Adjusted axes titlesize
    'legend.fontsize': 8, # Adjusted legend fontsize
    'xtick.labelsize': 8, # Adjusted xtick labelsize
    'ytick.labelsize': 8, # Adjusted ytick labelsize
    'axes.linewidth': 1,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'figure.figsize': (16, 14), # Increased figure size
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.sans-serif': ['Microsoft YaHei'],
    'axes.unicode_minus': False
})

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load Data
try:
    population_df = pd.read_csv("time_moe_population_data.csv")
    aging_df = pd.read_csv("time_moe_aging_data.csv")
    bed_df = pd.read_csv("time_moe_bed_data.csv")
    economic_df = pd.read_csv("time_moe_economic_data.csv")
except FileNotFoundError as e:
    print(f"Error: {e}. Please check if CSV files exist.")
    exit(1)

# Load StandardScaler
try:
    with open("standard_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    print("Error: standard_scaler.pkl not found.")
    exit(1)

# Merge data
try:
    df = population_df.merge(aging_df, on=["统计年度", "地区名称"]).merge(bed_df, on=["统计年度", "地区名称"]).merge(
        economic_df, on=["统计年度", "地区名称"])
except Exception as e:
    print(f"Error merging data: {e}")
    exit(1)

# 2. Data Processing - Smooth pension bed data
df = df.sort_values(by=["地区名称", "统计年度"])
provinces = df["地区名称"].unique()
n_years = df["统计年度"].nunique()
n_provinces = len(provinces)
print(f"Data loaded: {n_provinces} provinces, {n_years} years")

df["养老床位_smoothed"] = df.groupby("地区名称")["养老床位"].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean())
df["养老床位_growth_rate"] = df.groupby("地区名称")["养老床位_smoothed"].pct_change()
df.loc[df["养老床位_growth_rate"] > 1, "养老床位_smoothed"] = df["养老床位_smoothed"].shift(1) * 1.5
df.loc[df["养老床位_growth_rate"] < -0.5, "养老床位_smoothed"] = df["养老床位_smoothed"].shift(1) * 0.5
df["养老床位"] = df["养老床位_smoothed"]
df = df.drop(columns=["养老床位_smoothed", "养老床位_growth_rate"])

# Define input and target features
input_features = {
    "population": ["人口数", "15-64岁人口数", "65岁及以上人口数",
                   "人口数_lag1", "15-64岁人口数_lag1", "65岁及以上人口数_lag1"],
    "aging": ["老龄化率", "老年人口抚养比", "老龄化率_lag1", "老年人口抚养比_lag1",
              "老龄化率_3年均值", "老年人口抚养比_3年均值",
              "养保支出(亿元)", "养保支出(亿元)_lag1", "养保支出(亿元)_3年均值",
              "老龄化率增长率"],
    "bed": ["养老床位", "床位覆盖率", "养老床位_lag1", "养老床位_5年均值增长率"],
    "economic": ["经济(亿元)", "经济(亿元)_lag1", "经济(亿元)_3年均值", "经济(亿元)_3年标准差", "经济增长率"]
}
target_features = ["人口数", "15-64岁人口数", "65岁及以上人口数", "养老床位"]

input_cols = sum(input_features.values(), [])
missing_input_cols = [col for col in input_cols if col not in df.columns]
if missing_input_cols:
    print(f"Error: Input columns missing: {missing_input_cols}")
    exit(1)

scale_cols = [
    "人口数", "0-14岁人口数", "15-64岁人口数", "65岁及以上人口数",
    "总抚养比", "少年儿童抚养比", "老年人口抚养比", "老龄化率", "养老床位",
    "床位增长率", "床位覆盖率", "经济(亿元)", "养保支出(亿元)", "经济增长率", "养保支出增长率",
    "老龄化率_lag1", "老年人口抚养比_lag1", "养老床位_lag1", "65岁及以上人口数_lag1",
    "人口数_lag1", "15-64岁人口数_lag1", "经济(亿元)_lag1", "养保支出(亿元)_lag1",
    "老龄化率_3年均值", "老年人口抚养比_3年均值", "养老床位_3年均值", "65岁及以上人口数_3年均值",
    "经济(亿元)_3年均值", "养保支出(亿元)_3年均值",
    "老龄化率_3年标准差", "老年人口抚养比_3年标准差", "养老床位_3年标准差", "65岁及以上人口数_3年标准差",
    "经济(亿元)_3年标准差", "养保支出(亿元)_3年标准差",
    "养老床位_5年均值增长率", "老龄化率增长率"
]

# Ensure numeric types and handle NaNs
for col in input_cols + target_features:
    if not pd.api.types.is_numeric_dtype(df[col]):
        print(f"Warning: Column {col} is not numeric, converting to float")
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(0)

# Verify scaler compatibility
expected_n_features = len(scale_cols)
if scaler.n_features_in_ != expected_n_features:
    print(f"Warning: Scaler feature count ({scaler.n_features_in_}) does not match scale_cols ({expected_n_features}). Refitting scaler...")
    scaler_input = np.zeros((df.shape[0], len(scale_cols)))
    for col in scale_cols:
        if col in df.columns:
            scaler_input[:, scale_cols.index(col)] = df[col].values
    scaler.fit(scaler_input)
else:
    print(f"Scaler loaded with {scaler.n_features_in_} features, matching scale_cols.")

# 3. Feature Engineering - Lagged and Rolling Features (Simplified for correlation)
df_corr = df.copy()

def create_lagged_features(df, cols, lags):
    for col in cols:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df.groupby('地区名称')[col].shift(lag)
    return df

def create_rolling_features(df, cols, windows, min_periods, agg_func='mean', suffix=''):
    for col in cols:
        for window in windows:
            df[f'{col}_{window}年{suffix}'] = df.groupby('地区名称')[col].rolling(window=window, min_periods=min_periods).agg(agg_func).reset_index(level=0, drop=True)
    return df

lag_cols_base = ["人口数", "15-64岁人口数", "65岁及以上人口数", "老龄化率", "老年人口抚养比", "养老床位", "经济(亿元)", "养保支出(亿元)"]
df_corr = create_lagged_features(df_corr, lag_cols_base, lags=[1])
df_corr = create_rolling_features(df_corr, ["老龄化率", "老年人口抚养比", "养老床位", "经济(亿元)", "养保支出(亿元)"], windows=[3], min_periods=1, suffix='年均值')
df_corr = create_rolling_features(df_corr, ["经济(亿元)", "养保支出(亿元)"], windows=[3], min_periods=1, agg_func='std', suffix='年标准差')
df_corr['养老床位_5年均值增长率'] = df_corr.groupby('地区名称')['养老床位'].rolling(window=5, min_periods=1).mean().pct_change().reset_index(level=0, drop=True) * 100
df_corr['老龄化率增长率'] = df_corr.groupby('地区名称')['老龄化率'].pct_change() * 100

# Fill NaN values introduced by lagging and rolling with 0 for correlation calculation
df_corr = df_corr.fillna(0)

# Select all features used in the model
all_features = sorted(sum(input_features.values(), []))

# Calculate the correlation matrix
corr_matrix = df_corr[all_features].corr()

# Define the 'Spec' nodes and their representative features/related features
spec_nodes = {
    "老龄化率": "老龄化率",
    "养老床位数": "养老床位",
    "老龄人口数": "65岁及以上人口数"
}

# Define connections based on feature groups (adjust these based on your understanding)
feature_group_connections = {
    "老龄化率": ["老年人口抚养比", "老龄化率_lag1", "老年人口抚养比_lag1", "老龄化率_3年均值", "老年人口抚养比_3年均值", "老龄化率增长率"],
    "养老床位数": ["床位覆盖率", "养老床位_lag1", "养老床位_5年均值增长率"],
    "老龄人口数": ["65岁及以上人口数_lag1", "老年人口抚养比", "老年人口抚养比_lag1", "老年人口抚养比_3年均值"]
}



# 3. Data Splitting
train_df = df[df["统计年度"].between(2014, 2018)]
val_df = df[df["统计年度"].between(2019, 2020)]
test_df = df[df["统计年度"].between(2021, 2022)]

for dataset, name in [(train_df, "训练集"), (val_df, "验证集"), (test_df, "测试集")]:
    if dataset[input_cols + target_features].isna().any().any():
        print(f"Warning: {name} contains NaN values, filling with 0")
        dataset[input_cols + target_features] = dataset[input_cols + target_features].fillna(0)

print(f"Training data shape: {train_df.shape}")
print(f"Validation data shape: {val_df.shape}")
print(f"Test data shape: {test_df.shape}")

# 4. Define Dataset Class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_cols, target_cols, seq_length):
        self.data = data.reset_index(drop=True)
        self.input_cols = input_cols
        self.target_cols = target_cols
        self.seq_length = seq_length
        self.sequences = []
        self.indices = []
        self._create_sequences()

    def _create_sequences(self):
        for province in self.data["地区名称"].unique():
            province_data = self.data[self.data["地区名称"] == province].sort_values("统计年度")
            num_years = len(province_data)
            if num_years < self.seq_length:
                print(f"Warning: Province {province} has insufficient data ({num_years} years) for seq_length {self.seq_length}")
                continue
            for i in range(num_years - self.seq_length + 1):
                seq = province_data.iloc[i:i + self.seq_length][self.input_cols].values
                target = province_data.iloc[i + self.seq_length - 1][self.target_cols].values
                seq = np.array(seq, dtype=np.float32)
                target = np.array(target, dtype=np.float32)
                self.sequences.append((seq, target))
                self.indices.append(province_data.index[i + self.seq_length - 1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        return torch.FloatTensor(seq), torch.FloatTensor(target)

# 5. Create Datasets and Loaders
seq_length = 2
train_dataset = TimeSeriesDataset(train_df, input_cols, target_features, seq_length)
val_dataset = TimeSeriesDataset(val_df, input_cols, target_cols=target_features, seq_length=seq_length)
test_dataset = TimeSeriesDataset(test_df, input_cols, target_cols=target_features, seq_length=seq_length)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"训练集样本数: {len(train_dataset)}")
print(f"验证集样本数: {len(val_dataset)}")
print(f"测试集样本数: {len(test_dataset)}")

# 6. Define Neural Network Models
class TimeMoE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_length):
        super(TimeMoE, self).__init__()
        self.seq_length = seq_length
        self.expert1 = nn.Sequential(
            nn.Linear(input_size * seq_length, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.expert2 = nn.Sequential(
            nn.Linear(input_size * seq_length, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.gate = nn.Sequential(
            nn.Linear(input_size * seq_length, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        gate_weights = self.gate(x)
        expert1_out = self.expert1(x)
        expert2_out = self.expert2(x)
        combined = gate_weights[:, 0].unsqueeze(1) * expert1_out + gate_weights[:, 1].unsqueeze(1) * expert2_out
        return combined

class ConvTimeNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_length):
        super(ConvTimeNet, self).__init__()
        self.seq_length = seq_length
        kernel_size = min(2, seq_length)
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        conv_output_length = max(1, seq_length - kernel_size + 1)
        self.fc = nn.Linear(hidden_size * conv_output_length, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 7. Grid Search for Neural Networks
def grid_search(model_class, train_loader, val_loader, input_size, output_size, seq_length):
    learning_rates = [0.0005, 0.001, 0.005]
    hidden_sizes = [64, 128, 256]
    best_val_loss = float('inf')
    best_params = None
    best_model = None

    for lr in learning_rates:
        for hidden_size in hidden_sizes:
            print(f"Testing lr={lr}, hidden_size={hidden_size}")
            model = model_class(input_size, hidden_size, output_size, seq_length).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            for epoch in range(50):
                model.train()
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = {'lr': lr, 'hidden_size': hidden_size}
                    best_model = model.state_dict()
    print(f"Best params: {best_params}, Best val loss: {best_val_loss}")
    return best_params, best_model

# Train Neural Network Model
def train_model(model, train_loader, val_loader, lr, epochs=150):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        model.eval()
        val_loss = 0
        if len(val_loader) > 0:
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
        else:
            val_loss = float('inf')
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        if val_loss < best_val_loss and val_loss != float('inf'):
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_{model.__class__.__name__}.pth")

# Get Neural Network Predictions
def get_predictions(model, loader):
    model.eval()
    predictions = []
    actuals = []
    inputs_list = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.cpu().numpy())
            inputs_list.append(inputs.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(actuals), np.concatenate(inputs_list, axis=0)

# Initialize and train neural networks
input_size = len(input_cols)
output_size = len(target_features)
time_moe_params, time_moe_state = grid_search(TimeMoE, train_loader, val_loader, input_size, output_size, seq_length)
time_moe_model = TimeMoE(input_size, time_moe_params['hidden_size'], output_size, seq_length).to(device)
time_moe_model.load_state_dict(time_moe_state)
train_model(time_moe_model, train_loader, val_loader, time_moe_params['lr'])

conv_time_params, conv_time_state = grid_search(ConvTimeNet, train_loader, val_loader, input_size, output_size, seq_length)
conv_time_net = ConvTimeNet(input_size, conv_time_params['hidden_size'], output_size, seq_length).to(device)
conv_time_net.load_state_dict(conv_time_state)
train_model(conv_time_net, train_loader, val_loader, conv_time_params['lr'])

# Load best models
try:
    time_moe_model.load_state_dict(torch.load("best_TimeMoE.pth"))
    conv_time_net.load_state_dict(torch.load("best_ConvTimeNet.pth"))
except FileNotFoundError:
    print("Warning: Best model files not found, using current models")

# Get predictions for TimeMoE and ConvTimeNet
time_moe_preds, test_actuals, test_inputs = get_predictions(time_moe_model, test_loader)
conv_time_preds, _, _ = get_predictions(conv_time_net, test_loader)

# 8. Stacking Ensemble
def prepare_stacking_data(time_moe_model, conv_time_net, loader):
    time_moe_preds, actuals, inputs = get_predictions(time_moe_model, loader)
    conv_time_preds, _, _ = get_predictions(conv_time_net, loader)
    meta_features = np.hstack([time_moe_preds, conv_time_preds])
    return meta_features, actuals, inputs

train_meta_features, train_actuals, _ = prepare_stacking_data(time_moe_model, conv_time_net, train_loader)
val_meta_features, val_actuals, _ = prepare_stacking_data(time_moe_model, conv_time_net, val_loader)
test_meta_features, test_actuals, test_inputs = prepare_stacking_data(time_moe_model, conv_time_net, test_loader)

meta_model = LinearRegression()
meta_model.fit(train_meta_features, train_actuals)
stacking_preds = meta_model.predict(test_meta_features)

# 9. Traditional Models (RandomForest and XGBoost)
def prepare_traditional_data(dataset, input_cols, target_cols, seq_length):
    X, y = [], []
    for province in dataset["地区名称"].unique():
        province_data = dataset[dataset["地区名称"] == province].sort_values("统计年度")
        num_years = len(province_data)
        if num_years < seq_length:
            continue
        for i in range(num_years - seq_length + 1):
            seq = province_data.iloc[i:i + seq_length][input_cols].values.flatten()
            target = province_data.iloc[i + seq_length - 1][target_cols].values
            X.append(seq)
            y.append(target)
    return np.array(X), np.array(y)

X_train, y_train = prepare_traditional_data(train_df, input_cols, target_features, seq_length)
X_val, y_val = prepare_traditional_data(val_df, input_cols, target_features, seq_length)
X_test, y_test = prepare_traditional_data(test_df, input_cols, target_features, seq_length)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# 10. Ensemble Neural Network Predictions
def get_val_loss(model, val_loader):
    model.eval()
    criterion = nn.MSELoss()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    return val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')

time_moe_val_loss = get_val_loss(time_moe_model, val_loader)
conv_time_val_loss = get_val_loss(conv_time_net, val_loader)
total_loss = time_moe_val_loss + conv_time_val_loss
time_moe_weight = conv_time_val_loss / total_loss if total_loss != 0 else 0.5
conv_time_weight = time_moe_val_loss / total_loss if total_loss != 0 else 0.5
ensemble_preds = time_moe_weight * time_moe_preds + conv_time_weight * conv_time_preds

# 11. Calculate Metrics
def calculate_metrics(y_true, y_pred, scaler, feature_indices, raw=False):
    metrics = []
    for i, idx in enumerate(feature_indices):
        y_true_i = y_true[:, i]
        y_pred_i = y_pred[:, i]
        if raw:
            temp = np.zeros((len(y_true), scaler.n_features_in_))
            temp[:, idx] = y_true_i
            y_true_raw = scaler.inverse_transform(temp)[:, idx]
            temp[:, idx] = y_pred_i
            y_pred_raw = scaler.inverse_transform(temp)[:, idx]
        else:
            y_true_raw, y_pred_raw = y_true_i, y_pred_i
        mse = mean_squared_error(y_true_raw, y_pred_raw)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_raw, y_pred_raw)
        mae = mean_absolute_error(y_true_raw, y_pred_raw)
        epsilon = 1e-10
        rmspe = np.sqrt(np.mean(((y_true_raw - y_pred_raw) / (y_true_raw + epsilon)) ** 2))
        metrics.append((mse, rmse, r2, mae, rmspe))
    return metrics

feature_indices = [scale_cols.index(col) for col in target_features]
models = {
    "Ensemble (TimeMoE + ConvTimeNet)": ensemble_preds,
    "Stacking (TimeMoE + ConvTimeNet)": stacking_preds,
    "TimeMoE": time_moe_preds,
    "ConvTimeNet": conv_time_preds,
    "RandomForest": rf_preds,
    "XGBoost": xgb_preds
}

metrics_dict = {}
for model_name, preds in models.items():
    metrics = calculate_metrics(test_actuals, preds, scaler, feature_indices, raw=True)
    metrics_dict[model_name] = {
        "MSE": [m[0] for m in metrics],
        "RMSE": [m[1] for m in metrics],
        "R2": [m[2] for m in metrics],
        "MAE": [m[3] for m in metrics],
        "RMSPE": [m[4] for m in metrics]
    }

# Select best model based on average RMSE
avg_rmse = {model: np.mean(metrics_dict[model]["RMSE"]) for model in models}
best_model_name = min(avg_rmse, key=avg_rmse.get)
print(f"Best model based on average RMSE: {best_model_name}")

# 12. Visualize Model Comparison
output_dir = "model_comparison_plots"
Path(output_dir).mkdir(exist_ok=True)

# 自定义配色方案
custom_palette = ['#3083DB', '#8F9FBD', '#C9AED7', '#ECBAD4', '#FFBFCD', '#FFD1DC']  # 补充一个颜色

for metric_name in ["MSE", "RMSE", "R2", "MAE", "RMSPE"]:
    plt.figure(figsize=(14, 8))
    data = []
    for model in models:
        for i, feature in enumerate(target_features):
            data.append({
                "Model": model,
                "Feature": feature,
                metric_name: metrics_dict[model][metric_name][i]
            })
    df_metrics = pd.DataFrame(data)
    sns.barplot(x="Feature", y=metric_name, hue="Model", data=df_metrics, palette=custom_palette)
    plt.title(f"Model Comparison: {metric_name}", fontsize=18, pad=15)
    plt.xlabel("Feature", fontsize=16)
    plt.ylabel(metric_name, fontsize=16)
    plt.legend(title="Model", loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"model_comparison_{metric_name}.png"), dpi=300)
    plt.close()

# 13. Comprehensive Analysis of Best Model
best_preds = models[best_model_name]
test_inputs_tensor = torch.FloatTensor(test_inputs).to(device)
n_samples = test_inputs.shape[0]
input_size = len(input_cols)
feature_dim = seq_length * input_size
test_inputs_flat = test_inputs_tensor.view(n_samples, -1)

# Create a test dataset to get the correct indices
test_dataset = TimeSeriesDataset(test_df, input_cols, target_features, seq_length)
test_indices = test_dataset.indices
test_df_subset = test_df.iloc[test_indices].reset_index(drop=True)

# Inverse transform for raw values
transformed_input = np.zeros((n_samples, scaler.n_features_in_))
for i, idx in enumerate(feature_indices):
    transformed_input[:, idx] = best_preds[:, i]
best_preds_inv = scaler.inverse_transform(transformed_input)[:, feature_indices]
transformed_actuals = np.zeros((n_samples, scaler.n_features_in_))
for i, idx in enumerate(feature_indices):
    transformed_actuals[:, idx] = test_actuals[:, i]
test_actuals_inv = scaler.inverse_transform(transformed_actuals)[:, feature_indices]

# Predictions vs Actuals Scatter Plots
for i, feature in enumerate(target_features):
    plt.figure(figsize=(10, 6))
    plt.scatter(test_actuals_inv[:, i], best_preds_inv[:, i], alpha=0.5, color='navy')
    min_val = min(test_actuals_inv[:, i].min(), best_preds_inv[:, i].min())
    max_val = max(test_actuals_inv[:, i].max(), best_preds_inv[:, i].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title(f'{best_model_name}: {feature} 预测值 vs 实际值', fontsize=18)
    plt.xlabel('实际值', fontsize=16)
    plt.ylabel('预测值', fontsize=16)
    plt.text(0.05, 0.95, f"R² = {metrics_dict[best_model_name]['R2'][i]:.4f}\nRMSE = {metrics_dict[best_model_name]['RMSE'][i]:.2f}",
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{best_model_name}_{feature}_pred_vs_actual.png'), dpi=300)
    plt.close()

# Residual Analysis
for i, feature in enumerate(target_features):
    residuals = test_actuals_inv[:, i] - best_preds_inv[:, i]
    plt.figure(figsize=(12, 6))
    plt.scatter(best_preds_inv[:, i], residuals, alpha=0.5, color='navy')
    plt.hlines(y=0, xmin=best_preds_inv[:, i].min(), xmax=best_preds_inv[:, i].max(), colors='red', linestyles='--')
    plt.title(f'{best_model_name} {feature} 残差分布', fontsize=18)
    plt.xlabel('预测值', fontsize=16)
    plt.ylabel('残差', fontsize=16)
    stat, p_value = stats.normaltest(residuals)
    plt.text(0.05, 0.95, f"正态性检验 p值: {p_value:.4f}", transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{best_model_name}_{feature}_residuals.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'{best_model_name} {feature} 残差Q-Q图', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{best_model_name}_{feature}_residuals_qq.png'), dpi=300)
    plt.close()

# Prediction Error Distribution
for i, feature in enumerate(target_features):
    errors = test_actuals_inv[:, i] - best_preds_inv[:, i]
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True, bins=30, color='navy')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title(f'{best_model_name} {feature} 预测误差分布', fontsize=18)
    plt.xlabel('预测误差', fontsize=16)
    plt.ylabel('频率', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{best_model_name}_{feature}_error_dist.png'), dpi=300)
    plt.close()

# Error vs Feature Value
key_features = ['老年人口抚养比', '床位覆盖率']
for feature in key_features:
    for i, target_feature in enumerate(target_features):
        errors = test_actuals_inv[:, i] - best_preds_inv[:, i]
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=test_df_subset[feature], y=errors, hue=test_df_subset['统计年度'], palette='viridis', alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title(f'{best_model_name} {target_feature} 预测误差与 {feature} 的关系', fontsize=18)
        plt.xlabel(feature, fontsize=16)
        plt.ylabel('预测误差', fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{best_model_name}_{target_feature}_error_vs_{feature}.png'), dpi=300)
        plt.close()

# Regional Prediction Performance
key_regions = ['北京市', '上海市', '广东省', '四川省', '河南省']
region_metrics = {region: {'MSE': [], 'R2': []} for region in key_regions}
for region in key_regions:
    region_data = test_df[test_df['地区名称'] == region]
    if not region_data.empty:
        region_dataset = TimeSeriesDataset(region_data, input_cols, target_features, seq_length)
        region_loader = DataLoader(region_dataset, batch_size=32, shuffle=False)
        if best_model_name in ["TimeMoE", "ConvTimeNet", "Ensemble (TimeMoE + ConvTimeNet)"]:
            region_preds, region_actuals, _ = get_predictions(time_moe_model if best_model_name == "TimeMoE" else conv_time_net, region_loader)
            if best_model_name == "Ensemble (TimeMoE + ConvTimeNet)":
                region_preds = time_moe_weight * region_preds + conv_time_weight * get_predictions(conv_time_net, region_loader)[0]
        else:
            X_region, y_region = prepare_traditional_data(region_data, input_cols, target_features, seq_length)
            region_preds = (meta_model.predict(np.hstack([get_predictions(time_moe_model, region_loader)[0], get_predictions(conv_time_net, region_loader)[0]])[:, :4]) if best_model_name == "Stacking (TimeMoE + ConvTimeNet)" else
                            rf_model.predict(X_region) if best_model_name == "RandomForest" else xgb_model.predict(X_region))
            region_actuals = y_region
        # Inverse transform regional predictions
        transformed_region_preds = np.zeros((region_preds.shape[0], scaler.n_features_in_))
        for i, idx in enumerate(feature_indices):
            transformed_region_preds[:, idx] = region_preds[:, i]
        region_preds_inv = scaler.inverse_transform(transformed_region_preds)[:, feature_indices]
        transformed_region_actuals = np.zeros((region_actuals.shape[0], scaler.n_features_in_))
        for i, idx in enumerate(feature_indices):
            transformed_region_actuals[:, idx] = region_actuals[:, i]
        region_actuals_inv = scaler.inverse_transform(transformed_region_actuals)[:, feature_indices]
        for i in range(len(target_features)):
            mse = mean_squared_error(region_actuals_inv[:, i], region_preds_inv[:, i])
            r2 = r2_score(region_actuals_inv[:, i], region_preds_inv[:, i])
            region_metrics[region]['MSE'].append(mse)
            region_metrics[region]['R2'].append(r2)

for i, feature in enumerate(target_features):
    mse_values = [region_metrics[region]['MSE'][i] for region in key_regions if len(region_metrics[region]['MSE']) > i]
    r2_values = [region_metrics[region]['R2'][i] for region in key_regions if len(region_metrics[region]['R2']) > i]
    regions = [region for region in key_regions if len(region_metrics[region]['MSE']) > i]
    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(regions))
    width = 0.35
    ax1.bar(x - width / 2, mse_values, width, label='MSE', color='crimson')
    ax1.set_ylabel('MSE', color='crimson', fontsize=16)
    ax1.tick_params(axis='y', labelcolor='crimson')
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, r2_values, width, label='R²', color='navy')
    ax2.set_ylabel('R²', color='navy', fontsize=16)
    ax2.tick_params(axis='y', labelcolor='navy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions, rotation=45)
    plt.title(f'{best_model_name} {feature} 在不同地区的预测性能', fontsize=18)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{best_model_name}_{feature}_regional_performance.png'), dpi=300)
    plt.close()

# Yearly Prediction Accuracy
years = sorted(test_df['统计年度'].unique())
year_metrics = {year: {'MSE': [], 'R2': []} for year in years}
for year in years:
    # Use data up to the target year to create sequences
    year_data = df[df['统计年度'] <= year]
    if not year_data.empty:
        year_dataset = TimeSeriesDataset(year_data, input_cols, target_features, seq_length)
        if len(year_dataset) == 0:
            print(f"Warning: No valid sequences for year {year}")
            continue
        year_loader = DataLoader(year_dataset, batch_size=32, shuffle=False)
        # Filter predictions to only include the target year
        year_indices = [idx for idx, i in enumerate(year_dataset.indices) if year_data.iloc[i]['统计年度'] == year]
        if not year_indices:
            print(f"Warning: No data for year {year} in sequences")
            continue
        if best_model_name in ["TimeMoE", "ConvTimeNet", "Ensemble (TimeMoE + ConvTimeNet)"]:
            year_preds, year_actuals, _ = get_predictions(time_moe_model if best_model_name == "TimeMoE" else conv_time_net, year_loader)
            if best_model_name == "Ensemble (TimeMoE + ConvTimeNet)":
                year_preds = time_moe_weight * year_preds + conv_time_weight * get_predictions(conv_time_net, year_loader)[0]
            year_preds = year_preds[year_indices]
            year_actuals = year_actuals[year_indices]
        else:
            # For traditional models, prepare data for the target year
            year_subset = test_df[test_df['统计年度'] == year]
            if year_subset.empty:
                print(f"Warning: No test data for year {year}")
                continue
            X_year, y_year = prepare_traditional_data(year_subset, input_cols, target_features, seq_length)
            if len(X_year) == 0:
                print(f"Warning: No valid sequences for year {year} in traditional models")
                continue
            year_preds = (meta_model.predict(np.hstack([get_predictions(time_moe_model, year_loader)[0], get_predictions(conv_time_net, year_loader)[0]])[:, :4])[year_indices] if best_model_name == "Stacking (TimeMoE + ConvTimeNet)" else
                          rf_model.predict(X_year) if best_model_name == "RandomForest" else xgb_model.predict(X_year))
            year_actuals = y_year
        # Inverse transform yearly predictions
        transformed_year_preds = np.zeros((year_preds.shape[0], scaler.n_features_in_))
        for i, idx in enumerate(feature_indices):
            transformed_year_preds[:, idx] = year_preds[:, i]
        year_preds_inv = scaler.inverse_transform(transformed_year_preds)[:, feature_indices]
        transformed_year_actuals = np.zeros((year_actuals.shape[0], scaler.n_features_in_))
        for i, idx in enumerate(feature_indices):
            transformed_year_actuals[:, idx] = year_actuals[:, i]
        year_actuals_inv = scaler.inverse_transform(transformed_year_actuals)[:, feature_indices]
        for i in range(len(target_features)):
            mse = mean_squared_error(year_actuals_inv[:, i], year_preds_inv[:, i])
            r2 = r2_score(year_actuals_inv[:, i], year_preds_inv[:, i])
            year_metrics[year]['MSE'].append(mse)
            year_metrics[year]['R2'].append(r2)

for i, feature in enumerate(target_features):
    mse_by_year = [year_metrics[year]['MSE'][i] for year in years if len(year_metrics[year]['MSE']) > i]
    r2_by_year = [year_metrics[year]['R2'][i] for year in years if len(year_metrics[year]['R2']) > i]
    valid_years = [year for year in years if len(year_metrics[year]['MSE']) > i]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.plot(valid_years, mse_by_year, marker='o', linestyle='-', color='crimson')
    ax1.set_title(f'{best_model_name} {feature} 年份预测MSE变化趋势', fontsize=18)
    ax1.set_ylabel('MSE', fontsize=16)
    ax1.grid(True)
    ax2.plot(valid_years, r2_by_year, marker='o', linestyle='-', color='navy')
    ax2.set_title(f'{best_model_name} {feature} 年份预测R²变化趋势', fontsize=18)
    ax2.set_xlabel('年份', fontsize=16)
    ax2.set_ylabel('R²', fontsize=16)
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{best_model_name}_{feature}_yearly_accuracy.png'), dpi=300)
    plt.close()

# Time Series Predictions for Key Regions
for region in key_regions:
    region_data = df[df['地区名称'] == region]
    if not region_data.empty:
        region_dataset = TimeSeriesDataset(region_data, input_cols, target_features, seq_length)
        region_loader = DataLoader(region_dataset, batch_size=32, shuffle=False)
        if best_model_name in ["TimeMoE", "ConvTimeNet", "Ensemble (TimeMoE + ConvTimeNet)"]:
            region_preds, region_actuals, _ = get_predictions(time_moe_model if best_model_name == "TimeMoE" else conv_time_net, region_loader)
            if best_model_name == "Ensemble (TimeMoE + ConvTimeNet)":
                region_preds = time_moe_weight * region_preds + conv_time_weight * get_predictions(conv_time_net, region_loader)[0]
        else:
            X_region, y_region = prepare_traditional_data(region_data, input_cols, target_features, seq_length)
            region_preds = (meta_model.predict(np.hstack([get_predictions(time_moe_model, region_loader)[0], get_predictions(conv_time_net, region_loader)[0]])[:, :4]) if best_model_name == "Stacking (TimeMoE + ConvTimeNet)" else
                            rf_model.predict(X_region) if best_model_name == "RandomForest" else xgb_model.predict(X_region))
            region_actuals = y_region
        # Inverse transform time series predictions
        transformed_region_preds = np.zeros((region_preds.shape[0], scaler.n_features_in_))
        for i, idx in enumerate(feature_indices):
            transformed_region_preds[:, idx] = region_preds[:, i]
        region_preds_inv = scaler.inverse_transform(transformed_region_preds)[:, feature_indices]
        transformed_region_actuals = np.zeros((region_actuals.shape[0], scaler.n_features_in_))
        for i, idx in enumerate(feature_indices):
            transformed_region_actuals[:, idx] = region_actuals[:, i]
        region_actuals_inv = scaler.inverse_transform(transformed_region_actuals)[:, feature_indices]
        for i, feature in enumerate(target_features):
            plt.figure(figsize=(12, 6))
            plt.plot(region_data['统计年度'][:len(region_actuals)], region_actuals_inv[:, i], label='实际值', marker='o', color='navy')
            plt.plot(region_data['统计年度'][:len(region_preds)], region_preds_inv[:, i], label='预测值', marker='s', linestyle='--', color='crimson')
            plt.title(f'{region} {feature} {best_model_name} 预测与实际值对比 (2014-2022)', fontsize=18)
            plt.xlabel('年份', fontsize=16)
            plt.ylabel(feature, fontsize=16)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{best_model_name}_{region}_{feature}_time_series.png'), dpi=300)
            plt.close()
# SHAP Analysis for Best Model
if best_model_name in ["RandomForest", "XGBoost"]:
    explainer = shap.TreeExplainer(rf_model if best_model_name == "RandomForest" else xgb_model)
    shap_values = explainer.shap_values(X_test)
    feature_names = [f"{col}_t{k}" for k in range(seq_length) for col in input_cols]

    for i, feature in enumerate(target_features):
        shap_values_target = shap_values[:, :, i]  # Shape: (n_samples, n_features)

        # Compute mean absolute SHAP values for feature importance (for pie chart)
        mean_abs_shap = np.abs(shap_values_target).mean(axis=0)
        total_shap = mean_abs_shap.sum()
        shap_contributions = (mean_abs_shap / total_shap) * 100  # Percentage contribution

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 2]})

        # (a) Pie chart for feature contributions
        ax1.pie(shap_contributions, labels=feature_names, autopct='%1.1f%%', startangle=90,
                colors=plt.cm.Paired(np.arange(len(feature_names))))
        ax1.set_title(f'Contribution for {feature} (%)', fontsize=14)
        ax1.axis('equal')  # Equal aspect ratio ensures pie is circular

        # (b) Beeswarm plot for SHAP values
        shap.summary_plot(shap_values_target, features=X_test, feature_names=feature_names, plot_type="dot", show=False,
                          ax=ax2)
        ax2.set_title(f'Feature Value Impact on {feature}', fontsize=14)
        ax2.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_combined_{feature}_{best_model_name}.png'), dpi=300)
        plt.close()

elif best_model_name in ["TimeMoE", "ConvTimeNet", "Ensemble (TimeMoE + ConvTimeNet)",
                         "Stacking (TimeMoE + ConvTimeNet)"]:
    class WrappedModel(nn.Module):
        def __init__(self, model, seq_length, input_size, weight1=None, model2=None, weight2=None, meta_model=None):
            super(WrappedModel, self).__init__()
            self.model = model
            self.model2 = model2
            self.weight1 = weight1
            self.weight2 = weight2
            self.meta_model = meta_model
            self.seq_length = seq_length
            self.input_size = input_size

        def forward(self, x):
            if x.dim() == 2:
                batch_size = x.size(0)
                x = x.view(batch_size, self.seq_length, self.input_size)
            elif x.dim() == 3:
                batch_size = x.size(0)
            else:
                x = x.view(1, self.seq_length, self.input_size)
                batch_size = 1

            if self.meta_model is not None:
                with torch.no_grad():
                    out1 = self.model(x)
                    out2 = self.model2(x)
                meta_features = torch.hstack([out1, out2]).cpu().numpy()
                return torch.FloatTensor(self.meta_model.predict(meta_features)).to(x.device)
            elif self.model2 is not None and self.weight1 is not None and self.weight2 is not None:
                out1 = self.model(x)
                out2 = self.model2(x)
                return self.weight1 * out1 + self.weight2 * out2
            return self.model(x)


    wrapped_model = WrappedModel(
        model=time_moe_model if best_model_name in ["TimeMoE", "Ensemble (TimeMoE + ConvTimeNet)",
                                                    "Stacking (TimeMoE + ConvTimeNet)"] else conv_time_net,
        seq_length=seq_length,
        input_size=input_size,
        weight1=time_moe_weight if best_model_name == "Ensemble (TimeMoE + ConvTimeNet)" else None,
        model2=conv_time_net if best_model_name in ["Ensemble (TimeMoE + ConvTimeNet)",
                                                    "Stacking (TimeMoE + ConvTimeNet)"] else None,
        weight2=conv_time_weight if best_model_name == "Ensemble (TimeMoE + ConvTimeNet)" else None,
        meta_model=meta_model if best_model_name == "Stacking (TimeMoE + ConvTimeNet)" else None
    ).to(device)


    def model_predict(inputs):
        if isinstance(inputs, np.ndarray):
            inputs = torch.FloatTensor(inputs).to(device)
        with torch.no_grad():
            outputs = wrapped_model(inputs)
        return outputs.cpu().numpy()


    # Prepare background data with correct shape
    background_data = test_inputs[:10].reshape(-1, seq_length * input_size)

    for i, feature in enumerate(target_features):
        def single_output_predict(inputs):
            if isinstance(inputs, np.ndarray):
                inputs = torch.FloatTensor(inputs).to(device)
            with torch.no_grad():
                outputs = wrapped_model(inputs)[:, i]
            return outputs.cpu().numpy()


        explainer = shap.KernelExplainer(single_output_predict, background_data)
        try:
            shap_values = explainer.shap_values(test_inputs_flat.cpu().numpy(), nsamples=100)
            features_reshaped = test_inputs.reshape(n_samples, feature_dim)
            feature_names = [f"{col}_t{k}" for k in range(seq_length) for col in input_cols]

            # Compute mean absolute SHAP values for feature importance (for pie chart)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            total_shap = mean_abs_shap.sum()
            shap_contributions = (mean_abs_shap / total_shap) * 100  # Percentage contribution

            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 2]})

            # (a) Pie chart for feature contributions
            ax1.pie(shap_contributions, labels=feature_names, autopct='%1.1f%%', startangle=90,
                    colors=plt.cm.Paired(np.arange(len(feature_names))))
            ax1.set_title(f'Contribution for {feature} (%)', fontsize=14)
            ax1.axis('equal')

            # (b) Beeswarm plot for SHAP values
            shap.summary_plot(shap_values, features=features_reshaped, feature_names=feature_names, plot_type="dot",
                              show=False, ax=ax2)
            ax2.set_title(f'Feature Value Impact on {feature}', fontsize=14)
            ax2.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=12)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'shap_combined_{feature}_{best_model_name}.png'), dpi=300)
            plt.close()
        except Exception as e:
            print(f"SHAP analysis failed for {feature}: {e}")

# 14. Future Predictions (2025-2027)
def exp_trend(t, a, b):
    return a * np.exp(b * t)

historical_df = df[df["统计年度"].between(2014, 2022)].copy()
historical_input = np.zeros((historical_df.shape[0], len(scale_cols)))
historical_input[:, [scale_cols.index(col) for col in target_features]] = historical_df[target_features].values
for col in scale_cols:
    if col not in target_features and col in historical_df.columns:
        historical_input[:, scale_cols.index(col)] = historical_df[col].values
historical_raw = scaler.inverse_transform(historical_input)[:, [scale_cols.index(col) for col in target_features]]
historical_df[target_features] = historical_raw
for col in target_features:
    historical_df[col] = historical_df[col].clip(lower=1)
historical_df["老龄化率"] = historical_df["65岁及以上人口数"] / historical_df["人口数"] * 100
historical_df["老年人口抚养比"] = historical_df["65岁及以上人口数"] / historical_df["15-64岁人口数"] * 100
historical_df["老龄化率"] = historical_df["老龄化率"].clip(lower=0, upper=100)
historical_df["老年人口抚养比"] = historical_df["老年人口抚养比"].clip(lower=0, upper=100)

exp_params_per_province = {province: {} for province in provinces}
for province in provinces:
    province_data = historical_df[historical_df["地区名称"] == province].sort_values("统计年度")
    t = np.arange(len(province_data))
    for col in target_features:
        y = province_data[col].values
        try:
            params, _ = curve_fit(exp_trend, t, y, p0=[y[0], 0.01], maxfev=10000)
            exp_params_per_province[province][col] = params
        except RuntimeError:
            exp_params_per_province[province][col] = [y[0], 0]

bed_2024_data = {
    "北京市": 116629, "天津市": 64290, "河北省": 291670, "山西省": 103130, "内蒙古自治区": 75278,
    "辽宁省": 233365, "吉林省": 179085, "黑龙江省": 229252, "上海市": 155134, "江苏省": 415078,
    "浙江省": 279934, "安徽省": 329460, "福建省": 74999, "江西省": 195478, "山东省": 432824,
    "河南省": 450235, "湖北省": 311795, "湖南省": 299359, "广东省": 255598, "广西壮族自治区": 103417,
    "海南省": 16135, "重庆市": 168145, "四川省": 428705, "贵州省": 70388, "云南省": 89874,
    "西藏自治区": 13232, "陕西省": 144468, "甘肃省": 40632, "青海省": 13648, "宁夏回族自治区": 26734,
    "新疆维吾尔自治区": 97735
}

def linear_trend(t, m, c):
    return m * t + c

bed_data_2021_2022_2024 = {}
for province in provinces:
    province_data = historical_df[historical_df["地区名称"] == province].sort_values("统计年度")
    bed_2021 = province_data[province_data["统计年度"] == 2021]["养老床位"].values
    bed_2022 = province_data[province_data["统计年度"] == 2022]["养老床位"].values
    if province in bed_2024_data and len(bed_2021) > 0 and len(bed_2022) > 0:
        bed_2024 = bed_2024_data[province]
        bed_data_2021_2022_2024[province] = {
            "years": [2021, 2022, 2024],
            "beds": [bed_2021[0], bed_2022[0], bed_2024]
        }
    else:
        bed_data_2021_2022_2024[province] = None

linear_params_2021_2024 = {province: {} for province in provinces}
for province in provinces:
    if bed_data_2021_2022_2024[province] is None:
        continue
    data = bed_data_2021_2022_2024[province]
    t = np.array([year - 2021 for year in data["years"]])
    y = np.array(data["beds"])
    try:
        params, _ = curve_fit(linear_trend, t, y, p0=[10000, y[0]], maxfev=10000)
        linear_params_2021_2024[province]["养老床位"] = params
    except RuntimeError:
        linear_params_2021_2024[province]["养老床位"] = [0, y[0]]
    for col in [col for col in target_features if col != "养老床位"]:
        linear_params_2021_2024[province][col] = exp_params_per_province[province][col]

future_years = np.arange(2025, 2028)
future_df_adjusted = pd.DataFrame({
    "统计年度": np.repeat(future_years, n_provinces),
    "地区名称": np.tile(provinces, len(future_years))
})

for province in provinces:
    t_future = np.arange(4, 7)
    for col in target_features:
        if col == "养老床位" and province in linear_params_2021_2024 and "养老床位" in linear_params_2021_2024[province]:
            m, c = linear_params_2021_2024[province]["养老床位"]
            future_values = linear_trend(t_future, m, c)
            future_values = np.maximum(future_values, 1)
            future_df_adjusted.loc[future_df_adjusted["地区名称"] == province, col] = future_values
        elif col == "65岁及以上人口数":
            province_data = historical_df[historical_df["地区名称"] == province].sort_values("统计年度")
            t_historical = np.arange(len(province_data))
            y_historical = province_data["65岁及以上人口数"].values
            try:
                params, _ = curve_fit(exp_trend, t_historical, y_historical, p0=[y_historical[0], 0.01], maxfev=10000)
                a, b = params
            except RuntimeError:
                a, b = y_historical[0], 0
            t_future_elderly = np.arange(11, 14)
            future_values = exp_trend(t_future_elderly, a, b)
            future_df_adjusted.loc[future_df_adjusted["地区名称"] == province, col] = future_values
        else:
            a, b = linear_params_2021_2024[province][col]
            future_values = exp_trend(t_future, a, b)
            future_df_adjusted.loc[future_df_adjusted["地区名称"] == province, col] = future_values

future_df_adjusted["老龄化率"] = future_df_adjusted["65岁及以上人口数"] / future_df_adjusted["人口数"] * 100
future_df_adjusted["老年人口抚养比"] = future_df_adjusted["65岁及以上人口数"] / future_df_adjusted["15-64岁人口数"] * 100
future_df_adjusted["老龄化率"] = future_df_adjusted["老龄化率"].clip(lower=0, upper=100)
future_df_adjusted["老年人口抚养比"] = future_df_adjusted["老年人口抚养比"].clip(lower=0, upper=100)

historical_aging_rate = historical_df.groupby("统计年度")["老龄化率"].mean()
aging_rate_growth = historical_aging_rate.pct_change().mean()
for year in range(2026, 2028):
    future_df_adjusted.loc[future_df_adjusted["统计年度"] == year, "老龄化率"] = (
        future_df_adjusted.loc[future_df_adjusted["统计年度"] == year - 1, "老龄化率"].values * (1 + aging_rate_growth)
    )

future_df_adjusted.to_csv("future_predictions_2025_2027_linear_bed.csv", index=False)
print("未来预测 (2025-2027) 已保存至 future_predictions_2025_2027_linear_bed.csv")

# 15. Visualize Trends
plot_features = ["人口数", "15-64岁人口数", "65岁及以上人口数", "养老床位", "老龄化率", "老年人口抚养比"]
trend_dir = "province_trend_plots"
Path(trend_dir).mkdir(exist_ok=True)

historical_extended_df = historical_df.copy()
historical_2024 = pd.DataFrame({
    "统计年度": np.repeat([2024], n_provinces),
    "地区名称": np.tile(provinces, 1)
})

future_years_full = np.arange(2023, 2028)
future_df = pd.DataFrame({
    "统计年度": np.repeat(future_years_full, n_provinces),
    "地区名称": np.tile(provinces, len(future_years_full))
})

for province in provinces:
    t_future = np.arange(9, 14)
    for col in target_features:
        a, b = exp_params_per_province[province][col]
        future_values = exp_trend(t_future, a, b)
        future_df.loc[future_df["地区名称"] == province, col] = future_values

future_df["老龄化率"] = future_df["65岁及以上人口数"] / future_df["人口数"] * 100
future_df["老年人口抚养比"] = future_df["65岁及以上人口数"] / future_df["15-64岁人口数"] * 100
future_df["老龄化率"] = future_df["老龄化率"].clip(lower=0, upper=100)
future_df["老年人口抚养比"] = future_df["老年人口抚养比"].clip(lower=0, upper=100)

for province in provinces:
    historical_2024.loc[(historical_2024["地区名称"] == province) & (historical_2024["统计年度"] == 2024), "养老床位"] = bed_2024_data[province]
    for col in [col for col in plot_features if col != "养老床位"]:
        val = future_df[(future_df["地区名称"] == province) & (future_df["统计年度"] == 2024)][col].values
        if len(val) > 0:
            historical_2024.loc[(historical_2024["地区名称"] == province) & (historical_2024["统计年度"] == 2024), col] = val[0]

historical_extended_df = pd.concat([historical_extended_df, historical_2024], ignore_index=True)
combined_df = pd.concat([historical_extended_df, future_df_adjusted], ignore_index=True)
combined_df = combined_df[combined_df["统计年度"] != 2023]

for province in provinces:
    province_dir = os.path.join(trend_dir, province)
    Path(province_dir).mkdir(exist_ok=True)
    province_data = combined_df[combined_df["地区名称"] == province].sort_values("统计年度")
    for feature in plot_features:
        plt.figure(figsize=(12, 8))
        historical_mask = province_data["统计年度"] <= 2024
        plt.plot(province_data[historical_mask]["统计年度"], province_data[historical_mask][feature],
                 marker='o', linestyle='-', color='navy', label='历史数据 (2014-2022, 2024)')
        future_mask = province_data["统计年度"] >= 2025
        plt.plot(province_data[future_mask]["统计年度"], province_data[future_mask][feature],
                 marker='s', linestyle='--', color='crimson', label='预测数据 (2025-2027)')
        historical_std = province_data[historical_mask][feature].std()
        future_std = province_data[future_mask][feature].std()
        plt.fill_between(province_data[historical_mask]["统计年度"],
                         province_data[historical_mask][feature] - historical_std,
                         province_data[historical_mask][feature] + historical_std,
                         color='navy', alpha=0.1)
        plt.fill_between(province_data[future_mask]["统计年度"],
                         province_data[future_mask][feature] - future_std,
                         province_data[future_mask][feature] + future_std,
                         color='crimson', alpha=0.1)
        plt.title(f"{province} - {feature} 历史与预测趋势", fontsize=18)
        plt.xlabel("统计年度", fontsize=16)
        plt.ylabel(feature, fontsize=16)
        plt.xticks(np.arange(2014, 2028, 1), rotation=45)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        safe_feature_name = feature.replace("/", "_").replace(" ", "_")
        plt.savefig(os.path.join(province_dir, f"{safe_feature_name}_trend.png"), dpi=300)
        plt.close()

print(f"所有省份的趋势图已保存至 {trend_dir}")
print("所有模型评估和可视化完成！")