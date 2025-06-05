import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import shap
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
import pickle

# 特征重要性绘图函数
def plot_feature_importance(model, features, title):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    fig_width = max(10, len(features) * 0.6)
    plt.figure(figsize=(fig_width, 6))
    plt.bar(range(len(features)), importance[indices], align="center", color='skyblue')
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45, ha="right")
    plt.title(title)
    plt.xlabel("特征")
    plt.ylabel("重要性分数")
    plt.tight_layout()
    plt.show()

# 自定义特征相关性热图函数
def plot_feature_heatmap_custom(data, title):
    corr = data.corr()
    n = corr.shape[0]

    base_size = 12
    scale_factor = 1.8
    min_cell_size = 1.0

    fig_width = min(max(base_size, n * min_cell_size * scale_factor), 18)
    fig_height = min(max(base_size, n * min_cell_size * scale_factor), 18)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)

    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=-1, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    for i in range(n):
        for j in range(n):
            alpha = 0.3 if i >= j else 0.1
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor='white', edgecolor='lightgrey', alpha=alpha))

    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            cell_color = cmap(norm(val))
            if i >= j:
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=cell_color, alpha=0.5, edgecolor='none'))
                font_size = 10 if n < 15 else 8
                text_color = 'black' if abs(val) < 0.7 else 'white'
                ax.text(j + 0.5, i + 0.5, f"{val:.2f}", ha="center", va="center",
                        fontsize=font_size, color=text_color,
                        weight='bold' if abs(val) > 0.7 else 'normal')
            else:
                bubble_size = abs(val) * 3000
                ax.scatter(j + 0.5, i + 0.5, s=bubble_size, color=cell_color,
                           edgecolors='black', alpha=0.7, linewidths=0.5)

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)

    label_fontsize = 12 if n < 20 else (10 if n < 30 else 8)
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=label_fontsize)
    ax.set_yticklabels(corr.columns, fontsize=label_fontsize)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.invert_yaxis()
    ax.set_aspect('equal')

    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.08, aspect=40)
    cbar.ax.tick_params(labelsize=10)

    plt.title(title, fontsize=16, pad=10)
    plt.tight_layout(rect=[0.05, 0.15, 0.98, 0.95])
    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.15)
    plt.show()
    return corr

# 分析高度相关特征对并删除冗余特征
def analyze_high_correlation_pairs(corr_matrix, model, threshold=0.8, importance_threshold=0.05):
    features = corr_matrix.columns
    importance = pd.Series(model.feature_importances_, index=features)
    low_importance_features = importance[importance < importance_threshold].index
    high_corr_pairs = []
    for i in range(len(features)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col1 = features[i]
                col2 = features[j]
                high_corr_pairs.append((col1, col2))
    to_drop = set()
    for col1, col2 in high_corr_pairs:
        if col1 in low_importance_features and col2 in low_importance_features:
            if importance[col1] < importance[col2]:
                to_drop.add(col1)
            else:
                to_drop.add(col2)
        elif col1 in low_importance_features:
            to_drop.add(col1)
        elif col2 in low_importance_features:
            to_drop.add(col2)
    if to_drop:
        print("\n在高度相关特征对中，删除的重要性较低的特征:")
        for feature in to_drop:
            print(f"- {feature}")
    else:
        print("\n没有检测到需要删除的高度相关冗余特征")
    return list(to_drop)

# 数据变换、绘图及其它辅助函数
def plot_transformed_data(data, data_log, data_bc, col, lambdas):
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    sns.histplot(data[col], kde=True, color='skyblue', edgecolor='black')
    plt.title(f'{col} 原始分布')
    plt.xlabel('值')
    plt.ylabel('频率')

    plt.subplot(1, 3, 2)
    sns.histplot(data_log[col], kde=True, color='lightgreen', edgecolor='black')
    plt.title(f'{col} 对数变换后分布')
    plt.xlabel('值')
    plt.ylabel('频率')

    plt.subplot(1, 3, 3)
    sns.histplot(data_bc[col], kde=True, color='salmon', edgecolor='black')
    plt.title(f'{col} Box-Cox变换后分布 (λ={lambdas[col]:.3f})')
    plt.xlabel('值')
    plt.ylabel('频率')
    plt.tight_layout()
    plt.show()

def plot_transformed_boxplot(data, data_log, data_bc, col, lambdas):
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    sns.boxplot(y=data[col], color='skyblue')
    plt.title(f'{col} 原始箱线图')
    plt.ylabel('值')

    plt.subplot(1, 3, 2)
    sns.boxplot(y=data_log[col], color='lightgreen')
    plt.title(f'{col} 对数变换后箱线图')
    plt.ylabel('值')

    plt.subplot(1, 3, 3)
    sns.boxplot(y=data_bc[col], color='salmon')
    plt.title(f'{col} Box-Cox变换后箱线图 (λ={lambdas[col]:.3f})')
    plt.ylabel('值')
    plt.tight_layout()
    plt.show()

def calculate_skewness(data, data_log, data_bc, col):
    skew_original = stats.skew(data[col].dropna())
    skew_log = stats.skew(data_log[col].dropna())
    skew_bc = stats.skew(data_bc[col].dropna())
    return skew_original, skew_log, skew_bc

def train_random_forest(X, y, data_type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{data_type} 数据随机森林模型评估:")
    print(f"均方误差(MSE): {mse:.4f}")
    print(f"R2 分数: {r2:.4f}")
    plot_feature_importance(model, X.columns, f"{data_type} 数据特征重要性")
    return model

def train_random_forest_classifier(X, y, data_type, target_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(f"\n[{data_type}] 对 {target_name} 分类结果:")
    print(f"准确率: {acc:.4f}")
    print("分类报告:")
    print(report)
    return clf

def plot_prediction_comparison(y_test_bc, y_pred_bc, y_test_log, y_pred_log):
    y_test_bc = y_test_bc.astype(float)
    y_pred_bc = y_pred_bc.astype(float)
    y_test_log = y_test_log.astype(float)
    y_pred_log = y_pred_log.astype(float)

    plt.figure(figsize=(12, 12))

    ax1 = plt.subplot(2, 2, 1)
    ax1.scatter(y_test_bc, y_pred_bc, color='blue', alpha=0.6, label='预测值')
    lim = [min(np.min(y_test_bc), np.min(y_pred_bc)), max(np.max(y_test_bc), np.max(y_pred_bc))]
    ax1.plot(lim, lim, 'r--', label='理想线')
    ax1.set_xlabel('实际渗透率 (Box-Cox变换后)')
    ax1.set_ylabel('预测渗透率 (Box-Cox变换后)')
    ax1.set_title('Box-Cox 变换后预测渗透率')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.set_aspect('equal', adjustable='box')

    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(y_test_log, y_pred_log, color='green', alpha=0.6, label='预测值')
    lim = [min(np.min(y_test_log), np.min(y_pred_log)), max(np.max(y_test_log), np.max(y_pred_log))]
    ax2.plot(lim, lim, 'r--', label='理想线')
    ax2.set_xlabel('实际渗透率 (对数变换后)')
    ax2.set_ylabel('预测渗透率 (对数变换后)')
    ax2.set_title('对数变换后预测渗透率')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.set_aspect('equal', adjustable='box')

    residuals_bc = y_test_bc - y_pred_bc
    residuals_log = y_test_log - y_pred_log

    ax3 = plt.subplot(2, 2, 3)
    sns.histplot(residuals_bc, kde=True, color='purple', edgecolor='black', ax=ax3)
    ax3.set_xlabel('残差 (Box-Cox变换后)')
    ax3.set_title('Box-Cox 残差分布')

    ax4 = plt.subplot(2, 2, 4)
    sns.histplot(residuals_log, kde=True, color='orange', edgecolor='black', ax=ax4)
    ax4.set_xlabel('残差 (对数变换后)')
    ax4.set_title('对数残差分布')

    plt.tight_layout()
    plt.show()

# 修改后的预测结果绘制函数，坐标轴采用 symlog 显示
def plot_inverse_predictions_adjusted(y_test_log, y_pred_log, y_test_bc, y_pred_bc, lambda_target, title_prefix):
    # 先进行逆变换
    y_test_log_inv = np.expm1(y_test_log)
    y_pred_log_inv = np.expm1(y_pred_log)
    y_test_bc_inv = inv_boxcox(y_test_bc, lambda_target)
    y_pred_bc_inv = inv_boxcox(y_pred_bc, lambda_target)

    plt.figure(figsize=(10, 10))

    ax1 = plt.subplot(2, 2, 1)
    ax1.scatter(y_test_bc_inv, y_pred_bc_inv, color='blue', alpha=0.6, label='预测值')
    lim = [min(np.min(y_test_bc_inv), np.min(y_pred_bc_inv)), max(np.max(y_test_bc_inv), np.max(y_pred_bc_inv))]
    ax1.plot(lim, lim, 'r--', label='理想线')
    ax1.set_xlabel('实际渗透率 (Box-Cox逆变换, 单位: mD)')
    ax1.set_ylabel('预测渗透率 (Box-Cox逆变换, 单位: mD)')
    ax1.set_title(title_prefix + ' - Box-Cox逆变换')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()

    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(y_test_log_inv, y_pred_log_inv, color='green', alpha=0.6, label='预测值')
    lim = [min(np.min(y_test_log_inv), np.min(y_pred_log_inv)), max(np.max(y_test_log_inv), np.max(y_pred_log_inv))]
    ax2.plot(lim, lim, 'r--', label='理想线')
    ax2.set_xlabel('实际渗透率 (对数逆变换, 单位: mD)')
    ax2.set_ylabel('预测渗透率 (对数逆变换, 单位: mD)')
    ax2.set_title(title_prefix + ' - 对数逆变换')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()

    ax3 = plt.subplot(2, 2, 3)
    residuals_log = y_test_log_inv - y_pred_log_inv
    sns.histplot(residuals_log, kde=True, color='purple', edgecolor='black', ax=ax3)
    ax3.set_xlabel('残差 (实际 - 预测, 单位: mD)')
    ax3.set_title('对数逆变换残差分布')
    ax3.set_xscale('symlog')
    ax3.set_aspect('equal', adjustable='box')

    ax4 = plt.subplot(2, 2, 4)
    residuals_bc = y_test_bc_inv - y_pred_bc_inv
    sns.histplot(residuals_bc, kde=True, color='orange', edgecolor='black', ax=ax4)
    ax4.set_xlabel('残差 (实际 - 预测, 单位: mD)')
    ax4.set_title('Box-Cox逆变换残差分布')
    ax4.set_xscale('symlog')
    ax4.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()

def explain_rf_with_shap(model, X, model_name, sample_size=100):
    explainer = shap.TreeExplainer(model)
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X.copy()
    shap_values = explainer.shap_values(X_sample)
    print(f"\n【SHAP 解释】 {model_name} 的 SHAP 值：")
    shap.summary_plot(shap_values, X_sample, feature_names=X.columns, plot_type="bar", show=False)
    plt.title(f"{model_name} - SHAP 条形图(全局特征重要性)")
    plt.show()
    shap.summary_plot(shap_values, X_sample, feature_names=X.columns, show=False)
    plt.title(f"{model_name} - SHAP 摘要散点图")
    plt.show()

# 修改后的决策树可视化函数（最大深度固定为3，用于展示结构）
def train_and_visualize_decision_tree(X, y, data_type, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dt_model = DecisionTreeRegressor(max_depth=3, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n[{data_type}] 决策树模型评估:")
    print(f"均方误差(MSE): {mse:.4f}")
    print(f"R2 分数: {r2:.4f}")
    
    plt.figure(figsize=(20, 10))
    plot_tree(dt_model, feature_names=feature_names, filled=True, rounded=True, fontsize=12)
    plt.title(f"{data_type} 决策树可视化 (最大深度限制为3)")
    plt.show()
    
    return dt_model

def train_random_forest_grid_search(X, y, data_type):
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3,
                               scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    print(f"\n【超参数调优】 {data_type} 最佳参数: ", grid_search.best_params_)
    print(f"【超参数调优】 {data_type} 最佳交叉验证 R2: ", grid_search.best_score_)
    best_model = grid_search.best_estimator_
    return best_model

# 新增：基于百分位数边界的异常值检测函数
def count_percentile_outliers(df, lower_q=0.05, upper_q=0.95):
    outlier_flags = pd.DataFrame(index=df.index)
    for col in df.columns:
        lower = df[col].quantile(lower_q)
        upper = df[col].quantile(upper_q)
        outlier_flags[col] = ((df[col] < lower) | (df[col] > upper)).astype(int)
    return outlier_flags.sum(axis=1)

# 主函数：加载数据、预处理、建模及绘图
def main():
    sns.set_style("darkgrid")
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18

    file_path = r"C:/Users/aaalo/Desktop/渗透率预测数据体.xlsx"
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        print("生成模拟数据进行演示...")
        np.random.seed(42)
        data_sim = pd.DataFrame({
            '磁化率（10-3SI）': np.random.exponential(scale=1, size=1000),
            'P波(m/s)': np.random.exponential(scale=1000, size=1000),
            'S波（m/s）': np.random.normal(loc=100, scale=10, size=1000),
            '电阻率（Ω·m）': np.random.exponential(scale=50, size=1000),
            '极化率（%）': np.random.exponential(scale=10, size=1000),
            '渗透率（mD）': np.random.exponential(scale=5, size=1000),
            '地区': np.random.choice(['A区', 'B区'], size=1000),
            '岩性': np.random.choice(['1', '2', '3', '4'], size=1000),
            '状态': np.random.choice(['正常', '异常'], size=1000)
        })
        df = data_sim
    else:
        print("数据加载成功！")
        df = pd.read_excel(file_path)

    print("\n数据概览:")
    print(df.head())
    print("\n数据描述统计:")
    print(df.describe())
    print("\n缺失值统计:")
    print(df.isnull().sum())

    print("\n正在预处理数据...")
    if '序号' in df.columns:
        df = df.drop(columns=['序号'])
    unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    categorical_cols = ['地区', '岩性', '状态']
    numerical_cols = [col for col in df.columns if col not in categorical_cols]

    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    target_col = '渗透率（mD）'
    df = df.dropna(subset=[target_col])

    data = df[numerical_cols]
    data_log = data.copy()
    data_bc = data.copy()
    lambdas = {}
    for col in data.columns:
        data_log[col] = np.log1p(data[col])
        pos_data = data[col].clip(lower=1e-5)
        data_bc[col], lambda_opt = stats.boxcox(pos_data)
        lambdas[col] = lambda_opt

    lambda_str = ", ".join([f"{col}: {lambdas[col]:.3f}" for col in lambdas])
    print(f"\nBox-Cox变换最优参数合集: {{ {lambda_str} }}")

    outlier_counts_log = count_percentile_outliers(data_log, 0.05, 0.95)
    outlier_counts_bc = count_percentile_outliers(data_bc, 0.05, 0.95)
    
    deleted_log = (outlier_counts_log >= 4).sum()
    deleted_bc = (outlier_counts_bc >= 4).sum()
    print(f"\n删除的对数变换样本数: {deleted_log}")
    print(f"删除的 Box-Cox变换样本数: {deleted_bc}")

    to_remove = (outlier_counts_log >= 4) | (outlier_counts_bc >= 4)
    print(f"\n最终依据合并规则将删除异常样本数：{to_remove.sum()}")
    valid_idx = ~to_remove
    data      = data.loc[valid_idx].copy()
    data_log  = data_log.loc[valid_idx].copy()
    data_bc   = data_bc.loc[valid_idx].copy()
    df        = df.loc[valid_idx].copy()

    for col in data.columns:
        plot_transformed_data(data, data_log, data_bc, col, lambdas)
        plot_transformed_boxplot(data, data_log, data_bc, col, lambdas)

    for col in data.columns:
        skew_original, skew_log, skew_bc = calculate_skewness(data, data_log, data_bc, col)
        print(f"{col} 偏度对比: 原始={skew_original:.3f}, 对数变换后={skew_log:.3f}, Box-Cox变换后={skew_bc:.3f}")

    X_num_log = data_log.drop(columns=[target_col])
    X_num_bc = data_bc.drop(columns=[target_col])
    y_log = data_log[target_col]
    y_bc = data_bc[target_col]

    X_cat_dummies = pd.get_dummies(df[categorical_cols], drop_first=True)

    X_log_reg = pd.concat([X_num_log, X_cat_dummies], axis=1)
    X_bc_reg = pd.concat([X_num_bc, X_cat_dummies], axis=1)

    print("\n================== 回归预测（渗透率）模型 ==================")
    print("\n【第一步】使用完整特征训练模型（含分类信息）：")
    model_log = train_random_forest(X_log_reg, y_log, "对数变换")
    print("\n【第一步】Box-Cox变换数据模型:")
    model_bc = train_random_forest(X_bc_reg, y_bc, "Box-Cox变换")

    print("\n【SHAP 解释】对完整特征数据训练的随机森林回归模型进行解释:")
    explain_rf_with_shap(model_log, X_log_reg, "对数变换随机森林回归模型")
    explain_rf_with_shap(model_bc, X_bc_reg, "Box-Cox变换随机森林回归模型")

    print("\n【自定义热图】展示完整特征相关性热图（下三角数字，上三角气泡）:")
    plot_feature_heatmap_custom(X_log_reg, "渗透率预测模型相关性热图")

    print("\n【第二步】通过高度相关性分析，删除冗余特征（删除高相关性中重要性较低者）:")
    corr_matrix = X_log_reg.corr()
    feats_to_drop = analyze_high_correlation_pairs(corr_matrix, model_log, threshold=0.8, importance_threshold=0.05)
    X_log_reg_new = X_log_reg.drop(columns=feats_to_drop)
    X_bc_reg_new = X_bc_reg.drop(columns=feats_to_drop)
    print(f"\n删除后剩余的特征有：{list(X_log_reg_new.columns)}")

    print("\n【第三步】使用删除冗余特征后的数据训练回归模型")
    model_log_new = train_random_forest(X_log_reg_new, y_log, "对数变换(删除冗余特征)")
    model_bc_new = train_random_forest(X_bc_reg_new, y_bc, "Box-Cox变换(删除冗余特征)")

    print("\n【SHAP 解释】对降维后随机森林回归模型进行解释:")
    explain_rf_with_shap(model_log_new, X_log_reg_new, "对数变换随机森林回归模型(删除冗余特征)")
    explain_rf_with_shap(model_bc_new, X_bc_reg_new, "Box-Cox变换随机森林回归模型(删除冗余特征)")

    print("\n================== 决策树可视化（渗透率预测） ==================")
    print("【对数变换】 - 使用完整特征训练决策树回归模型并进行可视化:")
    dt_model_log = train_and_visualize_decision_tree(X_log_reg, y_log, "对数变换决策树", list(X_log_reg.columns))
    print("【Box-Cox变换】 - 使用完整特征训练决策树回归模型并进行可视化:")
    dt_model_bc = train_and_visualize_decision_tree(X_bc_reg, y_bc, "Box-Cox变换决策树", list(X_bc_reg.columns))

    print("\n================== 决策树对比 ==================")
    for common_depth in [3, 5, 10]:
        print(f"\n【决策树对比】 相同决策树层数（max_depth = {common_depth}）下：")
        # 对数变换数据决策树
        X_train_log_dt, X_test_log_dt, y_train_log_dt, y_test_log_dt = train_test_split(X_log_reg, y_log, test_size=0.2, random_state=42)
        dt_log = DecisionTreeRegressor(max_depth=common_depth, random_state=42)
        dt_log.fit(X_train_log_dt, y_train_log_dt)
        y_pred_log_dt = dt_log.predict(X_test_log_dt)
        mse_log_dt = mean_squared_error(y_test_log_dt, y_pred_log_dt)
        r2_log_dt = r2_score(y_test_log_dt, y_pred_log_dt)
        # 计算对数逆变换后的评估指标
        y_test_log_dt_inv = np.expm1(y_test_log_dt)
        y_pred_log_dt_inv = np.expm1(y_pred_log_dt)
        mse_log_dt_inv = mean_squared_error(y_test_log_dt_inv, y_pred_log_dt_inv)
        r2_log_dt_inv = r2_score(y_test_log_dt_inv, y_pred_log_dt_inv)

        # Box-Cox变换数据决策树
        X_train_bc_dt, X_test_bc_dt, y_train_bc_dt, y_test_bc_dt = train_test_split(X_bc_reg, y_bc, test_size=0.2, random_state=42)
        dt_bc = DecisionTreeRegressor(max_depth=common_depth, random_state=42)
        dt_bc.fit(X_train_bc_dt, y_train_bc_dt)
        y_pred_bc_dt = dt_bc.predict(X_test_bc_dt)
        mse_bc_dt = mean_squared_error(y_test_bc_dt, y_pred_bc_dt)
        r2_bc_dt = r2_score(y_test_bc_dt, y_pred_bc_dt)
        # Box-Cox逆变换到原始尺度
        y_test_bc_dt_inv = inv_boxcox(y_test_bc_dt, lambdas[target_col])
        y_pred_bc_dt_inv = inv_boxcox(y_pred_bc_dt, lambdas[target_col])
        mse_bc_dt_inv = mean_squared_error(y_test_bc_dt_inv, y_pred_bc_dt_inv)
        r2_bc_dt_inv = r2_score(y_test_bc_dt_inv, y_pred_bc_dt_inv)

        print(f"对数变换数据: MSE = {mse_log_dt:.4f}, R2 = {r2_log_dt:.4f}")
        print(f"对数原始数据域模型评估: MSE = {mse_log_dt_inv:.4f}, R2 = {r2_log_dt_inv:.4f}")
        print(f"Box-Cox变换数据: MSE = {mse_bc_dt:.4f}, R2 = {r2_bc_dt:.4f}")
        print(f"Box-Cox原始数据域模型评估: MSE = {mse_bc_dt_inv:.4f}, R2 = {r2_bc_dt_inv:.4f}")
        
        plot_inverse_predictions_adjusted(y_test_log_dt, y_pred_log_dt, 
                                            y_test_bc_dt, y_pred_bc_dt, 
                                            lambdas[target_col], 
                                            f"决策树预测 (树深度={common_depth})")

    print("\n================== 参数优化：GridSearchCV调参 ==================")
    # Box-Cox变换数据模型参数优化，以及逆变换后评估
    optimized_model_bc = train_random_forest_grid_search(X_bc_reg_new, y_bc, "Box-Cox变换(优化)")
    _, X_test_opt_bc, _, y_test_opt_bc = train_test_split(X_bc_reg_new, y_bc, test_size=0.2, random_state=42)
    y_pred_opt_bc = optimized_model_bc.predict(X_test_opt_bc)
    y_test_opt_inv_bc = inv_boxcox(y_test_opt_bc, lambdas[target_col])
    y_pred_opt_inv_bc = inv_boxcox(y_pred_opt_bc, lambdas[target_col])
    mse_inv_bc = mean_squared_error(y_test_opt_inv_bc, y_pred_opt_inv_bc)
    r2_inv_bc = r2_score(y_test_opt_inv_bc, y_pred_opt_inv_bc)
    print(f"【Box-Cox变换(优化) - 原始数据域】 模型评估: MSE = {mse_inv_bc:.4f}, R2 = {r2_inv_bc:.4f}")

    # 对数变换数据模型参数优化，以及逆变换后评估
    optimized_model_log = train_random_forest_grid_search(X_log_reg_new, y_log, "对数变换(优化)")
    _, X_test_opt, _, y_test_opt = train_test_split(X_log_reg_new, y_log, test_size=0.2, random_state=42)
    y_pred_opt = optimized_model_log.predict(X_test_opt)
    y_test_opt_inv_log = np.expm1(y_test_opt)
    y_pred_opt_inv_log = np.expm1(y_pred_opt)
    mse_inv_log = mean_squared_error(y_test_opt_inv_log, y_pred_opt_inv_log)
    r2_inv_log = r2_score(y_test_opt_inv_log, y_pred_opt_inv_log)
    print(f"【对数变换(优化) - 原始数据域】 模型评估: MSE = {mse_inv_log:.4f}, R2 = {r2_inv_log:.4f}")

    # 绘制优化后模型预测结果对比图（逆变换后）
    plot_inverse_predictions_adjusted(y_test_opt, y_pred_opt,
                                       y_test_opt_bc, y_pred_opt_bc,
                                       lambdas[target_col],
                                       "优化后模型预测渗透率")

    # 保存经过GridSearchCV调优后的模型
    model_log_filename = "trained_model_log_optimized.pkl"
    with open(model_log_filename, "wb") as f:
        pickle.dump(optimized_model_log, f)
    print(f"\n对数变换（经过调优的模型）已保存到：{model_log_filename}")
    
    model_bc_filename = "trained_model_bc_optimized.pkl"
    with open(model_bc_filename, "wb") as f:
        pickle.dump(optimized_model_bc, f)
    print(f"Box-Cox变换（经过调优的模型）已保存到：{model_bc_filename}")

if __name__ == '__main__':
    main()
