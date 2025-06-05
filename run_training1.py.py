import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import shap  # 添加shap库

# 预设Box‑Cox最优lambda参数合集
optimal_lambdas = {
    "密度(g/cm3)": 6.758,
    "磁化率（10-3SI）": -0.311,
    "P波(m/s)": -0.135,
    "S波（m/s）": 0.156,
    "电阻率（Ω·m）": -0.146,
    "极化率（%）": 0.026,
    "渗透率（mD）": -0.051,
    "孔隙度": 0.270
}

# ----------------- 辅助函数 -----------------

def _get_positive_limits(arr1, arr2):
    """
    从两个数组中提取所有正值，并返回 (最小正值, 最大正值)。
    如果没有正值，则返回默认 (1e-5, 1)。
    """
    combined = np.concatenate((arr1, arr2))
    pos = combined[combined > 0]
    if len(pos) == 0:
        return (1e-5, 1)
    return (np.min(pos), np.max(pos))

def preprocess_new_data(df, model_features, transform_type="log", remove_porosity=False):
    """
    数据预处理：
      1. 仅保留模型所需的特征（model_features中出现的列）。
      2. 对类别变量（如“地区”、“岩性”、“状态”）转换为字符串；
         对非类别型数值变量，根据 transform_type 进行转换：
             当 transform_type=="log" 时，先 clip 后使用 np.log1p；
             当 transform_type=="boxcox" 时，先转换为浮点型、clip，再使用 optimal_lambdas（如存在）进行 Box‑Cox 转换。
      3. 若 remove_porosity=True，则删除“孔隙度”列（如果存在）。
    """
    df = df.copy()
    # 保留存在的模型特征
    df = df[[col for col in model_features if col in df.columns]]
    
    # 删除孔隙度（如果需要且存在）
    if remove_porosity and "孔隙度" in df.columns:
        df = df.drop(columns=["孔隙度"])
    
    # 定义类别变量列表（依据实际需求调整）
    categorical_cols = ['地区', '岩性', '状态']
    for col in df.columns:
        if col in categorical_cols:
            df[col] = df[col].astype(str)
        else:
            # 转换为浮点型；若失败则产生 NaN
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            df[col] = df[col].clip(lower=1e-5)
            if transform_type == "log":
                df[col] = np.log1p(df[col])
            elif transform_type == "boxcox":
                if col in optimal_lambdas:
                    lam = optimal_lambdas[col]
                    try:
                        df[col] = boxcox(df[col].values, lmbda=lam)
                    except Exception as e:
                        print(f"对 {col} 进行 Box‑Cox 转换时出错: {e}")
                else:
                    # 没有最优参数则不转换
                    pass
    return df

def plot_combined_predictions(y_test_boxcox, y_pred_boxcox, lam_target,
                              y_test_log, y_pred_log, title_prefix="渗透率预测"):
    """
    将 Box‑Cox 与对数逆变换后的预测对比图及残差分布图合并在一张图里，采用 2×2 布局：
      • 左侧子图（上：Box‑Cox预测；下：Box‑Cox残差分布）
      • 右侧子图（上：对数逆变换预测；下：对数逆变换残差分布）
    参数：
      y_test_boxcox, y_pred_boxcox：目标变量经过 Box‑Cox 转换后的测试值与预测值（数组）
      lam_target：Box‑Cox 转换使用的 λ 参数
      y_test_log, y_pred_log：目标变量经过对数转换（log1p）后的测试值与预测值（数组）
      title_prefix：图表标题前缀
    """
    # 先分别获得逆变换后的结果
    # Box‑Cox逆变换
    y_test_inv_boxcox = inv_boxcox(y_test_boxcox, lam_target)
    y_pred_inv_boxcox = inv_boxcox(y_pred_boxcox, lam_target)
    # 对数逆变换
    y_test_inv_log = np.expm1(y_test_log)
    y_pred_inv_log = np.expm1(y_pred_log)
    
    # 创建2×2布局的子图
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))

    # ----------------- 第一列：Box‑Cox -----------------
    # 上左：Box‑Cox预测对比图
    ax = axes[0, 0]
    ax.scatter(y_test_inv_boxcox, y_pred_inv_boxcox, color='blue', alpha=0.6, label='预测值')
    lim_lower, lim_upper = _get_positive_limits(y_test_inv_boxcox, y_pred_inv_boxcox)
    ax.plot([lim_lower, lim_upper], [lim_lower, lim_upper], 'r--', label='完美预测')
    ax.set_xlabel('实际渗透率 (Box‑Cox逆变换, mD)')
    ax.set_ylabel('预测渗透率 (Box‑Cox逆变换, mD)')
    ax.set_title(title_prefix + ' - Box‑Cox预测')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lim_lower, lim_upper)
    ax.set_ylim(lim_lower, lim_upper)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    # 下左：Box‑Cox残差分布图
    ax = axes[1, 0]
    residuals_boxcox = y_test_inv_boxcox - y_pred_inv_boxcox
    residuals_boxcox = residuals_boxcox[~np.isnan(residuals_boxcox)]
    try:
        rmin, rmax = np.percentile(residuals_boxcox, [1, 99])
    except Exception:
        rmin, rmax = (-1, 1)
    if not (np.isfinite(rmin) and np.isfinite(rmax)) or (abs(rmax - rmin) < 1e-8):
        rmin, rmax = (-1, 1)
    if np.std(residuals_boxcox) < 1e-8:
        ax.set_xscale('linear')
    else:
        linthresh = max(np.max(np.abs(residuals_boxcox))/10., 1e-5)
        ax.set_xscale('symlog', linthresh=linthresh)
    ax.set_xlim(rmin, rmax)
    ax.set_xlabel('Box‑Cox逆变换残差 (mD)')
    ax.set_title('Box‑Cox逆变换残差分布')
    sns.histplot(residuals_boxcox, bins=50, kde=True, color='orange', edgecolor='black', ax=ax)
    
    # ----------------- 第二列：对数 -----------------
    # 上右：对数预测对比图
    ax = axes[0, 1]
    ax.scatter(y_test_inv_log, y_pred_inv_log, color='green', alpha=0.6, label='预测值')
    lim_lower, lim_upper = _get_positive_limits(y_test_inv_log, y_pred_inv_log)
    ax.plot([lim_lower, lim_upper], [lim_lower, lim_upper], 'r--', label='完美预测')
    ax.set_xlabel('实际渗透率 (对数逆变换, mD)')
    ax.set_ylabel('预测渗透率 (对数逆变换, mD)')
    ax.set_title(title_prefix + ' - 对数预测')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lim_lower, lim_upper)
    ax.set_ylim(lim_lower, lim_upper)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    # 下右：对数残差分布图
    ax = axes[1, 1]
    residuals_log = y_test_inv_log - y_pred_inv_log
    residuals_log = residuals_log[~np.isnan(residuals_log)]
    try:
        rmin_log, rmax_log = np.percentile(residuals_log, [1, 99])
    except Exception:
        rmin_log, rmax_log = (-1, 1)
    if not (np.isfinite(rmin_log) and np.isfinite(rmax_log)) or (abs(rmax_log - rmin_log) < 1e-8):
        rmin_log, rmax_log = (-1, 1)
    if np.std(residuals_log) < 1e-8:
        ax.set_xscale('linear')
    else:
        linthresh_log = max(np.max(np.abs(residuals_log))/10., 1e-5)
        ax.set_xscale('symlog', linthresh=linthresh_log)
    ax.set_xlim(rmin_log, rmax_log)
    ax.set_xlabel('对数逆变换残差 (mD)')
    ax.set_title('对数逆变换残差分布')
    sns.histplot(residuals_log, bins=50, kde=True, color='purple', edgecolor='black', ax=ax)
    
    plt.tight_layout()
    plt.show()

# ----------------- 主函数 -----------------

def main():
    # 配置字体及图表样式
    plt.rcParams['font.family'] = ['Microsoft YaHei']  # 注意这里是font.family
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size']        = 12
    plt.rcParams['axes.labelsize']   = 14
    plt.rcParams['axes.titlesize']   = 16
    plt.rcParams['xtick.labelsize']  = 12
    plt.rcParams['ytick.labelsize']  = 12
    plt.rcParams['legend.fontsize']  = 12
    plt.rcParams['figure.titlesize'] = 18

    # 数据文件路径；如果不存在则生成模拟数据进行演示
    file_path = r"C:/Users/aaalo/Desktop/渗透率预测数据体.xlsx"
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        print("生成模拟数据进行演示...")
        np.random.seed(42)
        data_sim = pd.DataFrame({
            '序号': np.arange(1, 1001),
            '地区': np.random.choice(['A区', 'B区'], size=1000),
            '状态': np.random.choice(['正常', '异常'], size=1000),
            '岩性': np.random.choice(['1', '2', '3', '4'], size=1000),
            '密度(g/cm3)': np.random.uniform(2.2, 2.6, size=1000),
            '磁化率（10-3SI）': np.random.exponential(scale=1, size=1000),
            'P波(m/s)': np.random.exponential(scale=1000, size=1000),
            'S波（m/s）': np.random.normal(loc=100, scale=10, size=1000),
            '电阻率（Ω·m）': np.random.exponential(scale=50, size=1000),
            '极化率（%）': np.random.exponential(scale=10, size=1000),
            '渗透率（mD）': np.random.exponential(scale=5, size=1000),
            'Unnamed: 11': np.nan,
            'Unnamed: 12': np.nan,
            'Unnamed: 13': np.nan,
            '孔隙度': np.random.uniform(0.05, 0.3, size=1000)
        })
        df = data_sim.copy()
    else:
        print("数据加载成功！")
        df = pd.read_excel(file_path)

    print("数据预览:")
    print(df.head())
    
    # 删除目标变量“渗透率（mD）”缺失值的样本
    df = df.dropna(subset=["渗透率（mD）"])
    
    # 定义模型所需的特征（不包含“孔隙度”）
    model_feature_names = np.array([
        "磁化率（10-3SI）", "S波（m/s）", "电阻率（Ω·m）", "极化率（%）",
        "地区", "岩性", "状态"
    ])
    
    # 划分训练集与测试集（目标变量为渗透率）
    X = df.drop(columns=["渗透率（mD）"])
    y = df["渗透率（mD）"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ---------- 1. 基于 Box‑Cox 预处理 ----------
    processed_X_train_boxcox = preprocess_new_data(X_train, model_feature_names,
                                                   transform_type="boxcox", remove_porosity=True)
    processed_X_test_boxcox = preprocess_new_data(X_test, model_feature_names,
                                                  transform_type="boxcox", remove_porosity=True)
    model_boxcox = RandomForestRegressor(n_estimators=100, random_state=42)
    model_boxcox.fit(processed_X_train_boxcox, y_train)
    y_pred_boxcox = model_boxcox.predict(processed_X_test_boxcox)
    print("【Box‑Cox模型】 R2 分数：", r2_score(y_test, y_pred_boxcox))

    # 添加 SHAP 部分: 基于 Box‑Cox 模型的特征重要性解释（条形柱状图）
    explainer = shap.TreeExplainer(model_boxcox)
    shap_values = explainer.shap_values(processed_X_train_boxcox)
    shap.summary_plot(shap_values, processed_X_train_boxcox, feature_names=processed_X_train_boxcox.columns, plot_type="bar")

    # 对目标变量渗透率使用 Box‑Cox 转换（预防负值，用 np.clip）
    lam_target = optimal_lambdas["渗透率（mD）"]  # -0.051
    y_test_boxcox_transformed = np.array([boxcox(val, lmbda=lam_target) for val in np.clip(y_test, 1e-5, None)])
    
    # ---------- 2. 基于对数转换预处理 ----------
    processed_X_train_log = preprocess_new_data(X_train, model_feature_names,
                                                transform_type="log", remove_porosity=True)
    processed_X_test_log = preprocess_new_data(X_test, model_feature_names,
                                               transform_type="log", remove_porosity=True)
    model_log = RandomForestRegressor(n_estimators=100, random_state=42)
    model_log.fit(processed_X_train_log, y_train)
    y_pred_log = model_log.predict(processed_X_test_log)
    print("【对数转换模型】 R2 分数：", r2_score(y_test, y_pred_log))
    
    # 对目标变量进行对数转换（log1p），以便后续逆变换使用 np.expm1
    y_test_log_transformed = np.log1p(y_test)

    # ---------- 3. 合并绘图：将Box‑Cox与对数两套预测及残差图显示在一张图中 ----------
    plot_combined_predictions(y_test_boxcox_transformed, y_pred_boxcox, lam_target,
                              y_test_log_transformed, y_pred_log,
                              title_prefix="渗透率预测 (无孔隙度)")
    
    # 可选：保存Box‑Cox模型（示例）
    model_filename = "trained_model_no_porosity_boxcox.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model_boxcox, f)
    print("模型已保存：", model_filename)

if __name__ == '__main__':
    main()
