import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import inv_boxcox  # 用于 Box-Cox 的逆变换
from scipy.stats import boxcox       # 用于 Box-Cox 正变换
import shap  # 添加 SHAP 库

# 配置 matplotlib 字体与样式
plt.rcParams['font.family']      = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size']        = 12
plt.rcParams['axes.labelsize']   = 14
plt.rcParams['axes.titlesize']   = 16
plt.rcParams['xtick.labelsize']  = 12
plt.rcParams['ytick.labelsize']  = 12
plt.rcParams['legend.fontsize']  = 12
plt.rcParams['figure.titlesize'] = 18

def preprocess_new_data(df, model_feature_names, transform_type="log",
                        target_col="渗透率（mD）", categorical_cols=["地区", "岩性", "状态"],
                        remove_porosity=False):
    """
    对新数据进行预处理：
      1. 删除不必要的列（如 '序号'、Unnamed列及目标变量列）；预测时会将目标变量剔除
         参数 remove_porosity 控制是否删除“孔隙度”，若模型训练包含则必须保留。
      2. 填充缺失值并转换数值变量；
      3. 数值变量进行对数或 Box-Cox 变换；
      4. 类别变量采用 one-hot 编码；
      5. 按照模型要求对齐特征顺序。
    """
    # 删除不必要的列
    if '序号' in df.columns:
        df = df.drop(columns=['序号'])
    unnamed = [col for col in df.columns if 'Unnamed' in col]
    if unnamed:
        df = df.drop(columns=unnamed)
    if target_col in df.columns:
        df = df.drop(columns=[target_col])
    
    if remove_porosity and "孔隙度" in df.columns:
        df = df.drop(columns=["孔隙度"])
    
    # 分离数值型与类别变量
    numeric_cols = [col for col in df.columns if col not in categorical_cols]
    df[numeric_cols] = df[numeric_cols].apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)
    
    # 数值变量变换
    if transform_type == "log":
        df_numeric = df[numeric_cols].apply(lambda col: np.log1p(col))
    elif transform_type == "boxcox":
        lam = 0.2  # 示例参数，按需调整
        df_numeric = df[numeric_cols].apply(lambda col: boxcox(col.clip(lower=1e-5), lmbda=lam)
                                              if np.all(col.clip(lower=1e-5) > 0) else col)
    else:
        raise ValueError("未知的变换类型，请选择 'log' 或 'boxcox'")
    
    # 类别变量 one-hot 编码（若存在则进行编码）
    if len(categorical_cols) > 0:
        df_cat = pd.get_dummies(df[categorical_cols], drop_first=True)
    else:
        df_cat = pd.DataFrame(index=df.index)
    
    # 拼接数值与类别变量，并按模型要求排列特征顺序
    processed_df = pd.concat([df_numeric, df_cat], axis=1)
    processed_df = processed_df.reindex(columns=model_feature_names, fill_value=0)
    return processed_df

def _get_positive_limits(arr1, arr2, default_min=1e-5, default_max=1e5):
    """
    从两个数组中过滤 NaN/Inf 后，取所有正值中的最小值作为下界，
    以及所有有限数中最大值作为上界；若无正值则用 default_min，
    若上界异常则采用 default_max。
    """
    data = np.concatenate((arr1.ravel(), arr2.ravel()))
    data = data[np.isfinite(data)]
    if data.size == 0:
        return default_min, default_max
    positive_data = data[data > 0]
    if positive_data.size > 0:
        lim_lower = np.min(positive_data)
    else:
        lim_lower = default_min
    lim_upper = np.max(data)
    if not np.isfinite(lim_upper) or lim_upper <= 0:
        lim_upper = default_max
    return lim_lower, lim_upper

def plot_inverse_predictions_adjusted(y_test_log, y_pred_log, y_test_bc, y_pred_bc, lambda_target, title_prefix):
    """
    绘制逆变换后的预测散点图与残差分布图：
      • 对数逆变换使用 np.expm1；
      • Box-Cox 逆变换使用 inv_boxcox；
    图中包含完美预测曲线 (y=x)，并采用对数坐标显示预测图，
    残差图利用直方图，并根据数据的1%和99%百分位自动调整X轴范围，
    若计算结果无效（NaN/Inf或上下界相等），则自动采用默认范围。
    """
    # 逆变换：对数用 np.expm1，Box-Cox 用 inv_boxcox
    y_test_log_inv = np.expm1(y_test_log)
    y_pred_log_inv = np.expm1(y_pred_log)
    y_test_bc_inv  = inv_boxcox(y_test_bc, lambda_target)
    y_pred_bc_inv  = inv_boxcox(y_pred_bc, lambda_target)
    
    plt.figure(figsize=(12, 12))
    
    # Box-Cox逆变换预测图
    ax1 = plt.subplot(2, 2, 1)
    ax1.scatter(y_test_bc_inv, y_pred_bc_inv, color='blue', alpha=0.6, label='预测值')
    lim_lower, lim_upper = _get_positive_limits(y_test_bc_inv, y_pred_bc_inv)
    ax1.plot([lim_lower, lim_upper], [lim_lower, lim_upper], 'r--', label='完美预测')
    ax1.set_xlabel('实际渗透率 (Box-Cox逆变换, mD)')
    ax1.set_ylabel('预测渗透率 (Box-Cox逆变换, mD)')
    ax1.set_title(title_prefix + ' - Box-Cox逆变换')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(lim_lower, lim_upper)
    ax1.set_ylim(lim_lower, lim_upper)
    ax1.set_aspect('equal', adjustable='box')
    ax1.legend()
    
    # Box-Cox逆变换残差图
    ax2 = plt.subplot(2, 2, 2)
    residuals_bc = y_test_bc_inv - y_pred_bc_inv
    sns.histplot(residuals_bc, bins=50, kde=True, color='orange', edgecolor='black', ax=ax2)
    ax2.set_xlabel('Box-Cox逆变换残差 (mD)')
    ax2.set_title('Box-Cox逆变换残差分布')
    # 计算1%和99%百分位数
    rmin, rmax = np.percentile(residuals_bc, [1, 99])
    if not (np.isfinite(rmin) and np.isfinite(rmax)) or (rmin == rmax):
        rmin, rmax = ax2.get_xlim()
    ax2.set_xlim(rmin, rmax)
    
    # 对数逆变换预测图
    ax3 = plt.subplot(2, 2, 3)
    ax3.scatter(y_test_log_inv, y_pred_log_inv, color='green', alpha=0.6, label='预测值')
    lim_lower_log, lim_upper_log = _get_positive_limits(y_test_log_inv, y_pred_log_inv)
    ax3.plot([lim_lower_log, lim_upper_log], [lim_lower_log, lim_upper_log], 'r--', label='完美预测')
    ax3.set_xlabel('实际渗透率 (对数逆变换, mD)')
    ax3.set_ylabel('预测渗透率 (对数逆变换, mD)')
    ax3.set_title(title_prefix + ' - 对数逆变换')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlim(lim_lower_log, lim_upper_log)
    ax3.set_ylim(lim_lower_log, lim_upper_log)
    ax3.set_aspect('equal', adjustable='box')
    ax3.legend()
    
    # 对数逆变换残差图
    ax4 = plt.subplot(2, 2, 4)
    residuals_log = y_test_log_inv - y_pred_log_inv
    sns.histplot(residuals_log, bins=50, kde=True, color='purple', edgecolor='black', ax=ax4)
    ax4.set_xlabel('对数逆变换残差 (mD)')
    ax4.set_title('对数逆变换残差分布')
    rmin_log, rmax_log = np.percentile(residuals_log, [1, 99])
    if not (np.isfinite(rmin_log) and np.isfinite(rmax_log)) or (rmin_log == rmax_log):
        rmin_log, rmax_log = ax4.get_xlim()
    ax4.set_xlim(rmin_log, rmax_log)
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance_filtered(model, features, title):
    """
    绘制模型特征重要性图，所有特征均显示（包括孔隙度）。
    要求模型具有 feature_importances_ 属性（例如基于树的模型）。
    """
    importance = model.feature_importances_
    filtered_features   = list(features)
    filtered_importance = list(importance)
    
    # 按重要性降序排列
    indices = np.argsort(filtered_importance)[::-1]
    fig_width = max(10, len(filtered_features) * 0.6)
    plt.figure(figsize=(fig_width, 6))
    plt.bar(range(len(filtered_features)), np.array(filtered_importance)[indices],
            align="center", color='skyblue')
    plt.xticks(range(len(filtered_features)), np.array(filtered_features)[indices], rotation=45, ha="right")
    plt.title(title)
    plt.xlabel("特征")
    plt.ylabel("重要性分数")
    plt.tight_layout()
    plt.show()

def main():
    # ----------------------- 数据加载与预处理 -----------------------
    file_path = r"C:/Users/aaalo/Desktop/渗透率预测数据体.xlsx"
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        print("生成模拟数据以作演示...")
        np.random.seed(42)
        data_sim = pd.DataFrame({
            '磁化率（10-3SI）': np.random.exponential(scale=1, size=1000),
            'P波(m/s)': np.random.exponential(scale=1000, size=1000),
            'S波（m/s）': np.random.normal(loc=100, scale=10, size=1000),
            '电阻率（Ω·m）': np.random.exponential(scale=50, size=1000),
            '极化率（%）': np.random.exponential(scale=10, size=1000),
            '渗透率（mD）': np.random.exponential(scale=5, size=1000),
            '孔隙度': np.random.uniform(0.05, 0.3, size=1000),
            '地区': np.random.choice(['A区', 'B区'], size=1000),
            '岩性': np.random.choice(['1', '2', '3', '4'], size=1000),
            '状态': np.random.choice(['正常', '异常'], size=1000)
        })
        df = data_sim.copy()
    else:
        print("数据加载成功！")
        df = pd.read_excel(file_path)
    
    print("数据预览:")
    print(df.head())
    
    # ----------------------- 模型加载 -----------------------
    log_model_filename    = "trained_model_log_optimized.pkl"
    boxcox_model_filename = "trained_model_bc_optimized.pkl"
    if not os.path.exists(log_model_filename):
        raise FileNotFoundError(f"找不到模型文件：{log_model_filename}")
    if not os.path.exists(boxcox_model_filename):
        raise FileNotFoundError(f"找不到模型文件：{boxcox_model_filename}")
    
    with open(log_model_filename, "rb") as f:
        model_log = pickle.load(f)
    with open(boxcox_model_filename, "rb") as f:
        model_boxcox = pickle.load(f)
    print("模型加载成功！")
    
    # 模型期望特征（包含孔隙度）
    if hasattr(model_log, "feature_names_in_"):
        model_feature_names_log = model_log.feature_names_in_
    else:
        model_feature_names_log = np.array([
            "磁化率（10-3SI）", "S波（m/s）", "电阻率（Ω·m）",
            "极化率（%）", "孔隙度", "地区", "岩性", "状态"
        ])
    if hasattr(model_boxcox, "feature_names_in_"):
        model_feature_names_boxcox = model_boxcox.feature_names_in_
    else:
        model_feature_names_boxcox = model_feature_names_log
    
    # ----------------------- 获取真实目标变量 -----------------------
    y_true = df["渗透率（mD）"].values
    df_without_target = df.drop(columns=["渗透率（mD）"])
    
    # ----------------------- 模拟孔隙度随机误差 -----------------------
    # 复制原始数据，生成两个扰动数据集，分别对应 ±10% 与 ±50% 的随机扰动
    df_porosity_10 = df_without_target.copy()
    df_porosity_50 = df_without_target.copy()
    np.random.seed(42)
    error_ratio_10 = np.random.uniform(low=-0.10, high=0.10, size=df_porosity_10.shape[0])
    error_ratio_50 = np.random.uniform(low=-0.50, high=0.50, size=df_porosity_50.shape[0])
    df_porosity_10["孔隙度"] = (df_porosity_10["孔隙度"] * (1 + error_ratio_10)).clip(lower=1e-5)
    df_porosity_50["孔隙度"] = (df_porosity_50["孔隙度"] * (1 + error_ratio_50)).clip(lower=1e-5)
    
    # ----------------------- 数据预处理 -----------------------
    # 对数变换预处理：原始数据、±10%扰动数据、±50%扰动数据
    processed_data_log_orig = preprocess_new_data(df_without_target,
                                                  model_feature_names_log,
                                                  transform_type="log",
                                                  remove_porosity=False)
    processed_data_log_10   = preprocess_new_data(df_porosity_10,
                                                  model_feature_names_log,
                                                  transform_type="log",
                                                  remove_porosity=False)
    processed_data_log_50   = preprocess_new_data(df_porosity_50,
                                                  model_feature_names_log,
                                                  transform_type="log",
                                                  remove_porosity=False)
    
    # Box-Cox 预处理
    processed_data_bc_orig = preprocess_new_data(df_without_target,
                                                 model_feature_names_boxcox,
                                                 transform_type="boxcox",
                                                 remove_porosity=False)
    processed_data_bc_10   = preprocess_new_data(df_porosity_10,
                                                 model_feature_names_boxcox,
                                                 transform_type="boxcox",
                                                 remove_porosity=False)
    processed_data_bc_50   = preprocess_new_data(df_porosity_50,
                                                 model_feature_names_boxcox,
                                                 transform_type="boxcox",
                                                 remove_porosity=False)
    
    # ----------------------- 模型预测 -----------------------
    # 对数变换模型的预测
    y_pred_log_orig = model_log.predict(processed_data_log_orig)
    y_pred_log_10   = model_log.predict(processed_data_log_10)
    y_pred_log_50   = model_log.predict(processed_data_log_50)
    
    # Box-Cox 模型的预测
    y_pred_bc_orig = model_boxcox.predict(processed_data_bc_orig)
    y_pred_bc_10   = model_boxcox.predict(processed_data_bc_10)
    y_pred_bc_50   = model_boxcox.predict(processed_data_bc_50)
    
    print("\n--- 预测结果（变换后空间） ---")
    print("对数变换 - 原始数据预测（前5个）：", y_pred_log_orig[:5])
    print("对数变换 - 孔隙度±10%预测（前5个）：", y_pred_log_10[:5])
    print("对数变换 - 孔隙度±50%预测（前5个）：", y_pred_log_50[:5])
    
    # ----------------------- 添加 SHAP 分析图 -----------------------
    # 使用对数变换模型，对 ±10% 与 ±50% 的预处理数据进行 SHAP 解释，绘制条形柱状图
    explainer_log = shap.TreeExplainer(model_log)
    
    # 孔隙度±10%数据的 SHAP 分析
    shap_values_log_10 = explainer_log.shap_values(processed_data_log_10)
    print("生成 SHAP 条形柱状图：对数变换模型（孔隙度±10%）")
    shap.summary_plot(shap_values_log_10, processed_data_log_10,
                      feature_names=model_feature_names_log, plot_type="bar", show=False)
    plt.title("SHAP 特征重要性（对数变换模型，孔隙度±10%）")
    plt.tight_layout()
    plt.show()
    
    # 孔隙度±50%数据的 SHAP 分析
    shap_values_log_50 = explainer_log.shap_values(processed_data_log_50)
    print("生成 SHAP 条形柱状图：对数变换模型（孔隙度±50%）")
    shap.summary_plot(shap_values_log_50, processed_data_log_50,
                      feature_names=model_feature_names_log, plot_type="bar", show=False)
    plt.title("SHAP 特征重要性（对数变换模型，孔隙度±50%）")
    plt.tight_layout()
    plt.show()
    
    # ----------------------- 真实值转换 -----------------------
    y_test_log = np.log1p(y_true)
    lam = 0.2  # Box-Cox 的 λ 与预处理时保持一致
    y_test_bc  = np.array([boxcox(np.clip(val, 1e-5, None), lmbda=lam) for val in y_true])
    
    # ----------------------- 绘制渗透率预测图 -----------------------
    # 分别对 ±10% 与 ±50% 的误差数据生成预测图
    title_prefix_10 = "渗透率预测 - 孔隙度±10%"
    plot_inverse_predictions_adjusted(y_test_log, y_pred_log_10, y_test_bc, y_pred_bc_10, lam, title_prefix=title_prefix_10)
    
    title_prefix_50 = "渗透率预测 - 孔隙度±50%"
    plot_inverse_predictions_adjusted(y_test_log, y_pred_log_50, y_test_bc, y_pred_bc_50, lam, title_prefix=title_prefix_50)
    
    # ----------------------- 绘制模型特征重要性图 -----------------------
    # 注意：这里分别输出对数变换模型和 Box-Cox 模型的特征重要性图，均包含“孔隙度”
    try:
        plot_feature_importance_filtered(model_log, model_feature_names_log, title="对数变换模型特征重要性（包含孔隙度）")
    except AttributeError:
        print("对数变换模型不支持 feature_importances_ 属性，无法绘制特征重要性图。")
    
    try:
        plot_feature_importance_filtered(model_boxcox, model_feature_names_boxcox, title="Box-Cox变换模型特征重要性（包含孔隙度）")
    except AttributeError:
        print("Box-Cox变换模型不支持 feature_importances_ 属性，无法绘制特征重要性图。")
    
    # ----------------------- 保存预测结果到 Excel -----------------------
    y_pred_log_orig_inv = np.expm1(y_pred_log_orig)
    y_pred_log_10_inv   = np.expm1(y_pred_log_10)
    y_pred_log_50_inv   = np.expm1(y_pred_log_50)
    
    results_df = pd.DataFrame({
        '实际渗透率(mD)': y_true,
        '对数变换-原始预测(mD)': y_pred_log_orig_inv,
        '对数变换-孔隙度±10%预测(mD)': y_pred_log_10_inv,
        '对数变换-孔隙度±50%预测(mD)': y_pred_log_50_inv
    })
    results_df.to_excel("预测结果_孔隙度不确定性.xlsx", index=False)
    print("\n预测结果已保存到预测结果_孔隙度不确定性.xlsx")

if __name__ == '__main__':
    main()
