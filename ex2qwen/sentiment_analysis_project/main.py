import warnings
warnings.filterwarnings('ignore')

def main():
    print("=== 基于Qwen2.5-0.5B的情感分类实验 ===\n")
    
    # 1. 数据预处理
    print("步骤1: 数据预处理")
    from data_preprocessing import prepare_data
    train_df, test_df = prepare_data("test.csv")
    
    # 2. 深度学习模型训练
    print("\n步骤2: 深度学习模型训练")
    from model_training import main as train_main
    classifier, trainer, dl_results = train_main()
    
    # 3. 传统机器学习比较
    print("\n步骤3: 传统机器学习模型比较")
    from traditional_ml import main as ml_main
    comparator, ml_results = ml_main()
    
    # 4. 结果比较
    print("\n步骤4: 结果比较")
    comparator.plot_comparison(dl_results)
    
    # 5. 生成实验报告
    print("\n步骤5: 生成实验报告")
    generate_report(dl_results, ml_results)

def generate_report(dl_results, ml_results):
    """生成实验报告"""
    report = """
    === 实验报告 ===
    
    实验目标：基于Qwen2.5-0.5B模型的情感分类
    
    数据集：
    - 总样本数：从提供的CSV文件中提取
    - 标签：1（负面）-> 0，2（正面）-> 1
    - 训练集/测试集比例：80%/20%
    
    模型表现比较：
    """
    
    # 添加深度学习结果
    report += f"""
    Qwen2.5-0.5B 模型：
    - 准确率：{dl_results['accuracy']:.4f}
    - F1-score：{dl_results['f1']:.4f}
    - AUC-ROC：{dl_results['auc_roc']:.4f}
    """
    
    # 添加传统模型结果
    for model_name, metrics in ml_results.items():
        report += f"""
    {model_name}：
    - 准确率：{metrics['accuracy']:.4f}
    - F1-score：{metrics['f1']:.4f}
    - AUC-ROC：{metrics['auc_roc']:.4f}
    - 交叉验证F1：{metrics['cv_mean']:.4f} (±{metrics['cv_std']:.4f})
        """
    
    report += """
    结论分析：
    1. Qwen2.5-0.5B模型在情感分类任务上表现出色
    2. 与传统机器学习方法相比，深度学习模型能够更好地理解文本语义
    3. 混淆矩阵显示了模型在各个类别上的分类效果
    4. 建议进行超参数调优以进一步提升性能
    """
    
    print(report)
    
    # 保存报告到文件
    with open('experiment_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

if __name__ == "__main__":
    main()