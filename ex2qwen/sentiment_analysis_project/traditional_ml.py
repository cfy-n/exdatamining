import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

class TraditionalMLComparator:
    def __init__(self):
        self.models = {
            'SVM': SVC(kernel='linear', probability=True, random_state=42),
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(random_state=42)
        }
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.results = {}
    
    def prepare_features(self, train_texts, test_texts):
        """准备TF-IDF特征"""
        # 拟合训练集并转换训练测试集
        X_train = self.vectorizer.fit_transform(train_texts)
        X_test = self.vectorizer.transform(test_texts)
        return X_train, X_test
    
    def train_and_evaluate(self, train_df, test_df):
        """训练和评估所有传统模型"""
        # 准备特征
        X_train, X_test = self.prepare_features(
            train_df['cleaned_text'].tolist(),
            test_df['cleaned_text'].tolist()
        )
        y_train = train_df['label'].values
        y_test = test_df['label'].values
        
        print("=== 传统机器学习模型比较 ===")
        
        for name, model in self.models.items():
            print(f"\n训练 {name}...")
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = y_pred  # 对于没有概率预测的模型
            
            # 计算指标
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            try:
                auc_roc = roc_auc_score(y_test, y_pred_proba)
            except:
                auc_roc = 0.5  # 如果无法计算AUC
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
            
            self.results[name] = {
                'accuracy': accuracy,
                'f1': f1,
                'auc_roc': auc_roc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model
            }
            
            print(f"{name} - 准确率: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc_roc:.4f}")
            print(f"交叉验证F1: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return self.results
    
    def plot_comparison(self, deep_learning_results=None):
        """绘制模型比较图"""
        models = list(self.results.keys())
        metrics = ['accuracy', 'f1', 'auc_roc']
        
        if deep_learning_results:
            models.append('Qwen2.5-0.5B')
            self.results['Qwen2.5-0.5B'] = deep_learning_results
        
        # 创建比较图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            scores = [self.results[model][metric] for model in models]
            axes[i].bar(models, scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
            axes[i].set_title(f'{metric.upper()} 比较')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            
            # 在柱子上添加数值
            for j, v in enumerate(scores):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    from data_preprocessing import prepare_data
    
    # 加载数据
    train_df, test_df = prepare_data("test.csv")
    
    # 比较传统模型
    comparator = TraditionalMLComparator()
    results = comparator.train_and_evaluate(train_df, test_df)
    
    return comparator, results

if __name__ == "__main__":
    comparator, results = main()