import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class SentimentClassifier:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B", use_ollama=False):
        self.model_name = model_name
        self.use_ollama = use_ollama
        self.tokenizer = None
        self.model = None
        
        if not use_ollama:
            self._load_huggingface_model()
    
    def _load_huggingface_model(self):
        """加载Hugging Face模型"""
        print("正在加载模型和分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # 如果分词器没有pad_token，设置为eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,  # 二分类
            ignore_mismatched_sizes=True
        )
        
        # 调整模型配置以适应分类任务
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def tokenize_function(self, examples):
        """分词函数"""
        return self.tokenizer(
            examples["cleaned_text"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
    
    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        predictions = torch.softmax(torch.tensor(predictions), dim=-1)
        
        # 预测类别
        preds = np.argmax(predictions.numpy(), axis=1)
        
        # 计算各项指标
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        auc_roc = roc_auc_score(labels, predictions[:, 1].numpy())
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'auc_roc': auc_roc
        }
    
    def train(self, train_df, test_df, output_dir="./results"):
        """训练模型"""
        if self.use_ollama:
            return self._train_with_ollama(train_df, test_df)
        
        # 准备数据集
        train_dataset = Dataset.from_pandas(train_df[['cleaned_text', 'label']])
        test_dataset = Dataset.from_pandas(test_df[['cleaned_text', 'label']])
        
        # 分词
        tokenized_train = train_dataset.map(self.tokenize_function, batched=True)
        tokenized_test = test_dataset.map(self.tokenize_function, batched=True)
        
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            seed=42
        )
        
        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        print("开始训练...")
        trainer.train()
        
        # 保存模型
        trainer.save_model(f"{output_dir}/best_model")
        self.tokenizer.save_pretrained(f"{output_dir}/best_model")
        
        return trainer
    
    def _train_with_ollama(self, train_df, test_df):
        """使用Ollama API进行训练（简化版）"""
        import ollama
        
        print("使用Ollama API进行训练（示例）")
        # 这里需要根据Ollama的具体API进行调整
        # 由于Ollama主要用于推理，微调可能需要其他方法
        
        # 示例：使用Ollama进行推理测试
        results = []
        for text in test_df['cleaned_text'].head(5):  # 测试前5个样本
            try:
                response = ollama.chat(
                    model='qwen2.5:0.5b',
                    messages=[{
                        'role': 'user',
                        'content': f"分析以下评论的情感是正面还是负面（只回答正面或负面）: {text}"
                    }]
                )
                results.append(response['message']['content'])
            except Exception as e:
                print(f"Ollama API错误: {e}")
                results.append("未知")
        
        return results

def evaluate_model(trainer, test_dataset, test_df):
    """评估模型性能"""
    print("\n=== 模型评估 ===")
    
    # 获取预测结果
    predictions = trainer.predict(test_dataset)
    pred_probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1)
    pred_labels = np.argmax(pred_probs.numpy(), axis=1)
    true_labels = test_df['label'].values
    
    # 计算指标
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    auc_roc = roc_auc_score(true_labels, pred_probs[:, 1].numpy())
    
    print(f"准确率: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    
    # 绘制混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['负面', '正面'],
                yticklabels=['负面', '正面'])
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'auc_roc': auc_roc,
        'confusion_matrix': cm
    }

# 主训练流程
def main():
    # 加载预处理好的数据
    from data_preprocessing import prepare_data
    
    train_df, test_df = prepare_data("test.csv")
    
    # 初始化分类器
    classifier = SentimentClassifier(use_ollama=False)  # 设置为True如果使用Ollama
    
    # 训练模型
    trainer = classifier.train(train_df, test_df)
    
    # 评估模型
    test_dataset = Dataset.from_pandas(test_df[['cleaned_text', 'label']])
    tokenized_test = test_dataset.map(classifier.tokenize_function, batched=True)
    
    results = evaluate_model(trainer, tokenized_test, test_df)
    
    return classifier, trainer, results

if __name__ == "__main__":
    classifier, trainer, results = main()