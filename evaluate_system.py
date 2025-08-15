import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def load_evaluation_data(base_dir="evaluation_data"):
    """加載所有評估數據"""
    all_data = []
    
    # 遍歷所有會話目錄
    for session_dir in os.listdir(base_dir):
        session_path = os.path.join(base_dir, session_dir)
        if not os.path.isdir(session_path):
            continue
        
        # 加載數據
        data_path = os.path.join(session_path, "data.json")
        if not os.path.exists(data_path):
            continue
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.append(data)
    
    return all_data

def analyze_reference_resolution(all_data):
    """分析參照解析性能"""
    resolution_results = []
    reference_types = Counter()
    
    for session_data in all_data:
        for interaction in session_data["interactions"]:
            if "reference_resolution" not in interaction:
                continue
            
            for resolution in interaction["reference_resolution"]:
                resolution_results.append(resolution["success"])
                
                # 提取參照類型（假設參照文本包含類型信息）
                ref_text = resolution["reference_text"].lower()
                if "左" in ref_text or "右" in ref_text or "上" in ref_text or "下" in ref_text:
                    reference_types["位置參照"] += 1
                elif "紅" in ref_text or "藍" in ref_text or "綠" in ref_text or "黃" in ref_text:
                    reference_types["特性參照"] += 1
                elif "這" in ref_text or "那" in ref_text:
                    reference_types["簡單指示詞"] += 1
                else:
                    reference_types["其他參照"] += 1
    
    # 計算總體成功率
    success_rate = np.mean(resolution_results) if resolution_results else 0
    
    return {
        "success_rate": success_rate,
        "total_references": len(resolution_results),
        "reference_types": dict(reference_types)
    }

def analyze_user_satisfaction(all_data):
    """分析用戶滿意度"""
    satisfaction_scores = []
    
    for session_data in all_data:
        for interaction in session_data["interactions"]:
            if "user_feedback" not in interaction:
                continue
            
            feedback = interaction["user_feedback"]
            satisfaction_scores.append(feedback["satisfaction_score"])
    
    # 計算平均滿意度
    avg_satisfaction = np.mean(satisfaction_scores) if satisfaction_scores else 0
    
    return {
        "average_satisfaction": avg_satisfaction,
        "total_feedbacks": len(satisfaction_scores),
        "score_distribution": Counter(satisfaction_scores)
    }

def generate_evaluation_report():
    """生成評估報告"""
    # 加載數據
    all_data = load_evaluation_data()
    
    if not all_data:
        print("未找到評估數據")
        return
    
    # 分析參照解析
    resolution_analysis = analyze_reference_resolution(all_data)
    
    # 分析用戶滿意度
    satisfaction_analysis = analyze_user_satisfaction(all_data)
    
    # 打印報告
    print("=== 系統評估報告 ===\n")
    
    print("參照解析性能:")
    print(f"總參照數: {resolution_analysis['total_references']}")
    print(f"成功率: {resolution_analysis['success_rate']:.2%}")
    print("參照類型分布:")
    for ref_type, count in resolution_analysis['reference_types'].items():
        print(f"  - {ref_type}: {count} ({count/resolution_analysis['total_references']:.2%})")
    
    print("\n用戶滿意度:")
    print(f"總反饋數: {satisfaction_analysis['total_feedbacks']}")
    print(f"平均滿意度: {satisfaction_analysis['average_satisfaction']:.2f}/5")
    print("評分分布:")
    for score in range(1, 6):
        count = satisfaction_analysis['score_distribution'].get(score, 0)
        percentage = count / satisfaction_analysis['total_feedbacks'] if satisfaction_analysis['total_feedbacks'] > 0 else 0
        print(f"  - {score}分: {count} ({percentage:.2%})")
    
    # 生成圖表
    plt.figure(figsize=(12, 5))
    
    # 參照類型分布
    plt.subplot(1, 2, 1)
    ref_types = resolution_analysis['reference_types']
    plt.pie(ref_types.values(), labels=ref_types.keys(), autopct='%1.1f%%')
    plt.title('參照類型分布')
    
    # 用戶滿意度分布
    plt.subplot(1, 2, 2)
    scores = [satisfaction_analysis['score_distribution'].get(i, 0) for i in range(1, 6)]
    plt.bar(range(1, 6), scores)
    plt.xlabel('滿意度評分')
    plt.ylabel('數量')
    plt.title('用戶滿意度分布')
    plt.xticks(range(1, 6))
    
    plt.tight_layout()
    plt.savefig('evaluation_report.png')
    plt.close()
    
    print("\n評估報告圖表已保存為 'evaluation_report.png'")

if __name__ == "__main__":
    generate_evaluation_report()