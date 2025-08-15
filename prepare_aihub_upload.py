# prepare_aihub_upload.py - 準備上傳到 Qualcomm AI Hub 的腳本

import os
import json
import shutil
from pathlib import Path

def create_aihub_package():
    """創建符合 AI Hub 要求的模型包"""
    
    # 創建上傳目錄
    upload_dir = Path("aihub_package")
    upload_dir.mkdir(exist_ok=True)
    
    # 1. 模型描述文件
    model_card = {
        "name": "Visual Reference Understanding System",
        "description": "A multimodal AI system that combines real-time speech recognition with visual understanding for contextual interaction with screen content",
        "use_cases": [
            "Educational assistance with screen content explanation",
            "Remote technical support with visual guidance", 
            "Accessibility support for screen reading",
            "Interactive presentation systems"
        ],
        "technical_details": {
            "model_type": "Multimodal (Vision + Speech)",
            "input_modalities": ["Image", "Audio"],
            "output_modalities": ["Text"],
            "frameworks": ["PyTorch", "Transformers"],
            "base_models": ["CLIP-ViT-Base", "RealtimeSTT"],
            "optimization": "INT8 quantization for edge deployment"
        },
        "performance": {
            "target_devices": ["Snapdragon 8 Gen 2", "Snapdragon X Elite"],
            "inference_time": "<100ms per frame",
            "memory_usage": "<500MB",
            "power_efficiency": "Optimized for mobile devices"
        },
        "requirements": {
            "python_version": "3.8+",
            "dependencies": [
                "torch>=1.13.0",
                "torchvision>=0.14.0", 
                "transformers>=4.21.0",
                "opencv-python>=4.6.0",
                "numpy>=1.21.0"
            ]
        },
        "license": "MIT",
        "tags": ["computer-vision", "speech-recognition", "multimodal", "edge-ai", "real-time"]
    }
    
    # 保存模型卡片
    with open(upload_dir / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2, ensure_ascii=False)
    
    # 2. 創建示例代碼
    example_code = '''
# 示例：在 Qualcomm 設備上使用視覺參照理解系統

import torch
import cv2
from qualcomm_deploy import QualcommVisionEncoder, QualcommSpeechRecognizer

# 初始化系統組件
vision_encoder = QualcommVisionEncoder("qualcomm_vision_model.pt")
speech_recognizer = QualcommSpeechRecognizer()

# 捕獲攝像頭畫面
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if ret:
    # 分析視覺場景
    segments = vision_encoder.segment_image_optimized(frame)
    
    # 啟動語音識別
    speech_recognizer.start_recording()
    
    # 獲取語音轉錄
    transcription = speech_recognizer.get_transcription()
    
    if transcription:
        print(f"用戶說: {transcription}")
        # 在此處添加參照解析邏輯
    
    speech_recognizer.stop_recording()

cap.release()
'''
    
    with open(upload_dir / "example.py", "w", encoding="utf-8") as f:
        f.write(example_code)
    
    # 3. 創建 requirements.txt
    requirements = [
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "transformers>=4.21.0", 
        "opencv-python>=4.6.0",
        "numpy>=1.21.0",
        "flask>=2.0.0",
        "PyQt6>=6.4.0"
    ]
    
    with open(upload_dir / "requirements.txt", "w") as f:
        f.write("\n".join(requirements))
    
    # 4. 創建部署說明
    deployment_guide = '''# Qualcomm AI Hub 部署指南

## 準備步驟

1. **模型量化**
   ```bash
   python qualcomm_deploy.py
   ```

2. **測試本地推理**
   ```bash
   python example.py
   ```

3. **驗證設備相容性**
   - 確保目標設備支援 Snapdragon NPU
   - 檢查記憶體需求 (<500MB)

## 上傳到 AI Hub

1. 登入 Qualcomm AI Hub
2. 選擇 "Upload Model"
3. 上傳 `qualcomm_vision_model.pt`
4. 填寫模型資訊（參考 model_card.json）
5. 設定目標設備為 Snapdragon 系列

## 性能優化

- 使用 INT8 量化減少模型大小
- 將圖像解析度降至 224x224
- 啟用 NPU 加速（如果可用）

## 故障排除

- 如果推理速度慢，檢查是否正確使用 NPU
- 記憶體不足時，可進一步量化至 INT4
'''
    
    with open(upload_dir / "DEPLOYMENT.md", "w", encoding="utf-8") as f:
        f.write(deployment_guide)
    
    # 5. 複製核心文件
    core_files = [
        "app.py",
        "vision_encoder.py", 
        "speech_recognition.py",
        "reference_resolver.py",
        "qualcomm_deploy.py"
    ]
    
    for file in core_files:
        if Path(file).exists():
            shutil.copy(file, upload_dir / file)
    
    print("🎉 AI Hub 上傳包已準備完成!")
    print(f"📁 上傳目錄: {upload_dir.absolute()}")
    print("\n📋 包含文件:")
    for file in upload_dir.iterdir():
        print(f"  - {file.name}")
    
    print(f"\n🚀 下一步:")
    print("1. 運行 'python qualcomm_deploy.py' 生成量化模型")
    print("2. 前往 https://aihub.qualcomm.com")
    print("3. 點擊 'Upload Model' 並上傳 aihub_package 資料夾中的文件")

if __name__ == "__main__":
    create_aihub_package()