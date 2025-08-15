# vision_encoder.py
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import time
import os

class VisionEncoder:
    def __init__(self):
        # 加載CLIP模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用設備: {self.device}")
        try:
            print("正在加載 CLIP 模型...")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("CLIP 模型加載成功")
        except Exception as e:
            print(f"加載 CLIP 模型時出錯: {e}")
            raise
        
        # 初始化攝像頭
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("無法打開攝像頭")
            print("攝像頭初始化成功")
        except Exception as e:
            print(f"初始化攝像頭時出錯: {e}")
            raise
        
        self.scene_cache = None
        self.cache_timestamp = None
        self.cache_duration = 5  # 緩存有效期（秒）
        
    def capture_frame(self):
        """捕獲當前畫面"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def segment_image(self, frame, grid_size=(3, 3)):
        """將畫面分割為網格"""
        height, width = frame.shape[:2]
        segments = []
        
        cell_h = height // grid_size[0]
        cell_w = width // grid_size[1]
        
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                y1 = i * cell_h
                y2 = (i + 1) * cell_h
                x1 = j * cell_w
                x2 = (j + 1) * cell_w
                
                segment = frame[y1:y2, x1:x2]
                segments.append({
                    "image": segment,
                    "position": (i, j),
                    "coordinates": (x1, y1, x2, y2)
                })
        
        return segments
    
    def encode_segments(self, segments):
        """為每個區域生成視覺特徵和描述"""
        results = []
        
        for segment in segments:
            # 處理圖像
            inputs = self.processor(images=segment["image"], return_tensors="pt").to(self.device)
            
            # 獲取特徵
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            
            # 將特徵轉換為numpy數組
            features_np = features.cpu().numpy()
            
            # 將區域添加到結果中
            results.append({
                "features": features_np,
                "position": segment["position"],
                "coordinates": segment["coordinates"],
                "image": segment["image"]
            })
        
        return results
    
    def describe_scene(self, force_refresh=False):
        """捕獲當前場景並生成區域描述"""
        current_time = time.time()
        if (not force_refresh and 
            self.scene_cache is not None and 
            self.cache_timestamp is not None and
            current_time - self.cache_timestamp < self.cache_duration):
            return self.scene_cache
        
        frame = self.capture_frame()
        if frame is None:
            return None
        
        # 分割圖像
        segments = self.segment_image(frame)
        
        # 編碼區域
        encoded_segments = self.encode_segments(segments)
        
        # 更新緩存
        self.scene_cache = {
            "frame": frame,
            "segments": encoded_segments
        }
        self.cache_timestamp = current_time
        
        return self.scene_cache
        
    
    def release(self):
        """釋放資源"""
        if hasattr(self, 'cap'):
            self.cap.release()