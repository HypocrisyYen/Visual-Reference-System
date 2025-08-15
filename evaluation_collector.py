import json
import os
import time
import cv2
import numpy as np

class EvaluationCollector:
    def __init__(self, output_dir="evaluation_data"):
        self.output_dir = output_dir
        self.session_id = f"session_{int(time.time())}"
        self.session_dir = os.path.join(output_dir, self.session_id)
        
        # 創建目錄
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "scenes"), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "audio"), exist_ok=True)
        
        # 初始化數據記錄
        self.data = {
            "session_id": self.session_id,
            "timestamp": time.time(),
            "interactions": []
        }
    
    def record_interaction(self, interaction_data):
        """記錄一次交互"""
        interaction_id = len(self.data["interactions"])
        
        # 保存場景圖像
        if "frame" in interaction_data:
            frame = interaction_data["frame"]
            frame_path = os.path.join(self.session_dir, "scenes", f"frame_{interaction_id}.jpg")
            cv2.imwrite(frame_path, frame)
            interaction_data["frame"] = frame_path
        
        # 保存音頻
        if "audio" in interaction_data:
            audio = interaction_data["audio"]
            audio_path = os.path.join(self.session_dir, "audio", f"audio_{interaction_id}.wav")
            with open(audio_path, 'wb') as f:
                f.write(audio)
            interaction_data["audio"] = audio_path
        
        # 添加時間戳
        interaction_data["timestamp"] = time.time()
        interaction_data["interaction_id"] = interaction_id
        
        # 添加到數據中
        self.data["interactions"].append(interaction_data)
        
        # 保存更新的數據
        self._save_data()
        
        return interaction_id
    
    def record_reference_resolution(self, interaction_id, reference_text, resolved_segment, success):
        """記錄參照解析結果"""
        if interaction_id >= len(self.data["interactions"]):
            return False
        
        interaction = self.data["interactions"][interaction_id]
        
        # 添加參照解析數據
        if "reference_resolution" not in interaction:
            interaction["reference_resolution"] = []
        
        resolution_data = {
            "reference_text": reference_text,
            "segment_position": resolved_segment["position"] if resolved_segment else None,
            "success": success,
            "timestamp": time.time()
        }
        
        interaction["reference_resolution"].append(resolution_data)
        
        # 保存更新的數據
        self._save_data()
        
        return True
    
    def record_user_feedback(self, interaction_id, satisfaction_score, comments):
        """記錄用戶反饋"""
        if interaction_id >= len(self.data["interactions"]):
            return False
        
        interaction = self.data["interactions"][interaction_id]
        
        # 添加用戶反饋
        interaction["user_feedback"] = {
            "satisfaction_score": satisfaction_score,  # 1-5
            "comments": comments,
            "timestamp": time.time()
        }
        
        # 保存更新的數據
        self._save_data()
        
        return True
    
    def _save_data(self):
        """保存數據到JSON文件"""
        data_path = os.path.join(self.session_dir, "data.json")
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)