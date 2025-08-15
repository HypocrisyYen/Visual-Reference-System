import cv2
import mediapipe as mp
import numpy as np

class GestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def detect_pointing(self, frame):
        """檢測指向手勢並返回指向的坐標"""
        # 轉換為RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 處理圖像
        results = self.hands.process(rgb_frame)
        
        # 沒有檢測到手
        if not results.multi_hand_landmarks:
            return None, frame
        
        # 獲取手部關鍵點
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # 繪製手部關鍵點
        annotated_frame = frame.copy()
        self.mp_drawing.draw_landmarks(
            annotated_frame, 
            hand_landmarks, 
            self.mp_hands.HAND_CONNECTIONS
        )
        
        # 獲取食指尖端和指關節的位置
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        
        # 計算畫面中的坐標
        h, w, _ = frame.shape
        tip_x, tip_y = int(index_tip.x * w), int(index_tip.y * h)
        pip_x, pip_y = int(index_pip.x * w), int(index_pip.y * h)
        
        # 計算方向向量
        dx = tip_x - pip_x
        dy = tip_y - pip_y
        
        # 延長線段以找到指向的點
        magnitude = np.sqrt(dx*dx + dy*dy)
        if magnitude < 1e-6:  # 避免除以零
            return None, annotated_frame
        
        # 標準化並延長
        scale = 50.0  # 延長倍數
        dx = dx / magnitude * scale
        dy = dy / magnitude * scale
        
        # 計算指向的點
        pointing_x = int(tip_x + dx)
        pointing_y = int(tip_y + dy)
        
        # 繪製指向的線
        cv2.line(annotated_frame, (tip_x, tip_y), (pointing_x, pointing_y), (0, 255, 0), 2)
        cv2.circle(annotated_frame, (pointing_x, pointing_y), 5, (0, 0, 255), -1)
        
        return (pointing_x, pointing_y), annotated_frame
    
    def find_pointed_segment(self, pointing_coords, segments):
        """找出被指向的區域段"""
        if pointing_coords is None:
            return None
        
        pointing_x, pointing_y = pointing_coords
        
        for segment in segments:
            x1, y1, x2, y2 = segment["coordinates"]
            if (x1 <= pointing_x <= x2) and (y1 <= pointing_y <= y2):
                return segment
        
        return None