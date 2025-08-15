# app.py
from flask import Flask, render_template, request, jsonify
import os
import base64
import numpy as np
import cv2
import threading
import time
import io
import sys
from PyQt6.QtCore import QCoreApplication

# 首先初始化Qt事件循環
def setup_qt_app():
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication(sys.argv)
    return app

# 在單獨的線程中運行Qt事件循環
class QtAppThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.app = setup_qt_app()
        self.daemon = True
        
    def run(self):
        self.app.exec()

# 啟動Qt事件循環
qt_thread = QtAppThread()
qt_thread.start()

from vision_encoder import VisionEncoder
from speech_recognition import SpeechRecognizer
from reference_resolver import ReferenceResolver

app = Flask(__name__)

# 初始化模塊
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key-here")  # 從環境變數讀取，或使用默認值

vision_encoder = VisionEncoder()
speech_recognizer = SpeechRecognizer(api_key=OPENAI_API_KEY)
speech_recognizer.set_language("zh")
reference_resolver = ReferenceResolver(api_key=OPENAI_API_KEY)

# 全局變量
current_scene = None
recording_active = False
recording_thread = None
recording_lock = threading.Lock()
# 防止重複語音處理
last_processed_text = ""
last_response_content = ""
duplicate_count = 0
MIN_TEXT_LENGTH = 5  # 最小有效文本長度
MAX_DUPLICATES = 2   # 最多允許的重複次數
last_process_time = 0
MIN_PROCESS_INTERVAL = 3  # 最小處理間隔(秒)

# 語言檢測
FORBIDDEN_CHARACTERS = set("뉴스이덕영")

# 存儲錄製會話的數據
session_data = {
    "scenes": [],       # 所有捕獲的場景
    "transcriptions": [],  # 所有轉錄的語音
    "timestamps": [],   # 每個場景和轉錄的時間戳
    "temp_responses": []  # 暫時性分析回應
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start_recording', methods=['POST'])
def start_recording():
    global recording_active, recording_thread, session_data
    
    # 如果已經在錄製中，返回錯誤
    if recording_active:
        return jsonify({"error": "錄製已經在進行中"}), 400
    
    # 重置會話數據
    session_data = {
        "scenes": [],
        "transcriptions": [],
        "timestamps": [],
        "temp_responses": []
    }
    last_processed_text = ""
    duplicate_count = 0
    last_process_time = 0
    # 啟動錄製
    recording_active = True
    if hasattr(speech_recognizer, 'cleanup'):
        speech_recognizer.cleanup()

    speech_recognizer.set_vad_callbacks(
        on_vad_start=lambda: print("語音活動開始"),
        on_vad_stop=lambda: print("語音活動結束")
    )
    # 開始語音錄製線程
    recording_thread = threading.Thread(target=continuous_speech_recording)
    recording_thread.daemon = True
    recording_thread.start()
    
    return jsonify({"message": "開始錄製會話"})

def is_valid_text(text):
    """檢查文本是否有效"""
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return False
        
    # 檢查是否包含韓文字符
    if any(char in FORBIDDEN_CHARACTERS for char in text):
        print(f"檢測到韓文字符，忽略: {text}")
        return False
        
    return True

def continuous_speech_recording():
    """持續錄製和轉錄語音的後台線程"""
    global recording_active, session_data, last_processed_text, duplicate_count, last_response_content, last_process_time
    speech_recognizer.start_recording()
    try:
        while recording_active:
            current_time = time.time()
            
            # 控制處理頻率 - 至少間隔1.5秒
            if current_time - last_process_time < MIN_PROCESS_INTERVAL:
                time.sleep(0.1)  # 短暫睡眠以減少CPU使用
                continue
                
            # 直接從 RealtimeSTT 獲取轉錄文本（無需模擬音頻處理）
            with recording_lock:
                transcription = speech_recognizer.get_latest_transcription()
                
                # 檢查文本是否有效
                if not is_valid_text(transcription):
                    continue
                
                # 檢查重複
                if transcription.strip() == last_processed_text:
                    duplicate_count += 1
                    if duplicate_count > MAX_DUPLICATES:
                        print(f"忽略重複文本 ({duplicate_count}): {transcription}")
                        continue
                else:
                    # 新文本，重置計數器
                    last_processed_text = transcription.strip()
                    duplicate_count = 0
                
                print(f"\n[用戶] {transcription}")
                
                # 更新處理時間戳
                last_process_time = time.time()
                
                # 添加到會話數據
                session_data["transcriptions"].append(transcription)
                session_data["timestamps"].append(current_time)
                
                # 如果有場景和轉錄，嘗試進行即時分析
                if len(session_data["scenes"]) > 0:
                    latest_scene = session_data["scenes"][-1]
                    try:
                        # 生成實時回應
                        response = reference_resolver.generate_response(transcription, latest_scene)
                        if response:
                            # 檢查回應是否與上次相同
                            if response["content"] == last_response_content:
                                print("忽略重複回應")
                                continue
                                
                            last_response_content = response["content"]
                            
                            session_data["temp_responses"].append({
                                "content": response["content"],
                                "type": response["type"],
                                "timestamp": current_time,
                                "segment": response.get("segment", None)
                            })
                    except Exception as e:
                        print(f"實時分析錯誤: {e}")
            
            # 適當休眠以減少CPU使用
            time.sleep(0.2)
    finally:
        # 確保錄音停止
        speech_recognizer.stop_recording()

@app.route('/api/capture_and_process', methods=['POST'])
def capture_and_process():
    global recording_active, session_data, current_scene
    
    # 如果未處於錄製狀態，返回錯誤
    if not recording_active:
        return jsonify({"error": "尚未開始錄製"}), 400
    
    # 捕獲當前場景
    current_scene = vision_encoder.describe_scene(force_refresh=True)
    
    if current_scene is None:
        return jsonify({"error": "無法捕獲場景"}), 400
    
    # 將場景添加到會話數據
    with recording_lock:
        session_data["scenes"].append(current_scene)
        session_data["timestamps"].append(time.time())
    
    # 將幀編碼為BASE64以便在前端顯示
    _, buffer = cv2.imencode('.jpg', current_scene["frame"])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # 準備分段預覽
    segments_preview = []
    for i, segment in enumerate(current_scene["segments"]):
        _, buffer = cv2.imencode('.jpg', segment["image"])
        segment_base64 = base64.b64encode(buffer).decode('utf-8')
        segments_preview.append({
            "id": i,
            "position": segment["position"],
            "image": segment_base64
        })
    
    # 獲取最新轉錄（如果有）
    latest_transcription = None
    if session_data["transcriptions"]:
        latest_transcription = session_data["transcriptions"][-1]
    
    # 獲取最新臨時響應（如果有）
    latest_temp_response = None
    latest_referenced_segment = None
    if session_data["temp_responses"]:
        latest_resp = session_data["temp_responses"][-1]
        latest_temp_response = latest_resp["content"]
        if latest_resp["segment"]:
            latest_referenced_segment = {"position": latest_resp["segment"]["position"]}
    
    return jsonify({
        "frame": frame_base64,
        "segments": segments_preview,
        "transcription": latest_transcription,
        "tempResponse": latest_temp_response,
        "referenced_segment": latest_referenced_segment
    })

@app.route('/api/stop_recording', methods=['POST'])
def stop_recording():
    global recording_active, recording_thread, session_data
    
    # 如果未在錄製，返回錯誤
    if not recording_active:
        return jsonify({"error": "尚未開始錄製"}), 400
    
    # 停止錄製
    recording_active = False
    speech_recognizer.stop_recording()
    # 等待錄製線程結束
    if recording_thread and recording_thread.is_alive():
        recording_thread.join(timeout=2)
    
    # 如果沒有收集到任何數據，返回錯誤
    if not session_data["scenes"]:
        return jsonify({"error": "未捕獲任何場景"}), 400
    
    # 進行整體分析
    try:
        final_summary = generate_session_summary(session_data)
        return jsonify({
            "summary": final_summary,
            "message": "錄製已停止並完成分析"
        })
    except Exception as e:
        return jsonify({"error": f"生成分析時出錯: {str(e)}"}), 500

def generate_session_summary(session_data):
    """生成整個會話的摘要分析"""
    # 如果沒有場景或轉錄，返回簡單消息
    if not session_data["scenes"] or not session_data["transcriptions"]:
        return "會話中沒有足夠的數據進行分析。"
    
    # 將所有轉錄合併為單個文本
    all_transcriptions = " ".join(session_data["transcriptions"])
    
    # 使用最後捕獲的場景進行整體分析
    latest_scene = session_data["scenes"][-1]
    
    # 向參照解析器提供更豐富的上下文
    context = {
        "full_transcription": all_transcriptions,
        "scene_count": len(session_data["scenes"]),
        "duration": session_data["timestamps"][-1] - session_data["timestamps"][0] if len(session_data["timestamps"]) > 1 else 0
    }
    
    # 生成最終響應
    response = reference_resolver.generate_response(
        all_transcriptions, 
        latest_scene, 
        additional_context=context,
        is_final_summary=True
    )
    
    return response["content"]

@app.route('/api/video_stream')
def video_stream():
    """即時視頻串流"""
    def generate_frames():
        while True:
            frame = vision_encoder.capture_frame()
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    
    from flask import Response
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/process_text', methods=['POST'])
def process_text():
    global current_scene, last_processed_text, last_response_content
    
    # 獲取文本輸入
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "文本不能為空"}), 400
    
    if text.strip() == last_processed_text:
        return jsonify({"error": "請勿重複提交相同文本"}), 400
    # 如果當前沒有場景，使用最近錄製的場景（如果有）
    scene_to_use = current_scene
    if scene_to_use is None and session_data["scenes"]:
        scene_to_use = session_data["scenes"][-1]
    
    if scene_to_use is None:
        return jsonify({"error": "請先捕獲場景或開始錄製"}), 400
    
    # 處理參照並生成回應
    response = reference_resolver.generate_response(text, scene_to_use)

    if response:
        last_response_content = response["content"]
    
    # 準備響應
    result = {
        "text": text,
        "response": response["content"],
        "type": response["type"]
    }
    
    # 如果是參照響應，添加參照區域的信息
    if response["type"] == "reference_response" and "segment" in response:
        segment = response["segment"]
        _, buffer = cv2.imencode('.jpg', segment["image"])
        segment_base64 = base64.b64encode(buffer).decode('utf-8')
        
        result["referenced_segment"] = {
            "position": segment["position"],
            "coordinates": segment["coordinates"],
            "image": segment_base64
        }
    
    return jsonify(result)

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        vision_encoder.release()