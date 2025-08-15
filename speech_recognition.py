# speech_recognition.py
from PyQt6.QtCore import QObject, pyqtSignal, QThread, QMutex, QMutexLocker
from RealtimeSTT import AudioToTextRecorder
import pyaudio
import threading
import numpy as np
import zhconv

class SpeechRecognizerThread(QThread):
    """語音識別線程"""
    
    # 定義信號
    text_received = pyqtSignal(str)         # 接收到文本信號
    vad_started = pyqtSignal()              # 語音活動開始信號
    vad_stopped = pyqtSignal()              # 語音活動結束信號
    error_occurred = pyqtSignal(str)        # 錯誤信號
    initialization_complete = pyqtSignal(bool)  # 初始化完成信號
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 狀態變數
        self.input_device_index = 0
        self.recorder = None
        self.is_active = False
        self.need_init = True
        self.language = "zh"
        
        # 互斥鎖
        self.mutex = QMutex()
    
    def run(self):
        """線程主循環"""
        print("語音識別線程已啟動")
        
        while not self.isInterruptionRequested():
            try:
                # 檢查是否需要初始化
                if self.need_init:
                    self._initialize_recorder()
                    self.need_init = False
                
                # 檢查是否活躍
                with QMutexLocker(self.mutex):
                    is_active = self.is_active
                    recorder = self.recorder
                
                # 如果不活躍，跳過此次循環
                if not is_active:
                    self.msleep(10)
                    continue
                
                # 獲取識別結果
                if recorder:
                    try:
                        input_text = zhconv.convert(recorder.text(), 'zh-cn')
                        
                        if input_text and input_text.strip():
                            print(f"[用戶說] {input_text}")
                            self.text_received.emit(input_text.strip())
                    except Exception as e:
                        print(f"語音識別錯誤: {str(e)}")
                        self.error_occurred.emit(f"語音識別錯誤: {str(e)}")
                
                self.msleep(10)
                
            except Exception as e:
                print(f"語音線程循環出錯: {str(e)}")
                self.error_occurred.emit(f"語音線程錯誤: {str(e)}")
                self.msleep(500)
    
    def _initialize_recorder(self):
        """初始化錄音器"""
        success = False
        
        try:
            print(f"初始化錄音器 (設備: {self.input_device_index})...")
            
            # 關閉現有錄音器
            if self.recorder:
                try:
                    self.recorder.shutdown()
                except Exception as e:
                    print(f"關閉舊錄音器失敗: {str(e)}")
                self.recorder = None
            
            # 創建新錄音器
            self.recorder = AudioToTextRecorder(
                spinner=False,
                model='large-v2',
                language=self.language,
                input_device_index=self.input_device_index,
                silero_sensitivity=0.5,
                silero_use_onnx=True,
                silero_deactivity_detection=True,
                webrtc_sensitivity=2,
                post_speech_silence_duration=0.3,
                no_log_file=True,
                on_vad_start=self._on_vad_start,
                on_vad_stop=self._on_vad_stop
            )
            
            print("錄音器初始化成功")
            success = True
            
        except Exception as e:
            print(f"初始化錄音器失敗: {str(e)}")
            self.error_occurred.emit(f"初始化語音錄音失敗: {str(e)}")
        
        self.initialization_complete.emit(success)
    
    def _on_vad_start(self):
        """檢測到語音活動開始"""
        print("檢測到語音活動開始")
        self.vad_started.emit()

    def _on_vad_stop(self):
        """檢測到語音活動結束"""
        print("檢測到語音活動結束")
        self.vad_stopped.emit()
    
    def cleanup(self):
        """清理資源"""
        with QMutexLocker(self.mutex):
            if self.recorder:
                try:
                    self.recorder.shutdown()
                except Exception as e:
                    print(f"關閉錄音器失敗: {str(e)}")
                finally:
                    self.recorder = None


class SpeechRecognizer(QObject):
    """新的語音識別器 - 使用RealtimeSTT和PyQt6"""
    
    # 定義信號
    text_received = pyqtSignal(str)
    vad_started = pyqtSignal()
    vad_stopped = pyqtSignal()
    error_occurred = pyqtSignal(str)
    initialization_complete = pyqtSignal(bool)
    
    def __init__(self, api_key=None, sample_rate=16000, input_device_index=0, parent=None):
        """初始化語音識別器
        
        注意: api_key參數保留但不使用，為了保持與舊版接口兼容
        """
        super().__init__(parent)
        
        # 創建工作線程
        self.thread = SpeechRecognizerThread(self)
        self.thread.input_device_index = input_device_index
        
        # 連接信號
        self.thread.text_received.connect(self.text_received)
        self.thread.vad_started.connect(self.vad_started)
        self.thread.vad_stopped.connect(self.vad_stopped)
        self.thread.error_occurred.connect(self.error_occurred)
        self.thread.initialization_complete.connect(self.initialization_complete)
        
        # 創建緩存隊列來模擬舊API行為
        self.text_queue = []
        self.text_queue_lock = threading.Lock()
        
        # 啟動線程
        self.thread.start()
    
    def set_language(self, language):
        """設置識別語言"""
        with QMutexLocker(self.thread.mutex):
            self.thread.language = language
            self.thread.need_init = True
    
    def set_vad_callbacks(self, on_vad_start=None, on_vad_stop=None):
        """設置VAD回調函數（保留兼容性）"""
        # 斷開之前的所有連接
        try:
            self.vad_started.disconnect()
            self.vad_stopped.disconnect()
        except:
            pass
            
        # 重新連接
        if on_vad_start:
            self.vad_started.connect(on_vad_start)
        if on_vad_stop:
            self.vad_stopped.connect(on_vad_stop)
            
        # 連接文本緩存處理
        try:
            self.text_received.disconnect(self._cache_text)
        except:
            pass
        self.text_received.connect(self._cache_text)
    
    def _cache_text(self, text):
        """緩存接收到的文本以便與舊API兼容"""
        with self.text_queue_lock:
            self.text_queue.append(text)
            # 只保留最新的5個文本
            if len(self.text_queue) > 5:
                self.text_queue.pop(0)
    
    def start_recording(self):
        """開始錄音"""
        with QMutexLocker(self.thread.mutex):
            if self.thread.is_active:
                return True
            
            if not self.thread.recorder:
                self.thread.need_init = True
            
            self.thread.is_active = True
        
        print("錄音已開始")
        return True
    
    def stop_recording(self):
        """停止錄音"""
        with QMutexLocker(self.thread.mutex):
            self.thread.is_active = False
        
        print("錄音已停止")
        return True
    
    def get_latest_transcription(self):
        """獲取最新的轉錄文本"""
        with self.text_queue_lock:
            if self.text_queue:
                # 取得最新文本並從隊列中移除，避免重複處理
                return self.text_queue.pop(0)
        return ""
    
    def record_audio(self, duration=5, silence_threshold=0.02, silence_duration=0.8):
        """模擬舊API的record_audio方法，但現在返回模擬音頻數據"""
        # 簡單的模擬音頻數據，不影響功能
        dummy_audio = np.zeros((int(16000 * 1), 1), dtype=np.int16)
        return dummy_audio if self.thread.is_active else None
    
    def transcribe_audio(self, audio_data=None):
        """模擬舊API，返回最新一條文本"""
        with self.text_queue_lock:
            if self.text_queue:
                return self.text_queue[-1]
        return ""
    
    def switch_device(self, device_index):
        """切換輸入設備"""
        with QMutexLocker(self.thread.mutex):
            if device_index == self.thread.input_device_index and self.thread.recorder:
                return True
            
            self.thread.input_device_index = device_index
            self.thread.need_init = True
        
        print(f"請求切換到設備 {device_index}")
        return True
    
    @staticmethod
    def get_input_devices():
        """獲取所有可用的音訊輸入設備"""
        devices = []
        p = pyaudio.PyAudio()
        
        try:
            info = p.get_host_api_info_by_index(0)
            num_devices = info.get('deviceCount')
            
            for i in range(num_devices):
                device_info = p.get_device_info_by_index(i)
                if device_info.get('maxInputChannels') > 0:
                    devices.append((i, device_info.get('name')))
        finally:
            p.terminate()
            
        return devices
    
    def cleanup(self):
        """清理資源"""
        print("正在清理語音識別資源...")
        
        with QMutexLocker(self.thread.mutex):
            self.thread.is_active = False
        
        self.thread.cleanup()
        self.thread.requestInterruption()
        
        if not self.thread.wait(2000):
            print("語音線程未能在超時時間內結束，強制終止")
            self.thread.terminate()
            self.thread.wait()
        
        print("語音線程已結束")