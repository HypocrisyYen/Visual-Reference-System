# 視覺參照理解系統 (Visual Reference Understanding System)

一個結合即時語音轉文字和視覺分析的智能對話系統，能夠理解用戶對畫面中特定區域的語音指令。

## ✨ 功能特色

- **即時視頻串流**: 30FPS 流暢的攝像頭畫面顯示，無卡頓體驗
- **語音轉文字**: 使用 RealtimeSTT 進行即時中文語音識別
- **視覺理解**: 基於 CLIP 模型的畫面分析和區域分割
- **智能對話**: 整合 OpenAI API 進行上下文理解和回應
- **參照解析**: 能夠理解「這個」、「那個」等指示詞並定位到畫面區域

## 📋 系統要求

- Python 3.8+
- 攝像頭設備
- 麥克風設備
- OpenAI API Key

## 🛠 安裝步驟

1. **克隆項目**
```bash
git clone https://github.com/HypocrisyYen/Visual-Reference-System.git
cd Visual-Reference-System
```

2. **創建虛擬環境**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **安裝依賴**
```bash
pip install -r requirements.txt
```

4. **設置環境變數**
```bash
# Windows
set OPENAI_API_KEY=your-openai-api-key-here

# macOS/Linux
export OPENAI_API_KEY=your-openai-api-key-here
```

## 🚀 使用方法

1. **啟動應用**
```bash
python app.py
```

2. **開啟瀏覽器**
   - 訪問 `http://localhost:5000`

3. **開始使用**
   - 點擊「開始」按鈕開始錄製
   - 對著麥克風說話，系統會即時轉錄並分析
   - 可以使用「這個」、「那個」等詞彙指向畫面中的物體
   - 點擊「停止」結束會話並獲得完整分析

## 📁 項目結構

```
visual-reference-system/
├── app.py                    # Flask 主應用
├── vision_encoder.py         # 視覺編碼器 (CLIP)
├── speech_recognition.py     # 語音識別模組
├── reference_resolver.py     # 參照解析器
├── gesture_recognizer.py     # 手勢識別器
├── requirements.txt          # 依賴列表
├── templates/
│   └── index.html           # 前端界面
└── static/
    └── sessions/            # 會話數據存儲
```

## 🔧 主要組件

### 視覺編碼器 (VisionEncoder)
- 使用 OpenAI CLIP 模型進行視覺理解
- 將畫面分割為 3x3 網格進行區域分析
- 提供即時視頻串流功能

### 語音識別器 (SpeechRecognizer)
- 基於 RealtimeSTT 的即時語音轉文字
- 支持中文語音識別
- 內建語音活動檢測 (VAD)

### 參照解析器 (ReferenceResolver)
- 整合 OpenAI GPT 模型
- 理解指示性語言並匹配視覺區域
- 提供上下文相關的智能回應

## 🎯 使用場景

- **教育輔助**: 講解屏幕內容時的智能問答
- **視頻會議**: 對屏幕分享內容進行實時討論
- **技術支援**: 遠程協助時的視覺指導
- **無障礙輔助**: 為視覺障礙者提供畫面描述

## ⚠️ 注意事項

- 首次運行時會下載 CLIP 模型，需要一些時間
- 確保攝像頭和麥克風權限已開啟
- OpenAI API 使用會產生費用，請注意用量控制
- 建議在較好的網絡環境下使用以確保模型運行流暢

## 🔍 故障排除

**攝像頭無法開啟**
- 檢查攝像頭是否被其他應用程式占用
- 確認攝像頭權限設置

**語音識別不工作**
- 檢查麥克風權限設置
- 確認音訊設備是否正常工作
- 檢查 PyQt6 相關依賴是否正確安裝

**API 錯誤**
- 確認 OpenAI API Key 是否正確設置
- 檢查網絡連接和 API 使用額度
