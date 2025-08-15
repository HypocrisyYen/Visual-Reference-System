# reference_resolver.py
import openai
import numpy as np
import cv2
from PIL import Image
import io
import base64

class ReferenceResolver:
    def __init__(self, api_key):
        self.openai_client = openai.OpenAI(api_key=api_key)
    
    def extract_references(self, text):
        """從文本中提取更複雜的指示性引用"""
        prompt = f"""
        從以下文本中提取任何指示性引用。識別以下類型的參照：
        1. 簡單指示詞（"這個"、"那個"）
        2. 位置參照（"左邊的"、"右上角的"）
        3. 特性參照（"紅色的"、"圓形的"）
        4. 組合參照（"左邊那個紅色的杯子"）
        
        文本: "{text}"
        
        輸出格式:
        引用類型: [簡單/位置/特性/組合]
        引用文本: [引用部分]
        位置信息: [任何位置詞，如左/右/上/下，若無則填"無"]
        特性信息: [任何描述物體特性的詞，如顏色、形狀，若無則填"無"]
        對象類型: [引用指向的對象類型，如杯子、書，若無法確定則填"物體"]
        """
        
        try:
            # 嘗試使用主要模型
            response = self.openai_client.responses.create(
                model="gpt-4.1-nano-2025-04-14",
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt}
                    ]
                }]
            )
            
            return response.output_text
        except Exception as e:
            print(f"使用 gpt-4.1-nano-2025-04-14 模型進行參照提取時出錯: {e}")
            
            try:
                # 嘗試使用替代模型
                response = self.openai_client.responses.create(
                    model="gpt-4.1-mini",
                    input=[{
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt}
                        ]
                    }]
                )
                
                return response.output_text
            except Exception as e2:
                print(f"使用替代模型進行參照提取時也失敗: {e2}")
                
                # 如果所有嘗試都失敗，提供一個基本的回應
                return "引用類型: 無引用\n引用文本: 無引用\n位置信息: 無\n特性信息: 無\n對象類型: 無"
    
    def resolve_reference(self, scene_data, reference_text):
        """解析參照並確定其指向的視覺區域"""
        # 提取每個區段的小圖片
        images_data = []
        for i, segment in enumerate(scene_data["segments"]):
            # 將OpenCV圖像(BGR)轉換為PIL圖像(RGB)
            img = cv2.cvtColor(segment["image"], cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            
            # 轉換為字節
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # 添加到圖像數據列表
            position = segment["position"]
            pos_text = f"位置({position[0]},{position[1]})"
            images_data.append({
                "image": img_byte_arr,
                "position": pos_text
            })
        
        # 創建提示信息
        content = [
            {"type": "input_text", "text": f"請根據以下提示確定指示性引用'{reference_text}'最可能指向哪個位置的物體。僅返回最可能的位置編號，格式為'位置(行,列)'。"}
        ]
        
        # 添加所有圖像
        for img_data in images_data:
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{base64.b64encode(img_data['image']).decode('utf-8')}"
            })
            content.append({"type": "input_text", "text": img_data["position"]})
        
        try:
            # 嘗試使用主要模型
            response = self.openai_client.responses.create(
                model="gpt-4.1-nano-2025-04-14",
                input=[{
                    "role": "user",
                    "content": content
                }]
            )
            
            # 解析回應
            position_text = response.output_text.strip()
            
        except Exception as e:
            print(f"使用 gpt-4.1-nano-2025-04-14 進行參照解析時出錯: {e}")
            
            try:
                # 嘗試使用替代模型
                response = self.openai_client.responses.create(
                    model="gpt-4.1-mini",
                    input=[{
                        "role": "user",
                        "content": content
                    }]
                )
                
                # 解析回應
                position_text = response.output_text.strip()
                
            except Exception as e2:
                print(f"使用替代模型進行參照解析時也失敗: {e2}")
                # 在失敗的情況下，直接返回第一個段落
                if len(scene_data["segments"]) > 0:
                    return scene_data["segments"][0]
                return None
        
        # 查找匹配的段
        for segment in scene_data["segments"]:
            seg_pos = segment["position"]
            seg_pos_text = f"位置({seg_pos[0]},{seg_pos[1]})"
            if seg_pos_text in position_text:
                return segment
        
        # 如果沒有找到匹配，但有網格，返回中間的段落
        if len(scene_data["segments"]) > 0:
            mid_index = len(scene_data["segments"]) // 2
            return scene_data["segments"][mid_index]
        
        return None
    
    def generate_response(self, text, scene_data, additional_context=None, is_final_summary=False):
        """生成對用戶查詢的回應"""
        # 如果是最終摘要，使用不同的處理邏輯
        if is_final_summary:
            return self.generate_session_summary(text, scene_data, additional_context)
        
        # 標準處理流程
        # 提取參照
        try:
            ref_info = self.extract_references(text)
        except Exception as e:
            print(f"提取參照時出錯: {e}")
            # 回退到簡單的錯誤處理
            return {"type": "text", "content": f"處理您的請求時遇到問題。錯誤: {str(e)}"}
        
        # 如果沒有參照，就直接使用GPT回答
        if "無引用" in ref_info:
            prompt = f"用戶說: {text}\n請提供適當的回應。"
            try:
                response = self.openai_client.responses.create(
                    model="gpt-4.1-nano-2025-04-14",
                    input=[{
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt}
                        ]
                    }]
                )
                return {"type": "text", "content": response.output_text}
            except Exception as e:
                print(f"使用 gpt-4.1-nano-2025-04-14 模型回應文本時出錯: {e}")
                try:
                    # 嘗試使用替代模型
                    response = self.openai_client.responses.create(
                        model="gpt-4.1-mini",
                        input=[{
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt}
                            ]
                        }]
                    )
                    return {"type": "text", "content": response.output_text}
                except Exception as e2:
                    return {"type": "text", "content": f"無法處理您的請求。請稍後再試。錯誤: {str(e2)}"}
        
        # 解析參照行獲取對象描述
        lines = ref_info.strip().split('\n')
        ref_text = ""
        for line in lines:
            if line.startswith("引用文本:"):
                ref_text = line.split("引用文本:")[1].strip()
                if ref_text == "無引用" or ref_text == "無":
                    return {"type": "text", "content": "我不確定你指的是什麼。"}
        
        # 如果找不到引用文本，嘗試其他格式
        if not ref_text:
            for line in lines:
                if "引用:" in line:
                    ref_text = line.split("引用:")[1].strip()
                    if ref_text == "無引用" or ref_text == "無":
                        return {"type": "text", "content": "我不確定你指的是什麼。"}
        
        # 解析參照到視覺區域
        try:
            referenced_segment = self.resolve_reference(scene_data, ref_text)
        except Exception as e:
            print(f"解析參照時出錯: {e}")
            return {"type": "text", "content": f"解析您指向的物體時遇到問題。錯誤: {str(e)}"}
        
        if referenced_segment is None:
            return {"type": "text", "content": "我不確定你指的是哪個物體。"}
        
        # 使用GPT-4.1分析該區域
        img = cv2.cvtColor(referenced_segment["image"], cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        
        # 轉換為字節
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # 創建請求
        prompt = f"用戶問: \"{text}\"\n\n分析這個圖像並回答用戶的問題。如果用戶是在詢問圖像中的物體，請描述該物體。請給出簡潔、信息豐富的回答。"
        
        try:
            # 嘗試使用主要模型
            response = self.openai_client.responses.create(
                model="gpt-4.1-nano-2025-04-14",
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64.b64encode(img_byte_arr).decode('utf-8')}"
                        }
                    ]
                }]
            )
            
            # 返回結果和參考區域
            return {
                "type": "reference_response",
                "content": response.output_text,
                "segment": referenced_segment
            }
        except Exception as e:
            print(f"使用 gpt-4.1-nano-2025-04-14 模型處理圖像時出錯: {e}")
            
            try:
                # 嘗試使用替代模型
                response = self.openai_client.responses.create(
                    model="gpt-4.1-mini",
                    input=[{
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64.b64encode(img_byte_arr).decode('utf-8')}"
                            }
                        ]
                    }]
                )
                
                return {
                    "type": "reference_response",
                    "content": response.output_text,
                    "segment": referenced_segment
                }
            except Exception as e2:
                return {
                    "type": "reference_response",
                    "content": f"分析圖像時遇到問題。錯誤: {str(e2)}",
                    "segment": referenced_segment
                }
    
    def generate_session_summary(self, full_transcription, final_scene, context=None):
        """生成整個錄製會話的綜合分析"""
        # 準備完整的場景圖像
        img = cv2.cvtColor(final_scene["frame"], cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        
        # 轉換為字節
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # 準備分段小圖像
        segment_images = []
        for segment in final_scene["segments"]:
            seg_img = cv2.cvtColor(segment["image"], cv2.COLOR_BGR2RGB)
            seg_pil = Image.fromarray(seg_img)
            
            seg_byte_arr = io.BytesIO()
            seg_pil.save(seg_byte_arr, format='JPEG')
            
            position = segment["position"]
            segment_images.append({
                "image": seg_byte_arr.getvalue(),
                "position": f"位置({position[0]},{position[1]})"
            })
        
        # 準備提示
        duration_text = ""
        scene_count_text = ""
        if context:
            if "duration" in context:
                duration_text = f"錄製持續了約 {context['duration']:.1f} 秒。"
            if "scene_count" in context:
                scene_count_text = f"共捕獲了 {context['scene_count']} 個場景。"
        
        prompt = f"""
        分析用戶在錄製過程中的所有語音內容，並結合場景圖像，生成一個全面的摘要報告。
        
        用戶在錄製過程中說了: "{full_transcription}"
        
        {duration_text}
        {scene_count_text}
        
        請提供以下內容:
        1. 場景的整體描述
        2. 識別用戶感興趣的主要物體或區域
        3. 根據用戶的語音內容，分析用戶可能想知道的信息
        4. 提供對用戶問題或關注點的全面回答
        
        結果應該是綜合性的，充分利用視覺和語音信息。
        """
        
        # 創建輸入內容
        content = [
            {"type": "input_text", "text": prompt},
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{base64.b64encode(img_byte_arr).decode('utf-8')}"
            }
        ]
        
        # 添加分段圖像
        for i, seg_data in enumerate(segment_images):
            if i < 3:  # 僅添加前幾個分段，以避免超出 token 限制
                content.append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64.b64encode(seg_data['image']).decode('utf-8')}"
                })
                content.append({"type": "input_text", "text": f"區域: {seg_data['position']}"})
        
        try:
            # 首先嘗試使用 gpt-4.1-nano-2025-04-14 模型
            response = self.openai_client.responses.create(
                model="gpt-4.1-nano-2025-04-14",
                input=[{
                    "role": "user",
                    "content": content
                }]
            )
            
            # 返回結果
            return {
                "type": "summary",
                "content": response.output_text
            }
            
        except Exception as e:
            print(f"使用 gpt-4.1-nano-2025-04-14 時發生錯誤: {e}")
            print("嘗試使用替代模型...")
            
            try:
                # 嘗試使用 gpt-4.1-mini 模型作為備選
                response = self.openai_client.responses.create(
                    model="gpt-4.1-mini",
                    input=[{
                        "role": "user",
                        "content": content
                    }]
                )
                
                return {
                    "type": "summary",
                    "content": response.output_text
                }
            except Exception as e2:
                print(f"使用替代模型時也失敗: {e2}")
                
                # 如果所有嘗試都失敗，回傳基本文本
                return {
                    "type": "summary",
                    "content": f"無法生成摘要分析。請檢查您的 OpenAI API 金鑰是否有效，或者聯繫系統管理員。\n錯誤詳情: {str(e2)}"
                }