#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM统一调用库
提供统一的接口来调用不同的大语言模型，方便后续更换和修改
"""

import base64
import os
import time
from typing import Optional, Dict, Any, List
from openai import OpenAI
import json
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# 加载环境变量
load_dotenv("Agent.env")


class LLMConfig:
    """LLM配置类"""
    
    # ModelScope API配置
    MODELSCOPE_BASE_URL = 'https://api-inference.modelscope.cn/v1'
    
    # 模型名称配置
    VL_MODEL = 'Qwen/Qwen3-VL-8B-Instruct' # 当前可用的VL模型
    TEXT_MODEL = 'Qwen/Qwen3-235B-A22B-Instruct-2507'
    
    @classmethod
    def get_api_key(cls) -> str:
        """获取API密钥，优先使用环境变量"""
        api_key = os.getenv('MODELSCOPE_API_KEY')
        if not api_key:
            raise ValueError("请在环境变量中设置 MODELSCOPE_API_KEY")
        return api_key


class BaseModelClient:
    """基础模型客户端"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or LLMConfig.get_api_key()
        self.base_url = base_url or LLMConfig.MODELSCOPE_BASE_URL
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
    
    def _make_request(self, model: str, messages: List[Dict], **kwargs) -> str:
        """发起API请求的通用方法，包含重试机制"""
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False,
                    **kwargs
                )
                return response.choices[0].message.content
                
            except Exception as e:
                error_str = str(e)
                
                # 如果是429错误（请求频率限制），等待后重试
                if "429" in error_str or "Request limit exceeded" in error_str:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # 指数退避
                        print(f"遇到请求频率限制，等待 {delay} 秒后重试... (尝试 {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"API请求失败，已达到最大重试次数: {e}")
                        raise e
                else:
                    # 其他错误直接抛出
                    print(f"API请求失败: {e}")
                    raise e


class VisionLanguageModel(BaseModelClient):
    """视觉语言模型客户端 - 用于图像识别和描述"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.model_name = LLMConfig.VL_MODEL
    
    def image_to_base64(self, image_data: bytes) -> str:
        """将图片数据转换为base64编码"""
        return base64.b64encode(image_data).decode('utf-8')
    
    def describe_image(self, image_data: bytes, prompt: str = None) -> str:
        """
        使用视觉语言模型描述图片内容
        
        Args:
            image_data: 图片的二进制数据
            prompt: 自定义提示词，如果为None则使用默认提示词
        
        Returns:
            图片描述文本
        """
        if prompt is None:
            prompt = '请详细描述这幅图片的内容，包括图片中的文字、图形、布局等所有可见元素。'
        
        try:
            # 使用Pillow检查并修正图片尺寸，避免发送过小图片导致400错误
            try:
                img = Image.open(BytesIO(image_data))
                width, height = img.size
                min_size = 16  # 模型要求最小尺寸>10，这里取16以留余量
                if width < min_size or height < min_size:
                    scale = max(min_size / max(width, 1), min_size / max(height, 1))
                    new_w = max(int(width * scale), min_size)
                    new_h = max(int(height * scale), min_size)
                    img = img.resize((new_w, new_h), Image.LANCZOS)
                # 始终以PNG编码，简化MIME处理
                buf = BytesIO()
                img.save(buf, format='PNG')
                safe_image_bytes = buf.getvalue()
                mime = 'image/png'
            except Exception:
                # 如果Pillow解析失败则退回原始数据
                safe_image_bytes = image_data
                mime = 'image/png'

            # 将图片转换为base64
            base64_image = self.image_to_base64(safe_image_bytes)
            image_url = f"data:{mime};base64,{base64_image}"
            
            messages = [{
                'role': 'user',
                'content': [{
                    'type': 'text',
                    'text': prompt,
                }, {
                    'type': 'image_url',
                    'image_url': {
                        'url': image_url,
                    },
                }],
            }]
            
            return self._make_request(self.model_name, messages)
            
        except Exception as e:
            print(f"图像识别失败: {e}")
            return "图片识别失败"
    
    def describe_image_with_context(self, image_data: bytes, context_before: str = "", 
                                  context_after: str = "", source_file: str = "") -> str:
        """
        结合上下文描述图片内容
        
        Args:
            image_data: 图片的二进制数据
            context_before: 图片前的文本内容
            context_after: 图片后的文本内容
            source_file: 源文件名
        
        Returns:
            结合上下文的图片描述
        """
        prompt = f"""
请根据以下信息，详细描述这幅图片的内容：

上文内容：{context_before}

下文内容：{context_after}

源文件：{source_file}

请描述图片中的所有可见元素，包括文字、图形、布局、界面元素等，并分析图片在文档中的作用和意义。
"""
        
        return self.describe_image(image_data, prompt)


class TextLanguageModel(BaseModelClient):
    """文本语言模型客户端 - 用于文本生成和处理"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.model_name = LLMConfig.TEXT_MODEL
    
    def enhance_description(self, image_description: str, context_before: str = "", 
                          context_after: str = "", source_file: str = "") -> str:
        """
        使用文本模型完善图片描述
        
        Args:
            image_description: 原始图片描述
            context_before: 图片前的文本内容
            context_after: 图片后的文本内容
            source_file: 源文件名
        
        Returns:
            完善后的图片描述
        """
        try:
            prompt = f"""
请根据以下信息，生成一段完整的图片内容描述：

图片识别结果：{image_description}

上文内容：{context_before}

下文内容：{context_after}

源文件：{source_file}

请结合上下文，分析这张图片的应用场景、主题和作用，并生成一段完整、准确的描述。描述应该包括：
1. 图片的具体内容
2. 图片在文档中的作用和意义
3. 与上下文的关联性
4. 应用场景和主题

请用一段话总结，不超过200字。
"""
            
            messages = [
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant that analyzes images in context.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
            
            return self._make_request(self.model_name, messages)
            
        except Exception as e:
            print(f"描述完善失败: {e}")
            return image_description  # 返回原始描述作为备选
    
    def generate_text(self, prompt: str, system_prompt: str = None) -> str:
        """
        生成文本内容
        
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
        
        Returns:
            生成的文本内容
        """
        messages = []
        
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })
        
        messages.append({
            'role': 'user',
            'content': prompt
        })
        
        return self._make_request(self.model_name, messages)


# ImageProcessor类已移除，图片处理功能已在Image_Process.py中实现


class LLMFactory:
    """LLM工厂类 - 提供统一的模型创建接口"""
    
    @staticmethod
    def create_vision_model(api_key: Optional[str] = None) -> VisionLanguageModel:
        """创建视觉语言模型实例"""
        return VisionLanguageModel(api_key)
    
    @staticmethod
    def create_text_model(api_key: Optional[str] = None) -> TextLanguageModel:
        """创建文本语言模型实例"""
        return TextLanguageModel(api_key)
    
    # create_image_processor方法已移除，图片处理功能已在Image_Process.py中实现
    
    @staticmethod
    def test_connection(api_key: Optional[str] = None) -> Dict[str, bool]:
        """测试模型连接状态"""
        results = {}
        
        try:
            vision_model = VisionLanguageModel(api_key)
            # 创建一个简单的测试图片（32x32像素的PNG），避免尺寸过小导致400错误
            img = Image.new('RGB', (32, 32), color=(255, 255, 255))
            buf = BytesIO()
            img.save(buf, format='PNG')
            test_image_data = buf.getvalue()
            response = vision_model.describe_image(test_image_data, "请简单描述这幅图片。")
            # 根据响应内容判断是否成功
            results['vision_model'] = bool(response) and ('失败' not in str(response))
        except Exception as e:
            print(f"视觉模型连接失败: {e}")
            results['vision_model'] = False
        
        try:
            text_model = TextLanguageModel(api_key)
            text_model.generate_text("请简单回答：你好")
            results['text_model'] = True
        except Exception as e:
            print(f"文本模型连接失败: {e}")
            results['text_model'] = False
        
        return results


# get_default_image_processor函数已移除，图片处理功能已在Image_Process.py中实现


def get_default_vision_model() -> VisionLanguageModel:
    """获取默认的视觉模型实例"""
    return LLMFactory.create_vision_model()


def get_default_text_model() -> TextLanguageModel:
    """获取默认的文本模型实例"""
    return LLMFactory.create_text_model()


if __name__ == "__main__":
    # 测试代码
    print("测试LLM库连接...")
    
    # 测试连接
    connection_results = LLMFactory.test_connection()
    print("连接测试结果:")
    for model, status in connection_results.items():
        print(f"  {model}: {'✅ 成功' if status else '❌ 失败'}")
    
    # 图像处理功能测试已移除，相关功能在Image_Process.py中实现
