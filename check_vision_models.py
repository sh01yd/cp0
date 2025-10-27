#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查可用的视觉模型
"""

from openai import OpenAI

def check_vision_models():
    """检查可用的视觉模型"""
    
    # API配置
    base_url = 'https://api-inference.modelscope.cn/v1'
    api_key = 'ms-56463e89-3008-4af3-a8d4-a400a154899f'
    
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    try:
        models = client.models.list()
        print("=== 可用的视觉模型 ===")
        
        # 查找视觉相关模型
        vision_keywords = ['vl', 'vision', 'visual', 'multimodal', 'mm']
        vision_models = []
        
        for model in models.data:
            model_id_lower = model.id.lower()
            if any(keyword in model_id_lower for keyword in vision_keywords):
                vision_models.append(model.id)
        
        if vision_models:
            print(f"找到 {len(vision_models)} 个视觉模型:")
            for i, model in enumerate(vision_models, 1):
                print(f"{i:2d}. {model}")
        else:
            print("未找到视觉模型")
            
        # 查找所有Qwen模型
        print("\n=== 所有Qwen模型 ===")
        qwen_models = [model.id for model in models.data if 'qwen' in model.id.lower()]
        for i, model in enumerate(qwen_models, 1):
            print(f"{i:2d}. {model}")
            
    except Exception as e:
        print(f"获取模型列表失败: {e}")

if __name__ == "__main__":
    check_vision_models()