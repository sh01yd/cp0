from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import torch
import numpy as np
from typing import List, Tuple, Optional
from faiss_store_y import FAISSVectorStore  # 引入FAISS向量存储

local_model_path = "./cross-encoder-model"
collection_name = "document_embeddings"

def retrieve_relevant_chunks(user_query: str, vector_store: FAISSVectorStore, 
                           top_k: int = 15, final_k: int = 5, 
                           embedding_model: Optional[SentenceTransformer] = None, 
                           cross_encoder1: Optional[CrossEncoder] = None) -> List:
    """使用FAISS检索相关文档片段的函数，优化了性能和错误处理，支持图片检索"""
    if embedding_model is None or cross_encoder1 is None:
        raise ValueError("embedding_model和cross_encoder1参数不能为None")
    
    model = embedding_model
    cross_encoder = cross_encoder1
    
    try:
        # 优化：生成查询向量，使用更高效的编码方式
        with torch.no_grad():  # 减少内存使用
            query_embedding = model.encode(
                user_query, 
                prompt_name="query",
                convert_to_tensor=False,  # 直接返回numpy数组
                normalize_embeddings=True  # 标准化嵌入向量
            ).tolist()

        # 从FAISS向量存储检索（返回为列表[{content, id, distance, score, ...}]）
        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=min(top_k, 50)  # 限制最大检索数量
        )

        # 解析返回结构，保留原始结果
        if not results:
            return []  # 无匹配结果

        # 检查是否有图片结果
        image_results = []
        text_results = []
        
        for item in results:
            content = item.get("content", "")
            if not content:
                continue
                
            # 检查是否是图片描述（以image_开头）
            if content.startswith('image_'):
                # 这是图片描述，保留原始结构
                image_results.append(item)
            else:
                # 这是普通文本
                text_results.append(content)
        
        print(f"检索到 {len(text_results)} 个文本片段，{len(image_results)} 个图片")
        
        # 如果没有足够的结果，直接返回
        if len(text_results) + len(image_results) <= final_k:
            # 处理图片结果
            formatted_images = []
            for img in image_results:
                content = img.get("content", "")
                parts = content.split(':', 1)
                if len(parts) >= 2:
                    image_id = parts[0].strip()
                    document_content = parts[1].strip()
                    formatted_images.append({
                        "type": 1,  # 图片描述
                        "document": document_content,
                        "source": image_id  # 图片ID，后续会被处理
                    })
            
            # 处理文本结果
            formatted_texts = [{"type": 0, "document": text, "source": ""} for text in text_results]
            
            # 合并结果
            return formatted_texts + formatted_images
        
        # 使用交叉编码器重新排序文本结果
        if text_results:
            pairs = [(user_query, chunk) for chunk in text_results]
            
            with torch.no_grad():  # 减少内存使用
                scores = cross_encoder.predict(pairs, batch_size=32)  # 批量预测
            
            # 优化：使用numpy进行更快的排序
            scores_array = np.array(scores)
            # 计算要保留的文本结果数量
            text_to_keep = min(len(text_results), max(1, final_k - len(image_results)))
            top_indices = np.argsort(scores_array)[::-1][:text_to_keep]  # 获取top_k索引
            
            # 格式化文本结果
            formatted_texts = [{"type": 0, "document": text_results[i], "source": ""} for i in top_indices]
        else:
            formatted_texts = []
        
        # 处理图片结果
        formatted_images = []
        for img in image_results[:min(len(image_results), final_k - len(formatted_texts))]:
            content = img.get("content", "")
            parts = content.split(':', 1)
            if len(parts) >= 2:
                image_id = parts[0].strip()
                document_content = parts[1].strip()
                formatted_images.append({
                    "type": 1,  # 图片描述
                    "document": document_content,
                    "source": image_id  # 图片ID，后续会被处理
                })
        
        # 合并结果
        return formatted_texts + formatted_images
        
    except Exception as e:
        print(f"检索过程中发生错误: {e}")
        return []

def batch_retrieve_relevant_chunks(queries: List[str], vector_store: FAISSVectorStore,
                                  top_k: int = 15, final_k: int = 5,
                                  embedding_model: Optional[SentenceTransformer] = None,
                                  cross_encoder1: Optional[CrossEncoder] = None) -> List[List[str]]:
    """批量检索多个查询的相关文档片段"""
    if embedding_model is None or cross_encoder1 is None:
        raise ValueError("embedding_model和cross_encoder1参数不能为None")
    
    results = []
    for query in queries:
        chunks = retrieve_relevant_chunks(
            query, vector_store, top_k, final_k, 
            embedding_model, cross_encoder1
        )
        results.append(chunks)
    
    return results

# 使用示例
if __name__ == "__main__":
    # 初始化模型
    model = SentenceTransformer(
        "./Qwen3-Embedding-0.6B",
        tokenizer_kwargs={"padding_side": "left"},
        trust_remote_code=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    cross_encoder = CrossEncoder(local_model_path)
    
    user_query = input("请输入你的问题: ")
    vector_store = FAISSVectorStore(index_path="./faiss_index", collection_name=collection_name)
    relevant_chunks = retrieve_relevant_chunks(
        user_query=user_query,
        vector_store=vector_store,
        embedding_model=model,
        cross_encoder1=cross_encoder
    )
    
    print(f"检索到 {len(relevant_chunks)} 个相关文档片段:")
    for i, chunk in enumerate(relevant_chunks, 1):
        print(f"\n片段 {i}:\n{chunk[:200]}...")
