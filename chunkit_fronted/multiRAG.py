#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态RAG系统封装类
提供build、insert、retrieve三个核心功能接口
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

# 导入现有模块
from faiss_store_y import FAISSVectorStore
from textsplitters import RecursiveCharacterTextSplitter
from Image_Processor.Image_Process import ImageExtractor
from Image_Processor.update_faiss_with_images import ImageFAISSUpdater
import tiktoken
import PyPDF2
from docx import Document


from Utils.Path import (
    PAPER_DOCS_DIR, CAMPUS_DOCS_DIR, FITNESS_DOCS_DIR, PSYCHOLOGY_DOCS_DIR,
    PAPER_INDEX_DIR, CAMPUS_INDEX_DIR, FITNESS_INDEX_DIR, PSYCHOLOGY_INDEX_DIR,
    ALL_PROCESSED_IMAGES_DIR,CAMPUS_IMAGES_DIR,PAPER_IMAGES_DIR,FITNESS_IMAGES_DIR,  PSYCHOLOGY_IMAGES_DIR,
    CAMPUS_PROCESSED_EXTRACTED_IMAGES,
    CAMPUS_EXTRACTED_IMAGES_JSON,
    CAMPUS_IMAGES_PATH,
    CAMPUS_IMAGES_MAPPING_PATH
)

class MultiRAG:
    """
    多模态RAG系统封装类
    
    功能:
    - build: 对source文件夹中的所有文件建立数据库，并做好图片的各种记录
    - insert: 将source文件夹中的所有文件加入知识库
    - retrieve: 检索相关度最高的topk个片段返回
    """
    
    def __init__(self, 
                 index_path: str = str(CAMPUS_INDEX_DIR),
                 collection_name: str = "campus_docs",
                 embedding_model_path: str = "./Qwen3-Embedding-0.6B",
                 cross_encoder_path: str = "./cross-encoder-model",
                 image_output_dir=str(CAMPUS_IMAGES_PATH),
                 image_mapping_file=str(CAMPUS_IMAGES_MAPPING_PATH)):
        """
        初始化MultiRAG系统
        
        Args:
            index_path: FAISS索引存储路径
            collection_name: 集合名称
            embedding_model_path: 嵌入模型路径
            cross_encoder_path: 交叉编码器路径
            image_output_dir: 图片输出目录
            image_mapping_file: 图片映射文件路径
        """
        self.index_path = index_path
        self.collection_name = collection_name
        self.embedding_model_path = embedding_model_path
        self.cross_encoder_path = cross_encoder_path
        self.image_output_dir = image_output_dir
        self.image_mapping_file = image_mapping_file
        
        # 确保所有必要的目录存在
        self._ensure_directories()
        
        # 初始化模型（延迟加载）
        self._embedding_model = None
        self._cross_encoder = None
        self._vector_store = None
        self._text_splitter = None
        
        # 图片映射数据
        self._image_mapping = {}
        
        # 初始化必要的文件
        self._initialize_files()
        
        print(f"MultiRAG系统初始化完成")
        print(f"索引路径: {index_path}")
        print(f"图片输出目录: {image_output_dir}")
    
    def _ensure_directories(self):
        """确保所有必要的目录存在"""
        directories = [
            self.index_path,
            self.image_output_dir,
            os.path.dirname(self.image_mapping_file) if os.path.dirname(self.image_mapping_file) else "."
        ]
        
        for directory in directories:
            if directory and directory != ".":
                os.makedirs(directory, exist_ok=True)
                print(f"确保目录存在: {directory}")
    
    def _initialize_files(self):
        """初始化必要的文件"""
        # 初始化图片映射文件
        if not os.path.exists(self.image_mapping_file):
            with open(self.image_mapping_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
            print(f"创建图片映射文件: {self.image_mapping_file}")
        
        # 加载现有的图片映射
        self._load_image_mapping()
    
    @property
    def embedding_model(self):
        """延迟加载嵌入模型"""
        if self._embedding_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._embedding_model = SentenceTransformer(
                self.embedding_model_path,
                device=device,
                tokenizer_kwargs={"padding_side": "left"},
                trust_remote_code=True
            )
            print(f"嵌入模型加载完成，使用设备: {device}")
        return self._embedding_model
    
    @property
    def cross_encoder(self):
        """延迟加载交叉编码器"""
        if self._cross_encoder is None:
            try:
                self._cross_encoder = CrossEncoder(self.cross_encoder_path)
                print("交叉编码器加载完成")
            except Exception as e:
                print(f"交叉编码器加载失败: {e}，将跳过重排序")
                self._cross_encoder = None
        return self._cross_encoder
    
    @property
    def vector_store(self):
        """延迟加载向量存储"""
        if self._vector_store is None:
            self._vector_store = FAISSVectorStore(
                index_path=self.index_path,
                collection_name=self.collection_name,
                dimension=1024,  # Qwen3-0.6B的固定维度
                reset=False
            )
            print(f"向量存储加载完成，当前文档数量: {self._vector_store.count()}")
        return self._vector_store
    
    @property
    def text_splitter(self):
        """延迟加载文本分割器"""
        if self._text_splitter is None:
            def token_length_function(text: str) -> int:
                encoding = tiktoken.get_encoding('cl100k_base')
                return len(encoding.encode(text))
            
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,  # 每个块最大 300 token
                chunk_overlap=50,  # 块之间重叠 50 token
                length_function=token_length_function,
                separators=None,  # 使用默认分隔符
            )
        return self._text_splitter
    
    def _read_file(self, filename: str) -> str:
        """读取文件内容，支持多种格式"""
        if filename.endswith('.txt') or filename.endswith('.md') or filename.endswith('.markdown'):
            try:
                with open(filename, 'r', encoding='utf-8') as file:
                    return file.read()
            except UnicodeDecodeError:
                try:
                    with open(filename, 'r', encoding='gbk') as file:
                        return file.read()
                except Exception as e:
                    raise Exception(f"读取文件出错: {str(e)}")
        
        elif filename.endswith('.pdf'):
            try:
                text = ""
                with open(filename, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() or ""
                return text
            except Exception as e:
                raise Exception(f"读取PDF文件出错: {str(e)}")
        
        elif filename.endswith('.docx'):
            try:
                doc = Document(filename)
                return "\n".join([para.text for para in doc.paragraphs])
            except Exception as e:
                raise Exception(f"读取Word文件出错: {str(e)}")
        
        else:
            raise ValueError(f"不支持的文件格式: {filename}")
    
    def _split_document(self, filename: str) -> List[str]:
        """分割文档为chunks"""
        content = self._read_file(filename)
        if not content.strip():
            print(f"警告: 文件 {filename} 内容为空")
            return []
        
        chunks = self.text_splitter.split_text(content)
        if not chunks:
            print(f"警告: 文档 {filename} 分割后未得到任何片段")
            return []
        
        print(f"文档 {os.path.basename(filename)} 分割完成，共得到 {len(chunks)} 个片段")
        return chunks
    
    def _process_images(self, source_folder: str) -> List[Dict]:
        """处理文件夹中的图片"""
        """source_folder是文档路径，与Build传入的参数是一致的，这里传入是知识库"""
        print("开始处理图片...")
        
        # 加载现有的图片映射
        self._load_image_mapping()
        
        # 使用ImageExtractor提取和处理图片
        extractor = ImageExtractor(source_folder)
        processed_data = extractor.process_all_documents()
        
        if not processed_data:
            print("没有找到任何图片")
            return []
        
        # 保存图片到processed_images文件夹
        os.makedirs(self.image_output_dir, exist_ok=True)
        
        for i, img_data in enumerate(processed_data):
            # 生成图片文件名
            source_file = os.path.basename(img_data['source_file'])

            img_filename = f"{source_file}_image_{i+1}.jpg"

            # 图片保存路径
            img_path = os.path.join(self.image_output_dir, img_filename)
            
            # 保存图片文件
            try:
                with open(img_path, 'wb') as f:
                    f.write(img_data['image_data'])
                
                # 更新图片数据中的路径信息
                img_data['saved_path'] = img_path
                
                # 更新图片数据，添加路径信息供ImageFAISSUpdater使用
                img_data['image_path'] = img_path
                img_data['processed_path'] = img_path
                img_data['image_filename'] = img_filename
                
            except Exception as e:
                print(f"保存图片失败: {e}")
                continue
        
        print(f"成功处理了 {len(processed_data)} 张图片")
        return processed_data
    
    def _load_image_mapping(self):
        """加载图片映射文件"""
        if os.path.exists(self.image_mapping_file):
            try:
                with open(self.image_mapping_file, 'r', encoding='utf-8') as f:
                    self._image_mapping = json.load(f)
                print(f"加载图片映射文件: {len(self._image_mapping)} 个图片")
            except Exception as e:
                print(f"加载图片映射文件失败: {e}")
                self._image_mapping = {}
        else:
            self._image_mapping = {}
    
    def _save_image_mapping(self):
        """保存图片映射文件"""
        try:
            with open(self.image_mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self._image_mapping, f, ensure_ascii=False, indent=2)
            print(f"保存图片映射文件: {len(self._image_mapping)} 个图片")
        except Exception as e:
            print(f"保存图片映射文件失败: {e}")
    
    def build(self, source: str):
        """
        对source文件夹中的所有文件建立数据库，并做好图片的各种记录
        
        Args:
            source: 源文件夹路径
        """
        if not os.path.isdir(source):
            raise NotADirectoryError(f"文件夹 {source} 不存在或不是一个有效的文件夹")
        
        print(f"=== 开始构建多模态RAG数据库 ===")
        print(f"源文件夹: {source}")
        
        # 确保所有必要的目录和文件存在
        self._ensure_directories()
        self._initialize_files()
        
        # 重置向量存储
        self._vector_store = FAISSVectorStore(
            index_path=self.index_path,
            collection_name=self.collection_name,
            dimension=1024,
            reset=True  # 重新构建时清空旧数据
        )
        
        # 1. 处理文本文档
        print("\n步骤1: 处理文本文档...")
        self._process_text_documents(source)
        
        # 2. 处理图片
        print("\n步骤2: 处理图片...")
        processed_images = self._process_images(source)
        
        # 3. 将图片描述添加到数据库
        if processed_images:
            print("\n步骤3: 将图片描述添加到数据库...")
            self._add_images_to_database(processed_images)
        
        print("\n=== 数据库构建完成 ===")
        self._print_database_stats()
    
    def insert(self, source: str):
        """
        将source文件夹中的所有文件加入知识库（增量添加）
        
        Args:
            source: 源文件夹路径
        """
        if not os.path.isdir(source):
            raise NotADirectoryError(f"文件夹 {source} 不存在或不是一个有效的文件夹")
        
        print(f"=== 开始增量添加文档到数据库 ===")
        print(f"源文件夹: {source}")
        
        # 确保所有必要的目录和文件存在
        self._ensure_directories()
        self._initialize_files()
        
        # 1. 处理文本文档（增量添加）
        print("\n步骤1: 增量添加文本文档...")
        self._process_text_documents(source)
        
        # 2. 处理图片（增量添加）
        print("\n步骤2: 处理新图片...")
        processed_images = self._process_images(source)
        
        # 3. 将图片描述添加到数据库
        if processed_images:
            print("\n步骤3: 将新图片描述添加到数据库...")
            self._add_images_to_database(processed_images, incremental=True)
        
        # 4. 保存图片映射文件
        self._save_image_mapping()
        
        print("\n=== 文档增量添加完成 ===")
        self._print_database_stats()
    
    # 在retrieve方法中，统一路径分隔符
    def retrieve(self, query: str, topk: int = 5) -> List[Dict[str, Any]]:
        """
        检索相关度最高的topk个片段返回
        
        Args:
            query: 查询文本
            topk: 返回的片段数量
            
        Returns:
            List[Dict]: 返回格式为字典列表，每个字典包含:
            {
                "type": 0 or 1,  # 0代表纯文字，1代表图片描述
                "document": str,  # 片段内容
                "source": str     # 图片地址，如果是纯文字则为空
            }
        """
        print(f"检索查询: {query}")
        
        # 加载图片映射
        self._load_image_mapping()
        
        # 生成查询向量
        with torch.no_grad():
            query_embedding = self.embedding_model.encode(
                query,
                prompt_name="query",
                convert_to_tensor=False,
                normalize_embeddings=True
            ).tolist()
        
        # 从FAISS检索
        initial_k = min(topk * 3, 50)  # 初始检索更多结果用于重排序
        results = self.vector_store.search(query_embedding, initial_k)
        
        if not results:
            print("没有找到相关结果")
            return []
        
        # 重排序（如果有交叉编码器）
        if self.cross_encoder and len(results) > topk:
            print("使用交叉编码器进行重排序...")
            results = self._rerank_results(query, results, topk)
        else:
            results = results[:topk]
        
        # 格式化输出
        formatted_results = []
        for result in results:
            content = result.get('content', '')
            
            # 检查是否是图片描述
            if content.startswith('image_'):
                # 提取图片ID
                parts = content.split(':', 1)
                if len(parts) >= 2:
                    image_id = parts[0].strip()
                    document_content = parts[1].strip()

#--------------------------lg开始修改-------------------------
                    # 获取图片路径
                    image_info = self._image_mapping.get(image_id, {})

                    # 从json文件中获取图片的文件名(例如:“体育场馆预约的攻略指南.docx_image_1.jpg”)
                    image_file_name = image_info.get('image_filename', '')

                    source_path = "" #初始化为空字符串
                    if image_file_name:
                        # 使用 os.path.join 来拼接路径
                        # self.image_output_dir的值来自于 CAMPUS_IMAGES_PATH
                        full_image_path = os.path.join(self.image_output_dir, image_file_name)
                        if os.path.exists(full_image_path):
                            # 如果存在，就是使用这个路径
                            source_path = full_image_path
                        else:
                            # 如果不存在，就报个错误
                            source_path = f"错误：图片文件在指定位置不存在: {full_image_path}"
                    else:
                        # 如果json中连文件名都没有，也返回错误信息
                        source_path = f"错误，在映射文件中找不到image_id'{image_id}的文件名'"
#------------------------------lg修改完毕----------------------

                    formatted_results.append({
                        "type": 1,  # 图片描述
                        "document": document_content,
                        "source": source_path
                    })
                else:
                    formatted_results.append({
                        "type": 1,
                        "document": content,
                        "source": ""
                    })
            else:
                # 纯文本
                formatted_results.append({
                    "type": 0,  # 纯文字
                    "document": content,
                    "source": ""
                })
        
        print(f"检索完成，返回 {len(formatted_results)} 个结果")
        return formatted_results
    
    def _process_text_documents(self, source_folder: str):
        """处理文本文档"""
        # 获取所有支持的文件
        supported_extensions = ['.txt', '.md', '.markdown', '.pdf', '.docx']
        all_files = []
        
        for file in os.listdir(source_folder):
            file_path = os.path.join(source_folder, file)
            if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in supported_extensions):
                all_files.append(file_path)
        
        if not all_files:
            print("没有找到支持的文本文件")
            return
        
        print(f"找到 {len(all_files)} 个文本文件")
        
        total_chunks = 0
        for file_idx, file_path in enumerate(all_files, 1):
            try:
                print(f"处理第 {file_idx}/{len(all_files)} 个文件: {os.path.basename(file_path)}")
                
                # 分割文档
                chunks = self._split_document(file_path)
                if not chunks:
                    continue
                
                total_chunks += len(chunks)
                
                # 生成嵌入向量
                embeddings = self.embedding_model.encode(chunks)
                
                # 准备数据
                documents = []
                embeddings_list = []
                ids = []
                
                for chunk_idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    unique_id = f"text_file_{file_idx}_chunk_{chunk_idx}"
                    documents.append(chunk)
                    embeddings_list.append(embedding.tolist())
                    ids.append(unique_id)
                
                # 添加到向量存储
                self.vector_store.add(
                    documents=documents,
                    embeddings=embeddings_list,
                    ids=ids
                )
                
                print(f"文件处理完成，新增 {len(chunks)} 个片段")
                
            except Exception as e:
                print(f"处理文件 {os.path.basename(file_path)} 时出错: {e}")
                continue
        
        print(f"文本文档处理完成，共生成 {total_chunks} 个片段")
    
    def _add_images_to_database(self, processed_images: List[Dict], incremental: bool = False):
        """将图片描述添加到数据库"""
        if not processed_images:
            return
        
        # 如果不是增量添加，先删除现有图片chunks
        if not incremental:
            self.vector_store.remove_by_id_prefix("image_")
            print("已删除现有图片描述chunks")
        
        # 创建图片chunks和映射
        image_chunks = []
        image_mapping = {}
        
        for i, img_data in enumerate(processed_images):
            image_id = f"image_{i}"
            
            # 在描述前加上image x:标签
            chunk_content = f"{image_id}: {img_data['enhanced_description']}"
            
            # 存储图片映射信息（包含完整路径）
            image_mapping[image_id] = {
                'image_path': img_data.get('image_path', ''),
                'image_filename': img_data.get('image_filename', ''),
                'processed_path': img_data.get('processed_path', ''),
                'source_file': img_data['source_file'],
                'context_before': img_data['context_before'],
                'context_after': img_data['context_after'],
                'ai_description': img_data.get('original_description', ''),
                'enhanced_description': img_data['enhanced_description'],
                'image_size': img_data.get('image_size', 0)
            }
            
            # 创建chunk
            chunk = {
                'content': chunk_content,
                'chunk_id': f"image_desc_{i}"
            }
            
            image_chunks.append(chunk)
        
        # 保存图片映射文件
        self._image_mapping = image_mapping
        self._save_image_mapping()
        
        # 直接添加到MultiRAG的vector_store
        added_count = 0
        batch_size = 10
        
        for i in range(0, len(image_chunks), batch_size):
            batch = image_chunks[i:i+batch_size]
            
            try:
                # 批量生成embedding
                contents = [chunk['content'] for chunk in batch]
                embeddings = self.embedding_model.encode(contents)
                
                # 批量添加到FAISS存储
                documents = [chunk['content'] for chunk in batch]
                ids = [chunk['chunk_id'] for chunk in batch]
                
                self.vector_store.add(
                    documents=documents,
                    embeddings=embeddings.tolist(),
                    ids=ids
                )
                
                added_count += len(batch)
                print(f"已添加图片描述批次 {i//batch_size + 1}: {len(batch)} 个描述")
                
            except Exception as e:
                print(f"添加图片描述批次 {i//batch_size + 1} 时出错: {e}")
                continue
        
        print(f"成功添加 {added_count} 个图片描述到数据库")
    
    def _rerank_results(self, query: str, results: List[Dict], topk: int) -> List[Dict]:
        """使用交叉编码器重排序结果"""
        if len(results) <= topk:
            return results
        
        # 准备重排序数据
        pairs = [(query, result.get('content', '')) for result in results]
        
        with torch.no_grad():
            scores = self.cross_encoder.predict(pairs, batch_size=32)
        
        # 按分数排序
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return [result for result, score in scored_results[:topk]]
    
    def _print_database_stats(self):
        """打印数据库统计信息"""
        total_docs = self.vector_store.count()
        
        # 统计图片和文本文档数量
        image_count = 0
        text_count = 0
        
        for doc_id in self.vector_store.ids:
            if doc_id.startswith('image_desc_') or doc_id.startswith('image_'):
                image_count += 1
            else:
                text_count += 1
        
        # 从图片映射文件读取图片映射数量
        image_mapping_count = 0
        if os.path.exists(self.image_mapping_file):
            try:
                with open(self.image_mapping_file, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                    image_mapping_count = len(mapping)
            except Exception:
                image_mapping_count = 0
        
        print(f"\n数据库统计信息:")
        print(f"  总文档数: {total_docs}")
        print(f"  文本片段数: {text_count}")
        print(f"  图片描述数: {image_count}")
        print(f"  图片映射数: {image_mapping_count}")


# 使用示例
if __name__ == "__main__":
    # 创建MultiRAG实例
    rag = MultiRAG(
        index_path=str(CAMPUS_INDEX_DIR),
        embedding_model_path="./Qwen3-Embedding-0.6B",
        cross_encoder_path="./cross-encoder-model",
        image_output_dir=str(CAMPUS_IMAGES_PATH)
    )
    
    # 构建数据库
    #rag.build("./办事攻略")
    
    # 或者增量添加
    # rag.insert("./新文档文件夹")
    
    # 检索示例
    results = rag.retrieve("校园邮箱如何使用", topk=3)
    
    print("\n检索结果:")
    for i, result in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"类型: {'图片描述' if result['type'] == 1 else '纯文字'}")
        print(f"内容: {result['document'][:200]}...")
        if result['source']:
            print(f"图片路径: {result['source']}")