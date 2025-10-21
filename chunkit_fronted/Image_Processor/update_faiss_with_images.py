import json
import base64
import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from faiss_store_y import FAISSVectorStore
import numpy as np

#从Path文件里面引入知识库文件地址,索引文件的地址
from Utils.Path import (
    PAPER_DOCS_DIR, CAMPUS_DOCS_DIR, FITNESS_DOCS_DIR, PSYCHOLOGY_DOCS_DIR,
    PAPER_INDEX_DIR, CAMPUS_INDEX_DIR, FITNESS_INDEX_DIR, PSYCHOLOGY_INDEX_DIR,
    ALL_PROCESSED_IMAGES_DIR,CAMPUS_IMAGES_DIR,PAPER_IMAGES_DIR,FITNESS_IMAGES_DIR,  PSYCHOLOGY_IMAGES_DIR,
    CAMPUS_PROCESSED_EXTRACTED_IMAGES,
    CAMPUS_EXTRACTED_IMAGES_JSON,
    CAMPUS_IMAGES_PATH
)



class ImageFAISSUpdater:
    """将图片描述添加到FAISS数据库的类"""

    def __init__(self, faiss_store_path: str = "faiss_index1"):
        self.faiss_store_path = faiss_store_path
        # 使用本地的Qwen3 embedding模型
        self.embedding_model = SentenceTransformer(
            "./Qwen3-Embedding-0.6B",
            tokenizer_kwargs={"padding_side": "left"},
            trust_remote_code=True
        )

        # 初始化或加载FAISS存储
        try:
            self.faiss_store = FAISSVectorStore(index_path=faiss_store_path)
            print(f"FAISS存储初始化完成: {faiss_store_path}")
        except Exception as e:
            print(f"初始化FAISS存储时出错: {e}")
            self.faiss_store = FAISSVectorStore(index_path=faiss_store_path)

    def load_processed_images(self, json_path: str) -> List[Dict]:
        """从JSON文件加载处理后的图片数据"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载JSON文件时出错: {e}")
            return []

    def remove_existing_image_chunks(self):
        """删除现有的图片描述chunks"""
        print("正在删除现有的图片描述chunks...")
        self.faiss_store.remove_by_id_prefix("image_")
        print("现有图片描述chunks已删除")

    def create_image_chunks(self, processed_images: List[Dict]) -> tuple[List[Dict], Dict[str, Dict]]:
        """将每个图片描述作为独立chunk，使用image x:前缀标签"""
        chunks = []
        image_mapping = {}  # 存储id到图片信息的映射

        for i, img_data in enumerate(processed_images):
            image_id = f"image_{i}"

            # 在描述前加上image x:标签
            chunk_content = f"{image_id}: {img_data['enhanced_description']}"

            # 存储图片映射信息
            image_mapping[image_id] = {
                'image_path': img_data.get('image_path', ''),
                'image_filename': img_data.get('image_filename', ''),
                'source_file': img_data['source_file'],
                'context_before': img_data['context_before'],
                'context_after': img_data['context_after'],
                'ai_description': img_data.get('ai_description', ''),
                'enhanced_description': img_data['enhanced_description'],
                'image_size': img_data.get('image_size', 0)
            }

            # 创建简化的chunk字典
            chunk = {
                'content': chunk_content,
                'metadata': {
                    'type': 'image_description',
                    'image_id': image_id
                },
                'chunk_id': f"image_desc_{i}"
            }

            chunks.append(chunk)

        return chunks, image_mapping

    def create_image_chunks_with_paths(self, processed_images: List[Dict]) -> tuple[List[Dict], Dict[str, Dict]]:
        """将每个图片描述作为独立chunk，包含完整的路径信息"""
        chunks = []
        image_mapping = {}  # 存储id到图片信息的映射

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
                'enhanced_description': img_data['enhanced_description']
            }

            # 创建简化的chunk字典
            chunk = {
                'content': chunk_content,
                'metadata': {
                    'type': 'image_description',
                    'image_id': image_id
                },
                'chunk_id': f"image_desc_{i}"
            }

            chunks.append(chunk)

        return chunks, image_mapping

    def add_image_chunks_to_faiss(self, image_chunks: List[Dict]):
        """将图片chunks添加到FAISS数据库"""
        print(f"开始将 {len(image_chunks)} 个图片描述添加到FAISS数据库...")

        added_count = 0
        # 批量处理以提高效率
        batch_size = 10
        for i in range(0, len(image_chunks), batch_size):
            batch = image_chunks[i:i + batch_size]

            try:
                # 批量生成embedding
                contents = [chunk['content'] for chunk in batch]
                embeddings = self.embedding_model.encode(contents)

                # 批量添加到FAISS存储
                documents = [chunk['content'] for chunk in batch]
                ids = [chunk['chunk_id'] for chunk in batch]

                self.faiss_store.add(
                    documents=documents,
                    embeddings=embeddings.tolist(),
                    ids=ids
                )

                added_count += len(batch)
                print(f"已添加图片描述批次 {i // batch_size + 1}: {len(batch)} 个描述")

            except Exception as e:
                print(f"添加图片描述批次 {i // batch_size + 1} 时出错: {e}")
                continue

        print("图片描述添加完成！")
        return added_count

    def save_image_mapping(self, image_mapping: Dict[str, Dict], mapping_path: str = "image_mapping.json"):
        """保存图片ID到路径的映射文件"""
        try:
            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump(image_mapping, f, ensure_ascii=False, indent=2)
            print(f"图片映射文件已保存到: {mapping_path}")
        except Exception as e:
            print(f"保存图片映射文件时出错: {e}")

    def save_faiss_index(self):
        """保存FAISS索引"""
        try:
            self.faiss_store.save()
            print(f"FAISS索引已保存到: {self.faiss_store_path}")
        except Exception as e:
            print(f"保存FAISS索引时出错: {e}")

    def get_faiss_stats(self):
        """获取FAISS数据库统计信息"""
        try:
            total_docs = self.faiss_store.count()
            print(f"FAISS数据库统计:")
            print(f"  总文档数: {total_docs}")

            # 统计图片类型的文档（通过ID前缀判断）
            image_count = 0
            for doc_id in self.faiss_store.ids:
                if doc_id.startswith('image_'):
                    image_count += 1

            print(f"  图片描述文档数: {image_count}")

            return {
                'total_chunks': total_docs,
                'image_chunks': image_count,
                'text_chunks': total_docs - image_count
            }

        except Exception as e:
            print(f"获取统计信息时出错: {e}")
            return None


def main():
    """主函数"""
    json_path = str( CAMPUS_EXTRACTED_IMAGES_JSON)
    faiss_store_path = str(CAMPUS_INDEX_DIR)

    # 检查JSON文件是否存在
    if not os.path.exists(json_path):
        print(f"错误: 找不到处理后的图片数据文件: {json_path}")
        print("请先运行 image_processor.py 来提取和处理图片")
        return

    # 创建FAISS更新器
    updater = ImageFAISSUpdater(faiss_store_path)

    # 首先删除现有的图片描述chunks
    updater.remove_existing_image_chunks()

    # 加载处理后的图片数据
    processed_images = updater.load_processed_images(json_path)

    if not processed_images:
        print("没有找到处理后的图片数据")
        return

    print(f"加载了 {len(processed_images)} 个图片描述")

    # 创建图片chunks（每个图片描述作为独立chunk）
    image_chunks, image_mapping = updater.create_image_chunks(processed_images)

    # 保存图片映射文件
    mapping_path = "d:\\core code\\chunkit_fronted\\image_mapping.json"
    updater.save_image_mapping(image_mapping, mapping_path)

    # 添加到FAISS数据库
    updater.add_image_chunks_to_faiss(image_chunks)

    # 保存索引
    updater.save_faiss_index()

    # 显示统计信息
    updater.get_faiss_stats()

    print("\n图片描述已成功重新添加到FAISS数据库！")


if __name__ == "__main__":
    main()