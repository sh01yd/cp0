from dashscope import Application
from http import HTTPStatus
import os
import json
from multiRAG import MultiRAG

#从Path文件里面引入知识库文件地址,索引文件的地址
from Utils.Path import (
    PAPER_DOCS_DIR, CAMPUS_DOCS_DIR, FITNESS_DOCS_DIR, PSYCHOLOGY_DOCS_DIR,
    PAPER_INDEX_DIR, CAMPUS_INDEX_DIR, FITNESS_INDEX_DIR, PSYCHOLOGY_INDEX_DIR,
    ALL_PROCESSED_IMAGES_DIR,CAMPUS_IMAGES_DIR,PAPER_IMAGES_DIR,FITNESS_IMAGES_DIR,  PSYCHOLOGY_IMAGES_DIR,
    CAMPUS_PROCESSED_EXTRACTED_IMAGES,
    CAMPUS_EXTRACTED_IMAGES_JSON,
    CAMPUS_IMAGES_PATH,
    CAMPUS_IMAGES_MAPPING_PATH
)

APP_ID = "c2affdebf6664d438a4043216ee15dea"
apiKey = "sk-f89e754d6cff4f31a25f609e82b3bce1"

class LLM_model:
    def __init__(self):
        """
        初始化LLM模型实例
        """
        self.session_id = "default_session"
        # 初始化MultiRAG系统
        self.multirag = MultiRAG(
            index_path=str(CAMPUS_INDEX_DIR),
            collection_name="campus_docs",
            embedding_model_path="./Qwen3-Embedding-0.6B",
            cross_encoder_path="./cross-encoder-model",
            image_output_dir=str(CAMPUS_IMAGES_PATH),
            image_mapping_file=str( CAMPUS_IMAGES_MAPPING_PATH)
        )
        print("MultiRAG系统初始化完成")

    def start_LLM(self):
        """
        启动LLM服务

        Returns:
            str: 启动状态信息
        """
        return "LLM model started successfully"

    def retrieve_and_answer(self, query: str, top_k: int = 8):
        """
        智能检索并回答问题

        Args:
            query (str): 用户问题
            top_k (int): 检索的片段数量

        Yields:
            str: 生成的文本段落
        """
        try:
            # 1. 使用MultiRAG检索相关片段
            print(f"正在检索与问题相关的top-{top_k}片段...")
            results = self.multirag.retrieve(query, topk=top_k)

            if not results:
                print("未找到相关片段，使用通用知识回答")
                yield from self.call_llm_stream(query, [])
                return

            # 2. 处理检索结果
            text_chunks = []
            image_info = []

            for result in results:
                result_type = result.get('type', 0)
                document = result.get('document', '')
                source = result.get('source', '')

                if result_type == 1:
                    if source and source != "":
                        image_info.append({
                            'description': document,
                            'path': source,
                            'score': 1.0
                        })
                        text_chunks.append(f"[图片内容] {document} [图片地址: {source}]")
                    else:
                        text_chunks.append(f"[图片内容] {document}")
                else:
                    text_chunks.append(document)

            print(f"检索到 {len(text_chunks)} 个文本片段，{len(image_info)} 个图片")

            # 3. 构建增强的prompt
            enhanced_chunks = self._enhance_chunks_with_images(text_chunks, image_info)

            # 4. 调用LLM生成回答
            yield from self.call_llm_stream(query, enhanced_chunks)

        except Exception as e:
            print(f"检索过程出错: {e}")
            import traceback
            traceback.print_exc()
            yield from self.call_llm_stream(query, [])

    def _enhance_chunks_with_images(self, text_chunks, image_info):
        """
        根据图片信息增强文本片段
        """
        enhanced_chunks = text_chunks.copy()

        if image_info:
            image_instruction = "\n注意：回答中如需引用图片，请直接使用图片地址，格式为：[具体路径]\n"
            enhanced_chunks.append(image_instruction)

            image_summary = "可用图片资源：\n"
            for i, img in enumerate(image_info[:3]):
                image_summary += f"{i + 1}. {img['description']} [地址: {img['path']}]\n"
            enhanced_chunks.append(image_summary)

        return enhanced_chunks

    def call_llm_stream(self, query, list):
        """
        流式生成回答，按段落返回。

        Args:
            query (str): 用户问题
            list (list): 相关文档片段列表

        Yields:
            str: 生成的文本段落
        """
        separator = "\n\n"
        # --- 修改部分：优化Prompt，要求分段并使用分隔符 ---
        prompt = f"""你是一个校园知识问答助手。你需要模仿学长学姐亲切的对话风格，例如"我们学校的体育馆是....、我们学校的饭堂是...我们学校绩点的计算是..."这样的话语将你的回答拆分成3到5个自然段落，首先要确保你的回答拼起来是连贯的，符合人类讲出来的，其次是段与段之间要在语义和逻辑上相互承接。每个段落结束后，必须使用特殊标记 `[NEW_PARAGRAPH]` 作为换段标志。

请根据用户的问题和下面的背景知识进行回答。

用户问题: {query}

背景知识:
{separator.join(list)}

回答要求：
1. 模仿人类口吻，友好自然地进行分段说明。
2. 将完整的回答分成3到5段，段与段之间要在语义和逻辑上相互承接，段落之间必须用 `[NEW_PARAGRAPH]` 分隔。
3. 如果背景知识中包含图片信息（标注为[图片内容]或[图片地址]），请在回答中适当引用。
4. 引用图片时，直接使用提供的图片地址，格式：[具体路径]，无需任何前缀或后缀。
5. 若用户问题与背景知识无关，则用通用知识解决问题。

请开始你的回答：
"""

        # --- 修改部分：调整流式逻辑为分段输出 ---
        # 使用非流式调用获取完整回答
        full_response_text = ""
        try:
            response = Application.call(
                api_key=apiKey,
                app_id=APP_ID,
                prompt=prompt,
                session_id=self.session_id,
                stream=False  # 改为非流式获取完整结果
            )
            if response.status_code == HTTPStatus.OK:
                request_id = response.request_id
                print(f"成功获取到回答，Request ID: {request_id}")  # 你可以在这里打印或记录它

                full_response_text = response.output.text
            else:
                error_message = f'API Error: {response.message}'
                print(error_message)
                yield error_message  # 返回错误信息
                return

        except Exception as e:
            error_message = f"调用LLM时发生异常: {e}"
            print(error_message)
            yield error_message
            return

        # 根据分隔符切分段落并依次返回，这里是核心逻辑，也就是说我把一大块内容分成几块，然后进行yield返回
        paragraphs = full_response_text.split('[NEW_PARAGRAPH]')
        for para in paragraphs:
            cleaned_para = para.strip()  # 去除可能存在的多余空格和换行
            if cleaned_para:  # 确保不返回空段落
                yield cleaned_para