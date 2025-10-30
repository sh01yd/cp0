from dashscope import Application
from http import HTTPStatus
import os
import json
from multiRAG import MultiRAG
from dotenv import load_dotenv

#从Path文件里面引入知识库文件地址,索引文件的地址
from Utils.Path import (
    PAPER_DOCS_DIR, CAMPUS_DOCS_DIR, FITNESS_DOCS_DIR, PSYCHOLOGY_DOCS_DIR,
    PAPER_INDEX_DIR, CAMPUS_INDEX_DIR, FITNESS_INDEX_DIR, PSYCHOLOGY_INDEX_DIR,
    ALL_PROCESSED_IMAGES_DIR,CAMPUS_IMAGES_DIR,PAPER_IMAGES_DIR,FITNESS_IMAGES_DIR,  PSYCHOLOGY_IMAGES_DIR,
    CAMPUS_PROCESSED_EXTRACTED_IMAGES,
    CAMPUS_EXTRACTED_IMAGES_JSON,
    CAMPUS_IMAGES_PATH,
    CAMPUS_IMAGES_MAPPING_PATH,
    FITNESS_IMAGES_PATH,
    FITNESS_IMAGES_MAPPING_PATH,
    PSYCHOLOGY_IMAGES_PATH,
    PSYCHOLOGY_IMAGES_MAPPING_PATH
)
# 加载环境变量
load_dotenv("Agent.env")

# 从环境变量读取 API Key 与 AppID（校园、健身、心理助手）
APP_ID_CAMPUS   = os.getenv("APP_ID_CAMPUS")
APP_ID_FITNESS  = os.getenv("APP_ID_FITNESS")
APP_ID_PSYCHOLOGY = os.getenv("APP_ID_PSYCHOLOGY")
apiKey = os.getenv("BAILIAN_API_KEY")

class LLM_model:
    """
    统一的多模态 RAG 回调类，支持：校园知识问答助手、健身饮食助手、心理助手。
    - 使用 MultiRAG 检索文本与图片
    - 先输出图片块（dict），再输出分段文本（str）
    - 根据不同智能体生成定制化提示词
    """

    def __init__(self, agent_type: str = "校园知识问答助手"):
        self.agent_type = agent_type
        self.session_id = f"{agent_type}_session"

        # 各智能体配置
        agent_config = {
            "校园知识问答助手": {
                "app_id_env": "APP_ID_CAMPUS",
                "index_dir": str(CAMPUS_INDEX_DIR),
                "collection": "campus_docs",
                "docs_dir": str(CAMPUS_DOCS_DIR),
                "images_dir": str(CAMPUS_IMAGES_PATH),
                "mapping_file": str(CAMPUS_IMAGES_MAPPING_PATH),
            },
            "健身饮食助手": {
                "app_id_env": "APP_ID_FITNESS",
                "index_dir": str(FITNESS_INDEX_DIR),
                "collection": "fitness_docs",
                "docs_dir": str(FITNESS_DOCS_DIR),
                "images_dir": str(FITNESS_IMAGES_PATH),
                "mapping_file": str(FITNESS_IMAGES_MAPPING_PATH),
            },
            "心理助手": {
                "app_id_env": "APP_ID_PSYCHOLOGY",
                "index_dir": str(PSYCHOLOGY_INDEX_DIR),
                "collection": "psychology_docs",
                "docs_dir": str(PSYCHOLOGY_DOCS_DIR),
                "images_dir": str(PSYCHOLOGY_IMAGES_PATH),
                "mapping_file": str(PSYCHOLOGY_IMAGES_MAPPING_PATH),
            },
        }

        cfg = agent_config.get(agent_type)
        if not cfg:
            raise ValueError(f"不支持的智能体类型: {agent_type}")

        self.app_id = os.getenv(cfg["app_id_env"]) or ""
        if not self.app_id or not apiKey:
            print(f"环境变量缺失：请在 Agent.env 中设置 {cfg['app_id_env']} 和 BAILIAN_API_KEY")

        # 初始化 MultiRAG
        self.multirag = MultiRAG(
            index_path=cfg["index_dir"],
            collection_name=cfg["collection"],
            embedding_model_path="./Qwen3-Embedding-0.6B",
            cross_encoder_path="./cross-encoder-model",
            image_output_dir=cfg["images_dir"],
            image_mapping_file=cfg["mapping_file"],
        )

        # 确保图片处理与索引
        try:
            self.multirag.ensure_image_index(cfg["docs_dir"])
        except Exception as e:
            print(f"初始化阶段确保图片索引失败({agent_type}): {e}")

    def start_LLM(self):
        return "Unified LLM model started successfully"

    def retrieve_and_answer(self, query: str, top_k: int = 8):
        try:
            results = self.multirag.retrieve(query, topk=top_k)
            if not results:
                yield from self.call_llm_stream(query, [])
                return

            text_chunks = []
            image_info = []
            for result in results:
                result_type = result.get('type', 0)
                document = result.get('document', '')
                source = result.get('source', '')

                if result_type == 1:
                    if source:
                        image_info.append({'description': document, 'path': source, 'score': 1.0})
                        text_chunks.append(f"[图片内容] {document} [图片地址: {source}]")
                    else:
                        text_chunks.append(f"[图片内容] {document}")
                else:
                    text_chunks.append(document)

            # 先输出图片信息块
            for img in image_info:
                try:
                    yield {
                        'type': 1,
                        'description': img.get('description', ''),
                        'source': img.get('path', ''),
                        'score': img.get('score', 1.0)
                    }
                except Exception as e:
                    print(f"输出图片信息时发生错误: {e}")

            enhanced_chunks = self._enhance_chunks_with_images(text_chunks, image_info)
            yield from self.call_llm_stream(query, enhanced_chunks)

        except Exception as e:
            print(f"检索过程出错({self.agent_type}): {e}")
            import traceback
            traceback.print_exc()
            yield from self.call_llm_stream(query, [])

    def _enhance_chunks_with_images(self, text_chunks, image_info):
        enhanced_chunks = text_chunks.copy()
        if image_info:
            image_instruction = "\n注意：回答中如需引用图片，请直接使用图片地址，格式为：[具体路径]\n"
            enhanced_chunks.append(image_instruction)
            image_summary = "可用图片资源：\n"
            for i, img in enumerate(image_info[:3]):
                image_summary += f"{i + 1}. {img['description']} [地址: {img['path']}]\n"
            enhanced_chunks.append(image_summary)
        return enhanced_chunks

    def _build_prompt(self, query: str, chunks: list[str]) -> str:
        separator = "\n\n"
        safe_chunks = separator.join(f"- {c}" for c in chunks[:10]) if chunks else "无"

        # 三种助手的提示词均参考校园知识助手的分段与风格要求，但用不同领域语气与侧重点
        if self.agent_type == "校园知识问答助手":
            return f"""
你是一个校园知识问答助手。你需要模仿学长学姐亲切的对话风格，例如"我们学校的体育馆是....、我们学校的饭堂是...我们学校绩点的计算是..."这样的话语将你的回答拆分成3到5个自然段落，首先要确保你的回答拼起来是连贯的，符合人类讲出来的，其次是段与段之间要在语义和逻辑上相互承接。每个段落结束后，必须使用特殊标记 `[NEW_PARAGRAPH]` 作为换段标志。

请根据用户的问题和下面的背景知识进行回答。

用户问题: {query}

背景知识:
{safe_chunks}

回答要求：
1. 模仿人类口吻，友好自然地进行分段说明。
2. 将完整的回答分成3到5段，段与段之间要在语义和逻辑上相互承接，段落之间必须用 `[NEW_PARAGRAPH]` 分隔。
3. 如果背景知识中包含图片信息（标注为[图片内容]或[图片地址]），请在回答中适当引用，引用时直接使用提供的图片地址，格式：[具体路径]。
4. 若用户问题与背景知识无关，则用通用知识解决问题。

请开始你的回答：
"""
        elif self.agent_type == "健身饮食助手":
            return f"""
你是一个健身与饮食助手。请用温和、鼓励且专业的语气，结合运动与营养学的常识，提供安全、可执行、循序渐进的建议。你的回答需要拆分成3到5个自然段落，段与段之间要在语义和逻辑上相互承接。每个段落结束后，必须使用特殊标记 `[NEW_PARAGRAPH]` 作为换段标志。

请根据用户的问题和下面的背景知识进行回答。

用户问题: {query}

背景知识:
{safe_chunks}

回答要求：
1. 语气友好鼓励，建议务必安全、具体、可执行（如强度、频次、时长、食材搭配）。
2. 使用 `[NEW_PARAGRAPH]` 分隔3到5个段落，并保持段落的逻辑承接（如先评估、再建议、后注意事项）。
3. 如有[图片内容]或[图片地址]，在合适位置引用，引用时直接使用提供的图片地址，格式：[具体路径]。
4. 强调个体差异与循序渐进，如用户情况不明，给出普适安全的默认建议。
5. 若用户问题与背景知识无关，则用通用健康知识回答。

请开始你的回答：
"""
        else:  # 心理助手
            return f"""
你是一个心理支持助手。请在尊重与共情的前提下，以支持性的语气提供清晰、可落实的建议或引导。你的回答需要拆分成3到5个自然段落，段与段之间要在语义和逻辑上相互承接。每个段落结束后，必须使用特殊标记 `[NEW_PARAGRAPH]` 作为换段标志。

请根据用户的问题和下面的背景知识进行回答。

用户问题: {query}

背景知识:
{safe_chunks}

回答要求：
1. 保持共情与尊重，避免诊断性结论，给出可实践的建议（如自我觉察、呼吸放松、日常习惯调整、寻求支持等）。
2. 使用 `[NEW_PARAGRAPH]` 分隔3到5段，并保持段落逻辑承接（如先理解感受、再提供策略、后给出资源与提醒）。
3. 如有[图片内容]或[图片地址]，在合适位置引用，引用时直接使用提供的图片地址，格式：[具体路径]。
4. 若问题涉及紧急风险或专业干预范围，提醒寻求专业帮助；无法确定时提供通用的自助与求助路径。
5. 若用户问题与背景知识无关，则用通用心理支持建议回答。

请开始你的回答：
"""

    def call_llm_stream(self, query: str, chunks: list[str]):
        prompt = self._build_prompt(query, chunks)
        full_response_text = ""
        try:
            response = Application.call(
                api_key=apiKey,
                app_id=self.app_id,
                prompt=prompt,
                session_id=self.session_id,
                stream=False
            )
            if response.status_code == HTTPStatus.OK:
                full_response_text = response.output.text
                # 将占位符 [具体路径] 替换为 chunks 中的图片地址
                full_response_text = self._substitute_image_placeholders(full_response_text, chunks)
            else:
                error_message = f"API Error: {response.message}"
                print(error_message)
                yield error_message
                return
        except Exception as e:
            error_message = f"调用LLM时发生异常: {e}"
            print(error_message)
            yield error_message
            return

        paragraphs = full_response_text.split('[NEW_PARAGRAPH]')
        for para in paragraphs:
            cleaned_para = para.strip()
            if cleaned_para:
                yield cleaned_para

    def _substitute_image_placeholders(self, response_text: str, chunks: list) -> str:
        """将响应中的 [具体路径] 顺序替换为 chunks 列表中的图片地址。
        解析 '[图片地址: ...]' 或 '[地址: ...]'，并按出现顺序替换。
        """
        if not response_text or not chunks:
            return response_text

        import re
        paths = []
        for c in chunks:
            try:
                s = str(c)
            except Exception:
                continue
            for m in re.finditer(r"\[图片地址:\s*(.*?)\]", s):
                p = m.group(1).strip()
                if p:
                    paths.append(p)
            for m in re.finditer(r"\[地址:\s*(.*?)\]", s):
                p = m.group(1).strip()
                if p:
                    paths.append(p)

        if not paths:
            return response_text

        result = response_text
        count = 0
        while "[具体路径]" in result and count < len(paths):
            result = result.replace("[具体路径]", paths[count], 1)
            count += 1
        return result