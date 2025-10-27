#=====================这是接受用户信息，获取回答的主函数===================
import os
from dotenv import load_dotenv
from IntentRecognition.Intent_by_Rag import RagQueryEnhancer
from RAGlibrary import RAG_psychology, RAG_fitness, RAG_compus, RAG_paper
from callback import LLM_model
# 加载 .env 文件
load_dotenv("Agent.env")

# 验证环境变量是否设置
required_env_vars = [
    "BAILIAN_API_KEY",
    "APP_ID_PSYCHOLOGY",
    "APP_ID_CAMPUS",
    "APP_ID_FITNESS",
    "APP_ID_PAPER"
]

missing_vars = []
for var in required_env_vars:
    if not os.getenv(var):
        missing_vars.append(var)

if missing_vars:
    print(f"请在.env文件中设置以下环境变量: {', '.join(missing_vars)}")
    exit(1)

print("所有环境变量配置验证成功")
print(f"使用的智能体应用:")
print(f"   - 心理助手: {os.getenv('APP_ID_PSYCHOLOGY')}")
print(f"   - 健身助手: {os.getenv('APP_ID_FITNESS')}")
print(f"   - 校园助手: {os.getenv('APP_ID_CAMPUS')}")
print(f"   - 论文助手: {os.getenv('APP_ID_PAPER')}")
print()

enhancer = RagQueryEnhancer()

class InteractiveAgent:
    def  __init__(self):
        try:
            # 初始化意图分类器,这里我删了

            print("意图分类器初始化成功")

            # 初始化 RAG 智能体（延迟初始化以提高启动速度）
            self.rag_agents = {}
            self.agent_classes = {
                "心理助手": RAG_psychology,
                "健身饮食助手": RAG_fitness,
                "校园知识问答助手": RAG_compus,
                "论文助手": RAG_paper
            }
            self.llm = LLM_model()
            self.llm.start_LLM()
            # 意图到头像的映射关系
            self.intent_avatar_mapping = {
                "心理助手": "007-gin tonic.svg",
                "健身饮食助手": "014-mojito.svg", 
                "校园知识问答助手": "042-milkshake.svg",
                "论文助手": "044-whiskey sour.svg",
                "其他": "050-lemon juice.svg"
            }
            
            print("RAG 智能体类加载成功")

        except Exception as e:
            print(f"初始化失败: {e}")
            raise

    def get_rag_agent(self, intent):
        """延迟初始化RAG智能体"""
        if intent not in self.rag_agents:
            if intent in self.agent_classes:
                print(f"🔧 正在初始化 {intent} RAG智能体...")
                try:
                    self.rag_agents[intent] = self.agent_classes[intent]()
                    print(f"{intent} RAG智能体初始化成功")
                except Exception as e:
                    print(f"{intent} RAG智能体初始化失败: {e}")
                    return None
            else:
                return None

        return self.rag_agents.get(intent)

    def check_rag_status(self, intent, rag_agent):
        """检查RAG知识库状态"""
        try:
            doc_count = rag_agent.vector_store.count()
            if doc_count == 0:
                print(f"{intent} 知识库中暂无文档")
                return False
            else:
                print(f"{intent} 知识库包含 {doc_count} 个文档片段")
                return True
        except Exception as e:
            print(f"检查 {intent} 知识库状态失败: {e}")
            return False

    def process_question_with_full_response(self, user_input: str, stream_mode: bool = False):
        """

        处理用户问题并返回一个或多个完整的回答。这是主聊天流程调用的方法。
        """
        try:
            # 1. 【第一步】进行意图识别和查询强化，这是所有后续操作的基础。
            enhancement_result = enhancer.enhance_query(user_input)

            # --- 【新增】可视化调试输出 ---
            if enhancement_result and enhancement_result.get("intent_distribution"):
                distribution = enhancement_result["intent_distribution"]
                total_docs = sum(distribution.values())

                # 构造调试信息字符串
                debug_parts = []
                for intent, count in distribution.items():
                    confidence = f"({count}/{total_docs})" if total_docs > 0 else ""
                    debug_parts.append(f"{intent} 有 {count} 份 {confidence}")

                print(f"🔍 [调试信息] 检索到的意图分布: {', '.join(debug_parts)}")
            # --- 可视化结束 ---

            if not enhancement_result or not enhancement_result.get("analysis_results"):
                # 如果没结果，根据模式返回错误信息
                if stream_mode: return self._stream_error("抱歉，未能识别出您问题的意图。")
                return [{"success": False, "message": "未能识别出意图"}]

            # 2. 【第二步】根据模式，调用对应的执行器
            if stream_mode:
                # 流式模式下，将分析结果交给专门的流式生成器处理
                return self._stream_answers_for_intents(enhancement_result)
            else:
                # 非流式模式下，将分析结果交给专门的批量处理器处理
                return self._get_batch_answers_for_intents(enhancement_result)

        except Exception as e:
            # 统一的顶层异常处理
            if stream_mode: return self._stream_error(f"处理过程中发生严重错误: {str(e)}")
            return [{"success": False, "message": f"处理过程中发生严重错误: {str(e)}"}]

    def _get_batch_answers_for_intents(self, enhancement_result: dict) -> list:
        """【内部执行器 - 非流式】接收分析结果，返回一个包含所有回答的列表。"""
        all_responses = []
        original_query = enhancement_result.get("original_query")
        for item in enhancement_result["analysis_results"]:
            if "error" in item: continue

            Rag_intent = item["intent"]
            rewritten_query = item["rewritten_query"]
            avatar = self.intent_avatar_mapping.get(Rag_intent, self.intent_avatar_mapping["其他"])

            try:
                # 根据意图选择Agent并调用
                if Rag_intent == "校园知识问答助手":
                    string_generator = self.llm.retrieve_and_answer(original_query, top_k=8)
                    answer = "".join(string_generator)
                else:
                    rag_agent = self.get_rag_agent(Rag_intent)
                    if rag_agent:
                        answer = rag_agent.call_RAG(original_query)
                    else:
                        answer = "抱歉，暂不支持此意图。"

                all_responses.append({"success": True, "intent": Rag_intent, "avatar": avatar, "answer": answer})
            except Exception as e:
                all_responses.append({"success": False, "intent": Rag_intent, "avatar": avatar, "error": str(e)})
        return all_responses

        # 在 InteractiveAgent 类中找到这个方法并替换它

    def _stream_answers_for_intents(self, enhancement_result: dict):
            """【内部执行器 - 流式】接收分析结果，返回一个依次处理所有意图的生成器。"""
            try:
                original_query = enhancement_result.get("original_query")
                if not original_query:
                    yield from self._stream_error("未能获取到原始用户问题。")
                    return

                for item in enhancement_result["analysis_results"]:
                    if "error" in item:
                        # 对于错误情况，保持原有格式或简化
                        yield {"type": "error", "intent": item.get("intent"), "message": item["error"]}
                        continue

                    Rag_intent = item["intent"]
                    rewritten_query = item["rewritten_query"]
                    avatar = self.intent_avatar_mapping.get(Rag_intent, self.intent_avatar_mapping["其他"])

                    # 定义一个生成器变量，用来接收来自不同智能体的段落流
                    paragraph_generator = None

                    try:
                        if Rag_intent == "校园知识问答助手":
                            paragraph_generator = self.llm.retrieve_and_answer(original_query, top_k=8)
                        else:
                            rag_agent = self.get_rag_agent(Rag_intent)
                            if rag_agent:
                                paragraph_generator = rag_agent.call_RAG_stream(original_query)
                            else:
                                # 如果智能体不存在，则生成一个包含错误信息的段落
                                paragraph_generator = iter(["抱歉，暂不支持此意图。"])

                        # 统一处理所有段落流
                        for paragraph in paragraph_generator:
                            # 检查是否是字典类型（可能包含图片信息）
                            if isinstance(paragraph, dict):
                                # 如果是图片类型的结果
                                if paragraph.get("type") == 1:  # 图片描述
                                    yield {
                                        "type": "image",
                                        "intent": Rag_intent,
                                        "avatar": avatar,
                                        "image_path": paragraph.get("source", ""),
                                        "description": paragraph.get("document", "")
                                    }
                                else:  # 普通文本
                                    yield {
                                        "type": "content",
                                        "intent": Rag_intent,
                                        "avatar": avatar,
                                        "delta": paragraph.get("document", "")
                                    }
                            else:
                                # 为普通文本创建一个包含所有信息的、完整的消息包
                                yield {
                                    "type": "content",
                                    "intent": Rag_intent,
                                    "avatar": avatar,
                                    "delta": paragraph  # paragraph 就是我们的一整段话
                                }

                    except Exception as e:
                        # 如果在生成过程中出错，也发送一个结构完整的错误消息
                        yield {
                            "type": "error",
                            "intent": Rag_intent,
                            "avatar": avatar,
                            "message": f"处理时发生错误: {str(e)}"
                        }
                    # --- 修改结束 ---

                    # 每个意图结束后发送一个分隔符
                    yield {"type": "break", "message": f"意图 {Rag_intent} 回答结束"}

                # 所有意图都结束后发送最终完成标志
                yield {"type": "finished", "finished": True}
            except Exception as e:
                yield from self._stream_error(f"流式处理时发生严重错误: {str(e)}")

    def _stream_error(self, message: str):
        """【辅助函数】用于在流式模式下返回一个标准的错误信息。"""
        yield {"type": "error", "message": message}
        yield {"type": "finished", "finished": True}

    def chat(self):
        print("=== 欢迎使用智能助手系统 ===")
        print("本系统使用本地RAG检索增强 + 远程智能体架构")
        print("支持交叉编码器精确检索和流式回答")
        print("输入你的问题（输入 'exit' 退出，'batch' 切换非流式模式）：\n")

        stream_mode = True

        while True:
            user_input = input("你：")

            if user_input.lower() in ["exit", "quit"]:
                print("再见！")
                break

            if user_input.lower() == "batch":
                stream_mode = not stream_mode
                print(f"模式已切换。当前流式输出: {'开启' if stream_mode else '关闭'}")
                continue

            results = self.process_question_with_full_response(user_input, stream_mode=stream_mode)
            # 根据模式处理并打印结果
            if stream_mode:
                # 处理流式生成器
                current_intent = "未知意图"
                print("--- 流式回答 (一段一段) ---")
                try:
                    for chunk in results:
                        # 处理不同类型的响应
                        if chunk.get('type') == 'content':
                            avatar = chunk.get('avatar', '🤖')
                            paragraph = chunk.get('delta', '')
                            # 模拟前端渲染：每一段都带上自己的头像信息
                            print(f"头像: {avatar} | 回答段落: {paragraph}")
                            
                        elif chunk.get('type') == 'image':
                            avatar = chunk.get('avatar', '🤖')
                            image_path = chunk.get('image_path', '')
                            description = chunk.get('description', '')
                            # 显示图片信息
                            print(f"头像: {avatar} | 图片: {image_path}")
                            print(f"图片描述: {description}")

                        elif chunk.get('type') == 'break':
                            print("--- (一个意图回答结束) ---\n")

                        elif chunk.get('type') == 'error':
                            print(f"处理时发生错误: {chunk.get('message')}")

                except Exception as e:
                    print(f"\n处理流式响应时发生错误: {e}")
                print("\n------------------\n")

            else:
                # 处理非流式（批量）结果
                print("--- 回答 ---")
                if not results:
                    print("抱歉，未能生成回答。")

                for response in results:
                    if response.get("success"):
                        intent = response.get('intent', '未知意图')
                        answer = response.get('answer', '（无回答）')
                        print(f"🤖 {intent} 回答：{answer}\n")
                    else:
                        intent = response.get('intent', '未知意图')
                        error_msg = response.get('error', '未知错误')
                        print(f"处理意图 '{intent}' 时出错: {error_msg}\n")
                print("------------\n")


def predict_intent_only(self, user_input):
        """
               进行意图识别，返回一个或多个意图及其对应的头像。

               Args:
                   user_input (str): 用户输入的问题

               Returns:
                   dict: 一个包含处理结果的字典。
                         - success (bool): 处理是否成功。
                         - results (list): 一个包含所有识别出的意图信息的列表。
                                           每个元素是一个字典，如:
                                           {"intent": "心理助手", "avatar": "🧠"}
                         - message (str): 描述信息。
               """
        try:
            # 进行意图识别
            enhancement_result = enhancer.enhance_query(user_input)
            # 检查是否有有效的分析结果
            if not enhancement_result or not enhancement_result.get("analysis_results"):
                return {
                    "success": False,
                    "results": [],
                    "message": "未能识别出任何意图"
                }

            # 2. 【关键】创建一个空列表，用于收集所有结果
            identified_intents = []

            #3.遍历所有分析出的意图
            for item in enhancement_result["analysis_results"]:
                if "error" in item:
                    print(f"处理意图 '{item.get('intent', '未知')}' 时出错: {item['error']}")
                    continue  # 跳过这个出错的结果，继续下一个
                #在循环内部获取每个意图
                Rag_intent = item["intent"]

                # 获取对应的头像
                avatar = self.intent_avatar_mapping.get(Rag_intent, self.intent_avatar_mapping["其他"])

                #保存结果
                identified_intents.append({
                    "intent": Rag_intent,
                    "avatar": avatar
                })

                #4.返回包含结果的列表
            if not identified_intents:
                 return {
                "success": False,
                "results": [],
                "message": "未能识别出任何有效意图"
                       }

            return {
               "success": True,
               "results": identified_intents,  # 返回包含一个或多个结果的列表
               "message": f"成功识别出 {len(identified_intents)} 个意图"
                 }

        except Exception as e:
         # 保持异常处理不变
                 return {
                     "success": False,
                     "results": [],
                     "error": str(e),
                        "message": "意图识别过程中发生未知错误"
                 }
    # 在process_question_with_intent方法中，找到校园知识问答的处理部分
    # 将原有的逻辑替换为：
    


if __name__ == "__main__":
    try:
        agent = InteractiveAgent()
        agent.chat() 
        
    except KeyboardInterrupt:
        print("\n 程序被用户中断，再见！")
    except Exception as e:
        print(f"程序运行失败: {e}")
