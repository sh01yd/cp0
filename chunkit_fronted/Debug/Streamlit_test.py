import streamlit as st
import sys
import time
import re
import os
from typing import List, Dict, Any
from pathlib import Path
from Agent_test import InteractiveAgent

# 项目内模块导入
from Utils.Path import (
    PAPER_DOCS_DIR, CAMPUS_DOCS_DIR, FITNESS_DOCS_DIR, PSYCHOLOGY_DOCS_DIR,
    PAPER_INDEX_DIR, CAMPUS_INDEX_DIR, FITNESS_INDEX_DIR, PSYCHOLOGY_INDEX_DIR,
    CAMPUS_IMAGES_PATH, CAMPUS_IMAGES_MAPPING_PATH,
)

from multiRAG import MultiRAG
from callback import LLM_model as CampusLLM  # 校园域：使用MultiRAG + LLM流式回答（含图片信息）

from RAGlibrary import (
    RAG_psychology,
    RAG_fitness,
    RAG_compus,
    RAG_paper,
)
from retrieve_model import retrieve_relevant_chunks

#运行方法;
#cd demo\back-end-python\chunkit_fronted到这个目录
#然后Streamlit run Debug\Streamlit_test.py可以运行此功能

# ----------------------------
# 工具函数
# ----------------------------

def parse_image_paths_from_text(text: str) -> List[str]:
    """从流式段落文本中解析出图片路径（兼容两种提示格式）。
    兼容：[图片地址: D:\...] 或 [地址: D:\...]
    """
    paths = []
    patterns = [r"\[图片地址:\s*([^\]]+)\]", r"\[地址:\s*([^\]]+)\]"]
    for pat in patterns:
        for m in re.finditer(pat, text):
            path = m.group(1).strip()
            if path:
                paths.append(path)
    return paths


def format_matches_for_display(results: List[Dict[str, Any]]) -> str:
    """格式化 MultiRAG.retrieve 的匹配结果（含文本与图片）为显示文本。"""
    if not results:
        return "未检索到相关内容。"
    
    output = f"检索到 {len(results)} 个相关结果：\n\n"
    for i, r in enumerate(results, 1):
        rtype = r.get('type', 0)
        content = r.get('document', '')
        source = r.get('source', '')
        label = '图片' if rtype == 1 else '文字'
        output += f"—— 结果 {i}（{label}）——\n"
        output += content[:300] + "\n"
        if rtype == 1 and source:
            output += f"图片路径: {source}\n"
        output += "\n"
    
    return output


# ----------------------------
# 查询与流式问答
# ----------------------------

def query_and_stream_answer(domain: str, question: str, topk: int = 5, show_sources: bool = True):
    """查询检索并流式输出答案。根据领域自动选择实现方式。返回结果和图片路径。"""
    domain = domain.lower()
    st.write(f"**领域**: {domain} | **问题**: {question} | **topk**: {topk}")
    
    results = []
    candidate_image_paths = []
    answer_text = ""
    
    if domain == 'campus':
        # 1) 检索（含图片路径）
        rag = MultiRAG(
            index_path=str(CAMPUS_INDEX_DIR),
            collection_name='campus_docs',
            image_output_dir=str(CAMPUS_IMAGES_PATH),
            image_mapping_file=str(CAMPUS_IMAGES_MAPPING_PATH),
        )
        results = rag.retrieve(question, topk=topk)
        
        # 2) 流式回答
        llm = CampusLLM()
        llm.start_LLM()
        
        # 创建一个空的占位符用于流式输出
        answer_placeholder = st.empty()
        full_text = ""
        
        try:
            for para in llm.retrieve_and_answer(question, top_k=max(5, topk)):
                full_text += para + "\n"
                answer_placeholder.markdown(full_text)
                # 从段落中解析可能出现的图片路径
                candidate_image_paths.extend(parse_image_paths_from_text(para))
            
            answer_text = full_text
        except Exception as e:
            st.error(f"LLM 调用失败：{e}")
            
    elif domain == 'all':
        st.subheader("批量查询多个领域")
        
        # 校园（含图片）
        st.markdown("### 领域: campus")
        rag = MultiRAG(
            index_path=str(CAMPUS_INDEX_DIR),
            collection_name='campus_docs',
            image_output_dir=str(CAMPUS_IMAGES_PATH),
            image_mapping_file=str(CAMPUS_IMAGES_MAPPING_PATH),
        )
        campus_results = rag.retrieve(question, topk=topk)
        if show_sources:
            st.markdown("#### 检索源:")
            st.markdown(format_matches_for_display(campus_results))
        
        st.markdown("#### 回答:")
        campus_answer_placeholder = st.empty()
        campus_full_text = ""
        
        llm = CampusLLM()
        llm.start_LLM()
        try:
            for para in llm.retrieve_and_answer(question, top_k=max(5, topk)):
                campus_full_text += para + "\n"
                campus_answer_placeholder.markdown(campus_full_text)
                candidate_image_paths.extend(parse_image_paths_from_text(para))
        except Exception as e:
            st.error(f"Campus LLM 调用失败：{e}")
        
        # 其他领域（文本）
        rag_map = {
            'paper': RAG_paper,
            'fitness': RAG_fitness,
            'psychology': RAG_psychology,
        }
        
        for domain_name in ('paper', 'fitness', 'psychology'):
            st.markdown(f"### 领域: {domain_name}")
            rag_cls = rag_map[domain_name]
            rag_agent = rag_cls()
            chunks = retrieve_relevant_chunks(
                user_query=question,
                vector_store=rag_agent.vector_store,
                embedding_model=rag_agent.model,
                cross_encoder1=rag_agent.cross_encoder,
            )
            
            if show_sources:
                st.markdown("#### 检索源:")
                source_text = f"检索到 {len(chunks)} 个相关文字片段：\n\n"
                for i, c in enumerate(chunks, 1):
                    source_text += f"—— 片段 {i} ——\n{c[:300]}\n\n"
                st.markdown(source_text)
            
            st.markdown("#### 回答:")
            domain_answer_placeholder = st.empty()
            domain_full_text = ""
            
            try:
                for para in rag_agent.call_RAG_stream(question):
                    domain_full_text += para + "\n"
                    domain_answer_placeholder.markdown(domain_full_text)
            except Exception as e:
                st.error(f"{domain_name} LLM 调用失败：{e}")
        
        # 合并所有结果
        results = campus_results
        answer_text = "查看各领域的详细回答"
            
    elif domain in ('paper', 'fitness', 'psychology', 'compus'):
        # 1) 检索（文字片段）
        rag_map = {
            'paper': RAG_paper,
            'fitness': RAG_fitness,
            'psychology': RAG_psychology,
            'compus': RAG_compus,
        }
        rag_cls = rag_map[domain]
        rag_agent = rag_cls()
        chunks = retrieve_relevant_chunks(
            user_query=question,
            vector_store=rag_agent.vector_store,
            embedding_model=rag_agent.model,
            cross_encoder1=rag_agent.cross_encoder,
        )
        
        # 2) 流式回答
        answer_placeholder = st.empty()
        full_text = ""
        
        try:
            for para in rag_agent.call_RAG_stream(question):
                full_text += para + "\n"
                answer_placeholder.markdown(full_text)
            
            answer_text = full_text
        except Exception as e:
            st.error(f"LLM 调用失败：{e}")
    
    else:
        st.error(f"不支持的领域: {domain}")
    
    # 去重图片路径
    unique_image_paths = []
    for p in candidate_image_paths:
        if p not in unique_image_paths:
            unique_image_paths.append(p)
    
    return results, answer_text, unique_image_paths


# 设置页面标题和图标
st.set_page_config(
    page_title="智能问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 1. 初始化智能体 ---
# 使用 st.cache_resource 确保模型和类只被初始化一次，提高应用性能
@st.cache_resource
def load_agent():
    """
    加载并初始化 InteractiveAgent。
    此函数的结果将被缓存，避免每次页面刷新时都重新加载模型。
    """
    with st.spinner("正在初始化智能体，请稍候..."):
        try:
            agent = InteractiveAgent()
            return agent
        except Exception as e:
            # 如果初始化失败，显示错误并停止应用
            st.error(f"智能体初始化失败: {e}")
            st.stop()


# --- 2. 侧边栏和模式选择 ---
st.sidebar.title("🛠️ 测试控制台")
st.sidebar.markdown("选择一个接口进行测试，或在主窗口直接开始对话。")

test_mode = st.sidebar.radio(
    "选择测试模式",
    ("完整聊天 (流式)", "仅意图识别", "RAG检索问答")
)

# --- 3. 根据不同模式显示不同界面 ---

# 模式一：完整聊天
if test_mode == "完整聊天 (流式)":
    # 加载智能体实例
    agent = load_agent()
    
    st.title("🤖 多智能体聊天系统")
    st.caption("这是一个用于测试多智能体 RAG 系统的交互界面。")

    # 初始化聊天记录
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 接收用户输入
    if prompt := st.chat_input("请输入您的问题..."):
        # 将用户消息添加到历史记录并显示
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 获取并显示助手回答
        with st.chat_message("assistant"):

            full_response = ""

            try:
                # 调用核心方法，获取段落生成器
                stream_generator = agent.process_question_with_full_response(prompt, stream_mode=True)

                # 遍历生成器，处理每个返回的段落
                for chunk in stream_generator:
                    if chunk.get("type") == "content":
                        # 从 chunk 中获取头像和段落内容
                        avatar = chunk.get("avatar", "🤖")
                        paragraph = chunk.get("delta", "")

                        if paragraph:
                            # 1. 按照你要求的格式构建输出行
                            output_line = f"头像: {avatar} | 回答段落: {paragraph}"

                            # 2. 使用 st.markdown 直接显示这一行
                            st.markdown(output_line)

                            # 3. 将生成的行添加到完整回复中，用于历史记录
                            full_response += output_line + "\n"

                    elif chunk.get("type") == "error":
                        error_message = chunk.get("message", "未知错误")
                        st.error(f"处理时发生错误: {error_message}")
                        full_response += f"\n\n**错误**: {error_message}"

            except Exception as e:
                st.error(f"调用智能体时发生严重错误: {e}")
                full_response = "抱歉，处理您的请求时发生了严重错误。"
                st.markdown(full_response)
            # --- 核心逻辑修改结束 ---

        # 将助手的完整回答添加到历史记录
        st.session_state.messages.append({"role": "assistant", "content": full_response})


# 模式二：仅意图识别
elif test_mode == "仅意图识别":
    # 加载智能体实例
    agent = load_agent()
    
    st.title("🎯 意图识别接口测试")
    st.info("在这里，你可以输入问题，系统将只调用 `predict_intent_only` 方法并返回识别出的意图。")

    user_input = st.text_area("输入要识别的问题:", height=100,
                              placeholder="例如：我想咨询心理方面的问题，并了解一下校园图书馆的开放时间。")

    if st.button("识别意图", use_container_width=True):
        if user_input:
            with st.spinner("正在识别..."):
                result = agent.predict_intent_only(user_input)

                st.subheader("识别结果 (原始JSON):")
                st.json(result)

                st.subheader("格式化展示:")
                if result.get("success") and result.get("results"):
                    for intent_info in result["results"]:
                        st.success(
                            f"**意图:** {intent_info.get('intent', 'N/A')} "
                            f"| **头像:** {intent_info.get('avatar', 'N/A')}"
                        )
                else:
                    st.warning(f"未能成功识别意图。消息: {result.get('message', '无')}")
        else:
            st.warning("请输入问题后再进行识别。")

# 模式三：RAG检索问答
elif test_mode == "RAG检索问答":
    st.title("🔍 RAG检索问答系统")
    st.markdown("基于RAG的多领域智能问答系统，支持文本和图片检索")
    
    # 侧边栏设置
    domain = st.sidebar.selectbox(
        "选择领域",
        ["campus", "paper", "fitness", "psychology", "compus", "all"],
        index=0,
        help="选择要查询的知识领域，选择'all'将在所有领域中查询"
    )
    
    topk = st.sidebar.slider(
        "检索数量 (TopK)",
        min_value=1,
        max_value=20,
        value=5,
        help="设置检索返回的结果数量"
    )
    
    show_sources = st.sidebar.checkbox(
        "显示检索源",
        value=True,
        help="是否显示检索到的原始内容"
    )
    
    # 主界面
    question = st.text_input("请输入您的问题:", placeholder="例如：校园邮箱如何使用？")
    
    if st.button("提交问题") and question:
        with st.spinner("正在思考中..."):
            if domain != "all":
                # 创建两列布局
                col1, col2 = st.columns([3, 1])
                
                # 执行查询
                results, answer, image_paths = query_and_stream_answer(
                    domain=domain,
                    question=question,
                    topk=topk,
                    show_sources=show_sources
                )
                
                # 在左侧列显示答案
                with col1:
                    st.subheader("回答")
                    st.markdown(answer)
                
                # 在右侧列显示检索源和图片
                with col2:
                    if show_sources and results:
                        st.subheader("检索源")
                        st.markdown(format_matches_for_display(results))
                    
                    # 显示图片
                    if image_paths:
                        st.subheader("相关图片")
                        for img_path in image_paths:
                            if os.path.exists(img_path):
                                st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)
                            else:
                                st.warning(f"图片不存在: {img_path}")
            else:
                # 全选模式下直接使用单列布局
                results, answer, image_paths = query_and_stream_answer(
                    domain=domain,
                    question=question,
                    topk=topk,
                    show_sources=show_sources
                )
                
                # 显示图片
                if image_paths:
                    st.subheader("相关图片")
                    image_cols = st.columns(min(3, len(image_paths)))
                    for i, img_path in enumerate(image_paths):
                        if os.path.exists(img_path):
                            with image_cols[i % 3]:
                                st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)
                        else:
                            st.warning(f"图片不存在: {img_path}")