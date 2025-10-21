import streamlit as st
import sys
import time
import re
from typing import List, Dict, Any
from pathlib import Path
import os

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


# ----------------------------
# Streamlit 应用界面
# ----------------------------

def main():
    st.set_page_config(
        page_title="智能问答系统",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("智能问答系统")
    st.markdown("基于RAG的多领域智能问答系统，支持文本和图片检索")
    
    # 侧边栏设置
    st.sidebar.title("设置")
    domain = st.sidebar.selectbox(
        "选择领域",
        ["campus", "paper", "fitness", "psychology", "compus"],
        index=0,
        help="选择要查询的知识领域"
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


if __name__ == "__main__":
    main()