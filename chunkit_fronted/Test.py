r"""
Test.py
一体化测试脚本：一次性实现整个项目的功能。
- 知识库处理：支持校园领域的文字+图片处理（MultiRAG），其他领域文字处理（builder）。
- 问答反馈：检索展示匹配的文字与图片路径 + 流式问答输出。
- 统一命令行：build/insert/query。

使用示例：
1）构建校园（含图片）：
   python Test.py build --domain campus --reset

2）增量插入校园：
   python Test.py insert --domain campus

3）查询并流式回答（校园，展示匹配源）：
   python Test.py query --domain campus --question "校园邮箱如何使用" --topk 5 --show_sources

4）查询并流式回答（其他领域）：
   python Test.py query --domain paper --question "如何写摘要" --topk 5 --show_sources
"""

import argparse
import sys
import time
import re
from typing import List, Dict, Any

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
    r"""从流式段落文本中解析出图片路径（兼容两种提示格式）。
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


def pretty_print_matches(results: List[Dict[str, Any]]) -> None:
    """打印 MultiRAG.retrieve 的匹配结果（含文本与图片）。"""
    if not results:
        print("未检索到相关内容。")
        return
    print(f"\n检索到 {len(results)} 个相关结果：")
    for i, r in enumerate(results, 1):
        rtype = r.get('type', 0)
        content = r.get('document', '')
        source = r.get('source', '')
        label = '图片' if rtype == 1 else '文字'
        print(f"\n—— 结果 {i}（{label}）——")
        print(content[:300])
        if rtype == 1 and source:
            print(f"图片路径: {source}")


# ----------------------------
# 构建/插入：校园（含图片） + 其他领域（文本）
# ----------------------------

def build_knowledge_base(domain: str, reset: bool = False) -> None:
    """构建指定领域的知识库。
    - campus：使用 MultiRAG，处理文字+图片，reset=True 将重置索引。
    - 其他领域：使用 builder 的各领域类，仅处理文字。
    - all：依次构建 campus/paper/fitness/psychology。
    """
    start = time.time()
    domain = domain.lower()

    if domain == 'campus':
        print("[构建-校园] 使用 MultiRAG 处理文字+图片...")
        rag = MultiRAG(
            index_path=str(CAMPUS_INDEX_DIR),
            collection_name='campus_docs',
            image_output_dir=str(CAMPUS_IMAGES_PATH),
            image_mapping_file=str(CAMPUS_IMAGES_MAPPING_PATH),
        )
        # MultiRAG.build() 内部始终重置索引（reset=True）
        rag.build(str(CAMPUS_DOCS_DIR))

    elif domain == 'all':
        print("[构建-全部] 依次构建 campus/paper/fitness/psychology...")
        # 校园（含图片）
        rag = MultiRAG(
            index_path=str(CAMPUS_INDEX_DIR),
            collection_name='campus_docs',
            image_output_dir=str(CAMPUS_IMAGES_PATH),
            image_mapping_file=str(CAMPUS_IMAGES_MAPPING_PATH),
        )
        rag.build(str(CAMPUS_DOCS_DIR))
        # 其他领域（文本）
        from builder import PaperAssistant, FitnessDietAssistant, PsychologyAssistant
        tasks = [
            ('paper', PaperAssistant, PAPER_DOCS_DIR),
            ('fitness', FitnessDietAssistant, FITNESS_DOCS_DIR),
            ('psychology', PsychologyAssistant, PSYCHOLOGY_DOCS_DIR),
        ]
        for name, cls, folder in tasks:
            print(f"[构建-{name}] 使用 builder 处理文字...")
            kb = cls()
            kb.process_folder(folder_name=str(folder), reset=reset)

    elif domain in ('paper', 'fitness', 'psychology'):
        print(f"[构建-{domain}] 使用 builder 处理文字...")
        # 延迟导入，避免不必要的依赖加载
        from builder import PaperAssistant, FitnessDietAssistant, PsychologyAssistant, CampusQnA
        agent_map = {
            'paper': (PaperAssistant, PAPER_DOCS_DIR),
            'fitness': (FitnessDietAssistant, FITNESS_DOCS_DIR),
            'psychology': (PsychologyAssistant, PSYCHOLOGY_DOCS_DIR),
        }
        cls, folder = agent_map[domain]
        kb = cls()
        kb.process_folder(folder_name=str(folder), reset=reset)
    else:
        print(f"不支持的领域: {domain}")
        return

    print(f"构建完成，耗时 {time.time() - start:.2f}s。")


def insert_knowledge_base(domain: str, override_source: str | None = None) -> None:
    """增量插入指定领域的知识库。
    - campus：使用 MultiRAG.insert，处理文字+图片的增量（并更新图片映射）。
    - 其他领域：当前仅支持校园图片增量，其它领域建议通过 builder 再次构建。
    """
    start = time.time()
    domain = domain.lower()

    if domain == 'campus':
        print("[增量-校园] 使用 MultiRAG 处理文字+图片的增量...")
        rag = MultiRAG(
            index_path=str(CAMPUS_INDEX_DIR),
            collection_name='campus_docs',
            image_output_dir=str(CAMPUS_IMAGES_PATH),
            image_mapping_file=str(CAMPUS_IMAGES_MAPPING_PATH),
        )
        src = override_source or str(CAMPUS_DOCS_DIR)
        rag.insert(src)
    else:
        print("目前增量插入仅支持校园（含图片）。其他领域请使用 build --reset 进行重建。")

    print(f"增量完成，耗时 {time.time() - start:.2f}s。")


# ----------------------------
# 查询与流式问答
# ----------------------------

def query_and_stream_answer(domain: str, question: str, topk: int = 5, show_sources: bool = True) -> None:
    """查询检索并流式输出答案。根据领域自动选择实现方式。"""
    domain = domain.lower()
    print(f"\n[查询] 领域: {domain} | 问题: {question} | topk: {topk}")
    candidate_image_paths: List[str] = []

    if domain == 'campus':
        # 1) 检索（含图片路径）
        rag = MultiRAG(
            index_path=str(CAMPUS_INDEX_DIR),
            collection_name='campus_docs',
            image_output_dir=str(CAMPUS_IMAGES_PATH),
            image_mapping_file=str(CAMPUS_IMAGES_MAPPING_PATH),
        )
        results = rag.retrieve(question, topk=topk)
        if show_sources:
            pretty_print_matches(results)

        # 2) 流式回答（由 callback.LLM_model.retrieve_and_answer 执行多模检索增强）
        print("\n流式回答（校园）：\n")
        llm = CampusLLM()
        llm.start_LLM()
        full_text = ""
        try:
            for para in llm.retrieve_and_answer(question, top_k=max(5, topk)):
                print(para)
                full_text += (para + "\n")
                # 从段落中解析可能出现的图片路径
                candidate_image_paths.extend(parse_image_paths_from_text(para))
        except Exception as e:
            print(f"LLM 调用失败：{e}")

    elif domain == 'all':
        print("\n[批量查询] 依次在 campus/paper/fitness/psychology 检索并回答。\n")
        # 校园（含图片）
        print("\n===== 领域: campus =====")
        rag = MultiRAG(
            index_path=str(CAMPUS_INDEX_DIR),
            collection_name='campus_docs',
            image_output_dir=str(CAMPUS_IMAGES_PATH),
            image_mapping_file=str(CAMPUS_IMAGES_MAPPING_PATH),
        )
        results = rag.retrieve(question, topk=topk)
        if show_sources:
            pretty_print_matches(results)
        print("\n流式回答（校园）：\n")
        llm = CampusLLM()
        llm.start_LLM()
        try:
            for para in llm.retrieve_and_answer(question, top_k=max(5, topk)):
                print(para)
                candidate_image_paths.extend(parse_image_paths_from_text(para))
        except Exception as e:
            print(f"LLM 调用失败：{e}")

        # 其他领域（文本）
        rag_map = {
            'paper': RAG_paper,
            'fitness': RAG_fitness,
            'psychology': RAG_psychology,
        }
        for d in ('paper', 'fitness', 'psychology'):
            print(f"\n===== 领域: {d} =====")
            rag_cls = rag_map[d]
            rag_agent = rag_cls()
            chunks = retrieve_relevant_chunks(
                user_query=question,
                vector_store=rag_agent.vector_store,
                embedding_model=rag_agent.model,
                cross_encoder1=rag_agent.cross_encoder,
            )
            if show_sources:
                print(f"\n检索到 {len(chunks)} 个相关文字片段：")
                for i, c in enumerate(chunks, 1):
                    print(f"\n—— 片段 {i} ——\n{c[:300]}")
            print("\n流式回答：\n")
            try:
                for para in rag_agent.call_RAG_stream(question):
                    print(para)
            except Exception as e:
                print(f"LLM 调用失败：{e}")

    elif domain in ('paper', 'fitness', 'psychology', 'compus'):
        # 1) 检索（文字片段）并打印
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
        if show_sources:
            print(f"\n检索到 {len(chunks)} 个相关文字片段：")
            for i, c in enumerate(chunks, 1):
                print(f"\n—— 片段 {i} ——\n{c[:300]}")

        # 2) 流式回答
        print("\n流式回答：\n")
        try:
            for para in rag_agent.call_RAG_stream(question):
                print(para)
        except Exception as e:
            print(f"LLM 调用失败：{e}")
    else:
        print(f"不支持的领域: {domain}")
        return

    # 汇总可能出现的图片地址（仅校园有效）：
    if candidate_image_paths:
        uniq = []
        for p in candidate_image_paths:
            if p not in uniq:
                uniq.append(p)
        print("\n可能引用的图片地址（解析自回答文本）：")
        for p in uniq:
            print(p)


# ----------------------------
# 交互模式与命令行入口
# ----------------------------

def interactive_main():
    print("\n进入交互模式：请选择操作\n")
    def prompt_choice(msg, choices, default=None):
        while True:
            s = input(f"{msg} ({'/'.join(choices)})" + (f" [默认: {default}]" if default else "") + ": ").strip().lower()
            if not s and default:
                return default
            if s in choices:
                return s
            print("无效选项，请重试。")

    def prompt_yes_no(msg, default=True):
        while True:
            s = input(f"{msg} (y/n) [默认: {'y' if default else 'n'}]: ").strip().lower()
            if not s:
                return default
            if s in ('y', 'yes'):
                return True
            if s in ('n', 'no'):
                return False
            print("请输入 y 或 n。")

    def prompt_int(msg, default=5):
        while True:
            s = input(f"{msg} [默认: {default}]: ").strip()
            if not s:
                return default
            try:
                v = int(s)
                return v
            except Exception:
                print("请输入整数。")

    while True:
        print("""
======== 菜单 ========
1) 构建知识库（campus/paper/fitness/psychology/all）
2) 增量插入校园知识库（文+图）
3) 检索并流式问答（campus/paper/fitness/psychology/compus/all）
4) 退出
=====================
""")
        choice = input("请输入选项编号: ").strip()

        if choice == '1':
            domain = prompt_choice("选择领域", ['campus', 'paper', 'fitness', 'psychology', 'all'], default='campus')
            reset = prompt_yes_no("是否重置后重建（仅文本构建有效，校园 MultiRAG 始终重置）", default=False)
            try:
                build_knowledge_base(domain, reset=reset)
            except Exception as e:
                print(f"构建失败：{e}")

        elif choice == '2':
            print("当前仅支持校园领域的增量插入（文+图）。")
            src = input("可选：覆盖默认校园知识库目录路径（留空使用默认）: ").strip()
            try:
                insert_knowledge_base('campus', override_source=(src or None))
            except Exception as e:
                print(f"增量插入失败：{e}")

        elif choice == '3':
            domain = prompt_choice("选择领域", ['campus', 'paper', 'fitness', 'psychology', 'compus', 'all'], default='campus')
            question = input("请输入问题: ").strip()
            if not question:
                print("问题不能为空。")
                continue
            topk = prompt_int("TopK（检索数量）", default=5)
            show_sources = prompt_yes_no("是否显示检索源", default=True)
            try:
                query_and_stream_answer(domain, question, topk=topk, show_sources=show_sources)
            except Exception as e:
                print(f"查询失败：{e}")

        elif choice in ('4', 'q', 'quit', 'exit'):
            print("已退出交互模式。")
            break
        else:
            print("无效选项，请输入 1/2/3/4。")


def main():
    parser = argparse.ArgumentParser(description="一体化测试：知识库处理（文+图）与问答反馈")
    parser.add_argument('--interactive', action='store_true', help='进入交互模式')
    subparsers = parser.add_subparsers(dest='command')

    # build
    p_build = subparsers.add_parser('build', help='构建指定领域的知识库')
    p_build.add_argument('--domain', choices=['campus', 'paper', 'fitness', 'psychology', 'all'], default='campus')
    p_build.add_argument('--reset', action='store_true', help='重置索引后重建（仅文本构建有效）')

    # insert
    p_insert = subparsers.add_parser('insert', help='增量插入校园知识库（文+图）')
    p_insert.add_argument('--domain', choices=['campus'], default='campus')
    p_insert.add_argument('--source', help='覆盖默认校园知识库目录（可选）')

    # query
    p_query = subparsers.add_parser('query', help='检索并流式问答')
    p_query.add_argument('--domain', choices=['campus', 'paper', 'fitness', 'psychology', 'compus', 'all'], default='campus')
    p_query.add_argument('--question', required=True)
    p_query.add_argument('--topk', type=int, default=5)
    p_query.add_argument('--show_sources', action='store_true')

    args = parser.parse_args()

    if args.interactive or args.command is None:
        interactive_main()
    elif args.command == 'build':
        build_knowledge_base(args.domain, reset=args.reset)
    elif args.command == 'insert':
        insert_knowledge_base(args.domain, override_source=args.source)
    elif args.command == 'query':
        query_and_stream_answer(args.domain, args.question, topk=args.topk, show_sources=args.show_sources)
    else:
        parser.print_help()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"运行出错: {e}")
        sys.exit(1)