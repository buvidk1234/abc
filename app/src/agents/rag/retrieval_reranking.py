"""
混合检索 vector + keyword -> Top 50
多路召回
重排序 reranking -> Top 5
查询重写 (LLM)
"""

import os
from typing import List, Any, TypedDict

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

# --- 关键依赖 ---
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_cohere import CohereRerank


# ==========================================
# 1. 🏆 最佳配置参数 (SOTA Configuration)
# ==========================================

# Qdrant 配置
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = "enterprise_knowledge_hybrid"  # 建议用新名字，因为数据结构变了

# Rerank 配置 (使用最新的 v3.5)
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "your-key")
RERANK_MODEL = "rerank-multilingual-v3.5"  # ✅ 升级到 v3.5

# 检索参数 (漏斗设计)
TOP_K_RECALL = 50  # 混合召回数量
TOP_K_RERANK = 8  # 最终给 LLM 的数量 (v3.5 支持更长的上下文，可以给多点)


# ==========================================
# 2. 🛠️ 初始化 SOTA 检索器
# ==========================================

def get_retriever():
    """
    构建 [BGE-M3 混合召回] + [Cohere v3.5 重排序] 的管道
    """

    # 1. 初始化 Qdrant 客户端
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # 2. 定义 Embedding 模型 (关键步骤)
    # 使用 FastEmbed 运行 BGE-M3，速度极快，无需 GPU
    # 作用: 生成语义向量 (Dense)
    dense_embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-m3"
    )

    # 3. 定义 Sparse Embedding 模型 (关键步骤)
    # 作用: 生成关键词向量 (Sparse/SPLADE)，替代传统的 BM25
    # 使用 Qdrant 推荐的 BM42 (基于 BGE 优化的稀疏模型)
    sparse_embeddings = FastEmbedSparse(
        model_name="Qdrant/bm42-all-minilm-l6-v2-attentions"
    )

    # 4. 初始化向量库 (开启混合检索模式)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=dense_embeddings,
        # ✅ 开启混合检索魔法
        sparse_embedding=sparse_embeddings,
        retrieval_mode="hybrid",
    )

    # 5. 定义基础检索器 (Hybrid Retriever)
    # 这一步会自动并发执行: 语义搜索 + 关键词搜索，并自动融合分数
    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RECALL}
    )

    # 6. 定义重排序器 (Cohere v3.5)
    compressor = CohereRerank(
        cohere_api_key=COHERE_API_KEY,
        model=RERANK_MODEL,
        top_n=TOP_K_RERANK
    )

    # 7. 组装管道
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    return final_retriever


# 全局单例
_GLOBAL_RETRIEVER = get_retriever()


# ==========================================
# 3. 🧩 图节点逻辑
# ==========================================

class GraphState(TypedDict):
    messages: List[Any]
    retrieved_context: str
    source_documents: List[Document]


def retrieve_node(state: GraphState, config: RunnableConfig):
    """
    企业级混合检索节点
    """
    print(f"--- 🚀 开始检索 (Model: BGE-M3 + Cohere v3.5) ---")

    query = state["messages"][-1].content

    try:
        # 这一行代码背后发生了：
        # 1. Query -> BGE-M3 -> [Dense Vector]
        # 2. Query -> BM42   -> [Sparse Vector]
        # 3. Qdrant -> Dense Search + Sparse Search -> Score Fusion -> Top 50
        # 4. Cohere -> Rerank -> Top 8
        docs = _GLOBAL_RETRIEVER.invoke(query)

        # 格式化输出
        context_parts = []
        for i, doc in enumerate(docs):
            # 获取元数据 (假设入库时存了 source 和 page)
            meta = doc.metadata
            source_info = f"{meta.get('source', 'unknown')} (P.{meta.get('page', '?')})"

            # 拼接: [1] 内容 (来源)
            context_parts.append(f"[{i + 1}] {doc.page_content}\n   Source: {source_info}")

        context_str = "\n\n".join(context_parts)

        print(f"✅ 检索成功: 最终保留 {len(docs)} 条高相关文档")

        return {
            "retrieved_context": context_str,
            "source_documents": docs
        }

    except Exception as e:
        print(f"❌ 检索严重错误: {e}")
        # 生产环境建议接入 Sentry 或 Log 监控
        return {
            "retrieved_context": "",
            "source_documents": []
        }
