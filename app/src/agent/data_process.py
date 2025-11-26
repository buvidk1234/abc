"""
TODO:
多模态解析 (Advanced Parsing)
父子索引 (Parent-Child Indexing)
语义切分 (Semantic Chunking)
"""

# ingestion.py (数据写入脚本)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# 1. 加载和切分
loader = PyPDFLoader("company_handbook.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(docs)

# 2. 初始化完全相同的模型
dense_embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-m3")
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")

# 3. 写入 Qdrant
url = "http://localhost:6333"
QdrantVectorStore.from_documents(
    documents=splits,
    embedding=dense_embeddings,      # 生成稠密向量
    sparse_embedding=sparse_embeddings, # 生成稀疏向量 (关键!)
    url=url,
    collection_name="enterprise_knowledge_hybrid",
    retrieval_mode="hybrid"
)

print("✅ 数据入库完成！稠密+稀疏向量已生成。")