import os
from llama_index.core import SimpleDirectoryReader
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import Document
from llama_index.core.embeddings import BaseEmbedding
import numpy as np

load_dotenv()

# 定义嵌入模型名称
MODEL_NAME = "nvidia/nv-embedqa-e5-v5"

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构建相对于脚本的documents目录路径
pdf_directory = os.path.join(script_dir, "documents")

print(f"尝试读取文档目录: {pdf_directory}")

# 检查目录是否存在
if not os.path.exists(pdf_directory):
    print(f"错误: 目录 {pdf_directory} 不存在")
    # 尝试使用备选路径（用于直接在hackathon_aiqtoolkit目录下运行的情况）
    pdf_directory = os.path.join(os.getcwd(), "documents")
    print(f"尝试备选路径: {pdf_directory}")

if os.path.exists(pdf_directory):
    documents = SimpleDirectoryReader(input_dir=pdf_directory).load_data()
    print(f"成功加载 {len(documents)} 个文档")
    
    splitter = TokenTextSplitter(
        chunk_size=512,  # 每个块的目标token数，与模型最大输入保持一致
        chunk_overlap=50,   # 相邻块之间的重叠token数
        backup_separators=["。", "！", "？"]  # 中文文档可添加这些分隔符
    )

    # 使用SimpleDirectoryReader读取PDF文件
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"文档分割为 {len(nodes)} 个节点")

    # 创建NVIDIA嵌入模型
    print("初始化NVIDIA嵌入模型...")
    nvidia_embedder = NVIDIAEmbeddings(model=MODEL_NAME, truncate="END")
    
    # 配置 Milvus 向量存储（维度需与模型输出一致！）
    print("连接Milvus向量存储...")
    vector_store = MilvusVectorStore(
        uri="http://127.0.0.1:19530",
        dim=1024,
        collection_name="my_knowledge_base",
        overwrite=True
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 创建索引，使用nvidia_embedder作为嵌入模型
    print("创建向量索引...")
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=nvidia_embedder  # 直接传入嵌入模型
    )
    print("索引创建成功！")
else:
    print(f"错误: 无法找到文档目录。请检查路径设置。")