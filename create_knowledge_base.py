# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import os
import logging
from uuid import uuid4

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# 导入必要的库
try:
    from langchain_community.document_loaders import BSHTMLLoader, TextLoader, PyPDFLoader
    from langchain_milvus import Milvus
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.node_parser import TokenTextSplitter
    from llama_index.vector_stores.milvus import MilvusVectorStore
    from llama_index.core import VectorStoreIndex, StorageContext
    from dotenv import load_dotenv
    
    # 导入网页抓取相关工具
    from scripts.web_utils import cache_html, get_file_path_from_url, scrape
    
    load_dotenv()
except ImportError as e:
    logger.error(f"导入必要库时出错: {e}")
    logger.info("请确保已安装所有依赖: pip install -r requirements.txt")
    exit(1)


def create_knowledge_base_from_urls(
    urls: list[str], 
    milvus_uri: str, 
    collection_name: str,
    embedding_model: str = "nvidia/nv-embedqa-e5-v5",
    clean_cache: bool = True,
    base_path: str = "./.tmp/data",
    chunk_size: int = 1000,
    chunk_overlap: int = 100
):
    """从URL列表创建知识库到Milvus"""
    logger.info(f"开始从URL创建知识库 '{collection_name}'...")
    
    # 初始化嵌入模型
    embedder = NVIDIAEmbeddings(model=embedding_model, truncate="END")

    # 创建Milvus向量存储
    vector_store = Milvus(
        embedding_function=embedder,
        collection_name=collection_name,
        connection_args={"uri": milvus_uri},
    )

    # 检查集合是否存在
    collection_existed_before = vector_store.col is not None
    
    if collection_existed_before:
        logger.info(f"使用已存在的Milvus集合: {collection_name}")
        try:
            num_entities = vector_store.client.query(collection_name=collection_name, 
                                                      filter="", 
                                                      output_fields=["count(*)"])
            entity_count = num_entities[0]["count(*)"] if num_entities else "unknown number of"
            logger.info(f"集合 '{collection_name}' 包含 {entity_count} 个文档")
        except Exception as e:
            logger.warning(f"无法获取集合信息: {e}")
    else:
        logger.info(f"集合 '{collection_name}' 不存在，将在添加文档时创建")

    # 检查是否有缓存的HTML文件
    filenames = [
        get_file_path_from_url(url, base_path)[0] for url in urls
        if os.path.exists(get_file_path_from_url(url, base_path)[0])
    ]
    urls_to_scrape = [url for url in urls if get_file_path_from_url(url, base_path)[0] not in filenames]
    
    if filenames:
        logger.info(f"从缓存加载 {len(filenames)} 个文件")
    
    # 抓取新的URL
    if urls_to_scrape:
        logger.info(f"正在抓取 {len(urls_to_scrape)} 个URL: {urls_to_scrape}")
        html_data, err = asyncio.run(scrape(urls_to_scrape))
        if err:
            logger.warning(f"抓取失败: {[f['url'] for f in err]}")
        filenames.extend([cache_html(data, base_path)[1] for data in html_data if data.get('content')])

    # 创建文本分割器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )

    # 处理文件并添加到Milvus
    doc_ids = []
    for filename in filenames:
        if not os.path.exists(filename):
            logger.warning(f"文件不存在: {filename}")
            continue
            
        logger.info(f"解析文件: {filename}")
        try:
            # 使用BSHTMLLoader加载HTML文件
            loader = BSHTMLLoader(filename)
            docs = loader.load()
            docs = splitter.split_documents(docs)
            
            if not isinstance(docs, list):
                docs = [docs]
                
            if not docs:
                logger.warning(f"文件没有产生任何文档块: {filename}")
                continue
                
            # 为文档生成唯一ID
            ids = [str(uuid4()) for _ in range(len(docs))]
            logger.info(f"将 {len(docs)} 个文档块添加到Milvus集合 '{collection_name}'")
            
            # 添加文档到Milvus
            added_ids = asyncio.run(vector_store.aadd_documents(documents=docs, ids=ids))
            doc_ids.extend(added_ids)
            
            # 清理缓存
            if clean_cache:
                logger.info(f"删除缓存文件: {filename}")
                os.remove(filename)
                
        except Exception as e:
                logger.error(f"处理文件 {filename} 时出错: {e}")
                continue
        
    # 输出最终状态
    if collection_existed_before:
        logger.info(f"成功向现有集合 '{collection_name}' 添加了 {len(doc_ids)} 个新文档")
    else:
        logger.info(f"成功创建集合 '{collection_name}' 并添加了 {len(doc_ids)} 个文档")
    
    return doc_ids


def create_knowledge_base_from_files(
    file_paths: list[str] or str, 
    milvus_uri: str, 
    collection_name: str,
    embedding_model: str = "nvidia/nv-embedqa-e5-v5",
    chunk_size: int = 512,
    chunk_overlap: int = 50
):
    """从本地文件创建知识库到Milvus (使用LlamaIndex)"""
    logger.info(f"开始从文件创建知识库 '{collection_name}'...")
    
    # 确保file_paths是列表
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    # 检查目录是否存在
    valid_paths = []
    for path in file_paths:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            logger.warning(f"路径不存在: {path}")
    
    if not valid_paths:
        logger.error("没有有效的文件路径")
        return []
    
    # 判断是目录还是文件
    documents = []
    for path in valid_paths:
        if os.path.isdir(path):
            # 如果是目录，使用SimpleDirectoryReader
            logger.info(f"从目录加载文档: {path}")
            dir_docs = SimpleDirectoryReader(input_dir=path).load_data()
            documents.extend(dir_docs)
        else:
            # 如果是单个文件，根据扩展名选择合适的加载器
            ext = os.path.splitext(path)[1].lower()
            logger.info(f"加载单个文件: {path} ({ext})")
            if ext == '.pdf':
                loader = PyPDFLoader(path)
                docs = loader.load()
                documents.extend(docs)
            elif ext in ['.txt', '.md']:
                loader = TextLoader(path, encoding='utf-8')
                docs = loader.load()
                documents.extend(docs)
            elif ext in ['.html', '.htm']:
                loader = BSHTMLLoader(path)
                docs = loader.load()
                documents.extend(docs)
            else:
                logger.warning(f"不支持的文件格式: {ext}")
    
    if not documents:
        logger.error("未能加载任何文档")
        return []
    
    logger.info(f"成功加载 {len(documents)} 个文档")
    
    # 创建文本分割器
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,  # 每个块的目标token数
        chunk_overlap=chunk_overlap,  # 相邻块之间的重叠token数
        backup_separators=["。", "！", "？"]  # 中文文档的分隔符
    )
    
    # 分割文档
    nodes = splitter.get_nodes_from_documents(documents)
    logger.info(f"文档分割为 {len(nodes)} 个节点")
    
    # 创建NVIDIA嵌入模型
    logger.info("初始化NVIDIA嵌入模型...")
    nvidia_embedder = NVIDIAEmbeddings(model=embedding_model, truncate="END")
    
    # 配置Milvus向量存储
    logger.info("连接Milvus向量存储...")
    vector_store = MilvusVectorStore(
        uri=milvus_uri,
        dim=1024,  # 维度需与模型输出一致
        collection_name=collection_name,
        overwrite=False  # 设置为True会覆盖现有集合
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 创建索引
    logger.info("创建向量索引...")
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=nvidia_embedder
    )
    
    logger.info("索引创建成功！")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="使用Milvus创建自定义知识库的工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 选择模式：URL或文件
    parser.add_argument(
        "--mode", 
        choices=["url", "file"], 
        default="url",
        help="创建知识库的模式: 从URL或本地文件"
    )
    
    # 通用参数
    parser.add_argument(
        "--milvus_uri", 
        default="http://localhost:19530",
        help="Milvus服务器URI"
    )
    parser.add_argument(
        "--collection_name", 
        default="my_knowledge_base",
        help="Milvus集合名称"
    )
    parser.add_argument(
        "--embedding_model", 
        default="nvidia/nv-embedqa-e5-v5",
        help="嵌入模型名称"
    )
    
    # URL模式参数
    parser.add_argument(
        "--urls", 
        nargs="*", 
        default=[],
        help="要抓取的URL列表"
    )
    parser.add_argument(
        "--url_file", 
        help="包含URL列表的文件路径，每行一个URL"
    )
    parser.add_argument(
        "--clean_cache", 
        action="store_true",
        help="是否清理缓存的HTML文件"
    )
    
    # 文件模式参数
    parser.add_argument(
        "--file_paths", 
        nargs="*", 
        default=[],
        help="要处理的文件或目录路径列表"
    )
    
    args = parser.parse_args()
    
    # 处理URL来源
    urls = args.urls
    if args.url_file and os.path.exists(args.url_file):
        with open(args.url_file, 'r', encoding='utf-8') as f:
            urls.extend([line.strip() for line in f if line.strip()])
    
    # 根据模式执行相应的函数
    if args.mode == "url":
        if not urls:
            parser.error("在URL模式下，必须提供--urls参数或--url_file参数")
        create_knowledge_base_from_urls(
            urls=urls,
            milvus_uri=args.milvus_uri,
            collection_name=args.collection_name,
            embedding_model=args.embedding_model,
            clean_cache=args.clean_cache
        )
    else:  # file模式
        if not args.file_paths:
            parser.error("在文件模式下，必须提供--file_paths参数")
        create_knowledge_base_from_files(
            file_paths=args.file_paths,
            milvus_uri=args.milvus_uri,
            collection_name=args.collection_name,
            embedding_model=args.embedding_model
        )


if __name__ == "__main__":
    main()