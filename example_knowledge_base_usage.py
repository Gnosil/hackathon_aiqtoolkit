# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import asyncio
from langchain_milvus import Milvus
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# 加载环境变量
try:
    load_dotenv()
except ImportError:
    logger.warning("未安装python-dotenv，但仍可继续")


def create_knowledge_base_example():
    """创建知识库的示例代码"""
    print("\n========== 创建知识库示例 ==========")
    print("以下是使用create_knowledge_base.py创建知识库的命令示例：\n")
    
    print("# 从URL创建知识库")
    print("python create_knowledge_base.py --mode url \n")
    print("    --urls https://example.com/article1 https://example.com/article2 \n")
    print("    --collection_name my_web_knowledge \n")
    print("    --milvus_uri http://localhost:19530 \n")
    print("    --embedding_model nvidia/nv-embedqa-e5-v5 \n")
    print("    --clean_cache\n")
    
    print("# 从URL文件列表创建知识库")
    print("python create_knowledge_base.py --mode url \n")
    print("    --url_file urls.txt \n")
    print("    --collection_name my_web_knowledge \n")
    print("    --milvus_uri http://localhost:19530\n")
    
    print("# 从本地文件创建知识库")
    print("python create_knowledge_base.py --mode file \n")
    print("    --file_paths documents/ my_document.pdf \n")
    print("    --collection_name my_file_knowledge \n")
    print("    --milvus_uri http://localhost:19530\n")


def query_knowledge_base(collection_name, milvus_uri="http://localhost:19530", query=""):
    """查询已创建的知识库"""
    if not query:
        query = "请简要总结知识库中的主要内容？"
    
    print(f"\n========== 查询知识库 '{collection_name}' ==========")
    print(f"查询内容: {query}\n")
    
    try:
        # 初始化嵌入模型
        embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", truncate="END")
        
        # 连接到Milvus向量存储
        vector_store = Milvus(
            embedding_function=embedder,
            collection_name=collection_name,
            connection_args={"uri": milvus_uri},
        )
        
        # 检查集合是否存在
        if not vector_store.col:
            print(f"错误: 集合 '{collection_name}' 不存在")
            return None
        
        # 创建检索器
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # 设置提示模板
        template = """使用提供的上下文来回答问题。如果上下文没有相关信息，直接说不知道。

上下文:
{context}

问题:
{question}

回答:
"""
        prompt = ChatPromptTemplate.from_template(template)
        
        # 选择LLM模型
        # 注意：这里默认使用OpenAI的模型，您可以根据需要更改为其他模型
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        # 构建检索-生成链
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # 执行查询
        print("正在查询知识库...")
        result = rag_chain.invoke(query)
        
        # 输出结果
        print(f"\n查询结果:\n{result}")
        
        # 输出检索到的相关文档
        print("\n检索到的相关文档:")
        docs = retriever.get_relevant_documents(query)
        for i, doc in enumerate(docs, 1):
            # 提取文档的元数据和部分内容
            metadata = doc.metadata
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            
            print(f"\n相关文档 {i}:")
            if "source" in metadata:
                print(f"来源: {metadata['source']}")
            print(f"内容预览: {content_preview}")
            
        return result
        
    except Exception as e:
            print(f"查询知识库时出错: {e}")
            return None


def main():
    # 显示创建知识库的示例命令
    create_knowledge_base_example()
    
    # 提供一个查询示例
    user_input = input("\n是否要进行知识库查询示例？(y/n): ")
    if user_input.lower() == 'y':
        collection_name = input("请输入知识库的Milvus集合名称: ")
        milvus_uri = input("请输入Milvus服务器URI [默认: http://localhost:19530]: ") or "http://localhost:19530"
        custom_query = input("请输入查询内容 [留空使用默认查询]: ")
        
        query_knowledge_base(
            collection_name=collection_name,
            milvus_uri=milvus_uri,
            query=custom_query if custom_query else ""
        )
    
    print("\n知识库创建和查询示例结束。")


if __name__ == "__main__":
    main()