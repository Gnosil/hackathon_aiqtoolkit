# 使用Milvus创建自定义知识库

本指南将帮助您使用提供的脚本在NeMo-Agent-Toolkit中创建和使用自定义Milvus知识库。

## 准备工作

在开始之前，请确保您已完成以下准备工作：

1. 已安装NeMo-Agent-Toolkit及其依赖
2. Milvus服务正在运行（默认连接到 http://localhost:19530）
3. 如需使用网页抓取功能，请确保网络连接正常
4. 如需使用NVIDIA嵌入模型，请确保您有有效的API密钥

## 提供的脚本文件

1. **create_knowledge_base.py** - 主要工具脚本，用于从URL或本地文件创建Milvus知识库
2. **example_knowledge_base_usage.py** - 示例脚本，展示如何使用上述工具和查询知识库
3. **README_knowledge_base.md** - 本文档，提供使用指南

## 创建知识库

### 从URL创建知识库

您可以直接指定URL列表或从文件中读取URL列表：

```bash
# 直接指定URL列表
python create_knowledge_base.py --mode url \
    --urls https://example.com/article1 https://example.com/article2 \
    --collection_name my_web_knowledge \
    --milvus_uri http://localhost:19530 \
    --embedding_model nvidia/nv-embedqa-e5-v5 \
    --clean_cache

# 从文件中读取URL列表
python create_knowledge_base.py --mode url \
    --url_file urls.txt \
    --collection_name my_web_knowledge \
    --milvus_uri http://localhost:19530
```

其中，`urls.txt`文件应包含每行一个URL：

```
https://example.com/article1
https://example.com/article2
https://example.com/article3
```

### 从本地文件创建知识库

您可以指定单个文件、多个文件或整个目录：

```bash
# 指定单个文件
python create_knowledge_base.py --mode file \
    --file_paths my_document.pdf \
    --collection_name my_file_knowledge \
    --milvus_uri http://localhost:19530

# 指定多个文件
python create_knowledge_base.py --mode file \
    --file_paths document1.pdf document2.txt \
    --collection_name my_file_knowledge \
    --milvus_uri http://localhost:19530

# 指定整个目录
python create_knowledge_base.py --mode file \
    --file_paths documents/ \
    --collection_name my_folder_knowledge \
    --milvus_uri http://localhost:19530
```

支持的文件格式包括：PDF、TXT、MD、HTML等。

## 参数说明

### 通用参数

- `--mode` - 创建知识库的模式（`url` 或 `file`）
- `--milvus_uri` - Milvus服务器URI，默认：`http://localhost:19530`
- `--collection_name` - Milvus集合名称，默认：`my_knowledge_base`
- `--embedding_model` - 嵌入模型名称，默认：`nvidia/nv-embedqa-e5-v5`

### URL模式特有参数

- `--urls` - 要抓取的URL列表
- `--url_file` - 包含URL列表的文件路径
- `--clean_cache` - 是否清理缓存的HTML文件

### 文件模式特有参数

- `--file_paths` - 要处理的文件或目录路径列表

## 查询知识库

您可以使用 `example_knowledge_base_usage.py` 脚本查询已创建的知识库：

```bash
python example_knowledge_base_usage.py
```

运行脚本后，按照提示输入Milvus集合名称、Milvus服务器URI和查询内容。

### 自定义查询示例

如果您想在自己的代码中查询知识库，可以参考以下代码片段：

```python
from langchain_milvus import Milvus
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# 初始化嵌入模型
embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", truncate="END")

# 连接到Milvus向量存储
vector_store = Milvus(
    embedding_function=embedder,
    collection_name="my_knowledge_base",
    connection_args={"uri": "http://localhost:19530"},
)

# 创建检索器
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 执行检索
query = "您的问题"
relevant_docs = retriever.get_relevant_documents(query)

# 处理检索结果
for doc in relevant_docs:
    print(f"来源: {doc.metadata.get('source', '未知')}")
    print(f"内容: {doc.page_content[:200]}...")
    print("---")
```

## 常见问题解答

### 1. Milvus连接失败

- 确保Milvus服务正在运行
- 检查Milvus服务器URI是否正确
- 验证网络连接和防火墙设置

### 2. 嵌入模型错误

- 确保您有有效的NVIDIA API密钥
- 检查网络连接是否正常
- 尝试使用其他可用的嵌入模型

### 3. 文件加载失败

- 确保文件路径正确
- 检查文件格式是否受支持
- 验证您是否有文件读取权限

### 4. URL抓取失败

- 确保网络连接正常
- 检查URL格式是否正确
- 某些网站可能有防爬虫措施

## 注意事项

1. 创建大型知识库时，可能需要较长时间，请耐心等待
2. Milvus集合名称在同一服务器中必须唯一
3. 使用NVIDIA嵌入模型可能会产生API费用，请留意您的使用情况
4. 建议为不同类型的知识创建不同的集合，以便更好地组织和管理

## 扩展建议

1. 您可以根据需要自定义嵌入模型和Milvus配置参数
2. 对于大型知识库，可以考虑调整文本分割的`chunk_size`和`chunk_overlap`参数
3. 可以将知识库集成到您自己的应用程序或聊天机器人中