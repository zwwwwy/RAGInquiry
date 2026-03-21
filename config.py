config_path_name = ".config"
md5_path = f"./{config_path_name}/md5_text.txt"

# chroma相关
collection_name = "rag"  # 数据库名
persist_directory = "./chromaDb"  # 数据库本地存储文件夹
history_database_path = "sqlite:///history/chat_history.db"  # 历史记录数据库
embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"


# 文本分割相关
chunk_size = 1000  # 分割后最大文本长度
chunk_overlap = 100  # 连续文本段间字符重叠数量
separators = ["\n", "\n\n", ".", "?", "!", "。", "？", "！", ",", "，", ""]
split_threshold = 1000

search_num = 40  # 输入向量匹配数
bm25_k = 40
rerank_n = 10

sim_threshold = 0.7  # 切分chunk时语义相似度阈值
min_chunk_len = 3    # chunk含有最短句子数量
max_chunk_len = 10

doc_tags = ["模板文件", "招股说明书", "合同", "资产负债表", "利润表", "现金流量表", "审计报告", "支持性文件"]