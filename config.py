config_path_name = ".config"
md5_path = f"./{config_path_name}/md5_text.txt"

# chroma相关
collection_name = "rag"  # 数据库名
persist_directory = "./chromaDb"  # 数据库本地存储文件夹
embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"
model_api = "sk-9b5057b6df9c4309a6146c115fcbc212"

# 文本分割相关
chunk_size = 1000  # 分割后最大文本长度
chunk_overlap = 100  # 连续文本段间字符重叠数量
separators = ["\n", "\n\n", ".", "?", "!", "。", "？", "！", ",", "，", ""]
split_threshold = 1000

search_num = 2  # 输入向量匹配数
