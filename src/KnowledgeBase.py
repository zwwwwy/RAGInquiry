import sys
import os

from numpy import record

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

import os
import config
import hashlib
from datetime import datetime
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def check_md5(md5_hex: str) -> bool:
    if not os.path.exists(config.md5_path):
        open(config.md5_path, "w", encoding="utf-8").close()
        return False
    else:
        for line in open(config.md5_path, "r", encoding="utf-8").readlines():
            line = line.strip()
            if line == md5_hex:
                return True
        return False


def save_md5(md5_hex: str) -> None:
    with open(config.md5_path, "a", encoding="utf-8") as f:
        f.write(md5_hex + "\n")


def string2md5(input_str: str, encoding="utf-8") -> str:
    str_bin = input_str.encode(encoding=encoding)

    md5_obj = hashlib.md5()
    md5_obj.update(str_bin)
    md5_hex = md5_obj.hexdigest()

    return md5_hex

def delete_md5(md5_hex: str):
    with open(config.md5_path, 'r') as file:
        lines = file.readlines()
    with open(config.md5_path, 'w') as file:
        for line in lines:
            if md5_hex not in line:
                file.write(line)

class KnowledgeBase(object):
    def __init__(self) -> None:
        os.makedirs(config.persist_directory, exist_ok=True)

        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=DashScopeEmbeddings(model=config.embedding_model_name),
            persist_directory=config.persist_directory,
        )  # 向量库对象

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,  # 长度统计
        )  # 文本分割器对象

    def uploadStr(self, data: str, filename: str, doc_tag:list) -> bool:
        md5_hex = string2md5(data)
        if check_md5(md5_hex):  # 文件重复
            return False
        else:  # 成功
            if len(data) > config.split_threshold:
                text_chunk: list[str] = self.spliter.split_text(data)
            else:
                text_chunk: list[str] = [data]

            meta_data = {"source": filename, "ctime": datetime.now().strftime("%Y-%m-%d %H:%H:%S"), "doc_tag": doc_tag, "md5":md5_hex}
            self.chroma.add_texts(texts=text_chunk, metadatas=[meta_data | {"chunk_id":string2md5(i)} for i in text_chunk])
            save_md5(md5_hex)
            return True
        
    def get_records_by_tag(self, tag):
        return self.chroma.get(where={"doc_tag": {"$contains": tag}})
    
    def delete_by_sourcce(self, source):
        records = self.chroma.get(where={"source": source})
        if records:
            self.chroma.delete(ids=records["ids"])
            delete_md5(records["metadatas"][0]["md5"])
            return 1
        
        return 0

    # def delete_by_tag(self, tag):
    #     records = self.chroma.get(where={"doc_tag": {"$contains": tag}})
    #     if records:
    #         self.chroma.delete(ids=records["ids"])
    #         delete_md5(records["metadatas"][0]["md5"])
    #         return 1
        
        # return 0
        
# if __name__ == "__main__":
#     db = KnowledgeBase()
#     db.uploadStr("asdasda", "aaa.a", ["aaa"])