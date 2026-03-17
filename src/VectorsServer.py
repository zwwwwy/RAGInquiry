import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

import config
from langchain_chroma import Chroma


class VectorsServer(object):  # 根据提问创建向量并匹配
    def __init__(self, embedding) -> None:
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": config.search_num})


if __name__ == "__main__":
    from langchain_community.embeddings import DashScopeEmbeddings

    retriever = VectorsServer(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()
    aaa = retriever.invoke(input="你目前掌握了哪些信息？")
    print(aaa, 666)
