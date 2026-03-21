from importlib import metadata
import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

import config
from langchain_chroma import Chroma
from langchain_core.documents import Document

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
    
    # def export_doc(self) -> list[Document]:
    # def export_doc(self):
    #     data = self.vector_store.get()
    #     print(data.keys())
        # return Document

    
    def export_doc(self) -> list[Document]:
        data = self.vector_store.get()
        docs = data.get("documents") or []
        metadata_list = data.get("metadatas") or []
        ids = data.get("ids") or[]
        ret = []
        n = min(len(docs), len(metadata_list), len(ids))
        for i in range(n):
            metadata = dict(metadata_list[i])
            metadata["ids"] = ids[i]
            ret.append(Document(page_content=docs[i], metadata=metadata))
        return ret

if __name__ == "__main__":
    from langchain_community.embeddings import DashScopeEmbeddings

    server = VectorsServer(DashScopeEmbeddings(model="text-embedding-v4"))
    # retriever = server.get_retriever()
#     aaa = retriever.invoke(input="你目前掌握了哪些信息？")
#     print(aaa, 666)
    server.export_doc()

    