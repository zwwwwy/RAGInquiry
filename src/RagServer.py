import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

import config
from src.VectorsServer import VectorsServer
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatTongyi
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

prompt_ = """
你是RAGInquiry，一个帮助券商工作人员回复证券交易所质询的助手。
以提供的已知参考资料（包括招股书、资产负债表、利润表、模板等）为主，
根据参考资料中提供的模板格式来回答用户所给出的质询，尽可能专业并符合模板的格式，
并在结尾给出引用的参考资料，
参考资料：{context}。
"""


def doclst_process(doclst: list[Document]) -> str:
    if not doclst:
        return "无参考资料"

    res = ""
    for doc in doclst:
        res += f"内容：{doc.page_content},元数据：{doc.metadata}\n"
    return res


class RagServer(object):
    def __init__(self) -> None:
        self.vector_server = VectorsServer(
            DashScopeEmbeddings(
                model=config.embedding_model_name,
            )
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    prompt_,
                ),
                ("human", "请回答提问：{input}。"),
            ]
        )
        self.chat_model = ChatTongyi(model=config.chat_model_name, api_key=None)
        self.chain = self.__get_chain()

    def __get_chain(self):
        retriever = self.vector_server.get_retriever()
        chain = (
            {
                "input": RunnablePassthrough(),
                "context": retriever | doclst_process,
            }
            | self.prompt_template
            | self.chat_model
            | StrOutputParser()
        )
        return chain


if __name__ == "__main__":
    res = RagServer().chain.invoke("你掌握了那些信息？")
    print(res)
