import sys
import os


root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

import config
from src.VectorsServer import VectorsServer
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatTongyi
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

prompt_ = """
你是RAGInquiry，一个帮助券商工作人员回复证券交易所质询的助手。
以提供的已知参考资料（包括招股书、资产负债表、利润表、模板等）为主，来回答问题、查找资料或核对信息。
若需回答问题，请根据参考资料中提供的模板格式来回答用户所给出的质询，尽可能专业并符合模板的格式；
若需查找资料，请准确无误，格式简洁地把查找到的数据提供给用户，若未找到数据，请如实告知，不得随意编造；
如需核对信息，请根据查找到的信息检查用户地输入是否相符，并准确无误，格式简洁地把查找到的数据提供给用户，若未找到数据，请如实告知，不得随意编造。
并在结尾给出引用的参考资料。
-----参考资料如下-----
{context}
-----参考资料结束-----
"""


def doclst_process(doclst: list[Document]) -> str:
    if not doclst:
        return "无参考资料"
    
    grouped = dict()
    for doc in doclst:
        tag_list = doc.metadata.get("doc_tag")
        if not tag_list:
            grouped.setdefault("未分类", []).append(doc)
        else:
            for doc_tag in tag_list:
                grouped.setdefault(str(doc_tag), []).append(doc)
    
    res = ""
    for doc_tag, doc_list in grouped.items():
        res += f"-----{doc_tag}类文档如下----\n"
        for idx, doc in enumerate(doc_list, start=1):
            res += f"{idx}、元数据：{doc.metadata}，内容：{doc.page_content}\n"
        res += f"-----{doc_tag}类文档结束----\n\n"


    return res

def history_process(history_list:list[str])->str:
    if not history_list:
        return "无历史资料"
    res = ""
    for history in history_list:
        res += f"内容：{history}"
    return res




def print_prompt(prompt):
    print("---------prompt-------------------")
    print(prompt.to_string())
    print("----------------------------------\n")
    return prompt

class RagServer(object):
    def __init__(self) -> None:
        self.vector_server = VectorsServer(
            DashScopeEmbeddings(model=config.embedding_model_name)
        )
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", prompt_),
            ("system", "-----下面是你与用户的最近五条聊天记录，请你在需要的情况下参考。-----"),
            MessagesPlaceholder(variable_name="history", n_messages=5), # 历史会话的占位符
            ("system", "-----历史记录结束-----"),
            ("human", "请回答提问：{input}。"),
        ])
        self.chat_model = ChatTongyi(model=config.chat_model_name, api_key=None)
        self.history_database = config.history_database_path
        
        base_chain = self.__get_chain()
        self.chain = RunnableWithMessageHistory(
            base_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        
    def get_session_history(self, session_id: str):
        return SQLChatMessageHistory(
            session_id=session_id, 
            connection=self.history_database 
        )
    
    def get_history_for_web(self, session_id):
        ret = []
        for message in self.get_session_history(session_id).messages:
            if "HumanMessage" in str(type(message)):
                ret.append({"role":"human", "content": message.content})
            else:
                ret.append({"role":"ai", "content": message.content})
        return ret

    def __get_chain(self):
        retriever = self.vector_server.get_retriever()
        
        chain = (
            {
                "context": (lambda x: x["input"]) | retriever | doclst_process,
                "input": lambda x: x["input"],
                "history": lambda x: x["history"]
            }
            | self.prompt_template 
            | print_prompt
            | self.chat_model
            | StrOutputParser()
        )
        return chain

if __name__ == "__main__":
    server = RagServer()
    
    # query = "你是谁？"
    query = "你能从你的知识库找到哪些类型的参考文件？各举一个例子概括一下你找到参考文件的大致内容。"
    
    res = server.chain.invoke({'input': query}, {'configurable': {'session_id': 'test'}})
    print(res)
    
    # print(server.get_session_history("test").messages)
    # for i in server.get_session_history("test").messages:
    #     print(i)
    
    # print(server.get_history_for_web("test"))
    