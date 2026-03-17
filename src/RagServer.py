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
以提供的已知参考资料（包括招股书、资产负债表、利润表、模板等）为主，
根据参考资料中提供的模板格式来回答用户所给出的质询，尽可能专业并符合模板的格式，
并在结尾给出引用的参考资料。
-----参考资料如下-----
{context}
-----参考资料结束-----
"""


def doclst_process(doclst: list[Document]) -> str:
    if not doclst:
        return "无参考资料"

    res = ""
    for doc in doclst:
        res += f"内容：{doc.page_content},元数据：{doc.metadata}\n"
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
    
    query = "你是谁？"
    
    # res = server.chain.invoke({'input': query}, {'configurable': {'session_id': 'abcde'}})
    # print(res)
    # for i in server.get_session_history("admin").messages:
    #     print(i.content)
    
    print(server.get_history_for_web("admin"))
    