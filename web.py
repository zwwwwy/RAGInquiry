import json

import streamlit as st
from src.RagServer import RagServer

st.title("RAGInquiry")

if "rag_server" not in st.session_state:  # 防止页面刷新导致重新创建服务对象
    st.session_state["rag_server"] = RagServer()

if "session_id" not in st.session_state:
    st.session_state["session_id"] = "admin"

if "messages" not in st.session_state:
    st.session_state.messages = st.session_state["rag_server"].get_history_for_web(st.session_state["session_id"])
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        
        
question = st.chat_input("在此输入问题")
if question:
    st.chat_message("human").markdown(question)

    with st.spinner("思考中"):
        response = st.session_state["rag_server"].chain.invoke({'input': question}, {'configurable': {'session_id': st.session_state["session_id"]}})
        st.chat_message("ai").markdown(response)
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "ai", "content": response, "token":"asdsd"})
        
        
        
    
