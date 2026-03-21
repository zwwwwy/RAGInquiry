
import streamlit as st
import config
from src.KnowledgeBase import KnowledgeBase
from src.utils import excel2string, pdf2string
import pandas as pd

class UploadFileInfo:
    def __init__(self) -> None:
        self.raw_file = None
        self.name: str = ""
        self.type: str = ""
        self.size: float = 0.0
        self.text: str = ""
        self.tags: list = []


def init_file(upload_file, tags: list) -> UploadFileInfo:
    file = UploadFileInfo()
    if upload_file is None:
        return file

    file.tags = tags
    file.name = upload_file.name
    file.type = upload_file.type
    file.size = upload_file.size
    file.raw_file = upload_file

    return file


def upload_str(file: UploadFileInfo):
    with st.spinner("上传中"):
        col_name = "不提供列名称"
        if file.name.split(".")[1] == "xlsx":
            col_name = ",".join(map(str,pd.read_excel(file.raw_file).columns))
        state = st.session_state["server"].uploadStr(file.text, file.name, file.tags, col_name)
        if state == 1:
            st.write(f"已上传{file.name}，大小为{file.size/1024:.2f}KB")
        else:
            st.write("文件重复上传")
        st.session_state["cnt_upload"] += 1


def get_str(file: UploadFileInfo) -> UploadFileInfo:
    if file.raw_file is None:
        return file
    st.write(f"file_name={file.name}, file_type={file.type}, file_size={file.size}")

    type_ = file.raw_file.name.split(".")[1]
    if "pdf" == type_:
        with st.spinner("读取中"):
            file.text = pdf2string(file.raw_file)
    elif "xlsx" == type_:
        with st.spinner("读取中"):
            file.text = excel2string(file.raw_file)
    elif "txt" == type_:
        with st.spinner("读取中"):
            file.text = file.raw_file.getvalue().decode("utf-8")

    return file


st.title("知识库更新")
if "server" not in st.session_state:
    st.session_state["server"] = KnowledgeBase()

if "cnt_upload" not in st.session_state:
    st.session_state["cnt_upload"] = 0


upload_file = st.file_uploader(label="请上传文件", type=["txt", "pdf", "xlsx"], accept_multiple_files=False)

if upload_file:
    tags = st.multiselect("请选择文件类型", config.doc_tags, default=[])
    file = init_file(upload_file, tags)
    if st.button("提交"):
        file = get_str(file=file)
        # st.write(file.text)
        upload_str(file=file)

# if upload_file:
#     file = init_file(upload_file, [])
#     file = get_str(file=file)
#     st.write(file.text)
st.write("---")
st.write("# 知识库管理")
for tag in config.doc_tags:
    metadatas = st.session_state["server"].get_records_by_tag(tag)["metadatas"]
    cnt = dict()
    for metadata in metadatas:
        cnt[metadata["source"]] = cnt.setdefault(metadata["source"], 0) + 1
        

    st.write(f"#### {tag}类文档")
    for i, n in cnt.items():
        col_text, col_btn = st.columns([4, 1])
        with col_text:
            st.write(f"{i}\t共{n}块")

        with col_btn:
            if st.button("删除", key=f"delete-source{i}"):
                st.session_state["server"].delete_by_sourcce(i)
                st.rerun()
        st.write("---")