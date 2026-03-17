from rchitect import init
import streamlit as st
from src.KnowledgeBase import KnowledgeBase
import pdfplumber
from src.pdf import pdf2strint


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
        state = st.session_state["server"].uploadStr(file.text, file.name)
        if state == 1:
            st.write(f"已上传{file.name}，大小为{file.size/1024:.2f}KB")
        else:
            st.write("文件重复上传")
        st.session_state["cnt_upload"] += 1


def get_str(file: UploadFileInfo) -> UploadFileInfo:
    if file.raw_file is None:
        return file
    # st.write(f"file_name={file.name}, file_type={file.type}, file_size={file.size}")

    if "pdf" in file.raw_file.type:
        pdf = pdfplumber.open(file.raw_file)
        with st.spinner("读取中"):
            file.text = pdf2strint(pdf)

    elif "text" in file.raw_file.type:
        with st.spinner("读取中"):
            file.text = file.raw_file.getvalue().decode("utf-8")

    if "模板文件" in file.tags:
        file.text = "下面是模板文件内容：" + file.text
        file.name = "template_" + file.name

    if "招股书" in file.tags:
        file.text = f"下面是招股书——{file.name.split('.')[0]}内容：" + file.text
        file.name = "招股书_" + file.name
    return file


st.title("知识库更新")
if "server" not in st.session_state:  # 防止页面刷新导致重新创建服务对象
    st.session_state["server"] = KnowledgeBase()

if "cnt_upload" not in st.session_state:
    st.session_state["cnt_upload"] = 0


upload_file = st.file_uploader(label="请上传文件", type=["txt", "pdf"], accept_multiple_files=False)

if upload_file:
    tags = st.multiselect("请选择文件类型", ["模板文件", "招股书"], default=[])
    file = init_file(upload_file, tags)
    if st.button("提交"):
        file = get_str(file=file)
        # st.write(file.text)
        upload_str(file=file)

# if upload_file:
#     file = init_file(upload_file, [])
#     file = get_str(file=file)
#     st.write(file.text)
