from posixpath import curdir
import sys
import os

from numpy import record
from openai import embeddings

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

import os
import re
import config
import hashlib
import numpy as np
from src.utils import similarity_calc
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
        

                
        
    def split_chunk(self, text:str) -> list[str]:
        sim_threshold = config.sim_threshold
        min_chunk_len = config.min_chunk_len
        max_chunk_len = config.max_chunk_len
        
        splited = re.split(f"([。！？.!?])", text)  # ["123", ".", "234", "。"...]
        sentence_list = []
        for i in range(0, len(splited), 2):
            sentence = splited[i]
            if not sentence:
                continue
            if i+1 < len(splited):  # 合并相邻的标点符号
                sentence += splited[i+1]
            sentence_list.append(sentence.strip())
        
        embedding_model = DashScopeEmbeddings(model=config.embedding_model_name)
        embeded = np.array(embedding_model.embed_documents(sentence_list))
        if embeded.shape[0] == 0:
            return [text]
        
        chunk_list = []
        tmp_list = []
        for i in range(len(sentence_list)-1):
            this_vec = embeded[i]
            next_vec = embeded[i+1]
            sim = similarity_calc(this_vec, next_vec)
            
            if len(tmp_list) > max_chunk_len:  # chunk形成
                chunk_list.append("".join(tmp_list))
                tmp_list = [sentence_list[i+1]]
                continue
            
            if sim >= sim_threshold:
                tmp_list.append(sentence_list[i+1])
            else:
                if len(tmp_list) >= min_chunk_len:   # chunk形成
                    chunk_list.append("".join(tmp_list))
                    tmp_list = [sentence_list[i+1]]
                    continue
                else:
                    tmp_list.append(sentence_list[i+1])
        
        if tmp_list:
            chunk_list.append("".join(tmp_list))
            
        print(chunk_list)
        return chunk_list
        

    def uploadStr(self, data: str, filename: str, doc_tag:list, col_name) -> bool:
        md5_hex = string2md5(data)
        if check_md5(md5_hex):  # 文件重复
            return False
        else:  # 成功
            if len(data) > config.split_threshold:
                # text_chunk: list[str] = self.spliter.split_text(data)
                text_chunk = self.split_chunk(data)
            else:
                text_chunk: list[str] = [data]

            meta_data = {"source": filename, "ctime": datetime.now().strftime("%Y-%m-%d %H:%H:%S"), "doc_tag": doc_tag, "md5":md5_hex, "col_name":col_name}
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
        
if __name__ == "__main__":
    
    ttt = """
   3.1管理者能力与投资效率
管理者在企业的日常投资中也起着重要作用。高阶梯队理论认为，高层管理团队的认知、观念、能力和素质会影响企业的战略行为和绩效。高能力的管理者能够在企业的投资效率上取得全方位的提升。第一，高能力管理者帮助企业在投资项目选择的过程中，会更加清楚投资项目产生风险与收益的大小，能够在众多可能选择的项目中寻找真正有正NPV的项目，避免盲目投资带来的投资错误的产生，这是由于信息的能力不足或是决策能力偏差引起的。第二，投资项目执行过程中，高能力的管理者能够更好的监管项目的执行情况，能够及早地发现和解决问题，更有效地利用项目的现有资源。第三，高能力的管理者一般具有更高的自信和声誉动机，他们更多的是通过实现自己的企业业绩来证明自己，而不希望通过操纵公司财务等问题谋求自身的私利。他们自身也更能减少信息的不对称，能够低成本地进行融资，支持更合适有潜力的投资。因此，在投资的决策中，能力更好的经理人会更多地识别和评价项目。
同时许宁宁（2017）的研究指出[1]，能力出众的管理者能够更有效地向外界传递信号，减少投资者的信息解读错误，降低沟通成本。这种认知优势使他们能精准捕捉正净现值项目，规避盲目投资，从而提升企业的投资效率。基于上述理论与文献分析，本文提出假设1。
H1：管理者能力可以提高投资效率。
3.2管理者能力与会计稳健性
会计稳健性是会计信息质量的一个重要组成部分，稳健性要求企业企业做账谨慎地记录资产、收益，慎重地确认负债和费用[3]。这种“提前确认坏消息，延迟确认好消息”的原则，虽然短期内会使报告利润减少，但从长远来看，它可提供更可靠的信息衡量业绩，限制管理者的机会主义行为。有能力和高效的管理者更容易地认识到并主动利用会计稳健性的价值。一是声誉效应，能力强的经理对自身声誉和企业长期业绩相联系的特点。采用稳健的会计政策是可以向投资者传达自己的并不好大喜功、关注长期业绩的信息，获取投资者和债权人的信任，有助于保证其在职业经理人市场中的声誉。二是风险管理，能力强的经理更有估计市场风险的能力。他们明白，激进的会计政策虽然能暂时美化报表，但会掩盖潜在风险，一旦风险爆发，将对企业和个人声誉造成严重的破坏和危害。因此，他们更倾向于通过稳健的会计政策提前释放风险信号，防患于未然。最后，稳健的会计政策也方便企业降低债权方、员工等主体之间的契约成本，为他们经营提供一个稳定有利的宏观环境。
会计稳健性不仅仅是保证会计信息质量的要求，更是管理者主动选择的一种治理工具。Demerjian（2012）认为，高能力者倾向于提供高质量会计信息以维持其在经理人市场的溢价[6]，他们通过促进企业提高会计稳健性水平来尽早释放风险，防止风险积聚引发毁灭性后果。基于上述理论与文献分析，本文提出假设2。
H2：有能力的管理者促进企业提高会计稳健性水平。
3.3会计稳健性对管理者能力和投资效率的中介作用
William R.Scott（2018）认为经济后果的含义是会计政策的选择会影响公司的价值，这种影响首先表现为对企业管理者的影响[5]。由于会计稳健性加速坏消息的确认，管理者无法通过掩盖亏损来推卸责任，因此有能力的管理者会利用会计稳健性的这一治理优势，通过提高会计稳健性来实现投资效率的提升。
会计稳健性作为一种重要的会计政策选择，其对投资决策的治理作用已得到广泛证实。一方面，稳健的会计政策使得投资项目的潜在亏损能够被及时确认，管理者无法将责任推卸至未来任期。这种硬性的业绩约束会迫使他们在投资前进行更审慎的评估，放弃那些短期盈利但长期净现值为负的项目。另一方面，当项目出现亏损迹象时，稳健性要求计提减值准备或预计负债，这将直接冲击当期利润，从而激励管理者尽快从亏损项目中撤出资金，避免损失扩大[2]。
因此，高能力的管理者认识到会计稳健性的上述治理优势，主动选择并实施更为稳健的会计政策。而更为稳健的会计政策能够有效抑制过度投资和缓解投资不足，从而提升了整体的投资效率。在这个过程中，会计稳健性成为了连接管理者个人能力与企业投资效率提升的关键桥梁，发挥了重要的“中介传导机制”。基于上述分析，本文提出假设3。
H3：管理者能力通过提高企业的会计稳健性水平来提高投资效率，会计稳健性在其中发挥中介作用。
    """
    
    db = KnowledgeBase()
    db.split_chunk(ttt)