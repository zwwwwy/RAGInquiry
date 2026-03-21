import io
import pdfplumber
import pandas as pd
import numpy as np

def pdf2string(pdf_file) -> str:
    ret = ""
    pdf = pdfplumber.open(pdf_file)
    for page in pdf.pages:
        ret += page.extract_text()

    return ret

def excel2string(excel_file) -> str:
    ret = ""
    df = pd.read_excel(excel_file)
    output = io.StringIO()
    df.to_csv(output, index=False, encoding='utf-8')
    ret += output.getvalue()
    return ret

def similarity_calc(a:np.ndarray, b:np.ndarray):
    a_norm = np.linalg.norm(a) + 1e-12
    b_norm = np.linalg.norm(b) +  1e-12
    return (a @ b) / (a_norm*b_norm)