import io
import pdfplumber
import pandas as pd

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