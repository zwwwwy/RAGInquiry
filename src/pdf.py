import pdfplumber


def pdf2strint(pdf_file: pdfplumber.pdf.PDF) -> str:
    ret = ""
    for page in pdf_file.pages:
        ret += page.extract_text()

    return ret
