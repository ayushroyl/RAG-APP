import PyPDF2

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

pdf_path = r"RAG_App\policy-booklet-0923.pdf"  
pdf_text = extract_text_from_pdf(pdf_path)

with open("RAG_App\policy_text.txt", "w", encoding="utf-8") as f:
    f.write(pdf_text)

print("Text extracted and saved to policy_text.txt")
