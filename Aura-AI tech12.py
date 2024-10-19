import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Function to extract text from PDFs
def extract_text_from_pdfs(pdf_files):
    text_data = []
    for pdf_file in pdf_files:
        with fitz.open(pdf_file) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            text_data.append(text)
    return text_data

# Function to create an index of the extracted text
def create_index(text_data):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = []

    for text in text_data:
        chunks = text.split('\n\n')  # Split by paragraphs
        embeddings.extend(model.encode(chunks))

    # Create a FAISS index
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings).astype('float32'))
    return index, chunks

# Function to generate questions from context
def generate_question(context):
    question_generator = pipeline("text2text-generation", model="valhalla/t5-small-qa-qg")
    questions = question_generator(f"generate question: {context}", max_length=30)
    return questions[0]['generated_text']

# Main function to run the application
def main(pdf_files):
    text_data = extract_text_from_pdfs(pdf_files)
    index, chunks = create_index(text_data)

    query = input("Enter your query: ")
    query_embedding = model.encode(query).astype('float32').reshape(1, -1)

    D, I = index.search(query_embedding, k=5)  # Get top 5 most similar chunks
    for idx in I[0]:
        context = chunks[idx]
        question = generate_question(context)
        print(f"Generated Question: {question}")

if __name__ == "__main__":
    pdf_files = ["C:\\Users\\anudeep balumuri\\Downloads\\CI Lecture PPT-3.pdf", "C:\\Users\\anudeep balumuri\\Downloads\\CI Lecture-2 .pdf", "C:\\Users\\anudeep balumuri\\Downloads\\CI Lecture-1 (1).pdf"]  # List your PDF files here
    main(pdf_files)
