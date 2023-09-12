from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain import DocumentLoader, VectorStores, ConversationalRetrievalQAChain
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
import os
import uvicorn

os.environ["OPENAI_API_KEY"
           ] 
loader=TextLoader("requirements.txt")
loader.load()
app = FastAPI()


document_loader = DocumentLoader(DocumentLoader)
vector_store = VectorStores.get_local_vector_store(Chroma)  # Veya Pinecone, Chroma vb. kullanabilirsiniz
qa_chain = ConversationalRetrievalQAChain(vector_store)
loader = PyPDFLoader("example_data/layout-parser-paper.pdf")

pages = loader.load_and_split()
@app.post("/upload/")
async def upload_pdf(file: UploadFile,query):
   


    try:
        chroma_index = Chroma.from_documents(pages, OpenAIEmbeddings())
        docs = chroma_index.similarity_search("How will the community be engaged?", k=2)
        raw_documents = TextLoader('../../../state_of_the_union.txt').load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(raw_documents)
        db = Chroma.from_documents(documents, OpenAIEmbeddings())
        for doc in docs:
            print(str(doc.metadata["page"]) + ":", doc.page_content[:300])
        pdf_bytes = await file.read()
        document = document_loader.load_document(pdf_bytes)

        embedding_vector = OpenAIEmbeddings().embed_query(query)
        docs = db.similarity_search_by_vector(embedding_vector)
        print(docs[0].page_content) 
        for part in document.parts:
            vector = qa_chain.embed(part.text)
            vector_store.put_vector(part.id, vector)

        return {"message": "PDF basariyla işlendi ve parçalara ayrildi."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/")
async def search_similarity(question: dict
                            ,db):
    try:
         
        query = "What did the president say about Ketanji Brown Jackson"
        docs = db.similarity_search(query)
        print(docs[0].page_content)
        question_text = question["question"]
        results = qa_chain.search(question_text)

        query = "What did the president say about Ketanji Brown Jackson"
        docs = db.similarity_search(query)
        print(docs[0].page_content)
        
        llm = OpenAI(openai_api_key="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")

      
        response = {
            "question": question_text,
            "results": results
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000)
