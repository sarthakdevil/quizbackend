from fastapi import FastAPI, Form, Request, File, HTTPException
from fastapi.responses import JSONResponse
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from huggingface_hub import login
from fastapi.middleware.cors import CORSMiddleware
import os
import aiofiles
from pymongo import MongoClient
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from bson import ObjectId
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# MongoDB configuration
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["question_db"]
questions_collection = db["questions"]

# Login to Hugging Face
login(token="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")

def load_llm():
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0.3,
        max_tokens=1024,
        timeout=None,
        max_retries=2,
    )
    return llm

def file_processing(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()

    content = ''
    for page in data:
        content += page.page_content

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(content)
    documents = [Document(page_content=t) for t in chunks]

    return documents

def llm_pipeline(file_path, num_ques):
    documents = file_processing(file_path)
    llm_pipeline = load_llm()

    prompt_template = f"""
    You are an expert at creating questions with single-word answers based on documentation.
    Your goal is to prepare a student for their exam and coding tests.
    You do this by asking questions about the text below, ensuring each question can be answered with one word only and no extra then one work is required to answer any question:

    ------------ 
    {{text}} 
    ------------ 

    Please create {num_ques} questions that can be answered with only one word.
    Focus on extracting key terms or specific information where a single word is sufficient.

    QUESTIONS AND ANSWERS:
    """

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

    refine_template = f"""
    You are an expert at generating comprehensive practice questions based on context.
    Your goal is to assist a student in preparing for a test by creating single-word answer questions from various topics.
    We have received practice questions to a certain extent: {{existing_answer}}.
    Refine these questions so they can be answered with one word if number of questions are less than {num_ques} add new concise questions that touch on different areas of the content.
    
    Please create exactly {num_ques} questions based on the content, ensuring each question has a one-word answer.

    ------------ 
    {{text}} 
    ------------ 

    Ensure each question has a one-word answer.
    QUESTIONS AND ANSWERS:
    """

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    ques_gen_chain = load_summarize_chain(
        llm=llm_pipeline,
        chain_type="refine",
        verbose=True,
        question_prompt=PROMPT_QUESTIONS,
        refine_prompt=REFINE_PROMPT_QUESTIONS
    )

    # Run the chain to get output
    output = ques_gen_chain.run(documents)

    # Split the output based on new lines and process questions and answers
    qa_pairs = output.split("\n")
    questions_and_answers = []
    
    for i in range(len(qa_pairs)):
        if "?" in qa_pairs[i]:  # Check if it's a question
            question = qa_pairs[i].strip()
            answer = qa_pairs[i + 1].strip() if (i + 1) < len(qa_pairs) else ""
            questions_and_answers.append({"question": question, "answer": answer})

    return {
        "pdf_name": file_path,
        "questions_and_answers": questions_and_answers
    }

@app.post("/upload")
async def upload_pdf(request: Request, pdf_file: bytes = File(...), filename: str = Form(...)):
    base_folder = 'docs/'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    pdf_filename = os.path.join(base_folder, filename)

    async with aiofiles.open(pdf_filename, 'wb') as f:
        await f.write(pdf_file)

    response_data = {"msg": 'success', "pdf_filename": pdf_filename}
    return JSONResponse(content=response_data)

def get_json(file_path, pdf_name, num_ques):
    try:
        # Generate questions and answers using the llm_pipeline with num_ques
        questions_and_answers = llm_pipeline(file_path, num_ques)
        if not questions_and_answers:
            raise ValueError("No questions and answers generated.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in generating questions and answers: {str(e)}")

    try:
        questions_dict = {
            "pdf_name": pdf_name,
            "questions_and_answers": questions_and_answers
        }

        # Insert the document into MongoDB and get the inserted_id
        result = questions_collection.insert_one(questions_dict)
        inserted_id = result.inserted_id  # Get the _id of the inserted document

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing to MongoDB: {str(e)}")

    # Return a success message with the inserted _id
    return {"msg": "Generated successfully and saved to MongoDB.", "inserted_id": str(inserted_id)}

@app.post("/analyze")
async def analyze_file(request: Request, pdf_filename: str = Form(...), num_ques: str = Form(...)):
    pdf_filename = f"docs/{pdf_filename}"
    if not os.path.exists(pdf_filename):
        raise HTTPException(status_code=404, detail="PDF file not found")

    try:
        # Pass the number of questions to get_json
        output_message = get_json(pdf_filename, os.path.basename(pdf_filename), num_ques)
        
        # Fetch the questions from MongoDB using the inserted _id
        inserted_id = output_message.get("inserted_id")  # Get the _id from the output message
        print(inserted_id)
        questions_data = questions_collection.find_one({"_id": ObjectId(inserted_id)})  # Query for the document
        
        if not questions_data:
            raise HTTPException(status_code=404, detail="Questions not found in the database.")
        
        # Prepare the response data
        response_data = {
            "pdf_name": pdf_filename,
            "questions_and_answers": questions_data.get("questions_and_answers")
        }
        
        return JSONResponse(content=response_data)

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host='127.0.0.1', port=8000, reload=True)
