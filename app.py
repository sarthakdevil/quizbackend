from fastapi import FastAPI, Form, Request, Response, File, HTTPException
from fastapi.responses import JSONResponse
from langchain.llms import CTransformers
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from huggingface_hub import login
import os
import json
import aiofiles
from pymongo import MongoClient
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

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
def llm_pipeline(file_path):
    documents = file_processing(file_path)
    llm_pipeline = load_llm()

    prompt_template = """
    You are an expert at creating questions with single-word answers based on documentation.
    Your goal is to prepare a student for their exam and coding tests.
    You do this by asking questions about the text below, ensuring each question can be answered with one word:

    ------------ 
    {text} 
    ------------ 

    Please create questions that can be answered with only one word.
    Focus on extracting key terms or specific information where a single word is sufficient.

    QUESTIONS AND ANSWERS:
    """

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

    refine_template = """
    You are an expert at generating comprehensive practice questions based on context.
    Your goal is to assist a student in preparing for a test by creating single-word answer questions from various topics.
    We have received practice questions to a certain extent: {existing_answer}.
    Refine these questions so they can be answered with one word if possible, or add new concise questions that touch on different areas of the content.

    ------------ 
    {text} 
    ------------ 

    Ensure each question has a one-word answer. Aim to create at least 5 additional questions based on the content, exploring different topics or perspectives.
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

    # Split output based on newline characters
    output_lines = output.split("\n")
    
    questions_and_answers = []
    current_question = None

    for line in output_lines:
        stripped_line = line.strip()
        if stripped_line:  # Only process non-empty lines
            if current_question is None:
                current_question = stripped_line  # The line is a question
            else:
                # Create a structured format for question and answer
                questions_and_answers.append({
                    "question": current_question,  # Assign the last question here
                    "answer": stripped_line  # The current line is the answer
                })
                current_question = None  # Reset for the next question

    # Adjust the format as per your requirement to remove any unnecessary information
    formatted_output = [
        {
            "question": qa["question"].replace("Answer:", "").strip(),  # Clean up question
            "answer": qa["answer"].replace("Answer:", "").strip()  # Clean up answer
        }
        for qa in questions_and_answers if qa["question"] and qa["answer"]
    ]

    return {
        "pdf_name": "ACM.pdf",
        "questions_and_answers": formatted_output
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

def get_json(file_path, pdf_name):
    # Create a base folder for output if it doesn't exist
    # base_folder = 'output/'
    # if not os.path.isdir(base_folder):
    #     os.makedirs(base_folder)  
    # output_file = os.path.join(base_folder, "questions.json")

    try:
        # Generate questions and answers using the llm_pipeline
        questions_and_answers = llm_pipeline(file_path)
        if not questions_and_answers:
            raise ValueError("No questions and answers generated.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in generating questions and answers: {str(e)}")

    try:
        # Prepare the document for MongoDB
        questions_dict = {
            "pdf_name": pdf_name,
            "questions_and_answers": questions_and_answers
        }

        # Insert the document into MongoDB
        questions_collection.insert_one(questions_dict)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing to MongoDB: {str(e)}")

    # Return a success message
    return {"msg": "Generated successfully and saved to MongoDB."}
@app.post("/analyze")
async def analyze_file(request: Request, pdf_filename: str = Form(...),num_ques:str):
    if not os.path.exists(pdf_filename):
        raise HTTPException(status_code=404, detail="PDF file not found")

    try:
        output_file = get_json(pdf_filename, os.path.basename(pdf_filename))
        response_data = {"output_file": output_file}
        return JSONResponse(content=response_data)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host='127.0.0.1', port=8000, reload=True)
