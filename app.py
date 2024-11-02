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

app = FastAPI()

# MongoDB configuration
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["question_db"]  # Replace with your database name
questions_collection = db["questions"]  # Replace with your collection name

# Login to Hugging Face
login(token="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")

def load_llm():
    llm = CTransformers(
        model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        model_type="mistral",
        max_new_tokens=1024,
        temperature=0.3
    )
    return llm

def file_processing(file_path):
    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ''
    for page in data:
        question_gen += page.page_content

    splitter_ques_gen = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)
    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    return document_ques_gen

def llm_pipeline(file_path):
    document_ques_gen = file_processing(file_path)
    llm_ques_gen_pipeline = load_llm()

    prompt_template = """
    You are an expert at creating questions with single-word answers based on documentation.
    Your goal is to prepare a student their exam and coding tests.
    You do this by asking questions about the text below, ensuring each question can be answered with one word:

    ------------ 
    {text} 
    ------------ 

    Please create questions that can be answered with only one word.
    Focus on extracting key terms or specific information where a single word is sufficient.

    QUESTIONS:
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
QUESTIONS:
"""

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    ques_gen_chain = load_summarize_chain(
        llm=llm_ques_gen_pipeline,
        chain_type="refine",
        verbose=True,
        question_prompt=PROMPT_QUESTIONS,
        refine_prompt=REFINE_PROMPT_QUESTIONS
    )

    ques = ques_gen_chain.run(document_ques_gen)

    # Split and filter the generated questions
    ques_list = ques.split("\n")
    filtered_ques_list = [element for element in ques_list if element.endswith('?') or element.endswith('.')]

    return filtered_ques_list

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

def get_json(file_path,pdf_name):
    base_folder = 'output/'
    if not os.path.isdir(base_folder):
        os.makedirs(base_folder)  
    output_file = os.path.join(base_folder, "questions.json")

    questions_list = []

    try:
        # Get the list of questions from the pipeline
        pipeline_result = llm_pipeline(file_path)
        
        # If multiple values are returned, process them accordingly
        if isinstance(pipeline_result, tuple):
            questions_list = pipeline_result[1] if len(pipeline_result) > 1 else []
        else:
            questions_list = pipeline_result if isinstance(pipeline_result, list) else []

        if not questions_list:
            raise ValueError("No questions generated.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in generating questions: {str(e)}")

    # Save the questions to the MongoDB collection
    try:
        # Convert the list to a dictionary
        questions_dict = {
            "pdf_name": pdf_name,
            "questions": {f"question{i+1}": question for i, question in enumerate(questions_list)}
        }

        
        # Insert the questions dictionary into the MongoDB collection
        questions_collection.insert_one(questions_dict)

        # Optionally save to a JSON file in the required format
        with open(output_file, "w", encoding="utf-8") as jsonfile:
            json.dump(questions_dict, jsonfile, ensure_ascii=False, indent=4)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing to MongoDB: {str(e)}")

    return output_file

@app.post("/analyze")
async def analyze_file(request: Request, pdf_filename: str = Form(...)):
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
