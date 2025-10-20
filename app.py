from fastapi import FastAPI, Form, Request, File, HTTPException
from fastapi.responses import JSONResponse
from langchain.document_loaders import PyPDFLoader
from fastapi.middleware.cors import CORSMiddleware
import os
import aiofiles
from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from bson import ObjectId
from typing import Optional
import logging
from answer_matcher import answer_matcher
from cron import start_interval_scheduler, clean_docs
load_dotenv()
app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to hold the scheduler instance
scheduler = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://a-iquizapp.vercel.app","http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Scheduler lifecycle events
@app.on_event("startup")
async def startup_event():
    global scheduler
    try:
        # Start the interval-based scheduler to run every 24 hours
        scheduler = start_interval_scheduler()
        logger.info("24-hour interval cleanup scheduler started successfully")
        
        # Optionally run an initial cleanup on startup
        deleted_count = clean_docs()
        logger.info(f"Initial cleanup completed: {deleted_count} files removed from docs/")
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    global scheduler
    if scheduler:
        try:
            scheduler.shutdown(wait=True)
            logger.info("Scheduler shutdown completed")
        except Exception as e:
            logger.error(f"Error shutting down scheduler: {e}")

# MongoDB configuration
mongo_client = MongoClient("mongodb+srv://sarthakrajesh2005:Sarthak123@cluster0.lvcrr.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = mongo_client["question_db"]
questions_collection = db["questions"]

def load_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        max_tokens=1024,
        timeout=None,
        max_retries=2,
    )
    return llm

def llm_pipeline(file_path, num_ques):
    """
    Enhanced two-stage pipeline:
    1. Generate questions from all PDF chunks
    2. Use Gemini to select and refine the best N questions
    """
    import re
    
    # Load and process PDF content
    loader = PyPDFLoader(file_path)
    data = loader.load()

    # Combine all pages into one content string
    full_content = ''
    for page in data:
        full_content += page.page_content + '\n'

    logger.info(f"PDF content length: {len(full_content)} characters")
    
    llm = load_llm()
    num_ques = int(num_ques)
    
    # Configuration for chunking
    chunk_size = 6000  # Characters per chunk (safe for LLM context)
    chunk_overlap = 200  # Overlap to maintain context continuity
    
    # If content is small enough, process as single chunk
    if len(full_content) <= chunk_size:
        logger.info("PDF is small, processing as single chunk")
        return _generate_questions_from_chunk(llm, full_content, num_ques)
    
    # STAGE 1: Generate questions from all chunks
    logger.info("STAGE 1: Generating questions from all chunks")
    
    # Split content into chunks with overlap
    chunks = _split_content_into_chunks(full_content, chunk_size, chunk_overlap)
    logger.info(f"Split PDF into {len(chunks)} chunks")
    
    # Generate questions from each chunk (more questions per chunk to have variety)
    all_candidate_questions = []
    questions_per_chunk = max(2, num_ques // 2)  # Generate more questions than needed
    
    for chunk in chunks:
        logger.info(f"Processing chunk {chunk['number']}: generating {questions_per_chunk} questions")
        
        chunk_questions = _generate_questions_from_chunk(
            llm, 
            chunk['content'], 
            questions_per_chunk,
            chunk_number=chunk['number']
        )
        
        all_candidate_questions.extend(chunk_questions)
        logger.info(f"Chunk {chunk['number']} generated {len(chunk_questions)} questions")
    
    logger.info(f"STAGE 1 complete: Generated {len(all_candidate_questions)} candidate questions from {len(chunks)} chunks")
    
    # If we didn't get enough questions, return what we have
    if len(all_candidate_questions) == 0:
        logger.warning("No questions generated from any chunks")
        return []
    
    if len(all_candidate_questions) <= num_ques:
        logger.info(f"Generated questions ({len(all_candidate_questions)}) <= requested ({num_ques}), returning all")
        return all_candidate_questions
    
    # STAGE 2: Use Gemini to select and refine the best questions
    logger.info(f"STAGE 2: Using Gemini to select best {num_ques} questions from {len(all_candidate_questions)} candidates")
    
    selected_questions = _select_best_questions(llm, all_candidate_questions, num_ques)
    
    if not selected_questions:
        logger.warning("No questions selected in stage 2")
        return []
    
    # STAGE 3: Convert to multiple-choice questions with options
    logger.info(f"STAGE 3: Converting {len(selected_questions)} questions to multiple-choice format")
    
    final_mcq_questions = _create_multiple_choice_questions(llm, selected_questions)
    
    logger.info(f"Final result: Created {len(final_mcq_questions)} multiple-choice questions")
    
    return final_mcq_questions


def _split_content_into_chunks(full_content, chunk_size, chunk_overlap):
    """
    Split content into chunks with smart sentence boundary breaking
    """
    chunks = []
    start = 0
    chunk_num = 1
    
    while start < len(full_content):
        end = start + chunk_size
        
        # If not the last chunk, try to break at sentence boundary
        if end < len(full_content):
            # Look for sentence endings in the last 200 characters of the chunk
            sentence_end = max(
                full_content.rfind('. ', start + chunk_size - 200, end),
                full_content.rfind('.\n', start + chunk_size - 200, end),
                full_content.rfind('! ', start + chunk_size - 200, end),
                full_content.rfind('? ', start + chunk_size - 200, end)
            )
            
            if sentence_end > start:
                end = sentence_end + 1
        
        chunk_content = full_content[start:end].strip()
        if chunk_content:
            chunks.append({
                'content': chunk_content,
                'start': start,
                'end': end,
                'number': chunk_num
            })
            chunk_num += 1
        
        # Move start position with overlap
        start = end - chunk_overlap if end < len(full_content) else end
    
    return chunks


def _select_best_questions(llm, candidate_questions, target_count):
    """
    Use Gemini to select and refine the best questions from candidates
    """
    # Format candidate questions for the prompt
    candidates_text = ""
    for i, qa in enumerate(candidate_questions, 1):
        candidates_text += f"{i}. Q: {qa['question']}\n   A: {qa['answer']}\n\n"
    
    selection_prompt = f"""
You are an expert quiz curator. Below are {len(candidate_questions)} candidate questions generated from different parts of a document.

Your task is to select and refine exactly {target_count} questions that:
1. Cover diverse topics/concepts from the document
2. Have clear, unambiguous questions
3. Have short, accurate answers (1-3 words)
4. Are not duplicates or very similar to each other
5. Represent the most important information from the document

CANDIDATE QUESTIONS:
{candidates_text}

INSTRUCTIONS:
1. Select exactly {target_count} questions from the candidates above
2. You may slightly rephrase questions to improve clarity
3. Ensure answers remain short and accurate
4. Prioritize diversity of topics covered
5. Use this EXACT format for your final selection:

Q: [your selected/refined question]?
A: [short answer]

Generate exactly {target_count} question-answer pairs:
"""

    try:
        response = llm.invoke(selection_prompt)
        output = response.content if hasattr(response, 'content') else str(response)
        
        logger.info(f"DEBUG - Selection LLM Output (first 500 chars):\n{output[:500]}")
        
        # Parse the refined questions
        selected_questions = _parse_llm_output(output, target_count)
        
        logger.info(f"Selection stage: extracted {len(selected_questions)} questions from {len(candidate_questions)} candidates")
        
        return selected_questions
        
    except Exception as e:
        logger.error(f"Error in question selection: {str(e)}")
        # Return empty list if selection fails
        return []


def _create_multiple_choice_questions(llm, questions_list):
    """
    Convert simple Q&A questions into multiple-choice questions with 4 options
    """
    if not questions_list:
        return []
    
    # Format questions for the prompt
    questions_text = ""
    for i, qa in enumerate(questions_list, 1):
        questions_text += f"{i}. Q: {qa['question']}\n   Correct Answer: {qa['answer']}\n\n"
    
    mcq_prompt = f"""
You are an expert at creating multiple-choice quiz questions. Convert the following questions into multiple-choice format with 4 options each.

REQUIREMENTS:
1. Create exactly 4 options (A, B, C, D) for each question
2. Only ONE option should be correct (the given answer)
3. The 3 incorrect options should be plausible but clearly wrong
4. Make incorrect options related to the topic but not correct
5. Keep all options concise (1-3 words each)
6. Randomize the position of the correct answer (don't always put it as option A)

FORMAT FOR EACH QUESTION:
Q: [question]?
A) [option 1]
B) [option 2]  
C) [option 3]
D) [option 4]
Correct: [A/B/C/D]

QUESTIONS TO CONVERT:
{questions_text}

Convert each question to multiple-choice format:
"""

    try:
        response = llm.invoke(mcq_prompt)
        output = response.content if hasattr(response, 'content') else str(response)
        
        logger.info(f"DEBUG - MCQ LLM Output (first 800 chars):\n{output[:800]}")
        
        # Parse the multiple-choice questions
        mcq_questions = _parse_mcq_output(output)
        
        logger.info(f"MCQ conversion: extracted {len(mcq_questions)} multiple-choice questions")
        
        return mcq_questions
        
    except Exception as e:
        logger.error(f"Error creating multiple-choice questions: {str(e)}")
        # Return empty list if MCQ conversion fails
        return []


def _parse_mcq_output(output):
    """
    Parse LLM output to extract multiple-choice questions
    """
    import re
    
    mcq_questions = []
    lines = output.split('\n')
    
    current_question = None
    current_options = {}
    correct_answer = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Look for question
        if line.startswith('Q:'):
            # Save previous question if complete
            if current_question and len(current_options) == 4 and correct_answer:
                mcq_questions.append({
                    "question": current_question,
                    "options": current_options,
                    "correct_answer": correct_answer,
                    "answer": correct_answer,  # Store the letter (A/B/C/D) for consistency
                    "correct_option_text": current_options.get(correct_answer, "")  # Store the actual text
                })
            
            # Start new question
            current_question = line[2:].strip()
            current_options = {}
            correct_answer = None
            
        # Look for options A) B) C) D)
        elif re.match(r'^[ABCD]\)', line):
            option_letter = line[0]
            option_text = line[2:].strip()
            current_options[option_letter] = option_text
            
        # Look for correct answer
        elif line.startswith('Correct:'):
            correct_answer = line.split(':', 1)[1].strip()
    
    # Don't forget the last question
    if current_question and len(current_options) == 4 and correct_answer:
        mcq_questions.append({
            "question": current_question,
            "options": current_options,
            "correct_answer": correct_answer,
            "answer": correct_answer,  # Store the letter (A/B/C/D) for consistency
            "correct_option_text": current_options.get(correct_answer, "")  # Store the actual text
        })
    
    return mcq_questions





def _generate_questions_from_chunk(llm, content, num_ques, chunk_number=1):
    """
    Generate questions from a single chunk of content
    """
    import re
    
    prompt_template = f"""
You are an expert at creating quiz questions with short answers based on documentation.
Your goal is to create exactly {num_ques} questions from the following text chunk.

IMPORTANT INSTRUCTIONS:
1. Create EXACTLY {num_ques} questions (no more, no less)
2. Each question must have a short answer (1-3 words maximum)
3. Focus on key terms, names, numbers, concepts, definitions from this specific text
4. Questions should be clear and unambiguous
5. Avoid questions that require context from other parts of the document
6. Use this EXACT format for each question-answer pair:

Q: [your question here]?
A: [short answer here]

Example format:
Q: What is the capital of France?
A: Paris

Q: What year was mentioned?
A: 1985

TEXT CHUNK TO ANALYZE:
------------
{content}
------------

Now generate exactly {num_ques} question-answer pairs using the format above:
"""

    try:
        # Generate questions using the LLM
        response = llm.invoke(prompt_template)
        output = response.content if hasattr(response, 'content') else str(response)
        
        # Parse the output to extract questions and answers
        questions_and_answers = _parse_llm_output(output, num_ques)
        
        return questions_and_answers

    except Exception as e:
        logger.error(f"Error processing chunk {chunk_number}: {str(e)}")
        return []


def _parse_llm_output(output, expected_count):
    """
    Parse LLM output to extract question-answer pairs
    """
    import re
    
    questions_and_answers = []
    lines = output.split('\n')
    
    # Standard Q: A: format
    current_question = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for question pattern
        if line.startswith('Q:') or line.startswith('Question'):
            if ':' in line:
                current_question = line.split(':', 1)[1].strip()
                if not current_question.endswith('?'):
                    current_question += '?'
                    
        # Look for answer pattern
        elif line.startswith('A:') or line.startswith('Answer'):
            if current_question and ':' in line:
                answer = line.split(':', 1)[1].strip()
                questions_and_answers.append({
                    "question": current_question,
                    "answer": answer
                })
                current_question = None

    return questions_and_answers


def _deduplicate_questions(questions_list):
    """
    Remove duplicate or very similar questions to improve quality
    """
    import re
    
    if len(questions_list) <= 1:
        return questions_list
    
    deduplicated = []
    seen_questions = set()
    
    for qa in questions_list:
        question = qa['question'].lower().strip()
        
        # Simple deduplication based on question text
        question_key = re.sub(r'[^\w\s]', '', question)  # Remove punctuation
        question_key = ' '.join(question_key.split())    # Normalize whitespace
        
        if question_key not in seen_questions and len(question_key) > 5:  # Avoid very short questions
            seen_questions.add(question_key)
            deduplicated.append(qa)
    
    return deduplicated

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
def get_json(file_path, pdf_name, quiz_name, num_ques, time):
    try:
        # Generate questions and answers using the llm_pipeline with num_ques
        questions_and_answers = llm_pipeline(file_path, num_ques)
        if not questions_and_answers:
            raise ValueError("No questions and answers generated.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in generating questions and answers: {str(e)}")

    try:
        if time is not None:
            try:
                time = int(time)  # Convert time to integer
            except ValueError:
                raise ValueError("Invalid time format. Please provide a valid number.")
        # Use a default message if time is None
        questions_dict = {
            "pdf_name": pdf_name,
            "quiz_name": quiz_name,
            "questions_and_answers": questions_and_answers,
            "quiz_time": time if time is not None else "No time specified"
        }

        # Insert the document into MongoDB and get the inserted_id
        result = questions_collection.insert_one(questions_dict)
        inserted_id = result.inserted_id  # Get the _id of the inserted document
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing to MongoDB: {str(e)}")

    # Return a success message with the inserted _id
    return {"msg": "Generated successfully and saved to MongoDB.", "inserted_id": str(inserted_id)}

@app.post("/check-answer")
async def check_answer_endpoint(
    request: Request,
    user_answer: str = Form(...),
    correct_answer: str = Form(...),
    threshold: Optional[float] = Form(0.7)
):
    """
    Test endpoint for fuzzy answer matching
    """
    try:
        result = answer_matcher.check_answer(user_answer, correct_answer, threshold)
        return JSONResponse(content={
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "match_result": result,
            "threshold": threshold
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking answer: {str(e)}")

@app.post("/debug-questions")
async def debug_question_generation(
    request: Request,
    pdf_filename: str = Form(...),
    num_ques: str = Form(...)
):
    """
    Debug endpoint to test question generation and see exactly how many questions are created
    """
    try:
        pdf_filepath = f"docs/{pdf_filename}"
        logger.info(f"Debug: Generating {num_ques} questions from {pdf_filename}")
        
        if not os.path.exists(pdf_filepath):
            raise HTTPException(status_code=404, detail="PDF file not found")
        
        # Generate questions
        questions_and_answers = llm_pipeline(pdf_filepath, num_ques)
        
        # Format preview for multiple-choice questions
        preview_formatted = []
        for i, qa in enumerate(questions_and_answers[:3]):
            if 'options' in qa:  # Multiple-choice format
                preview_formatted.append({
                    "question": qa['question'],
                    "options": qa['options'],
                    "correct_answer": qa['correct_answer'],
                    "answer": qa['answer'],  # This is now the letter (A/B/C/D)
                    "correct_option_text": qa.get('correct_option_text', '')  # The actual text
                })
            else:  # Simple Q&A format (fallback)
                preview_formatted.append(qa)
        
        return JSONResponse(content={
            "requested_questions": int(num_ques),
            "generated_questions": len(questions_and_answers),
            "success": len(questions_and_answers) > 0,
            "questions_preview": preview_formatted,
            "all_questions": questions_and_answers,
            "format": "multiple_choice" if questions_and_answers and 'options' in questions_and_answers[0] else "simple_qa",
            "message": f"Successfully generated {len(questions_and_answers)} multiple-choice questions out of {num_ques} requested"
        })
        
    except Exception as e:
        logger.error(f"Error in debug question generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")

@app.post("/analyze")
async def analyze_file(
    request: Request,
    pdf_filename: str = Form(...),
    quiz_name: str = Form(...),  # New field for quiz name
    num_ques: str = Form(...),
    time: Optional[str] = Form(None)
):
    # Construct the file path
    pdf_filepath = f"docs/{pdf_filename}"
    logger.info(f"Received PDF filename for analysis: {pdf_filename}")
    # Check if the PDF file exists
    if not os.path.exists(pdf_filepath):
        raise HTTPException(status_code=404, detail="PDF file not found")

    try:
        # Pass parameters to get_json, including optional `time`
        output_message = get_json(pdf_filepath, os.path.basename(pdf_filename), quiz_name, num_ques, time)
        
        # Get the inserted MongoDB document ID
        inserted_id = output_message.get("inserted_id")
        
        # Retrieve the document from MongoDB by ID
        questions_data = questions_collection.find_one({"_id": ObjectId(inserted_id)})
        
        if not questions_data:
            raise HTTPException(status_code=404, detail="Questions not found in the database.")
        
        # Prepare the response data, including `quiz_time` and `quiz_name` fields
        response_data = {
            "pdf_name": pdf_filename,
            "quiz_name": questions_data.get("quiz_name", "Untitled Quiz"),
            "questions_and_answers": questions_data.get("questions_and_answers"),
            "quiz_time": questions_data.get("quiz_time", "No time specified")  # Default if time is None
        }
        
        # Return the response as JSON
        return JSONResponse(content=response_data)

    except HTTPException as e:
        # Raise HTTP errors as is for specific handling
        raise e
    except Exception as e:
        # Handle any other unexpected exceptions
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
