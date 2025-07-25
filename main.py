import dotenv
import asyncio
import traceback
import tempfile
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Annotated
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from constants import BATCH_SIZE, LANGUAGES, SOURCE_LANGUAGES
from summarizer import Summarizer
from googletrans import Translator

dotenv.load_dotenv()
app = FastAPI()
summarizer = Summarizer()
translator = Translator()

@app.post("/upload-pdf")
async def upload_pdf(
    file: Annotated[UploadFile, File(...)],
    source_lang: Annotated[str, Form(...)],
    target_lang: Annotated[str, Form(...)]):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        contents = await file.read()
        result_string = await process_pdf_with_langchain(
            contents, target_lang, source_lang
        )

        return JSONResponse(content={"message": "PDF processed successfully", "result": result_string})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get('/get-volunteer-information/{volunteer_id}')
async def get_volunteer_information(volunteer_id: str):
    feedbacks = get_volunteer_feedbacks(volunteer_id)
    formatted_feedbacks = "\n".join([f'- {feedback['message']}' for feedback in feedbacks])

    prompt = f'''
Saya mempunyai data saran untuk sebuah pengajar, bantukan saya membuat ringkasannya

{formatted_feedbacks}

Tolong berikan jawaban yang tidak bertele - tele
    '''

    return summarizer.answer(prompt)

def get_volunteer_feedbacks(volunteer_id: str):
    url = f"http://localhost:6969/api/feedback/volunteer/{volunteer_id}"  # Change to your actual URL and port
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        feedbacks = response.json()
        return feedbacks
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

async def process_pdf_with_langchain(
        pdf_bytes: bytes,
        target_language: str,
        source_language: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_path = Path(tmp_file.name)

    try:
        loader = PyPDFLoader(str(tmp_path))
        documents = loader.load()
        full_text = "\n\n".join([doc.page_content for doc in documents])
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " "]
        )
        chunks = text_splitter.split_text(full_text)
        
        chunks = [chunk.replace('\n', ' ') for chunk in chunks]
        full_result = ''
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            result = conclude_chunks(batch  )
            full_result += '\n' + result
        
        with open("result.txt", "w", encoding="utf-8") as f:
            f.write(full_result)

        # # ! RUNNING WITHOUT TRANSLATION TO PREVENT RATE LIMIT FROM GOOGLE TRANSLATE
        # return full_result

        translated_result = await translate_with_google(
                text=full_result,
                target_language=target_language,
                source_language=source_language
            )

        return translated_result.text
    except Exception as e:
        print(f"Error processing pdf {e}")
        raise e
    finally:
        tmp_path.unlink(missing_ok=True)  # Clean up temp file

def conclude_chunks(chunks: List[str]) -> str:
    formatted_chunks = "\n".join(f"- {chunk}" for chunk in chunks)
    prompt = f"""
I have chunks of paragraph from a pdf can you help me conclude it

{formatted_chunks}

please make sure your answer has no additional introduction just the summarized text
    """

    return summarizer.answer(prompt)

async def translate_with_google(text: str, target_language: str, source_language: str):
    try:
        destination = LANGUAGES[target_language]
        source = SOURCE_LANGUAGES[source_language]
        result = await translator.translate(text, src=source, dest=destination)

        return result
    except KeyError as key_error:
        print(f'Target: {target_language}, Source: {source_language}')
        traceback.print_exc()
        raise key_error
    except Exception as e:
        print(f"Error translating: {e}")
        traceback.print_exc()
        raise e
