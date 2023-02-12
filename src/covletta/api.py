import io
import logging
import os
import json

import openai
import PyPDF2
from fastapi import FastAPI, File, HTTPException, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class ModelInputs(BaseModel):
    job_ad_text: str = None
    temperature: float = 0.9
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.6
    first_name: str
    last_name: str
    email: str = None
    phone: str = None
    bool_title: bool = True


class LetterInputs(BaseModel):
    letter_text: str
    first_name: str
    last_name: str
    email: str = None
    phone: str = None
    bool_title: bool = True


if os.environ.get("K_SERVICE"):
    # Setup logging if we're in a cloud run environment
    from google.cloud.logging import Client as LoggingClient

    logging_client = LoggingClient()
    logging_client.setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

## TODO: ADD OPEN AI API KEY
openai.api_key = "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"


@app.get("/")
async def root():
    return {"message": "Covletta API"}


@app.post("/predict")
async def upload_file_and_read(
    modelInputs: ModelInputs = Depends(), cv_file: UploadFile = File(...)
):
    allowed_contents = ["pdf", "officedocument"]
    if not any(x in cv_file.content_type for x in allowed_contents):
        raise HTTPException(
            status_code=400, detail=f"File '{cv_file.filename}' is not a valid document."
        )

    content = await cv_file.read()
    document = io.BytesIO(content)
    pdf = PyPDF2.PdfReader(document)
    cv_string = parse_pdf(pdf)


    prompt = f""" The following is my resume:\n {cv_string}.\n\n
    Please write a fitting cover letter for the following job ad:\n\n
    {modelInputs.job_ad_text}.
    """
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=modelInputs.temperature,
        max_tokens=2000,
        top_p=modelInputs.top_p,
        frequency_penalty=modelInputs.frequency_penalty,
        presence_penalty=modelInputs.presence_penalty,
    )

    letter_raw_text = response["choices"][0]["text"]


    response = {
        "raw":  json.dumps(letter_raw_text, ensure_ascii=False),
    }

    return response

def parse_pdf(pdf):
    output = ""
    for page in range(len(pdf.pages)):
        page_obj = pdf.pages[page]
        output += page_obj.extract_text()
    return output
