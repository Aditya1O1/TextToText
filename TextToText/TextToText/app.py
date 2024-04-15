from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import MarianMTModel, MarianTokenizer
import logging
import os
import zipfile
from azure.storage.blob import BlobServiceClient
from tqdm import tqdm


def is_directory_empty(directory):
    return not os.listdir(directory)


def download_blob_with_progress_and_extract(storage_connection_string, container_name, blob_name, extract_path):
    # Check if the extraction directory is empty
    if not is_directory_empty(extract_path):
        print("Files are already downloaded.")
        return

    # Create the BlobServiceClient object which will be used to create a container client
    blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)

    # Get a specific container
    container_client = blob_service_client.get_container_client(container_name)

    # Get a specific blob
    blob_client = container_client.get_blob_client(blob_name)

    # Get the blob properties to determine the blob size
    blob_properties = blob_client.get_blob_properties()
    total_bytes = blob_properties.size

    # Initialize tqdm with the total number of bytes to download
    progress_bar = tqdm(total=total_bytes, unit='B', unit_scale=True, desc="Downloading")

    # Download the blob data and save it to a temporary file
    temp_zip_file_path = os.path.join(extract_path, "temp.zip")
    with open(temp_zip_file_path, "wb") as f:
        download_stream = blob_client.download_blob()
        bytes_downloaded = 0
        for chunk in download_stream.chunks():
            bytes_downloaded += len(chunk)
            f.write(chunk)
            progress_bar.update(len(chunk))

    # Close the progress bar
    progress_bar.close()

    # Extract the contents of the zip file
    with zipfile.ZipFile(temp_zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Delete the temporary zip file
    os.remove(temp_zip_file_path)


# Replace these variables with your Azure storage connection string, container name, blob name, and local file path
storage_connection_string = "DefaultEndpointsProtocol=https;AccountName=translationapi123;AccountKey=PRvaIk1zNuG+Rok6ZrkgmeQTAacr8z0CMPUwrWmyWEpB4LaDJCHbN+R6xNtwrbuIjew2Nz1gQvAv+AStb3BA3w==;EndpointSuffix=core.windows.net"
container_name = "translationapp"
blob_name = "Fine_Tune_Weight.zip"
extract_path = "./Weight"  # Extract to this directory

download_blob_with_progress_and_extract(storage_connection_string, container_name, blob_name, extract_path)

# Code for translation

# Created a FastAPI application instance named app
app = FastAPI()

# Configures the logging system to output log messages with
# the DEBUG level or higher.
logging.basicConfig(level=logging.DEBUG)

# Allowing CORS for development purposes
# Adds CORS middleware to the FastAPI application.
# This middleware allows cross-origin requests from any origin ("*"),
# including credentials (cookies, authorization headers), for POST requests,
# and allows any headers to be sent
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Load fine-tuned models and tokenizers for each language
models = {
    "hi": MarianMTModel.from_pretrained("./Weight/Fine_Tune_Weight/FineTuneHelsinkiTransformer_en_hi"),
    "pl": MarianMTModel.from_pretrained("./Weight/Fine_Tune_Weight/FineTunedHelsinkiTransformer_en_pol"),
    "fr": MarianMTModel.from_pretrained("./Weight/Fine_Tune_Weight/FineTunedHelsinkiTransformer_en_Fr"),
    "zh": MarianMTModel.from_pretrained("./Weight/Fine_Tune_Weight/FineTunedHelsinkiTransformer_en_zh"),
    "de": MarianMTModel.from_pretrained("./Weight/Fine_Tune_Weight/FineTunedHelsinkiTransformer_en_de"),
    "jap": MarianMTModel.from_pretrained("./Weight/Fine_Tune_Weight/FineTuneHelsinkiTransformer_en_jap"),
    "ar": MarianMTModel.from_pretrained("./Weight/Fine_Tune_Weight/FineTuneHelsinkiTransformer_en_ar"),
    "tl": MarianMTModel.from_pretrained("./Weight/Fine_Tune_Weight/FineTuneHelsinkiTransformer_en_tl"),
    "ur": MarianMTModel.from_pretrained("./Weight/Fine_Tune_Weight/FineTunedHelsinkiTransformer_en_ur")
}

tokenizers = {
    "hi": MarianTokenizer.from_pretrained("./Weight/Fine_Tune_Weight/FineTuneHelsinkiTransformer_en_hi"),
    "pl": MarianTokenizer.from_pretrained("./Weight/Fine_Tune_Weight/FineTunedHelsinkiTransformer_en_pol"),
    "fr": MarianTokenizer.from_pretrained("./Weight/Fine_Tune_Weight/FineTunedHelsinkiTransformer_en_Fr"),
    "zh": MarianTokenizer.from_pretrained("./Weight/Fine_Tune_Weight/FineTunedHelsinkiTransformer_en_zh"),
    "de": MarianTokenizer.from_pretrained("./Weight/Fine_Tune_Weight/FineTunedHelsinkiTransformer_en_de"),
    "jap": MarianTokenizer.from_pretrained("./Weight/Fine_Tune_Weight/FineTuneHelsinkiTransformer_en_jap"),
    "ar": MarianTokenizer.from_pretrained("./Weight/Fine_Tune_Weight/FineTuneHelsinkiTransformer_en_ar"),
    "tl": MarianTokenizer.from_pretrained("./Weight/Fine_Tune_Weight/FineTuneHelsinkiTransformer_en_tl"),
    "ur":MarianTokenizer.from_pretrained("./Weight/Fine_Tune_Weight/FineTunedHelsinkiTransformer_en_ur")
}


# Define translation function
def translate_text(text, language):
    model = models.get(language)
    tokenizer = tokenizers.get(language)
    if model is None or tokenizer is None:
        raise HTTPException(status_code=400, detail=f"Language '{language}' not supported")

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated_ids = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
    return translated_text[0]


@app.post("/translate/")
async def translate_text_api(text: str = Form(...), language: str = Form(...)):
    if not text:
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    if len(text) > 512:
        raise HTTPException(status_code=400, detail="Text input is too long, maximum length is 512 characters")
    try:
        translated_text = translate_text(text, language)
        return {"translated_text": translated_text}
    except Exception as e:
        logging.exception("Translation failed")
        raise HTTPException(status_code=500, detail="Translation failed")


