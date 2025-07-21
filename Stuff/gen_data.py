import warnings
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true")

import json
import os
from typing import List 
from pydantic import BaseModel
from litellm import completion
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from colorama import Fore 
from short_prompt import prompt_template

class Record(BaseModel):
    question: str
    answer: str

class Response(BaseModel):
    generated: List[Record]

def llm_call(data: str, num_records: int = 5) -> dict:
    stream = completion(
        model="ollama_chat/phi3-mini:latest",
        messages=[
            {
                "role": "user",
                "content": prompt_template(data, num_records),
            }
        ],
        stream=True,
        options={"num_predict": 2000},
        format=Response.model_json_schema()
    )

    response_data = ""  # accumulator for streamed output
    for x in stream:
        delta = x['choices'][0]["delta"]["content"]
        if delta is not None:
            print(Fore.LIGHTBLUE_EX + delta + Fore.RESET, end="")
            response_data += delta  # accumulate the streamed chunks

    # Now response_data should be a complete JSON string
    return json.loads(response_data)

def get_dataset(pdf_path: str):
    """
    Process a PDF file to generate question-answer pairs and save as JSON.

    Args:
        pdf_path (str): Path to the PDF file to process

    Returns:
        dict: The complete dataset containing generated Q&A pairs and their context
    """

    print("chunking data.....")

    # Initialize converter and process the PDF
    converter = DocumentConverter()
    doc = converter.convert(pdf_path).document
    chunker = HybridChunker()
    chunks = chunker.chunk(doc)

    print("processing_chunks.......")

    # Process each chunk
    dataset = {}
    all_instructions = []

    for i, chunk in enumerate(chunks):
        enriched_text = chunker.contextualize(chunk=chunk)
        data = llm_call(enriched_text)

        # Store the generated data (uncommented from original notebook)
        dataset[i] = {"generated": data["generated"], "context": enriched_text}

        # Add to instructions list for JSON file
        for pairs in data['generated']:
            all_instructions.append(pairs)

    # Generate output filename based on input PDF name
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_filename = f"{pdf_name}_dataset.json"

    # Save the instructions to JSON file (same as original notebook)
    with open(output_filename, 'w') as f:
        json.dump(all_instructions, f, indent=2)

    # Print completion message
    num_records = len(all_instructions)
    print(f"completed. {output_filename} created with {num_records} Question answer pairs")

    return dataset

if __name__ == "__main__":
    # Example usage (uncomment to test)
    result = get_dataset("Split - test.pdf")
    print(f"Dataset contains {len(result)} chunks")
    pass
