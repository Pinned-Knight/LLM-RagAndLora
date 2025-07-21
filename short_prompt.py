def prompt_template(data: str, num_records: int = 5):

    return f"""You are an expert data curator assisting a machine learning engineer in creating a high-quality instruction tuning dataset. Your task is to transform 
    the provided data chunk into **diverse question and answer (Q&A) pairs** for fine-tuning a language model.  

    For each of the {num_records} entries, generate **one short, well-structured question** that captures key information from the data. Keep questions concise (one sentence max).  
    Ensure the **answer is extremely brief—no more than 4 words.** Focus on precision and clarity.  

    Structure your output in JSON format, where each object contains 'question' and 'answer' fields. The JSON structure should look like this:

        "question": "Your concise question here...",
        "answer": "Your very short answer here..."

    Guidelines:
    - **Questions:** Simple, direct, and easy to understand.
    - **Answers:** Extremely short (1–4 words max), factual, and accurate.
    - **Diversity:** Cover different aspects of the data to avoid repetitive pairs.
    - **Neutrality:** Avoid any sensitive or biased content.

    By following these instructions, you will help build a compact and efficient dataset tailored for models optimized for brief responses.

    ---

    Data:
    {data}
    """
