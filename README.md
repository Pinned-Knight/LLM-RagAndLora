# 🤖 FLAN-T5 Fine-tuning Playground

A complete end-to-end web application for fine-tuning Google's FLAN-T5 Small model with your own PDF documents. Upload a PDF, generate a question-answer dataset, and fine-tune the model using either LoRA or full fine-tuning - or skip training and answer questions straight from the PDF with **Retrieval-Augmented Generation (RAG)** - all through an intuitive web interface. A benchmark script compares the approaches on held-out questions.

## ✨ Features

- **📄 PDF Processing**: Upload PDF documents and automatically generate question-answer datasets
- **🔎 RAG**: PDFs are chunked, embedded with a Hugging Face sentence-transformer and indexed in FAISS; tick "Use RAG" in the chat to answer from the retrieved chunks (the sources are shown)
- **📏 Benchmark**: `benchmark.py` compares the base model, RAG, LoRA and RAG + LoRA on held-out questions (exact match, F1, ROUGE-L, latency)
- **💬 Interactive Chat**: Chat with the FLAN-T5 model before and after fine-tuning
- **🎯 Two Training Methods**: Choose between LoRA (efficient) or full fine-tuning
- **📊 Real-time Training Logs**: Watch training progress with live streaming logs
- **🔍 Dataset Preview**: Preview generated Q&A pairs before training
- **🎛️ Configurable Parameters**: Adjust epochs, LoRA rank, and other training settings

## 🏗️ Project Structure

```
flan_t5_playground/
├── app.py                  # Flask backend server
├── gen_data.py            # PDF processing and dataset generation
├── short_prompt.py        # LLM prompt templates (user provided)
├── rag.py                 # RAG: chunk -> embed (Hugging Face) -> FAISS -> retrieve -> generate
├── benchmark.py           # base vs RAG vs LoRA on held-out questions
├── tests/                 # unit tests: chunking, metrics, prompt building, retrieval
├── results/               # benchmark output (json + markdown)
├── Stuff/                 # experiments: notebooks, sample PDFs, generated Q&A sets
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Single-page web interface
├── static/
│   ├── style.css          # Styling and responsive design
│   └── app.js             # Frontend JavaScript functionality
├── uploads/               # PDF upload directory (auto-created)
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (Python 3.10+ recommended)
- CUDA-compatible GPU (optional but recommended for faster training)
- Ollama with phi3-mini model (for dataset generation)

### Installation

1. **Clone or create the project directory**:
```bash
mkdir flan_t5_playground
cd flan_t5_playground
```

2. **Create and activate a virtual environment**:
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up Ollama** (for dataset generation):
```bash
# Install Ollama from https://ollama.ai/
ollama pull phi3-mini:latest
```

5. **Optional - Hugging Face Login**:
```bash
huggingface-cli login
```
*Note: Only required if you want to push trained models to Hugging Face Hub*

### Running the Application

1. **Start the Flask server**:
```bash
python app.py
```

2. **Open your browser** and navigate to:
```
http://localhost:5000
```

3. **Start using the application**:
   - Upload a PDF document
   - Wait for dataset generation to complete
   - Chat with the base model
   - Configure training settings
   - Start fine-tuning
   - Chat with the fine-tuned model

## 📖 Usage Guide

### Step 1: PDF Upload & Dataset Generation

1. Click the upload area or drag-and-drop a PDF file
2. Wait for the system to:
   - Process and chunk the PDF content
   - Generate question-answer pairs using the phi3-mini model
   - Save the dataset as JSON
3. Click **"Preview Dataset"** to review the generated Q&A pairs

### Step 2: Base Model Chat

- The FLAN-T5 Small model is loaded automatically on startup
- Use the chat interface to interact with the base model
- This helps you understand the model's baseline performance

### Step 2b: Chat with RAG (no training needed)

- Every uploaded PDF is indexed for retrieval as soon as it is uploaded (several PDFs share one index)
- Tick **"Use RAG"** above the chat box: the question is embedded, the top-3 most similar chunks are retrieved, and the model answers from them
- Click **Sources** under an answer to see which chunks were used and their similarity scores
- RAG uses whichever model is currently loaded, so it also works on top of a fine-tuned model

### Step 3: Fine-tuning Configuration

Choose your training method:

**LoRA (Low-Rank Adaptation) - Recommended**:
- More efficient and faster
- Requires less memory
- Adjust the rank parameter (default: 8)
- Good results with fewer epochs

**Full Fine-tuning**:
- Trains all model parameters
- Requires more memory and time
- Potentially better performance on domain-specific tasks

**Training Parameters**:
- **Epochs**: Number of training iterations (1-10)
- **LoRA Rank**: Only for LoRA method (1-64, default: 8)

### Step 4: Training Process

1. Click **"Start Training"** 
2. Monitor real-time logs in the training section
3. Training progress includes:
   - Current epoch and step
   - Training loss
   - Estimated time remaining

### Step 5: Testing Fine-tuned Model

- Once training completes, the chat interface automatically switches to the fine-tuned model
- Ask questions related to your PDF content
- Compare responses with your memory of the base model performance

## 🔧 Configuration

### Training Parameters

The training configuration can be adjusted in the web interface:

- **Batch Size**: Fixed at 4 (configurable in `app.py`)
- **Learning Rate**: 1e-2 (configurable in `app.py`)  
- **FP16**: Off by default (CPU-friendly); set `fp16=True` on a CUDA GPU
- **Logging Steps**: Every 2 steps

### Model Configuration

- **Base Model**: `google/flan-t5-small`
- **Max Input Length**: 128 tokens
- **Max Output Length**: 100 tokens
- **Temperature**: 0.7

### Dataset Generation

The `gen_data.py` script uses:
- **Docling** for PDF processing and chunking
- **phi3-mini** via Ollama for Q&A generation
- **Hybrid Chunking** for optimal content segmentation

## 📁 File Descriptions

### Core Files

- **`app.py`**: Flask web server with all API endpoints
- **`gen_data.py`**: PDF processing and dataset generation logic
- **`short_prompt.py`**: Prompt templates for LLM-based dataset generation

### Frontend Files

- **`templates/index.html`**: Complete web interface
- **`static/style.css`**: Modern, responsive styling
- **`static/app.js`**: Frontend JavaScript with real-time features

## 🛠️ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve main web interface |
| `/upload_pdf` | POST | Upload PDF and generate dataset |
| `/dataset` | GET | Retrieve generated dataset (first 20 pairs) |
| `/chat` | POST | Chat with current model |
| `/rag_chat` | POST | Answer from the uploaded PDF(s): retrieve top-k chunks, then generate (returns the sources too) |
| `/train` | POST | Start training process |
| `/train_stream` | GET | Server-sent events for training logs |

## 🔎 RAG pipeline (`rag.py`)

1. **Chunk**: PDF text (pypdf) is split into sentence-aware chunks of about 90 words with 20 words of overlap
2. **Embed**: each chunk is embedded with `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face)
3. **Index**: vectors go into a FAISS inner-product index; vectors are normalised, so the score is cosine similarity
4. **Retrieve**: the question is embedded and the top-k chunks (default 3) are fetched
5. **Generate**: chunks + question go into a prompt that says "answer using only the context, otherwise say I don't know"; FLAN-T5 answers with greedy decoding

Design notes: chunks are not numbered in the prompt (FLAN-T5-small answered "[1]" when they were); the lowest-ranked chunks are dropped if the prompt would not fit the 512-token window; several PDFs share one index, so questions can span documents from different domains.

## 📏 Benchmark: base vs RAG vs LoRA

`python benchmark.py` (add `--skip-lora` for a quick run) scores four ways of answering questions about a PDF with FLAN-T5-small on **held-out** questions: exact match (EM), token F1, ROUGE-L and latency. The gold Q&A pairs are the ones `gen_data.py` generated from each PDF with phi3-mini. Bolzano: 186 pairs, fixed-seed 70/30 split, LoRA trained on the 70%. Federer: 16 pairs, scored without any training. Numbers below are from one CPU run (raw output in `results/`).

**Philosophy of math (Bolzano), held-out n=56**

| System | Exact match | F1 | ROUGE-L | Latency / question |
|---|---|---|---|---|
| Base FLAN-T5-small, no context | 0.089 | 0.124 | 0.124 | 200 ms |
| RAG (top-3) + base FLAN-T5-small | 0.071 | 0.169 | 0.169 | 308 ms |
| LoRA fine-tuned, no context | 0.107 | 0.128 | 0.128 | 92 ms |
| RAG (top-3) + LoRA model | 0.143 | 0.207 | 0.207 | 240 ms |

LoRA: r=8, alpha=32, lr=1e-3, 8 epochs on 130 pairs; 344,064 of 77,305,216 weights trainable (0.45%); 66 s on CPU.

**Sports (Federer), n=16**

| System | Exact match | F1 | ROUGE-L | Latency / question |
|---|---|---|---|---|
| Base FLAN-T5-small, no context | 0.000 | 0.000 | 0.000 | 133 ms |
| RAG (top-3) + base FLAN-T5-small | 0.250 | 0.648 | 0.627 | 237 ms |

**How to read this**

- **Factual text (Federer): RAG is the clear win.** F1 goes from 0.00 to 0.65. Without retrieval the model guesses ("18 December 1897" for the birth date); with retrieval it answers "8 August 1981", "237", "103", "1968", "Laver Cup". EM stays low only because of phrasing ("five" vs "Five times").
- **Abstract text (Bolzano): everything is weak** (F1 0.12 to 0.21) and the differences are within noise at n=56 (one question is 1.8 points). RAG + base answered "I don't know" on 14 of 56 questions. RAG + LoRA scored best, partly because it almost never abstains (1 of 56), and an abstention scores zero.
- **Retrieval is the bottleneck.** The gold answer appears in the top-3 chunks for 50% (Federer) and 34% (Bolzano) of questions. This is a strict string match, so it undercounts correct paraphrases.
- RAG adds about 100 ms per question on CPU.

**Limits**: gold answers were written by a small LLM from the same chunks, so some are vague or wrong; EM / F1 / ROUGE-L punish correct paraphrases; the test sets are small; LoRA hyperparameters were not tuned; one run, no confidence intervals. The notebooks in `Stuff/` are earlier, separate experiments.

## 🎯 Advanced Usage

### Custom Dataset Format

If you want to use your own dataset instead of PDF generation, create a JSON file with this structure:

```json
[
  {
    "question": "Your question here?",
    "answer": "The corresponding answer."
  },
  {
    "question": "Another question?", 
    "answer": "Another answer."
  }
]
```

### Training Configuration

Modify training parameters in `app.py`:

```python
training_args = TrainingArguments(
    output_dir=f'flan-t5-small-{method}',
    per_device_train_batch_size=4,  # Adjust based on GPU memory
    num_train_epochs=epochs,
    learning_rate=1e-2,            # Experiment with different rates
    logging_steps=2,
    save_strategy='no',
    fp16=False,                    # Set to True on a CUDA GPU
    dataloader_pin_memory=False,
)
```

### Model Persistence

Trained models are saved to:
- LoRA: `./flan_t5_small_lora_adapter/`
- Full: `./flan-t5-small-full-{epochs}epochs/`

## ⚠️ Troubleshooting

### Common Issues

**1. CUDA/GPU Issues**:
```bash
# If you encounter GPU memory issues, try:
# - Reduce batch size in app.py
# - Set fp16=False if needed
# - Use LoRA instead of full fine-tuning
```

**2. Dataset Generation Fails**:
```bash
# Make sure Ollama is running:
ollama serve

# Verify phi3-mini is available:
ollama list
```

**3. Model Loading Errors**:
```bash
# Clear cache and retry:
rm -rf ~/.cache/huggingface/
huggingface-cli login
```

**4. Port Already in Use**:
```python
# In app.py, change the port:
app.run(debug=True, threaded=True, port=5001)
```

### Performance Tips

- **GPU Recommended**: Training is much faster with CUDA
- **Memory Management**: Use LoRA for large datasets or limited memory
- **Dataset Size**: Start with smaller datasets (50-200 pairs) for testing
- **Epochs**: 3-5 epochs usually sufficient for good results

## 🔒 Security Notes

- Files are uploaded to local `./uploads/` directory
- No data is sent to external services except for dataset generation (Ollama)
- Models and datasets remain on your local machine
- Clean up uploaded files periodically

## 🤝 Contributing

Feel free to submit issues and enhancement requests! Areas for improvement:

- Support for other model architectures
- Advanced training scheduling
- Multiple dataset management
- Model comparison features
- Batch processing capabilities

## 📄 License

This project is open source. Feel free to use, modify, and distribute as needed.

## 🙏 Acknowledgments

- **Hugging Face**: For the transformers library and FLAN-T5 model
- **Microsoft**: For the PEFT library enabling efficient fine-tuning
- **Docling**: For excellent PDF processing capabilities
- **Sentence-Transformers** and **FAISS**: For embeddings and vector search in the RAG pipeline
- **Ollama**: For local LLM inference capabilities

---

*Built with ❤️ for the AI community*
