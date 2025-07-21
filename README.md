# 🤖 FLAN-T5 Fine-tuning Playground

A complete end-to-end web application for fine-tuning Google's FLAN-T5 Small model with your own PDF documents. Upload a PDF, generate a question-answer dataset, and fine-tune the model using either LoRA or full fine-tuning - all through an intuitive web interface.

## ✨ Features

- **📄 PDF Processing**: Upload PDF documents and automatically generate question-answer datasets
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
- **Learning Rate**: 1e-4 (configurable in `app.py`)  
- **FP16**: Enabled by default for efficiency
- **Logging Steps**: Every 5 steps

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
| `/train` | POST | Start training process |
| `/train_stream` | GET | Server-sent events for training logs |

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
    learning_rate=1e-4,            # Experiment with different rates
    logging_steps=5,
    evaluation_strategy='no',
    save_strategy='no',
    fp16=True,                     # Set to False if no GPU
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
- **Ollama**: For local LLM inference capabilities

---

*Built with ❤️ for the AI community*
