
import os
import json
import queue
import threading
from pathlib import Path
from werkzeug.utils import secure_filename
import torch
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import gen_data

app = Flask(__name__)
CORS(app)

# Global variables
model_name = 'google/flan-t5-small'
tokenizer = None
base_model = None
current_model = None
DATA_PATH = None
DATA_SIZE = 0
log_queue = queue.Queue()
is_training = False

def load_base_model():
    """Load the base FLAN-T5 model and tokenizer"""
    global tokenizer, base_model, current_model
    print("Loading FLAN-T5 Small model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    current_model = base_model
    print("Model loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        pdf = request.files['file']
        if pdf.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save uploaded file
        os.makedirs("uploads", exist_ok=True)
        filename = secure_filename(pdf.filename)
        path = os.path.join("uploads", filename)
        pdf.save(path)

        # Generate dataset using gen_data.py
        global DATA_PATH, DATA_SIZE
        dataset = gen_data.get_dataset(path)

        # Set dataset path
        pdf_name = Path(filename).stem
        DATA_PATH = f"{pdf_name}_dataset.json"

        # Count dataset size
        if os.path.exists(DATA_PATH):
            with open(DATA_PATH, 'r') as f:
                data = json.load(f)
                DATA_SIZE = len(data)

        return jsonify({'pairs': DATA_SIZE, 'message': 'Dataset generated successfully!'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dataset')
def get_dataset():
    try:
        if DATA_PATH and os.path.exists(DATA_PATH):
            with open(DATA_PATH, 'r') as f:
                data = json.load(f)
            # Return first 20 pairs for preview
            return jsonify(data[:20])
        return jsonify({'error': 'No dataset available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        global current_model, tokenizer

        if current_model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.get_json()
        prompt = data.get('prompt', '')

        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        # Generate response
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)

        with torch.no_grad():
            outputs = current_model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def training_worker(method, epochs, rank):
    """Background training worker"""
    global current_model, is_training, log_queue

    try:
        is_training = True
        log_queue.put("Starting training preparation...")

        # Load fresh base model for training
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Load dataset
        if not DATA_PATH or not os.path.exists(DATA_PATH):
            log_queue.put("ERROR: No dataset found!")
            return

        with open(DATA_PATH, 'r') as f:
            data = json.load(f)

        log_queue.put(f"Loaded {len(data)} Q&A pairs")

        # Prepare dataset
        def preprocess(example):
            input_text = example['question']
            target_text = example['answer']
            model_inputs = tokenizer(input_text, truncation=True, padding='max_length', max_length=128)
            labels = tokenizer(target_text, truncation=True, padding='max_length', max_length=128)
            model_inputs['labels'] = labels['input_ids']
            return model_inputs

        dataset = Dataset.from_list(data)
        processed_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

        log_queue.put("Dataset preprocessed")

        # Configure model based on training method
        if method == "lora":
            log_queue.put(f"Setting up LoRA training with rank {rank}...")
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=rank * 4,
                target_modules=['q', 'v'],
                lora_dropout=0.1,
                bias='none',
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        else:
            log_queue.put("Setting up full fine-tuning...")

        from transformers import GenerationConfig
        gen_config = GenerationConfig.from_pretrained(model_name)
        model.generation_config = gen_config

        # Training arguments
        training_args = TrainingArguments(
            output_dir=f'flan-t5-small-{method}',
            per_device_train_batch_size=4,
            num_train_epochs=epochs,
            learning_rate=1e-2,
            logging_steps=2,
            save_strategy='no',
            fp16=False,
            dataloader_pin_memory=False,
        )
        training_args.generation_config = gen_config

        collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        # # Custom logging callback
        # class CustomLoggingCallback:
        #     def __init__(self, log_queue):
        #         self.log_queue = log_queue
        #         self.step_count = 0

        #     def on_log(self, logs):
        #         if 'loss' in logs:
        #             self.step_count += 1
        #             epoch = logs.get('epoch', 0)
        #             loss = logs.get('loss', 0)
        #             self.log_queue.put(f"Epoch {epoch:.1f} | Step {self.step_count} | Loss: {loss:.4f}")


        from transformers import TrainerCallback

        class CustomLoggingCallback(TrainerCallback):
            def __init__(self, log_queue):
                self.log_queue = log_queue
                self.step_count = 0

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and 'loss' in logs:
                    self.step_count += 1
                    epoch = state.epoch
                    loss = logs['loss']
                    self.log_queue.put(f"Epoch {epoch:.1f} | Step {self.step_count} | Loss: {loss:.4f}")

            def on_train_begin(self, args, state, control, **kwargs):
                self.log_queue.put("Training has started.")

            def on_train_end(self, args, state, control, **kwargs):
                self.log_queue.put("Training has ended.")


        # Create trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset,
            data_collator=collator,
        )

        # Add custom callback
        callback = CustomLoggingCallback(log_queue)
        trainer.add_callback(callback)

        log_queue.put("Starting training...")
        trainer.train()

        # Save model and update current_model
        if method == "lora":
            model.save_pretrained('flan_t5_small_lora_adapter')
            log_queue.put("LoRA adapter saved")
            # Merge and update current model
            merged_model = model.merge_and_unload()
            current_model = merged_model
        else:
            model.save_pretrained(f'flan-t5-small-full-{epochs}epochs')
            log_queue.put("Full model saved")
            current_model = model

        log_queue.put("Training complete! Model updated for chat.")

    except Exception as e:
        log_queue.put(f"Training error: {str(e)}")
    finally:
        is_training = False

@app.route('/train', methods=['POST'])
def start_training():
    global is_training

    if is_training:
        return jsonify({'error': 'Training already in progress'}), 400

    try:
        data = request.get_json()
        method = data.get('method', 'lora')  # 'lora' or 'full'
        epochs = int(data.get('epochs', 3))
        rank = int(data.get('rank', 8))

        if method not in ['lora', 'full']:
            return jsonify({'error': 'Invalid training method'}), 400

        if not DATA_PATH or not os.path.exists(DATA_PATH):
            return jsonify({'error': 'No dataset available for training'}), 400

        # Clear log queue
        while not log_queue.empty():
            log_queue.get()

        # Start training in background thread
        thread = threading.Thread(target=training_worker, args=(method, epochs, rank))
        thread.daemon = True
        thread.start()

        return jsonify({'status': 'started', 'message': f'{method.upper()} training started'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train_stream')
def training_stream():
    """Server-sent events for training logs"""
    def generate():
        while True:
            try:
                # Get log message with timeout
                message = log_queue.get(timeout=1)
                yield f"data: {message}\n\n"
                if "Training complete" in message or "Training error" in message:
                    break
            except queue.Empty:
                if not is_training:
                    break
                continue

    return Response(generate(), content_type='text/event-stream')

if __name__ == '__main__':
    # Load model on startup
    load_base_model()
    app.run(debug=True, threaded=True)
