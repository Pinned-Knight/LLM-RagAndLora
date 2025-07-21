// Global state
let isTraining = false;
let datasetAvailable = false;
let eventSource = null;

// DOM elements
const pdfFile = document.getElementById('pdfFile');
const dropZone = document.getElementById('dropZone');
const uploadStatus = document.getElementById('uploadStatus');
const previewBtn = document.getElementById('previewBtn');
const chatInput = document.getElementById('chatInput');
const chatMessages = document.getElementById('chatMessages');
const sendBtn = document.getElementById('sendBtn');
const chatStatus = document.getElementById('chatStatus');
const trainBtn = document.getElementById('trainBtn');
const trainBtnText = document.getElementById('trainBtnText');
const trainSpinner = document.getElementById('trainSpinner');
const trainingLogs = document.getElementById('trainingLogs');
const logsSection = document.getElementById('logsSection');
const datasetModal = document.getElementById('datasetModal');
const closeModal = document.getElementById('closeModal');
const datasetPreview = document.getElementById('datasetPreview');
const methodRadios = document.querySelectorAll('input[name="method"]');
const loraRankGroup = document.getElementById('loraRankGroup');
const epochs = document.getElementById('epochs');
const loraRank = document.getElementById('loraRank');

// Initialize the app
function init() {
    setupEventListeners();  
    enableChat(); // Enable chat on startup
}

// Setup event listeners
function setupEventListeners() {
    // File upload
    dropZone.addEventListener('click', () => pdfFile.click());
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);
    pdfFile.addEventListener('change', handleFileSelect);

    // Chat
    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Training
    trainBtn.addEventListener('click', startTraining);
    methodRadios.forEach(radio => {
        radio.addEventListener('change', toggleLoraRank);
    });

    // Modal
    closeModal.addEventListener('click', hideDatasetModal);
    datasetModal.addEventListener('click', (e) => {
        if (e.target === datasetModal) hideDatasetModal();
    });

    // Dataset preview
    previewBtn.addEventListener('click', showDatasetPreview);
}

// File upload handlers
function handleDragOver(e) {
    e.preventDefault();
    dropZone.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    dropZone.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    dropZone.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
}

function handleFile(file) {
    if (!file.type === 'application/pdf') {
        showStatus('error', 'Please select a PDF file.');
        return;
    }

    uploadPDF(file);
}

// Upload PDF and generate dataset
async function uploadPDF(file) {
    const formData = new FormData();
    formData.append('file', file);

    showStatus('info', '📤 Uploading PDF and generating dataset...');
    // disableTraining();
    previewBtn.style.display = 'none';

    try {
        const response = await fetch('/upload_pdf', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            showStatus('success', `✅ Dataset generated successfully! Created ${data.pairs} Q&A pairs.`);
            previewBtn.style.display = 'inline-block';
            // enableTraining();
            datasetAvailable = true;
        } else {
            showStatus('error', `❌ Error: ${data.error}`);
        }
    } catch (error) {
        showStatus('error', `❌ Upload failed: ${error.message}`);
    }
}

// Show status message
function showStatus(type, message) {
    uploadStatus.innerHTML = message;
    uploadStatus.className = `status-message status-${type}`;
    uploadStatus.style.display = 'block';
}

// Chat functions
function enableChat() {
    chatInput.disabled = false;
    sendBtn.disabled = false;
    chatInput.placeholder = 'Type your message...';
    chatStatus.innerHTML = '';
}

function disableChat(reason = 'Training in progress...') {
    chatInput.disabled = true;
    sendBtn.disabled = true;
    chatInput.placeholder = reason;
    chatStatus.innerHTML = `<div class="status-message status-info">💡 ${reason}</div>`;
}

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    addMessage('user', message);
    chatInput.value = '';
    chatInput.disabled = true;
    sendBtn.disabled = true;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prompt: message })
        });

        const data = await response.json();

        if (response.ok) {
            addMessage('bot', data.response);
        } else {
            addMessage('bot', `Error: ${data.error}`);
        }
    } catch (error) {
        addMessage('bot', `Error: ${error.message}`);
    }

    if (!isTraining) {
        chatInput.disabled = false;
        sendBtn.disabled = false;
    }
}

function addMessage(sender, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    if (sender === 'bot') {
        contentDiv.innerHTML = `<strong>FLAN-T5:</strong> ${content}`;
    } else {
        contentDiv.innerHTML = content;
    }

    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Training functions
// function enableTraining() {
//     if (datasetAvailable && !isTraining) {
//         // trainBtn.disabled = false;
//         trainBtn.removeAttribute('disabled'); // 🔥 this one clears DOM attribute
//         trainBtn.classList.remove('disabled');
//     }
// }

// function disableTraining() {
//     trainBtn.setAttribute('disabled', true);

// }

function toggleLoraRank() {
    const method = document.querySelector('input[name="method"]:checked').value;
    loraRankGroup.style.display = method === 'lora' ? 'block' : 'none';
}

async function startTraining() {
    if (!datasetAvailable) {
        alert('Please upload a PDF and generate a dataset first.');
        return;
    }

    const method = document.querySelector('input[name="method"]:checked').value;
    const epochsValue = parseInt(epochs.value);
    const rankValue = parseInt(loraRank.value);

    const payload = {
        method: method,
        epochs: epochsValue
    };

    if (method === 'lora') {
        payload.rank = rankValue;
    }

    try {
        const response = await fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (response.ok) {
            startTrainingUI();
            connectTrainingStream();
        } else {
            alert(`Training failed: ${data.error}`);
        }
    } catch (error) {
        alert(`Training failed: ${error.message}`);
    }
}

function startTrainingUI() {
    isTraining = true;
    
    // trainBtn.setAttribute('disabled', true);

    trainBtnText.textContent = 'Training...';
    trainSpinner.style.display = 'inline-block';
    disableChat('Training in progress...');
    logsSection.style.display = 'block';
    trainingLogs.textContent = '';
}

function finishTrainingUI() {
    isTraining = false;
    // trainBtn.disabled = false;
    // trainBtn.removeAttribute('disabled'); // 🔥 this one clears DOM attribute
    // trainBtn.classList.remove('disabled');
    trainBtnText.textContent = 'Start Training';
    trainSpinner.style.display = 'none';
    enableChat();

    // Add completion message to chat
    addMessage('bot', '🎉 Training completed! I\'ve been fine-tuned on your dataset. Try asking me questions related to your PDF!');
}

function connectTrainingStream() {
    if (eventSource) {
        eventSource.close();
    }

    eventSource = new EventSource('/train_stream');

    eventSource.onmessage = function(event) {
        const message = event.data;
        trainingLogs.textContent += message + '\n';
        trainingLogs.scrollTop = trainingLogs.scrollHeight;

        if (message.includes('Training complete') || message.includes('Training error')) {
            eventSource.close();
            finishTrainingUI();
        }
    };

    eventSource.onerror = function(event) {
        console.error('EventSource failed:', event);
        eventSource.close();
        finishTrainingUI();
    };
}

// Dataset preview functions
async function showDatasetPreview() {
    try {
        const response = await fetch('/dataset');
        const data = await response.json();

        if (response.ok) {
            renderDatasetPreview(data);
            showDatasetModal();
        } else {
            alert(`Failed to load dataset: ${data.error}`);
        }
    } catch (error) {
        alert(`Failed to load dataset: ${error.message}`);
    }
}

function renderDatasetPreview(data) {
    if (!data || data.length === 0) {
        datasetPreview.innerHTML = '<p>No data available</p>';
        return;
    }

    let html = `
        <div class="dataset-info">
            <p><strong>Showing first 20 question-answer pairs</strong></p>
        </div>
        <table class="preview-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Question</th>
                    <th>Answer</th>
                </tr>
            </thead>
            <tbody>
    `;

    data.forEach((item, index) => {
        html += `
            <tr>
                <td>${index + 1}</td>
                <td>${item.question || 'N/A'}</td>
                <td>${item.answer || 'N/A'}</td>
            </tr>
        `;
    });

    html += `
            </tbody>
        </table>
    `;

    datasetPreview.innerHTML = html;
}

function showDatasetModal() {
    datasetModal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

function hideDatasetModal() {
    datasetModal.style.display = 'none';
    document.body.style.overflow = 'auto';
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', init);

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (eventSource) {
        eventSource.close();
    }
});
