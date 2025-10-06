from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import torch
import torch.nn as nn
import librosa
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = 'models/raga_lstm.pth'
LABEL_ENCODER_PATH = 'models/label_encoder_classes.npy'
HIDDEN_DIM = 128
MAX_LEN = 500      # max frames per sequence
HOP_LEN = 250      # frames per chunk

# -----------------------
# LSTM Model
# -----------------------
class RagaLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = torch.cat((hn[-2], hn[-1]), dim=1)
        return self.fc(out)

# -----------------------
# Helper Functions
# -----------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_features_chunk(y_chunk, sr):
    mfccs = librosa.feature.mfcc(y=y_chunk, sr=sr, n_mfcc=13).T
    chroma = librosa.feature.chroma_stft(y=y_chunk, sr=sr).T
    mel = librosa.feature.melspectrogram(y=y_chunk, sr=sr).T
    contrast = librosa.feature.spectral_contrast(y=y_chunk, sr=sr).T
    tonnetz = librosa.feature.tonnetz(y=y_chunk, sr=sr).T
    return np.hstack([mfccs, chroma, mel, contrast, tonnetz])

def extract_features_stream(file_path, max_len=MAX_LEN, hop_len=HOP_LEN):
    y, sr = librosa.load(file_path, sr=None)
    frame_hop = int(len(y) / max_len)
    sequences = []

    # Slide window over audio in hop_len steps
    for start in range(0, len(y), frame_hop * hop_len):
        end = start + frame_hop * hop_len
        y_chunk = y[start:end]
        if len(y_chunk) == 0:
            continue
        feat_chunk = extract_features_chunk(y_chunk, sr)
        # pad or truncate
        if feat_chunk.shape[0] < max_len:
            pad_width = max_len - feat_chunk.shape[0]
            feat_chunk = np.pad(feat_chunk, ((0, pad_width), (0,0)), mode='constant')
        else:
            feat_chunk = feat_chunk[:max_len, :]
        sequences.append(feat_chunk)

    # Combine sequences along batch dimension
    return np.stack(sequences, axis=0).astype(np.float32)  # (num_chunks, max_len, feat_dim)

# -----------------------
# Prediction
# -----------------------
def predict_raga(file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        le_classes = np.load(LABEL_ENCODER_PATH, allow_pickle=True)
        features = extract_features_stream(file_path)
        features = torch.tensor(features).to(device)

        input_dim = features.shape[2]
        model = RagaLSTMClassifier(input_dim, HIDDEN_DIM, len(le_classes)).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

        with torch.no_grad():
            outputs = model(features)  # (num_chunks, num_classes)
            probs = torch.softmax(outputs, dim=1)
            avg_probs = probs.mean(dim=0)
            pred_idx = torch.argmax(avg_probs).item()
            confidence = avg_probs[pred_idx].item()

        return le_classes[pred_idx], confidence

    except Exception as e:
        return None, str(e)

# -----------------------
# Flask Routes
# -----------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    raga, confidence = predict_raga(filepath)
    os.remove(filepath)

    if raga is None:
        return jsonify({'error': f'Prediction failed: {confidence}'}), 500

    return jsonify({'raga': str(raga), 'confidence': float(confidence)*100})

# -----------------------
# Run App
# -----------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
