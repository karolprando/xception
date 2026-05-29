"""
=============================================================
  Deepfake Detector -- Backend Flask
  Baixa o modelo do Hugging Face automaticamente
=============================================================

Dependencias:
    pip install flask flask-cors torch torchvision timm pillow requests
"""

import io
import os
import requests
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# -------------------------------------------------------------
#  CONFIGURACOES
# -------------------------------------------------------------

MODEL_URL  = "https://huggingface.co/karolprando/xception/resolve/main/xception_140k.pth"
MODEL_PATH = "xception_140k.pth"
IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------
#  DOWNLOAD DO MODELO
# -------------------------------------------------------------

def baixar_modelo():
    if os.path.exists(MODEL_PATH):
        print(f"Modelo ja existe localmente: {MODEL_PATH}")
        return

    print(f"Baixando modelo do Hugging Face...")
    print(f"  URL: {MODEL_URL}")

    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    baixado = 0

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            baixado += len(chunk)
            if total:
                pct = baixado / total * 100
                print(f"  {pct:.1f}%  ({baixado/1e6:.1f} MB / {total/1e6:.1f} MB)", end="\r")

    print(f"\nModelo baixado com sucesso: {MODEL_PATH}")


# -------------------------------------------------------------
#  CARREGA MODELO
# -------------------------------------------------------------

def carregar_modelo():
    model = timm.create_model("xception", pretrained=False)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 2),
    )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "test_metrics" in checkpoint:
            acc = checkpoint["test_metrics"].get("accuracy", "N/A")
            print(f"  Acuracia no teste: {acc:.4f}")
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    print(f"  Modelo carregado! Dispositivo: {DEVICE}")
    return model


# Transformacao para inferencia
inferencia_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Inicializa
print("=" * 50)
print("  Deepfake Detector -- Iniciando")
print("=" * 50)
baixar_modelo()
model = carregar_modelo()
print("=" * 50)


# -------------------------------------------------------------
#  API FLASK
# -------------------------------------------------------------

app = Flask(__name__, static_folder=".")
CORS(app)


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Arquivo vazio"}), 400

    try:
        img_bytes = file.read()
        img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor    = inferencia_transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        fake_prob = probs[0].item()
        real_prob = probs[1].item()
        label     = "Real" if real_prob > fake_prob else "Fake"
        confianca = max(real_prob, fake_prob) * 100

        return jsonify({
            "label":     label,
            "confianca": round(confianca, 2),
            "real_prob": round(real_prob * 100, 2),
            "fake_prob": round(fake_prob * 100, 2),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status":      "online",
        "modelo":      MODEL_PATH,
        "dispositivo": str(DEVICE),
        "img_size":    IMG_SIZE,
    })


if __name__ == "__main__":
    print(f"\nServidor iniciado em http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)