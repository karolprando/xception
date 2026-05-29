"""
=============================================================
  Deepfake Detector -- Backend Flask
  Baixa o modelo do Hugging Face automaticamente
  Baixa imagens de validacao do Kaggle automaticamente
=============================================================
"""

import io
import os
import random
import requests
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

# -------------------------------------------------------------
#  CONFIGURACOES
# -------------------------------------------------------------

MODEL_URL          = "https://huggingface.co/karolprando/xception/resolve/main/xception_140k.pth"
MODEL_PATH         = "xception_140k.pth"
IMG_SIZE           = 224
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALIDATION_DIR     = "validation"
SAMPLES_PER_CLASS  = 50  # 50 real + 50 fake = 100 imagens

# -------------------------------------------------------------
#  DOWNLOAD DO MODELO
# -------------------------------------------------------------

def baixar_modelo():
    if os.path.exists(MODEL_PATH):
        print(f"Modelo ja existe localmente: {MODEL_PATH}")
        return

    print(f"Baixando modelo do Hugging Face...")
    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()

    total   = int(response.headers.get("content-length", 0))
    baixado = 0

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            baixado += len(chunk)
            if total:
                pct = baixado / total * 100
                print(f"  {pct:.1f}%  ({baixado/1e6:.1f} MB / {total/1e6:.1f} MB)", end="\r")

    print(f"\nModelo baixado com sucesso!")


# -------------------------------------------------------------
#  DOWNLOAD IMAGENS DE VALIDACAO DO KAGGLE
# -------------------------------------------------------------

def baixar_validacao():
    real_dir = os.path.join(VALIDATION_DIR, "real")
    fake_dir = os.path.join(VALIDATION_DIR, "fake")

    real_ok = os.path.exists(real_dir) and len(os.listdir(real_dir)) >= SAMPLES_PER_CLASS
    fake_ok = os.path.exists(fake_dir) and len(os.listdir(fake_dir)) >= SAMPLES_PER_CLASS

    if real_ok and fake_ok:
        print(f"Imagens de validacao ja existem localmente.")
        return

    token = os.environ.get("KAGGLE_TOKEN")
    if not token:
        print("AVISO: KAGGLE_TOKEN nao definido. Imagens de validacao nao serao baixadas.")
        return

    print("Baixando imagens de validacao do Kaggle...")

    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    import zipfile
    import tempfile

    headers = {"Authorization": f"Bearer {token}"}
    url     = "https://www.kaggle.com/api/v1/datasets/download/xhlulu/140k-real-and-fake-faces"

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "dataset.zip")

        print("  Conectando ao Kaggle...")
        response = requests.get(url, headers=headers, stream=True)

        if response.status_code != 200:
            print(f"  Erro ao baixar do Kaggle: {response.status_code}")
            return

        total   = int(response.headers.get("content-length", 0))
        baixado = 0

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                baixado += len(chunk)
                if total:
                    pct = baixado / total * 100
                    print(f"  {pct:.1f}%  ({baixado/1e6:.1f} MB)", end="\r")

        print(f"\n  Extraindo imagens...")

        with zipfile.ZipFile(zip_path, "r") as z:
            todos = z.namelist()
            reais = [f for f in todos if "/real/" in f.lower() and f.lower().endswith((".jpg", ".jpeg", ".png"))]
            fakes = [f for f in todos if "/fake/" in f.lower() and f.lower().endswith((".jpg", ".jpeg", ".png"))]

            amostra_real = random.sample(reais, min(SAMPLES_PER_CLASS, len(reais)))
            amostra_fake = random.sample(fakes, min(SAMPLES_PER_CLASS, len(fakes)))

            for i, f in enumerate(amostra_real):
                nome = f"real_{i:03d}.jpg"
                with z.open(f) as src, open(os.path.join(real_dir, nome), "wb") as dst:
                    dst.write(src.read())

            for i, f in enumerate(amostra_fake):
                nome = f"fake_{i:03d}.jpg"
                with z.open(f) as src, open(os.path.join(fake_dir, nome), "wb") as dst:
                    dst.write(src.read())

    print(f"  {SAMPLES_PER_CLASS} imagens reais e {SAMPLES_PER_CLASS} fakes baixadas!")


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


inferencia_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

print("=" * 50)
print("  Deepfake Detector -- Iniciando")
print("=" * 50)
baixar_modelo()
baixar_validacao()
model = carregar_modelo()
print("=" * 50)


# -------------------------------------------------------------
#  API FLASK
# -------------------------------------------------------------

app = Flask(__name__, static_folder=".")
CORS(app)


def gerar_pagina_diretorio(label):
    """Gera pagina HTML estilo FTP para um diretorio de imagens."""
    pasta = os.path.join(VALIDATION_DIR, label)

    if not os.path.exists(pasta):
        return Response("Pasta nao encontrada.", status=404)

    arquivos = sorted([f for f in os.listdir(pasta)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    linhas = ""
    for f in arquivos:
        tamanho = os.path.getsize(os.path.join(pasta, f))
        tamanho_str = f"{tamanho/1024:.1f} KB"
        linhas += f"""
        <tr>
            <td><a href="/validation/{label}/{f}" download="{f}">&#128444; {f}</a></td>
            <td>{tamanho_str}</td>
            <td><a href="/validation/{label}/{f}" target="_blank">visualizar</a></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>Imagens de Validacao -- {label.upper()}</title>
  <style>
    body {{ font-family: monospace; background: #0a0a0f; color: #e8e8f0; padding: 32px; }}
    h1 {{ color: {'#00f5a0' if label == 'real' else '#ff3c5f'}; }}
    a {{ color: {'#00f5a0' if label == 'real' else '#ff3c5f'}; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
    th {{ text-align: left; padding: 8px 16px; border-bottom: 1px solid #1e1e2e; color: #7070a0; }}
    td {{ padding: 8px 16px; border-bottom: 1px solid #1e1e2e; }}
    tr:hover td {{ background: #111118; }}
    .nav {{ margin-bottom: 24px; font-size: 13px; color: #7070a0; }}
    .nav a {{ color: #7070a0; }}
    .count {{ color: #7070a0; font-size: 13px; margin-top: 8px; }}
  </style>
</head>
<body>
  <div class="nav">
    <a href="/">inicio</a> /
    <a href="/files">validacao</a> /
    {label}
  </div>
  <h1>/{label}</h1>
  <p class="count">{len(arquivos)} imagens disponiveis para download</p>
  <table>
    <thead>
      <tr>
        <th>Nome</th>
        <th>Tamanho</th>
        <th>Acao</th>
      </tr>
    </thead>
    <tbody>
      {linhas}
    </tbody>
  </table>
</body>
</html>"""
    return Response(html, mimetype="text/html")


@app.route("/files")
def listar_raiz():
    """Pagina raiz do diretorio de validacao."""
    real_count = 0
    fake_count = 0

    real_path = os.path.join(VALIDATION_DIR, "real")
    fake_path = os.path.join(VALIDATION_DIR, "fake")

    if os.path.exists(real_path):
        real_count = len([f for f in os.listdir(real_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if os.path.exists(fake_path):
        fake_count = len([f for f in os.listdir(fake_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>Imagens de Validacao</title>
  <style>
    body {{ font-family: monospace; background: #0a0a0f; color: #e8e8f0; padding: 32px; }}
    h1 {{ color: #e8e8f0; }}
    a {{ color: #00f5a0; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
    th {{ text-align: left; padding: 8px 16px; border-bottom: 1px solid #1e1e2e; color: #7070a0; }}
    td {{ padding: 8px 16px; border-bottom: 1px solid #1e1e2e; }}
    tr:hover td {{ background: #111118; }}
    .nav {{ margin-bottom: 24px; font-size: 13px; color: #7070a0; }}
    .nav a {{ color: #7070a0; }}
    .desc {{ color: #7070a0; font-size: 13px; margin-top: 8px; margin-bottom: 24px; }}
  </style>
</head>
<body>
  <div class="nav"><a href="/">inicio</a> / validacao</div>
  <h1>/validacao</h1>
  <p class="desc">Imagens de exemplo para testar o detector. Fonte: 140k Real and Fake Faces (Kaggle).</p>
  <table>
    <thead>
      <tr>
        <th>Pasta</th>
        <th>Imagens</th>
        <th>Descricao</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><a href="/files/real">&#128194; real/</a></td>
        <td>{real_count} imagens</td>
        <td>Fotos reais de pessoas</td>
      </tr>
      <tr>
        <td><a href="/files/fake" style="color:#ff3c5f">&#128194; fake/</a></td>
        <td>{fake_count} imagens</td>
        <td>Faces geradas por StyleGAN</td>
      </tr>
    </tbody>
  </table>
</body>
</html>"""
    return Response(html, mimetype="text/html")


@app.route("/files/<label>")
def listar_label(label):
    if label not in ["real", "fake"]:
        return Response("Pasta invalida.", status=404)
    return gerar_pagina_diretorio(label)


@app.route("/validation/<label>/<filename>")
def servir_imagem(label, filename):
    pasta = os.path.join(VALIDATION_DIR, label)
    return send_from_directory(pasta, filename)


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
