# Deepfake Detector

Detector de imagens deepfake usando Xception com fine-tuning.
Desenvolvido como TCC.

## Como rodar com Docker

### Pre-requisitos
- Docker instalado

### Rodar

```bash
# 1. Clona o repositorio
git clone https://github.com/SEU_USUARIO/deepfake-detector
cd deepfake-detector

# 2. Builda a imagem
docker build -t deepfake-detector .

# 3. Inicia o container
docker run -p 5000:5000 deepfake-detector
```

O modelo e baixado automaticamente do Hugging Face na primeira execucao.

### Usar a interface

Apos iniciar o container, abra o `index.html` no navegador.

## Modelo

- **Arquitetura**: Xception (Chollet, 2017)
- **Pre-treinamento**: ImageNet
- **Fine-tuning**: 140k Real and Fake Faces dataset
- **Acuracia**: 81.56%
- **AUC-ROC**: 0.90
- **Modelo**: https://huggingface.co/karolprando/xception

## Tecnologias

- Python 3.11
- PyTorch
- Flask
- Docker
