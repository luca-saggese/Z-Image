# Z-Image Gradio Docker Deployment

Interfaccia Gradio per generare immagini da prompt testuali utilizzando Z-Image-Turbo con supporto NVIDIA CUDA.

## Prerequisiti

- **Docker** con supporto NVIDIA GPU
- **Docker Compose** (opzionale ma consigliato)
- **NVIDIA Container Runtime** installato
- GPU NVIDIA con compute capability >= 7.0 (es. Tesla V100, A100, H800)
- Almeno 16GB di VRAM (consigliati 24GB+ per migliori prestazioni)

### Verificare configurazione NVIDIA

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 nvidia-smi
```

## Build dell'immagine Docker

### Opzione 1: Con Docker Compose (consigliato)

```bash
docker-compose build
```

### Opzione 2: Con Docker CLI

```bash
docker build -t z-image-gradio:latest .
```

## Esecuzione

### Opzione 1: Con Docker Compose

```bash
docker-compose up
```

Per eseguire in background:
```bash
docker-compose up -d
```

Visualizzare i log:
```bash
docker-compose logs -f z-image-gradio
```

Stoppare il servizio:
```bash
docker-compose down
```

### Opzione 2: Con Docker CLI

```bash
docker run --rm \
  --gpus all \
  -p 7860:7860 \
  -v $(pwd)/ckpts:/app/ckpts \
  -v $(pwd)/models:/app/models \
  --shm-size=16gb \
  z-image-gradio:latest
```

## Accesso all'interfaccia

Una volta che il container è in esecuzione, apri il browser e vai a:

```
http://localhost:7860
```

## Funzionalità

L'interfaccia Gradio offre:

- **Prompt**: Campo di testo per inserire la descrizione dell'immagine
- **Height/Width**: Controllo delle dimensioni dell'immagine (512-1024px)
- **Inference Steps**: Numero di step diffusion (1-50, default 8 per velocità)
- **Guidance Scale**: Controllo della forza del prompt (0-7.5, default 0 per Turbo)
- **Seed**: Seed per la generazione (per risultati riproducibili)
- **Esempi pre-caricati**: Quick start con prompt di esempio

## Dettagli tecnici

### Image base
- `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04`
- Python 3.11
- PyTorch 2.5.0+
- CUDA 12.8, cuDNN 9.x

### Modello
- **Z-Image-Turbo** (8 step, generazione sub-second su H800)
- Scaricamento automatico da Hugging Face
- Optimization: Flash Attention v3 per massima velocità

### Storage
- Modelli cachati in `/app/models/huggingface/`
- Checkpoint PyTorch in `/app/models/torch/`
- Volume montato in `./models/` locale per persistenza

### Configurazione GPU
- Allocazione dinamica memoria CUDA
- Shared memory: 16GB
- Supporto multi-GPU (se disponibili)

## Profiling e Optimization

### Variabili ambiente modificabili

Editare `docker-compose.yml` o passare con `-e`:

```bash
ZIMAGE_ATTENTION="_native_flash"  # Backend attention (default per H100/H800)
PYTORCH_CUDA_ALLOC_CONF           # Allocazione memoria CUDA
```

### Per velocità massima (su H800):

Gli algorithm di attention sono auto-selezionati. Per GPU A100/V100:
- Usa `_native_flash` per migliore compatibilità

## Troubleshooting

### "CUDA out of memory"
```bash
# Aumentare shared memory e memory pool
docker-compose.yml: shm_size: 24gb
```

### Modello non si scarica
```bash
# Controllare connessione internet e credenziali HF
docker exec -it z-image-gradio bash
pip install --upgrade huggingface-hub
huggingface-cli login  # Se il modello è privato
```

### GPU non rilevata nel container
```bash
# Verificare runtime NVIDIA
docker run --rm --gpus all ubuntu nvidia-smi

# Aggiungere utente al gruppo docker
sudo usermod -aG docker $USER
newgrp docker
```

### Porta 7860 già in uso
```bash
# Modificare mappatura in docker-compose.yml:
ports:
  - "8080:7860"  # Accesso via localhost:8080
```

## Performance esperato

| GPU | Batch Size | Time/Image | VRAM Usage |
|-----|-----------|-----------|-----------|
| H800 | 1 | ~0.5s | 12GB |
| H100 | 1 | ~0.6s | 12GB |
| A100 | 1 | ~0.8s | 14GB |
| RTX 4090 | 1 | ~1.2s | 16GB |

## Development

Per modifiche all'app Gradio:

```bash
# Rebuild con cache bypass
docker-compose build --no-cache

# Sviluppo locale (senza Docker)
pip install -e .
pip install gradio>=4.0.0
python app_gradio.py
```

## License

Vedi [LICENSE](LICENSE) nel repository principale di Z-Image.
