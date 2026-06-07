# Teste de STT — Vosk vs Whisper

Teste simples que captura seu microfone e transcreve usando Vosk ou Whisper.

## Estrutura

| Arquivo | O que faz |
|---|---|
| `mic.py` | Captura de áudio do microfone (streaming e gravação) |
| `engine_vosk.py` | Transcrição em tempo real com Vosk |
| `engine_whisper.py` | Transcrição com Whisper (faster-whisper) |
| `main.py` | Menu de escolha + configurações |

## 1. Instalar dependências

```bash
pip install -r requirements.txt
```

No Linux, talvez precise da lib de áudio do sistema:

```bash
sudo apt install portaudio19-dev
```

## 2. Baixar o modelo do Vosk

Pegue um modelo em https://alphacephei.com/vosk/models (procure por português, ex.: `vosk-model-small-pt-0.3`), descompacte dentro da pasta `models/` e confira se o caminho bate com `VOSK_MODEL_PATH` no `main.py`.

paths + resume:
vosk-model-small-pt-0.3 -> Lightweight wideband model for Android and RPi
vosk-model-pt-fb-v0.1.1-20220516_2113 -> Big model from FalaBrazil


O Whisper **não** precisa de download manual — o faster-whisper baixa o modelo sozinho na primeira execução.

## 3. Rodar

```bash
python main.py
```

Escolha `1` para Vosk (fala em tempo real) ou `2` para Whisper (grava, aperta Enter, transcreve).

## Ajustes rápidos

- Trocar o tamanho do modelo Whisper: edite `WHISPER_MODEL_SIZE` no `main.py`.
- Rodar Whisper sem GPU: em `engine_whisper.py`, mude `DEVICE = "cpu"` e `COMPUTE_TYPE = "int8"`.
- Na sua RTX 2060, o padrão (`cuda` + `int8_float16`) já funciona com o modelo `large-v3` consumindo ~4 GB de VRAM.
