# Immersive Tool for Games

Pipeline de entrada de fala/texto e classificação de intenção para NPCs. O modelo de linguagem atua como um roteador: recebe a fala do jogador e retorna somente um estado de jogo permitido, sem gerar diálogo livre.

## Funcionalidades

- Entrada por teclado, Vosk ou Whisper.
- Whisper em tempo quase real com detecção de silêncio.
- Contexto de NPCs e jogadores definido em arquivos YAML independentes.
- Regras de classificação, estados válidos, viés e few-shots definidos em YAML.
- Prompt com contexto dinâmico e temperatura `0.0` para classificação determinística.
- Avaliação com `LLMEvaluator`, incluindo tratamento de respostas inválidas como `INVALID_STATE`.

## Estrutura

```text
.
├── data/
│   ├── npcs/                       # Um YAML por NPC
│   ├── players/                    # Um YAML por jogador
│   └── regras/                     # Task, estados, viés e few-shots
├── evaluation/
│   └── llm_evaluator.py            # Métricas e parser das previsões
├── jobs/
│   ├── stt_jobs/
│   │   ├── stt.py                  # Contrato speech_to_text(engine)
│   │   └── engines/
│   └── gen_jobs/
│       ├── prompt/prompt.py        # Carregamento YAML e montagem do prompt
│       └── models/                 # Adaptadores dos modelos LLM
├── main.py
└── requirements.txt
```

## Instalação

```bash
pip install -r requirements.txt
```

No Linux, o áudio pode exigir a biblioteca do sistema:

```bash
sudo apt install portaudio19-dev
```

O `faster-whisper` baixa o modelo Whisper na primeira execução. Para usar Vosk, baixe um modelo em [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) e coloque-o no caminho configurado por `VOSK_MODEL_PATH`.

O modelo generativo usado por `main.py` deve estar em `models/gen_models/Qwen/Qwen3-4B-Q4_K_M.gguf`, ou o caminho `QWEN_MODEL_PATH` deve ser ajustado.

## Execução

```bash
python main.py
```

O menu oferece:

1. Teclado — digita o texto diretamente.
2. Vosk — transcrição de áudio em streaming.
3. Whisper — grava até pressionar Enter e transcreve.
4. Whisper em tempo real — transcreve após pausas na fala.

Todas as opções de STT implementam `transcribe()` e são compatíveis com `speech_to_text()`.

## Contexto em YAML

O prompt é montado a partir dos IDs informados em `PromptData`:

```python
from jobs.gen_jobs.prompt.prompt import PromptData, build_prompt

prompt = build_prompt(PromptData(
    npc_id="gareth",
    player_id="ladino_01",
    player_input="Quero passar pelo portão.",
))
```

Os arquivos usados são:

```text
data/npcs/gareth.yaml
data/players/ladino_01.yaml
data/regras/estado_base.yaml
```

NPCs e jogadores podem conter qualquer chave definida pelo Game Designer. O código não depende de campos fixos: os objetos inteiros são serializados novamente para YAML e injetados no prompt.

O arquivo de regras define as chaves de controle do classificador:

```yaml
task: "Classifique a intenção da fala. Não gere diálogo."
options:
  - conversar
  - atacar
  - fugir
  - ignorar
bias: "Prefira conversar quando houver dúvida."
few_shots:
  - fala: "Quero conversar."
    estado: conversar
```

O template instrui o modelo a retornar apenas o nome de um estado válido. O `BaseModel` usa `temperature=0.0`.

## Avaliação do classificador

O módulo [`evaluation/llm_evaluator.py`](evaluation/llm_evaluator.py) normaliza as respostas do LLM, ignorando caixa e espaços extras. Qualquer saída fora de `allowed_classes` vira `INVALID_STATE`.

```python
from evaluation.llm_evaluator import LLMEvaluator

evaluator = LLMEvaluator([
    "conversar",
    "atacar",
    "fugir",
    "ignorar",
])

result = evaluator.evaluate(
    y_true=["conversar", "atacar", "fugir"],
    y_pred_raw=[" CONVERSAR ", "atacar", "resposta fora do formato"],
)
```

São exibidos:

- Taxa de Falha de Formatação.
- Accuracy.
- Precision macro e weighted.
- Recall macro e weighted.
- `classification_report` por classe, incluindo `INVALID_STATE` quando aplicável.

Para executar o exemplo embutido no avaliador:

```bash
python -m evaluation.llm_evaluator
```

## Ajustes rápidos

- Para usar Whisper sem GPU, altere `DEVICE = "cpu"` e `COMPUTE_TYPE = "int8"` nos engines Whisper.
- Para trocar o tamanho do Whisper, ajuste `model_size` em `main.py`.
- Para adicionar estados ou exemplos, altere apenas `data/regras/estado_base.yaml`.
