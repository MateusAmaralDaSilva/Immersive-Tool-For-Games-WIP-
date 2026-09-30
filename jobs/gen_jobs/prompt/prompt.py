"""Carregamento de contexto e montagem do prompt de classificacao de estado."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


PROMPT_TEMPLATE = """OBJETIVO:
{task}

=== CONTEXTO DO NPC ===
{npc_yaml}

=== CONTEXTO DO JOGADOR ===
{player_yaml}

=== REGRAS DE ESTADO ===
Você deve mapear a fala do jogador para UMA das opções abaixo.
Opções válidas: {options}
Viés de Decisão: {bias}

=== EXEMPLOS ===
{few_shots}

=== TAREFA FINAL ===
Fala do Jogador: "{player_input}"
Retorne APENAS o nome do estado válido, sem formatação."""


@dataclass(frozen=True)
class PromptData:
    """IDs das entidades e da regra usada para montar uma classificação."""

    npc_id: str
    player_id: str
    player_input: str
    data_dir: Path = field(default=DEFAULT_DATA_DIR)
    rules_id: str = "estado_base"


def _load_yaml(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo YAML nao encontrado: {path}")

    with path.open("r", encoding="utf-8") as yaml_file:
        value = yaml.safe_load(yaml_file)

    return {} if value is None else value


def _entity_path(data_dir: Path, category: str, entity_id: str) -> Path:
    # IDs sao nomes de arquivo, nunca caminhos arbitrarios.
    if not entity_id or Path(entity_id).name != entity_id:
        raise ValueError(f"ID de entidade invalido: {entity_id!r}")
    return data_dir / category / f"{entity_id}.yaml"


def _yaml_string(value: Any) -> str:
    return yaml.safe_dump(
        value,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).strip()


def _rule_value(rules: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in rules:
            return rules[name]
    return default


def build_prompt(data: PromptData) -> str:
    """Le entidades/regras e injeta todo o contexto no template imutavel."""
    data_dir = Path(data.data_dir)
    npc = _load_yaml(_entity_path(data_dir, "npcs", data.npc_id))
    player = _load_yaml(_entity_path(data_dir, "players", data.player_id))
    rules = _load_yaml(data_dir / "regras" / f"{data.rules_id}.yaml")

    if not isinstance(npc, dict):
        raise ValueError("O YAML do NPC deve conter um objeto/mapeamento.")
    if not isinstance(player, dict):
        raise ValueError("O YAML do jogador deve conter um objeto/mapeamento.")
    if not isinstance(rules, dict):
        raise ValueError("O YAML de regras deve conter um objeto/mapeamento.")

    task = _rule_value(rules, "task", "tarefa", default="")
    options = _rule_value(
        rules,
        "options",
        "opcoes_estado",
        "opções_estado",
        "estados",
        default=[],
    )
    bias = _rule_value(rules, "bias", "viés", "vies", default="")
    few_shots = _rule_value(rules, "few_shots", "few-shots", "exemplos", default=[])

    return PROMPT_TEMPLATE.format(
        task=task,
        npc_yaml=_yaml_string(npc),
        player_yaml=_yaml_string(player),
        options=_yaml_string(options),
        bias=bias,
        few_shots=_yaml_string(few_shots),
        player_input=data.player_input,
    )
