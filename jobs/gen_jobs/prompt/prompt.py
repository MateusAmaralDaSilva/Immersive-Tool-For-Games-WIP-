from dataclasses import dataclass
from typing import Dict


@dataclass
class PromptData:
    context: str
    player_profile: str
    npc_profile: str
    available_states: list[str]


def build_prompt(data: PromptData) -> str:
    states = "\n".join(f"- {s}" for s in data.available_states)

    return f"""
        Você é o cérebro de um NPC em um RPG.
        CONTEXTO:
        {data.context}
        PERFIL DO JOGADOR:
        {data.player_profile}
        PERFIL DO NPC:
        {data.npc_profile}
        PRÓXIMOS ESTADOS POSSÍVEIS:
        {states}
        Escolha apenas UM estado.
        Responda somente com o nome do estado.
        """.strip()