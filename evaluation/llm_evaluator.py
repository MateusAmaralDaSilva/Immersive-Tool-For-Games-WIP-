"""Métricas e tratamento de saídas para o classificador de intenções via LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
)


INVALID_STATE = "INVALID_STATE"


def _normalize(value: str) -> str:
    """Normaliza caixa e espaços para comparação, sem alterar a classe final."""
    return " ".join(value.strip().casefold().split())


@dataclass
class EvaluationResult:
    """Resultado estruturado da avaliação, além da saída exibida no terminal."""

    cleaned_predictions: list[str]
    formatting_failure_rate: float
    accuracy: float
    precision_macro: float
    precision_weighted: float
    recall_macro: float
    recall_weighted: float
    classification_report: str


class LLMEvaluator:
    """Avalia um LLM que roteia falas para estados discretos."""

    def __init__(self, allowed_classes: Sequence[str]):
        self.allowed_classes = self._validate_allowed_classes(allowed_classes)
        self._normalized_classes = {
            _normalize(label): label for label in self.allowed_classes
        }

    @staticmethod
    def _validate_allowed_classes(allowed_classes: Sequence[str]) -> list[str]:
        classes = list(allowed_classes)
        if not classes:
            raise ValueError("allowed_classes não pode ser vazio.")
        if any(not isinstance(label, str) or not label.strip() for label in classes):
            raise ValueError("Todas as classes permitidas devem ser strings não vazias.")
        if INVALID_STATE in classes:
            raise ValueError(
                f"{INVALID_STATE!r} é reservado para previsões inválidas."
            )
        return classes

    def clean_predictions(
        self,
        y_pred_raw: Sequence[Any],
        allowed_classes: Sequence[str] | None = None,
    ) -> list[str]:
        """Converte previsões para classes canônicas ou ``INVALID_STATE``.

        A comparação ignora maiúsculas/minúsculas e espaços extras. O valor
        retornado usa a grafia original definida em ``allowed_classes``.
        Valores que não sejam strings também são considerados inválidos.
        """
        if allowed_classes is not None:
            evaluator = LLMEvaluator(allowed_classes)
            return evaluator.clean_predictions(y_pred_raw)

        cleaned = []
        for prediction in y_pred_raw:
            if not isinstance(prediction, str):
                cleaned.append(INVALID_STATE)
                continue
            cleaned.append(
                self._normalized_classes.get(_normalize(prediction), INVALID_STATE)
            )
        return cleaned

    def evaluate(
        self,
        y_true: Sequence[str],
        y_pred_raw: Sequence[Any],
        allowed_classes: Sequence[str] | None = None,
    ) -> EvaluationResult:
        """Limpa previsões, calcula métricas e exibe o relatório completo."""
        if len(y_true) != len(y_pred_raw):
            raise ValueError("y_true e y_pred_raw devem ter o mesmo tamanho.")
        if not y_true:
            raise ValueError("y_true e y_pred_raw não podem ser vazios.")

        evaluator = self if allowed_classes is None else LLMEvaluator(allowed_classes)
        cleaned_predictions = evaluator.clean_predictions(y_pred_raw)
        failure_count = sum(
            prediction == INVALID_STATE for prediction in cleaned_predictions
        )
        failure_rate = 100 * failure_count / len(cleaned_predictions) if cleaned_predictions else 0.0

        labels = list(evaluator.allowed_classes)
        for label in y_true:
            if label not in labels:
                labels.append(label)
        if (
            (INVALID_STATE in cleaned_predictions or INVALID_STATE in y_true)
            and INVALID_STATE not in labels
        ):
            labels.append(INVALID_STATE)

        result = EvaluationResult(
            cleaned_predictions=cleaned_predictions,
            formatting_failure_rate=failure_rate,
            accuracy=accuracy_score(y_true, cleaned_predictions),
            precision_macro=precision_score(
                y_true,
                cleaned_predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            ),
            precision_weighted=precision_score(
                y_true,
                cleaned_predictions,
                labels=labels,
                average="weighted",
                zero_division=0,
            ),
            recall_macro=recall_score(
                y_true,
                cleaned_predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            ),
            recall_weighted=recall_score(
                y_true,
                cleaned_predictions,
                labels=labels,
                average="weighted",
                zero_division=0,
            ),
            classification_report=classification_report(
                y_true,
                cleaned_predictions,
                labels=labels,
                target_names=[str(label) for label in labels],
                zero_division=0,
            ),
        )

        self._print_result(result)
        return result

    @staticmethod
    def _print_result(result: EvaluationResult) -> None:
        print(f"Taxa de Falha de Formatação: {result.formatting_failure_rate:.2f}%")
        print(f"Accuracy: {result.accuracy:.4f}")
        print(f"Precision (Macro): {result.precision_macro:.4f}")
        print(f"Precision (Weighted): {result.precision_weighted:.4f}")
        print(f"Recall (Macro): {result.recall_macro:.4f}")
        print(f"Recall (Weighted): {result.recall_weighted:.4f}")
        print("\nRelatório de Classificação:")
        print(result.classification_report)


if __name__ == "__main__":
    evaluator = LLMEvaluator(["conversar", "atacar", "fugir", "ignorar"])
    evaluator.evaluate(
        y_true=["conversar", "atacar", "fugir", "ignorar"],
        y_pred_raw=[" CONVERSAR ", "atacar", "responder livremente", "ignorar"],
    )
