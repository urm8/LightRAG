from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractionPromptContext:
    language: str
    entity_types: list[str]
    relation_labels: list[str]
    max_total_records: int
    max_entity_records: int

    def prompt_vars(self) -> dict[str, str | int]:
        return {
            "language": self.language,
            "entity_types": ", ".join(self.entity_types),
            "relation_labels": ", ".join(self.relation_labels),
            "max_total_records": self.max_total_records,
            "max_entity_records": self.max_entity_records,
        }
