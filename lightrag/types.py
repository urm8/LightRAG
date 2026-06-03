from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field
from typing import Any, Optional


class GPTKeywordExtractionFormat(BaseModel):
    high_level_keywords: list[str]
    low_level_keywords: list[str]


class ExtractionEntity(BaseModel):
    entity_name: str = Field(validation_alias=AliasChoices("entity_name", "name"))
    entity_type: str = Field(validation_alias=AliasChoices("entity_type", "type"))
    entity_description: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "entity_description",
            "description",
            "summary",
        ),
    )


class ExtractionRelation(BaseModel):
    source_entity: str = Field(
        validation_alias=AliasChoices("source_entity", "source", "src")
    )
    target_entity: str = Field(
        validation_alias=AliasChoices("target_entity", "target", "tgt")
    )
    relationship_keywords: str = Field(
        validation_alias=AliasChoices(
            "relationship_keywords",
            "relation_type",
            "relation_label",
            "label",
            "verb",
            "keywords",
        )
    )
    relationship_description: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "relationship_description",
            "relation_description",
            "description",
            "summary",
        ),
    )


class ExtractionStructuredOutput(BaseModel):
    entities: list[ExtractionEntity] = Field(default_factory=list)
    relations: list[ExtractionRelation] = Field(
        default_factory=list,
        validation_alias=AliasChoices("relations", "relationships", "edges"),
    )


class KnowledgeGraphNode(BaseModel):
    id: str
    labels: list[str]
    properties: dict[str, Any]  # anything else goes here


class KnowledgeGraphEdge(BaseModel):
    id: str
    type: Optional[str]
    source: str  # id of source node
    target: str  # id of target node
    properties: dict[str, Any]  # anything else goes here


class KnowledgeGraph(BaseModel):
    nodes: list[KnowledgeGraphNode] = []
    edges: list[KnowledgeGraphEdge] = []
    is_truncated: bool = False
