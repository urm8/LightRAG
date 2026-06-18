from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field
from typing import Any, Optional


class GPTKeywordExtractionFormat(BaseModel):
    high_level_keywords: list[str]
    low_level_keywords: list[str]


class ExtractedEntity(BaseModel):
    """A single entity extracted from text by the LLM."""

    entity_name: str = Field(
        description="Name of the entity. Use title case for case-insensitive names."
    )
    entity_type: str = Field(description="Type/category of the entity.")
    entity_description: str = Field(
        description="Concise yet comprehensive description of the entity based on the input text."
    )


class ExtractedRelationship(BaseModel):
    """A single relationship between two entities extracted from text."""

    source_entity: str = Field(
        description="Name of the source entity in the relationship."
    )
    target_entity: str = Field(
        description="Name of the target entity in the relationship."
    )
    relationship_keywords: str = Field(
        description="Comma-separated high-level keywords summarizing the relationship."
    )
    relationship_description: str = Field(
        description="Concise explanation of the relationship between source and target entities."
    )


class EntityExtractionResult(BaseModel):
    """Structured output format for entity and relationship extraction from text."""

    entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description="List of entities extracted from the input text.",
    )
    relationships: list[ExtractedRelationship] = Field(
        default_factory=list,
        description="List of relationships between entities extracted from the input text.",
    )


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
