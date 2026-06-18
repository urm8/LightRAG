from __future__ import annotations

from lightrag.types import ExtractionStructuredOutput


EXTRACTION_TOOL_NAME = "submit_extraction"

EXTRACTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": EXTRACTION_TOOL_NAME,
            "description": (
                "Return extracted entities and relations for the current chunk. "
                "Descriptions are required for every entity and relation."
            ),
            "parameters": ExtractionStructuredOutput.model_json_schema(),
        },
    }
]

EXTRACTION_TOOL_CHOICE = {
    "type": "function",
    "function": {"name": EXTRACTION_TOOL_NAME},
}
