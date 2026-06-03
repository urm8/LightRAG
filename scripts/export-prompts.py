import json
from pathlib import Path

import lightrag.prompt as prompts_module

Path("evals").mkdir(exist_ok=True)

Path("evals/prompts.active.json").write_text(
    json.dumps(
        {
            "tuple_delimiter": prompts_module.DEFAULT_TUPLE_DELIMITER,
            "completion_delimiter": prompts_module.DEFAULT_COMPLETION_DELIMITER,
            "system": prompts_module.ENTITY_EXTRACTION_SYSTEM_PROMPT,
            "user": prompts_module.ENTITY_EXTRACTION_USER_PROMPT,
            "enrichment_user": prompts_module.ENTITY_CONTINUE_EXTRACTION_USER_PROMPT,
            "legacy_system": prompts_module.ENTITY_EXTRACTION_SYSTEM_PROMPT,
            "legacy_user": prompts_module.ENTITY_EXTRACTION_USER_PROMPT,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
