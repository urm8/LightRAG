from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Mapping, TypedDict

import yaml
from lightrag.extraction.prompts import (
    ENTITY_EXTRACTION_CONTINUE_USER_PROMPT,
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    ENTITY_EXTRACTION_USER_PROMPT,
)


PROMPTS: dict[str, Any] = {}

# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

# Default entity type guidance injected into extraction prompts via {entity_types_guidance}.
# Users can override this by passing entity_types_guidance in addon_params, or by
# replacing the full prompt template string in PROMPTS.
PROMPTS[
    "default_entity_types_guidance"
] = """Classify each entity using one of the following types. If no type fits, use `Other`.

- Person: Human individuals, real or fictional
- Creature: Non-human living beings (animals, mythical beings, etc.)
- Organization: Companies, institutions, government bodies, groups
- Location: Geographic places (cities, countries, buildings, regions)
- Event: Occurrences, incidents, ceremonies, meetings
- Concept: Abstract ideas, theories, principles, beliefs
- Method: Procedures, techniques, algorithms, workflows
- Content: Creative or informational works (books, articles, films, reports)
- Data: Quantitative or structured information (statistics, datasets, measurements)
- Artifact: Physical or digital objects created by humans (tools, software, devices)
- NaturalObject: Natural non-living objects (minerals, celestial bodies, chemical compounds)"""

PROMPTS["entity_extraction_system_prompt"] = ENTITY_EXTRACTION_SYSTEM_PROMPT
PROMPTS["entity_extraction_user_prompt"] = ENTITY_EXTRACTION_USER_PROMPT
PROMPTS["entity_continue_extraction_user_prompt"] = (
    ENTITY_EXTRACTION_CONTINUE_USER_PROMPT
)

PROMPTS["entity_extraction_examples"] = [
    """---Entity Types---
- Person: Human individuals, real or fictional
- Artifact: Physical or digital objects created by humans (tools, software, devices)
- Concept: Abstract ideas, theories, principles, beliefs

---Input Text---
```
Dr. Elena Vasquez led a field expedition to the Borneo rainforest to document the population decline of the Bornean orangutan. Using transect sampling -- a method where researchers walk predetermined line paths and record every animal sighting within a fixed distance -- her team estimated that fewer than 1,500 individuals remained in the surveyed region.

The expedition was funded by the Global Wildlife Conservation Institute and produced a landmark report titled "Primate Decline in Insular Southeast Asia." Vasquez attributed the collapse primarily to peat-soil destruction caused by palm oil plantation expansion, which had converted over 40% of the surveyed forest area within a decade.
```

---Output---
entity{tuple_delimiter}Dr. Elena Vasquez{tuple_delimiter}Person{tuple_delimiter}Dr. Elena Vasquez is a field researcher who led an expedition to document orangutan population decline in Borneo.
entity{tuple_delimiter}Borneo Rainforest{tuple_delimiter}Location{tuple_delimiter}The Borneo rainforest is the field site of the expedition and the primary habitat of the Bornean orangutan.
entity{tuple_delimiter}Bornean Orangutan{tuple_delimiter}Creature{tuple_delimiter}The Bornean orangutan is a primate species whose population was found to have declined to fewer than 1,500 individuals in the surveyed region.
entity{tuple_delimiter}Transect Sampling{tuple_delimiter}Method{tuple_delimiter}Transect sampling is a wildlife survey technique where researchers walk predetermined paths and record animal sightings within a fixed lateral distance.
entity{tuple_delimiter}Global Wildlife Conservation Institute{tuple_delimiter}Organization{tuple_delimiter}The Global Wildlife Conservation Institute funded the expedition led by Dr. Vasquez.
entity{tuple_delimiter}Primate Decline in Insular Southeast Asia{tuple_delimiter}Content{tuple_delimiter}A landmark research report produced by Vasquez's expedition documenting primate population decline in the region.
entity{tuple_delimiter}Peat Soil{tuple_delimiter}NaturalObject{tuple_delimiter}Peat soil is a natural substrate in the Borneo rainforest that has been destroyed by palm oil plantation expansion.
relation{tuple_delimiter}Dr. Elena Vasquez{tuple_delimiter}Bornean Orangutan{tuple_delimiter}field research, population survey{tuple_delimiter}Dr. Vasquez led the expedition that documented the population decline of the Bornean orangutan.
relation{tuple_delimiter}Dr. Elena Vasquez{tuple_delimiter}Transect Sampling{tuple_delimiter}methodology, research application{tuple_delimiter}Dr. Vasquez's team used transect sampling to estimate the orangutan population.
relation{tuple_delimiter}Global Wildlife Conservation Institute{tuple_delimiter}Dr. Elena Vasquez{tuple_delimiter}funding, research support{tuple_delimiter}The institute funded the expedition led by Dr. Vasquez.
relation{tuple_delimiter}Dr. Elena Vasquez{tuple_delimiter}Primate Decline in Insular Southeast Asia{tuple_delimiter}authorship, research output{tuple_delimiter}Dr. Vasquez's expedition produced the landmark report on primate decline.
relation{tuple_delimiter}Peat Soil{tuple_delimiter}Borneo Rainforest{tuple_delimiter}habitat composition, ecological destruction{tuple_delimiter}Peat soil destruction in the Borneo rainforest was caused by palm oil plantation expansion and is a primary driver of orangutan decline.
{completion_delimiter}


Output JSON:
{{
  "entities": [
    {{"entity_name": "NVIDIA", "entity_type": "Company", "entity_description": "NVIDIA is a company mentioned in connection with an earnings report."}},
    {{"entity_name": "analysts", "entity_type": "Organization", "entity_description": "analysts are tracking market sentiment and inflation risk."}},
    {{"entity_name": "earnings report", "entity_type": "Event", "entity_description": "earnings report is the event associated with NVIDIA rising."}},
    {{"entity_name": "CPI release", "entity_type": "Event", "entity_description": "CPI release is the event associated with increased BTC volatility."}},
    {{"entity_name": "market sentiment", "entity_type": "Concept", "entity_description": "market sentiment is being tracked by analysts."}},
    {{"entity_name": "inflation risk", "entity_type": "Concept", "entity_description": "inflation risk is being tracked by analysts."}}
  ],
  "relations": [
    {{"source_entity": "NVIDIA", "target_entity": "earnings report", "relationship_keywords": "RELATED_TO", "relationship_description": "NVIDIA rose after the earnings report."}},
    {{"source_entity": "BTC", "target_entity": "CPI release", "relationship_keywords": "RELATED_TO", "relationship_description": "BTC volatility increased after the CPI release."}},
    {{"source_entity": "analysts", "target_entity": "market sentiment", "relationship_keywords": "TRACKS", "relationship_description": "Analysts are tracking market sentiment."}},
    {{"source_entity": "analysts", "target_entity": "inflation risk", "relationship_keywords": "TRACKS", "relationship_description": "Analysts are tracking inflation risk."}}
  ]
}}
""",
]

###############################################################################
# JSON Structured Output Prompts for Entity Extraction
# Used when entity_extraction_use_json is enabled for higher extraction quality
###############################################################################

PROMPTS["entity_extraction_json_system_prompt"] = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the `---Input Text---` session of user prompt.

---Instructions---
1. **Entity Extraction:**
  - **Identification:** Identify clearly defined and meaningful entities in the `---Input Text---` session of user prompt.
  - **Entity Details:** For each identified entity, extract the following information:
    - `name`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
    - `type`: Categorize the entity using the type guidance provided in the `---Entity Types---` section below. If none of the provided entity types apply, classify it as `Other`.
    - `description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.

2. **Relationship Extraction:**
  - **Identification:** Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
  - **N-ary Relationship Decomposition:** If a single statement describes a relationship involving more than two entities (an N-ary relationship), decompose it into multiple binary (two-entity) relationship pairs for separate description.
    - Example: For "Alice, Bob, and Carol collaborated on Project X," extract binary relationships such as "Alice collaborated with Project X," "Bob collaborated with Project X," and "Carol collaborated with Project X," or "Alice collaborated with Bob," based on the most reasonable binary interpretations.
  - **Relationship Details:** For each binary relationship, extract the following fields:
    - `source`: The name of the source entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
    - `target`: The name of the target entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
    - `keywords`: One or more high-level keywords summarizing the overarching nature, concepts, or themes of the relationship, separated by commas.
    - `description`: A concise explanation of the nature of the relationship between the source and target entities, providing a clear rationale for their connection.

3. **Relationship Direction & Duplication:**
  - Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities for an undirected relationship does not constitute a new relationship.
  - Avoid outputting duplicate relationships.

4. **Output Limits & Prioritization:**
  - Output at most {max_total_records} total records across `entities` and `relationships` in this response.
  - Output at most {max_entity_records} entity objects in this response.
  - Output fewer records if fewer high-value items are present. Do not try to fill the limit.
  - Only output relationship objects whose `source` and `target` are both included in the selected `entities` list for this response.
  - Within the list of relationships, prioritize and output those relationships that are **most significant** to the core meaning of the input text first.

5. **Context & Objectivity:**
  - Ensure all entity names and descriptions are written in the **third person**.
  - Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.

6. **Language & Proper Nouns:**
  - The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
  - Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

7. **JSON Contract:**
  - Return one valid JSON object with `entities` and `relationships` arrays only.
  - If the record limit is reached, stop adding new objects immediately and return the JSON object with the allowed items only.

---Entity Types---
{entity_types_guidance}

---Examples---
{examples}
"""

PROMPTS["entity_extraction_json_user_prompt"] = """---Task---
Extract entities and relationships from the `---Input Text---` session below.

---Instructions---
1. **Strict Adherence to JSON Format:** Your output MUST be a valid JSON object with `entities` and `relationships` arrays. Do not include any introductory or concluding remarks, explanations, markdown code fences, or any other text before or after the JSON.
2. **Quantity Limits:** In this response, output at most {max_total_records} total records and at most {max_entity_records} entity objects. Output fewer records if fewer high-value items are present. Only output relationship objects whose `source` and `target` are both included in this response.
3. **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

---Entity Types---
{entity_types_guidance}

---Input Text---
```
{input_text}
```

---Output---
"""

PROMPTS["entity_continue_extraction_json_user_prompt"] = """---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly described** entities and relationships from the `---Input Text---` session.

---Instructions---
1. **Focus on Corrections/Additions:**
  - **Do NOT** re-output entities and relationships that were **correctly and fully** extracted in the last task.
  - If an entity or relationship was **missed** in the last task, extract and output it now.
  - If an entity or relationship was **incorrectly described** in the last task, re-output the *corrected and complete* version.
2. **Strict Adherence to JSON Format:** Your output MUST be a valid JSON object with `entities` and `relationships` arrays. Do not include any introductory or concluding remarks, explanations, markdown code fences, or any other text before or after the JSON.
3. **Quantity Limits:** In this response, output at most {max_total_records} total records and at most {max_entity_records} entity objects. Output fewer records if fewer high-value corrections or additions remain. A relationship object may reference entities already extracted correctly in the previous response. Do not repeat those entity objects unless they were missing or need correction.
4. **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.
5. **If nothing was missed or needs correction**, output: `{{"entities": [], "relationships": []}}`

---Output---
"""

PROMPTS["entity_extraction_json_examples"] = [
    """---Entity Types---
- Person: Human individuals, real or fictional
- Artifact: Physical or digital objects created by humans (tools, software, devices)
- Concept: Abstract ideas, theories, principles, beliefs

---Input Text---
```
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
```

---Output---
{
  "entities": [
    {"name": "Alex", "type": "Person", "description": "Alex is a character who experiences frustration and is observant of the dynamics among other characters."},
    {"name": "Taylor", "type": "Person", "description": "Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."},
    {"name": "Jordan", "type": "Person", "description": "Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."},
    {"name": "Cruz", "type": "Person", "description": "Cruz is associated with a vision of control and order, influencing the dynamics among other characters."},
    {"name": "The Device", "type": "Artifact", "description": "The Device is central to the story, with potential game-changing implications, and is revered by Taylor."},
    {"name": "Discovery", "type": "Concept", "description": "Discovery represents the shared intellectual pursuit that unites Jordan and Alex in opposition to Cruz's controlling worldview."}
  ],
  "relationships": [
    {"source": "Alex", "target": "Taylor", "keywords": "power dynamics, observation", "description": "Alex observes Taylor's authoritarian behavior and notes changes in Taylor's attitude toward the device."},
    {"source": "Alex", "target": "Jordan", "keywords": "shared goals, rebellion", "description": "Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."},
    {"source": "Taylor", "target": "Jordan", "keywords": "conflict resolution, mutual respect", "description": "Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."},
    {"source": "Jordan", "target": "Cruz", "keywords": "ideological conflict, rebellion", "description": "Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."},
    {"source": "Taylor", "target": "The Device", "keywords": "reverence, technological significance", "description": "Taylor shows reverence towards the device, indicating its importance and potential impact."}
  ]
}

""",
    """---Entity Types---
- Person: Human individuals, real or fictional
- Location: Geographic places (cities, countries, buildings, regions)
- Creature: Non-human living beings (animals, mythical beings, etc.)
- Method: Procedures, techniques, algorithms, workflows
- Organization: Companies, institutions, government bodies, groups
- Content: Creative or informational works (books, articles, films, reports)
- NaturalObject: Natural non-living objects (minerals, celestial bodies, chemical compounds)

---Input Text---
```
Dr. Elena Vasquez led a field expedition to the Borneo rainforest to document the population decline of the Bornean orangutan. Using transect sampling -- a method where researchers walk predetermined line paths and record every animal sighting within a fixed distance -- her team estimated that fewer than 1,500 individuals remained in the surveyed region.

The expedition was funded by the Global Wildlife Conservation Institute and produced a landmark report titled "Primate Decline in Insular Southeast Asia." Vasquez attributed the collapse primarily to peat-soil destruction caused by palm oil plantation expansion, which had converted over 40% of the surveyed forest area within a decade.
```

---Output---
{
  "entities": [
    {"name": "Dr. Elena Vasquez", "type": "Person", "description": "Dr. Elena Vasquez is a field researcher who led an expedition to document orangutan population decline in Borneo."},
    {"name": "Borneo Rainforest", "type": "Location", "description": "The Borneo rainforest is the field site of the expedition and the primary habitat of the Bornean orangutan."},
    {"name": "Bornean Orangutan", "type": "Creature", "description": "The Bornean orangutan is a primate species whose population was found to have declined to fewer than 1,500 individuals in the surveyed region."},
    {"name": "Transect Sampling", "type": "Method", "description": "Transect sampling is a wildlife survey technique where researchers walk predetermined paths and record animal sightings within a fixed lateral distance."},
    {"name": "Global Wildlife Conservation Institute", "type": "Organization", "description": "The Global Wildlife Conservation Institute funded the expedition led by Dr. Vasquez."},
    {"name": "Primate Decline in Insular Southeast Asia", "type": "Content", "description": "A landmark research report produced by Vasquez's expedition documenting primate population decline in the region."},
    {"name": "Peat Soil", "type": "NaturalObject", "description": "Peat soil is a natural substrate in the Borneo rainforest that has been destroyed by palm oil plantation expansion."}
  ],
  "relationships": [
    {"source": "Dr. Elena Vasquez", "target": "Bornean Orangutan", "keywords": "field research, population survey", "description": "Dr. Vasquez led the expedition that documented the population decline of the Bornean orangutan."},
    {"source": "Dr. Elena Vasquez", "target": "Transect Sampling", "keywords": "methodology, research application", "description": "Dr. Vasquez's team used transect sampling to estimate the orangutan population."},
    {"source": "Global Wildlife Conservation Institute", "target": "Dr. Elena Vasquez", "keywords": "funding, research support", "description": "The institute funded the expedition led by Dr. Vasquez."},
    {"source": "Dr. Elena Vasquez", "target": "Primate Decline in Insular Southeast Asia", "keywords": "authorship, research output", "description": "Dr. Vasquez's expedition produced the landmark report on primate decline."},
    {"source": "Peat Soil", "target": "Borneo Rainforest", "keywords": "habitat composition, ecological destruction", "description": "Peat soil destruction in the Borneo rainforest was caused by palm oil plantation expansion and is a primary driver of orangutan decline."}
  ]
}

""",
    """---Entity Types---
- Content: Creative or informational works (books, articles, films, reports)
- Artifact: Physical or digital objects created by humans (tools, software, devices)
- Person: Human individuals, real or fictional
- Organization: Companies, institutions, government bodies, groups
- Method: Procedures, techniques, algorithms, workflows
- Data: Quantitative or structured information (statistics, datasets, measurements)
- Concept: Abstract ideas, theories, principles, beliefs

---Input Text---
```
The 2023 edition of "Advances in Neural Architecture Search" synthesized findings from over 200 peer-reviewed papers and introduced a new benchmarking framework called NASBench-360, designed to evaluate search algorithms across diverse task domains. The publication was co-authored by Dr. Priya Nair and Dr. Luca Ferretti of the DeepSystems Research Lab.

NASBench-360 measures three key metrics: search efficiency (time-to-solution), model accuracy on held-out test sets, and computational cost in GPU-hours. Early results showed that evolutionary search algorithms outperformed gradient-based methods by 12% on accuracy while consuming 30% fewer GPU-hours on vision tasks.
```

---Output---
{
  "entities": [
    {"name": "Advances in Neural Architecture Search", "type": "Content", "description": "A 2023 publication that synthesizes findings from over 200 papers and introduces the NASBench-360 benchmarking framework."},
    {"name": "NASBench-360", "type": "Artifact", "description": "NASBench-360 is a benchmarking framework introduced to evaluate neural architecture search algorithms across diverse task domains."},
    {"name": "Dr. Priya Nair", "type": "Person", "description": "Dr. Priya Nair is a co-author of the publication and a researcher at the DeepSystems Research Lab."},
    {"name": "Dr. Luca Ferretti", "type": "Person", "description": "Dr. Luca Ferretti is a co-author of the publication and a researcher at the DeepSystems Research Lab."},
    {"name": "DeepSystems Research Lab", "type": "Organization", "description": "The DeepSystems Research Lab is the institution where the co-authors of the publication are affiliated."},
    {"name": "Evolutionary Search", "type": "Method", "description": "Evolutionary search is a class of neural architecture search algorithms that outperformed gradient-based methods in the NASBench-360 evaluation."},
    {"name": "Gradient-Based Search", "type": "Method", "description": "Gradient-based search is a class of neural architecture search algorithms that was benchmarked against evolutionary search in NASBench-360."},
    {"name": "GPU-Hours", "type": "Data", "description": "GPU-hours is a metric used in NASBench-360 to measure the computational cost of neural architecture search algorithms."},
    {"name": "Neural Architecture Search", "type": "Concept", "description": "Neural architecture search is the automated process of designing optimal neural network architectures, the central topic of the publication."}
  ],
  "relationships": [
    {"source": "Dr. Priya Nair", "target": "Advances in Neural Architecture Search", "keywords": "authorship", "description": "Dr. Priya Nair co-authored the publication."},
    {"source": "Dr. Luca Ferretti", "target": "Advances in Neural Architecture Search", "keywords": "authorship", "description": "Dr. Luca Ferretti co-authored the publication."},
    {"source": "Advances in Neural Architecture Search", "target": "NASBench-360", "keywords": "introduces, benchmarking", "description": "The publication introduced the NASBench-360 framework."},
    {"source": "Evolutionary Search", "target": "Gradient-Based Search", "keywords": "performance comparison", "description": "Evolutionary search outperformed gradient-based methods by 12% on accuracy and used 30% fewer GPU-hours on vision tasks."},
    {"source": "NASBench-360", "target": "GPU-Hours", "keywords": "evaluation metric", "description": "NASBench-360 uses GPU-hours as one of three key metrics to measure computational cost."}
  ]
}

""",
]

PROMPTS["summarize_entity_descriptions"] = """---Role---
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Your task is to synthesize a list of descriptions of a given entity or relation into a single, comprehensive, and cohesive summary.

---Instructions---
1. Input Format: The description list is provided in JSON format. Each JSON object (representing a single description) appears on a new line within the `Description List` section.
2. Output Format: The merged description will be returned as plain text, presented in multiple paragraphs, without any additional formatting or extraneous comments before or after the summary.
3. Comprehensiveness: The summary must integrate all key information from *every* provided description. Do not omit any important facts or details.
4. Context: Ensure the summary is written from an objective, third-person perspective; explicitly mention the name of the entity or relation for full clarity and context.
5. Context & Objectivity:
  - Write the summary from an objective, third-person perspective.
  - Explicitly mention the full name of the entity or relation at the beginning of the summary to ensure immediate clarity and context.
6. Conflict Handling:
  - In cases of conflicting or inconsistent descriptions, first determine if these conflicts arise from multiple, distinct entities or relationships that share the same name.
  - If distinct entities/relations are identified, summarize each one *separately* within the overall output.
  - If conflicts within a single entity/relation (e.g., historical discrepancies) exist, attempt to reconcile them or present both viewpoints with noted uncertainty.
7. Length Constraint:The summary's total length must not exceed {summary_length} tokens, while still maintaining depth and completeness.
8. Language: The entire output must be written in {language}. Proper nouns (e.g., personal names, place names, organization names) may in their original language if proper translation is not available.
  - The entire output must be written in {language}.
  - Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

---Input---
{description_type} Name: {description_name}

Description List:

```
{description_list}
```

---Output---
"""

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

PROMPTS["rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph and Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize both `Knowledge Graph Data` and `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a references section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Every grounded answer MUST include at least one reference entry. If no supporting reference can be cited from the provided context, say you do not have enough information to answer.
  - Never answer by summarizing, paraphrasing, or enumerating the `Reference Document List` itself unless the user explicitly asks about sources or references.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `- [n] Source`.
  - `Source` MUST be copied from the matching entry in the `Reference Document List` exactly as written.
  - If `Source` is an `http://` or `https://` URL, render it as a Markdown link using the same URL for both label and destination: `- [n] [https://example.com](https://example.com)`.
  - If `Source` is not a URL, output it verbatim after the reference number without inventing a title, alias, or summary.
  - Never replace a URL with prose like "Document Title" or "Source Article".
  - Output each citation on an individual line.
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] https://example.com/doc1
- [2] Workspace Documentation v2.3
- [3] https://docs.example.com/api
```

6. Additional Instructions: {user_prompt}


---Context---

{context_data}
"""

PROMPTS["apfel_rag_response"] = """Answer only from Context.

Rules:
- `### Answer`: 1-3 bullets, max 90 words total.
- Same language as user.
- If evidence is weak, say what is uncertain.
- Never output references only.
- End with `### References`.
- Use only real sources from `Reference Document List`; never invent URLs.
- No text after references.

Format: {response_type}
User instructions: {user_prompt}

Context:
{context_data}
"""

PROMPTS["apfel_iterative_rag_system"] = """You are the fast local answer synthesizer for LightRAG.
Use only the supplied retrieved portion and carry summary.
Do not invent facts, URLs, filenames, or source ids.
Do not output a references section; the server appends canonical references.
Return exactly two sections:
### Answer Delta
### Carry Summary
Keep both sections concise."""

PROMPTS["apfel_iterative_rag_user"] = """User query:
{query}

Response style:
{response_type}

Carry summary from earlier portions:
{carry_summary}

Retrieved portion {portion_index}/{portion_count}:
{context_data}

Write new useful facts from this portion only. If this portion adds nothing useful, leave Answer Delta empty.
Carry Summary must be a compact summary that preserves facts needed by later portions."""

PROMPTS["apfel_iterative_rag_final_user"] = """User query:
{query}

Response style:
{response_type}

Answer deltas produced from all retrieved portions:
{accumulated_answer}

Carry summary:
{carry_summary}

Write the final concise answer. Do not include references.
Return exactly:
### Answer Delta
<final answer>
### Carry Summary
done"""

PROMPTS["naive_rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a **References** section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Every grounded answer MUST include at least one reference entry. If no supporting reference can be cited from the provided context, say you do not have enough information to answer.
  - Never answer by summarizing, paraphrasing, or enumerating the `Reference Document List` itself unless the user explicitly asks about sources or references.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `- [n] Source`.
  - `Source` MUST be copied from the matching entry in the `Reference Document List` exactly as written.
  - If `Source` is an `http://` or `https://` URL, render it as a Markdown link using the same URL for both label and destination: `- [n] [https://example.com](https://example.com)`.
  - If `Source` is not a URL, output it verbatim after the reference number without inventing a title, alias, or summary.
  - Never replace a URL with prose like "Document Title" or "Source Article".
  - Output each citation on an individual line.
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] https://example.com/doc1
- [2] Workspace Documentation v2.3
- [3] https://docs.example.com/api
```

6. Additional Instructions: {user_prompt}


---Context---

{content_data}
"""

PROMPTS["kg_query_context"] = """
Knowledge Graph Data (Entity):

```json
{entities_str}
```

Knowledge Graph Data (Relationship):

```json
{relations_str}
```

Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["naive_query_context"] = """
Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["keywords_extraction"] = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), comments, or any other text before or after the JSON.
2. **Exact JSON Shape**: The JSON object must contain exactly these two keys:
   - `"high_level_keywords"`: an array of strings
   - `"low_level_keywords"`: an array of strings
3. **JSON Boundary**: The first character of your response must be `{{` and the last character must be `}}`.
4. **Source of Truth**: All keywords must be explicitly derived from the user query. Do not infer unsupported facts. Do not invent entities, products, organizations, dates, or technical terms that are not grounded in the query.
5. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
6. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), return:
   `{{"high_level_keywords": [], "low_level_keywords": []}}`
7. **No Duplicates**: Do not repeat the same keyword within a list. Keep the lists short and high-signal.
8. **Language**: All extracted keywords MUST be in {language}. Proper nouns (e.g., personal names, place names, organization names) should be kept in their original language.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"

Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}

""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"

Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}

""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"

Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}

    """,
]


PROMPTS["agent_tool_protocol_query"] = """You have access to tools for retrieving information. Use them to gather evidence before answering.

To call a tool, reply with exactly one XML block and nothing else:

<tool_call>
{"tool":"tool_name","args":{"key":"value"}}
</tool_call>

Available tools:
- search_entities(query, top_k=10): semantic search for entities relevant to the query.
- search_relations(query, top_k=10): semantic search for relationships/edges relevant to the query.
- search_chunks(query, top_k=10): semantic search for document text chunks relevant to the query.
- get_entity_detail(entity_name): retrieve full metadata and description for a specific entity.
- get_relations_for_entity(entity_name): retrieve all relationships connected to a specific entity.
- web_search(query, num_results=5): search the web using DuckDuckGo for current or external information.

Rules:
- Use tools when you need specific evidence. search_* tools use semantic similarity -- try different phrasings if initial results are poor.
- One tool call per turn. After receiving the result, either call another tool or produce the final grounded answer.
- Web search is available for current events, external facts, or when the knowledge graph lacks sufficient information.
- Final answers must be grounded in tool results. Always cite sources.
- For URL sources, render the citation as a Markdown link: `[url](url)`.
- always end with a `### References` section listing all cited sources.
- Final answers must not contain tool_call XML blocks."""


# ---------------------------------------------------------------------------
# Top-level convenience aliases -- kept in sync with the PROMPTS dict above
# Importing these directly (from lightrag.prompt import RAG_RESPONSE) is
# equivalent to PROMPTS["rag_response"].
# ---------------------------------------------------------------------------

DEFAULT_TUPLE_DELIMITER: str = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
DEFAULT_COMPLETION_DELIMITER: str = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
ENTITY_EXTRACTION_SYSTEM_PROMPT: str = PROMPTS["entity_extraction_system_prompt"]
ENTITY_EXTRACTION_USER_PROMPT: str = PROMPTS["entity_extraction_user_prompt"]
ENTITY_CONTINUE_EXTRACTION_USER_PROMPT: str = PROMPTS["entity_continue_extraction_user_prompt"]
SUMMARIZE_ENTITY_DESCRIPTIONS: str = PROMPTS["summarize_entity_descriptions"]
FAIL_RESPONSE: str = PROMPTS["fail_response"]
RAG_RESPONSE: str = PROMPTS["rag_response"]
NAIVE_RAG_RESPONSE: str = PROMPTS["naive_rag_response"]
KG_QUERY_CONTEXT: str = PROMPTS["kg_query_context"]
NAIVE_QUERY_CONTEXT: str = PROMPTS["naive_query_context"]
KEYWORDS_EXTRACTION: str = PROMPTS["keywords_extraction"]
KEYWORDS_EXTRACTION_EXAMPLES: list = PROMPTS["keywords_extraction_examples"]
AGENT_TOOL_PROTOCOL_QUERY: str = PROMPTS["agent_tool_protocol_query"]

class EntityExtractionPromptProfile(TypedDict):
    entity_types_guidance: str
    entity_extraction_examples: list[str]
    entity_extraction_json_examples: list[str]


def get_default_entity_extraction_prompt_profile() -> EntityExtractionPromptProfile:
    """Return a copy of the built-in entity extraction prompt profile."""

    return {
        "entity_types_guidance": PROMPTS["default_entity_types_guidance"].rstrip(),
        "entity_extraction_examples": [
            example.rstrip() for example in PROMPTS["entity_extraction_examples"]
        ],
        "entity_extraction_json_examples": [
            example.rstrip() for example in PROMPTS["entity_extraction_json_examples"]
        ],
    }


_ALLOWED_PROMPT_SUFFIXES = frozenset({".yml", ".yaml"})
_DEFAULT_PROMPT_DIR = "./prompts"
_ENTITY_TYPE_SUBDIR = "entity_type"


def get_entity_type_prompt_dir() -> Path:
    """Return the directory for entity type prompt profiles.

    Resolves ``PROMPT_DIR`` (defaults to ``./prompts`` relative to the current
    working directory, mirroring ``INPUT_DIR`` / ``WORKING_DIR``) and appends
    the hard-coded ``entity_type`` subdirectory. Profile files are provided by
    the user at runtime and are not shipped with the distribution. The
    file-name sandbox in :func:`resolve_entity_type_prompt_path` ensures
    user-supplied file names cannot escape the resolved directory.
    """

    configured = os.getenv("PROMPT_DIR", "").strip() or _DEFAULT_PROMPT_DIR
    return (Path(configured).expanduser() / _ENTITY_TYPE_SUBDIR).resolve()


def resolve_entity_type_prompt_path(prompt_file_name: str | Path) -> Path:
    """Resolve an allowlisted prompt profile file name to an absolute path."""

    file_name = str(prompt_file_name).strip()
    if not file_name:
        raise ValueError(
            "ENTITY_TYPE_PROMPT_FILE must be a file name such as "
            "'entity_type_prompt.sample.yml'."
        )
    if "\\" in file_name:
        raise ValueError(
            "ENTITY_TYPE_PROMPT_FILE must not contain directory separators. "
            "Only file names inside PROMPT_DIR/entity_type are allowed."
        )

    candidate = Path(file_name)
    if (
        candidate.is_absolute()
        or candidate.name != file_name
        or ".." in candidate.parts
    ):
        raise ValueError(
            "ENTITY_TYPE_PROMPT_FILE must be a file name only. "
            "Files are loaded from PROMPT_DIR/entity_type "
            "(PROMPT_DIR defaults to ./prompts)."
        )
    if candidate.suffix.lower() not in _ALLOWED_PROMPT_SUFFIXES:
        raise ValueError(
            "ENTITY_TYPE_PROMPT_FILE must use a '.yml' or '.yaml' extension."
        )

    return get_entity_type_prompt_dir() / candidate.name


def _normalize_prompt_examples(
    value: Any, field_name: str, profile_path: Path
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(
            f"ENTITY_TYPE_PROMPT_FILE '{profile_path}' field '{field_name}' "
            "must be a list of strings."
        )
    normalized: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"ENTITY_TYPE_PROMPT_FILE '{profile_path}' field '{field_name}' "
                f"item {index} must be a non-empty string."
            )
        normalized.append(item.rstrip())
    return normalized


def load_entity_extraction_prompt_profile(
    prompt_file: str | Path,
) -> dict[str, Any]:
    """Load and validate an entity extraction prompt profile from YAML."""

    profile_path = Path(prompt_file)
    if not profile_path.exists():
        raise FileNotFoundError(
            f"ENTITY_TYPE_PROMPT_FILE '{profile_path}' does not exist."
        )
    if not profile_path.is_file():
        raise ValueError(
            f"ENTITY_TYPE_PROMPT_FILE '{profile_path}' must point to a file."
        )

    try:
        content = profile_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(
            f"Failed to read ENTITY_TYPE_PROMPT_FILE '{profile_path}': {exc}"
        ) from exc

    try:
        raw_profile = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"ENTITY_TYPE_PROMPT_FILE '{profile_path}' contains invalid YAML: {exc}"
        ) from exc

    if raw_profile is None:
        raw_profile = {}
    if not isinstance(raw_profile, dict):
        raise ValueError(
            f"ENTITY_TYPE_PROMPT_FILE '{profile_path}' must contain a YAML mapping."
        )

    profile: dict[str, Any] = {}

    guidance = raw_profile.get("entity_types_guidance")
    if guidance is not None:
        if not isinstance(guidance, str) or not guidance.strip():
            raise ValueError(
                f"ENTITY_TYPE_PROMPT_FILE '{profile_path}' field "
                "'entity_types_guidance' must be a non-empty string."
            )
        profile["entity_types_guidance"] = guidance.rstrip()

    for field_name in (
        "entity_extraction_examples",
        "entity_extraction_json_examples",
    ):
        if field_name in raw_profile:
            profile[field_name] = _normalize_prompt_examples(
                raw_profile[field_name], field_name, profile_path
            )

    return profile


def resolve_entity_extraction_prompt_profile(
    addon_params: Mapping[str, Any] | None,
    use_json: bool,
) -> EntityExtractionPromptProfile:
    """Resolve and merge the configured entity extraction prompt profile."""

    default_profile = get_default_entity_extraction_prompt_profile()
    addon_params = addon_params or {}
    prompt_file = addon_params.get("entity_type_prompt_file")

    file_profile: dict[str, Any] = {}
    if prompt_file:
        prompt_path = resolve_entity_type_prompt_path(prompt_file)
        file_profile = load_entity_extraction_prompt_profile(prompt_path)
        required_examples_key = (
            "entity_extraction_json_examples"
            if use_json
            else "entity_extraction_examples"
        )
        if required_examples_key not in file_profile:
            mode_name = "json" if use_json else "text"
            raise ValueError(
                f"ENTITY_TYPE_PROMPT_FILE '{prompt_file}' must define "
                f"'{required_examples_key}' when entity extraction runs in "
                f"{mode_name} mode."
            )

    guidance = addon_params.get("entity_types_guidance")
    if guidance is None:
        guidance = file_profile.get(
            "entity_types_guidance", default_profile["entity_types_guidance"]
        )
    elif not isinstance(guidance, str) or not guidance.strip():
        raise ValueError(
            "addon_params['entity_types_guidance'] must be a non-empty string."
        )

    return {
        "entity_types_guidance": guidance,
        "entity_extraction_examples": list(
            file_profile.get(
                "entity_extraction_examples",
                default_profile["entity_extraction_examples"],
            )
        ),
        "entity_extraction_json_examples": list(
            file_profile.get(
                "entity_extraction_json_examples",
                default_profile["entity_extraction_json_examples"],
            )
        ),
    }


def validate_entity_extraction_prompt_profile_for_mode(
    prompt_profile: Mapping[str, Any],
    use_json: bool,
    prompt_file_name: str | None = None,
) -> EntityExtractionPromptProfile:
    """Validate that the resolved profile contains the active-mode examples."""

    required_examples_key = (
        "entity_extraction_json_examples" if use_json else "entity_extraction_examples"
    )
    if (
        required_examples_key not in prompt_profile
        or not prompt_profile[required_examples_key]
    ):
        mode_name = "json" if use_json else "text"
        source = (
            f"ENTITY_TYPE_PROMPT_FILE '{prompt_file_name}'"
            if prompt_file_name
            else "the resolved prompt profile"
        )
        raise ValueError(
            f"{source} must define '{required_examples_key}' when entity extraction "
            f"runs in {mode_name} mode."
        )

    return {
        "entity_types_guidance": str(prompt_profile["entity_types_guidance"]).rstrip(),
        "entity_extraction_examples": [
            str(example).rstrip()
            for example in prompt_profile["entity_extraction_examples"]
        ],
        "entity_extraction_json_examples": [
            str(example).rstrip()
            for example in prompt_profile["entity_extraction_json_examples"]
        ],
    }
