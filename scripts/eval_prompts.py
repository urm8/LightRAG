#!/usr/bin/env python3
import argparse

import requests

TESTS = [
    """In 1859, Charles Darwin published On the Origin of Species in London. The book introduced natural selection as a method for explaining biological evolution during the Victorian period. The Royal Society later debated the concept, while the microscope remained an important technology for observing organisms.""",
    """В 1938 году Сергей Королёв работал в Москве над ракетными технологиями. Позднее организация ОКБ-1 использовала метод многоступенчатой ракеты для разработки космического аппарата «Спутник-1». Запуск «Спутника-1» стал важным событием советского космического периода.""",
    """During the Renaissance, Leonardo da Vinci studied anatomy in Florence and recorded observations in his notebooks. His drawing Vitruvian Man became an artifact of humanist thought. The Medici family supported artists, while observational drawing functioned as a method for studying the body.""",
    """В эпоху Петра I Санкт-Петербург стал центром реформ. Сенат как государственная организация участвовал в управлении, а Табель о рангах была документом, закреплявшим новую систему службы. Строительство флота стало событием и технологическим проектом ранней Российской империи.""",
    """In 1969, NASA conducted the Apollo 11 mission. Neil Armstrong walked on the Moon, while the Saturn V rocket demonstrated advanced launch technology. The mission report documented the lunar landing procedure, and the Cold War period shaped the political meaning of the event.""",
]

PROMPT = """Extract entities and relations from the text and return JSON.

Allowed types:
Person
Organization
Location
Event
Concept
Technology
Method
Artifact
Document
Period

Return a JSON object with:
- entities: [{entity_name, entity_type, entity_description}]
- relations: [{source_entity, target_entity, relationship_keywords, relationship_description}]

Rules:
- Use exact text spans from the input when possible.
- Keep descriptions factual and grounded in the text.
- relationship_keywords must be a short list of strings.
- Do not add commentary or wrapper text.

Text:
{text}
"""


def ask_ollama(model: str, text: str, host: str) -> str:
    response = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": PROMPT.format(text=text),
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_k": 50,
                "repeat_penalty": 1.05,
                "num_predict": 500,
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        nargs="?",
        default="huihui_ai/granite4.1-abliterated:3b",
    )
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
    )
    args = parser.parse_args()

    for i, text in enumerate(TESTS, 1):
        print(f"\n===== TEST {i} =====")
        print(ask_ollama(args.model, text, args.host))


if __name__ == "__main__":
    main()
