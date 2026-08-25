from lightrag import observability


class _Histogram:
    def __init__(self):
        self.records = []

    def record(self, value, attributes):
        self.records.append((value, attributes))


def test_gen_ai_usage_records_tokens_and_duration(monkeypatch):
    tokens = _Histogram()
    duration = _Histogram()
    monkeypatch.setattr(
        observability,
        "_instruments",
        {"token_usage": tokens, "llm_duration": duration},
    )

    observability.record_gen_ai_usage(
        operation="chat",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        input_tokens=120,
        output_tokens=30,
        duration_s=1.25,
        role="query",
    )

    assert [record[0] for record in tokens.records] == [120, 30]
    assert [record[1]["gen_ai.token.type"] for record in tokens.records] == [
        "input",
        "output",
    ]
    assert duration.records[0][0] == 1.25
    assert duration.records[0][1]["gen_ai.provider.name"] == "deepseek"
    assert duration.records[0][1]["lightrag.llm.role"] == "query"


def test_setup_telemetry_is_disabled_without_endpoint(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.setattr(observability, "_initialized", False)

    assert observability.setup_telemetry(object()) is False
