from typing import Mapping, Optional, Tuple


DOLLY_SCHEMA = "dolly"
INSTRUCTION_OUTPUT_SCHEMA = "instruction_output"
ALPACA_SCHEMA = "alpaca"


def detect_dataset_schema(record: Mapping[str, object]) -> str:
    keys = set(record.keys())

    if {"instruction", "context", "response"}.issubset(keys):
        return DOLLY_SCHEMA
    if {"instruction", "input", "output"}.issubset(keys):
        return ALPACA_SCHEMA
    if {"instruction", "output"}.issubset(keys) and "response" not in keys:
        return INSTRUCTION_OUTPUT_SCHEMA

    raise ValueError(
        "Unsupported dataset schema. Expected Dolly "
        "(instruction/context/response), instruction-output "
        "(instruction/output), or Alpaca-style (instruction/input/output)."
    )


def prompt_fields(record: Mapping[str, object]) -> Tuple[str, Optional[str], str, str]:
    schema = detect_dataset_schema(record)

    if schema == DOLLY_SCHEMA:
        return record["instruction"], record.get("context"), record["response"], schema
    if schema == INSTRUCTION_OUTPUT_SCHEMA:
        return record["instruction"], None, record["output"], schema

    return record["instruction"], record.get("input"), record["output"], schema
