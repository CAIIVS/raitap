---
title: "Adding an input parser"
description: "How to add a new data/inputs variant to RAITAP: write a class that satisfies the InputParser protocol, add an *InputsConfig schema, register it with @input_parser, and select it via defaults: [data/inputs: <name>]. Walkthrough uses a fictional SuperDoc parser."
myst:
  html_meta:
    "description": "How to add a new data/inputs variant to RAITAP: write a class that satisfies the InputParser protocol, add an *InputsConfig schema, register it with @input_parser, and select it via defaults: [data/inputs: <name>]. Walkthrough uses a fictional SuperDoc parser."
---

# Adding an input parser

An **input parser** turns a raw source (a CSV column, a JSONL field, a
directory of files, ...) into `list[str]`, one string per sample. RAITAP
uses this today for text: `model.tokenizer` set plus a `data.inputs` block
tells `Data._load_data` to read the input parser, then tokenise its output
into `input_ids` + `attention_mask`. Image and tabular sources still load
directly from `data.source`; they do not go through this seam.

The registry lives in `src/raitap/data/input_parsers/` and is a straight
clone of `src/raitap/data/label_parsers/` (the label-parser seam, see
{doc}`/contributor/modules/data`): same `FamilyConfig` shape, same
decorator-based registration, same factory dispatch. `TextCsvInputParser`,
`TextJsonlInputParser`, and `TextDirInputParser` are the three in-tree
variants; copy whichever is closest to your source format.

This walkthrough uses a fictional `superdoc-lib` that reads `.superdoc`
transcript files. We will add a `SuperDocInputParser` that extracts the text
from each file. `superdoc-lib` is an optional dependency, so the example also
shows the `extra` wiring (the three built-in text parsers use core deps and
skip it).

## The contract

Your parser is any class that satisfies the `InputParser` protocol
(`src/raitap/data/input_parsers/base.py`) by structure. There is no base
class to inherit. The protocol asks for exactly two things:

- `supported_modalities: frozenset[InputModality]` (attribute)
- `parse(self, *, source: str, sample_ids: list[str] | None) -> list[str]` (method)

You never write `class InputParser(Protocol)` yourself, that is the framework
contract. You write a concrete class (`SuperDocInputParser` below) and the
`@input_parser` decorator both registers it and, because the protocol is
`runtime_checkable`, lets the factory verify it satisfies the contract.

## 1. Add a `SuperDocInputsConfig` schema

In `src/raitap/configs/schema.py`, subclass `InputsConfig`:

```python
@dataclass
class SuperDocInputsConfig(InputsConfig):
    _target_: str = "SuperDocInputParser"
    source: str = MISSING
    section: str = "body"   # add only the fields this variant actually uses
```

`_target_` must match the class name from step 2 (resolved against
`raitap.data.input_parsers.` by the factory, see step 4).

## 2. Write the parser class

Create `src/raitap/data/input_parsers/superdoc.py`, decorated with
`@input_parser`:

```python
from raitap.configs.schema import SuperDocInputsConfig
from raitap.data.input_parsers.registration import input_parser
from raitap.data.types import InputModality


@input_parser(registry_name="superdoc", schema=SuperDocInputsConfig, extra="superdoc")
class SuperDocInputParser:
    supported_modalities = frozenset({InputModality.text})

    def __init__(self, *, source: str, section: str = "body") -> None:
        self.source = source
        self.section = section

    def parse(self, *, source: str, sample_ids: list[str] | None) -> list[str]:
        import superdoc  # lazy: optional dep, only imported when this parser runs

        docs = superdoc.load(self.source)
        return [doc.text(self.section) for doc in docs]
```

Notes:

- `registry_name` is required.
- `extra="superdoc"` names the uv extra that ships `superdoc-lib`; deps
  inference then knows a config using this parser needs it. Omit `extra`
  (it defaults to `""`) when your parser only uses core dependencies, like
  the built-in `TextCsvInputParser`.
- Import the optional library **inside** `parse`, never at module top level,
  so the core install can import the module without `superdoc-lib` present.
- `sample_ids` is passed for parity with the label seam; ignore it if your
  format has no per-sample id.

## 3. Import in `__init__.py`

The decorator only runs when the module is imported. Add the import to
`src/raitap/data/input_parsers/__init__.py` and export the class:

```python
from .superdoc import SuperDocInputParser  # pyright: ignore[reportUnusedImport]

__all__ = [..., "SuperDocInputParser"]
```

## 4. Select it from a config

`data/inputs` is a Hydra config-group (`FamilyConfig(group="data/inputs", ...)`
in `registration.py`). Pick the variant with `defaults:`, then set its
fields under `data.inputs:`:

```yaml
defaults:
  - raitap_schema
  - data/inputs: superdoc
  - _self_

model:
  tokenizer: some-hf-model-id   # required: selects the text modality

data:
  source: "./data/transcripts"
  inputs:
    source: "./data/transcripts"
    section: "body"
```

`create_input_parser` (`src/raitap/data/input_parsers/factory.py`) resolves
`_target_` against `raitap.data.input_parsers.` at call time, the same
dispatch `create_label_parser` uses for `data/labels`.

## 5. Add tests

Add tests in `src/raitap/data/tests/`. `test_input_parser_registry.py` covers
the registry mechanics (decorator registration, factory dispatch, config
composition); `test_text_input_parsers.py` covers the three built-in variants.
Follow their shape: a small fixture source, then assert `parse(...)` returns
the expected `list[str]`.

## Worked example: text

The three in-tree variants (`src/raitap/data/input_parsers/text_csv.py`,
`text_jsonl.py`, `text_dir.py`) all satisfy `InputModality.text` and use core
deps (no `extra`). A complete runnable config lives at
`contributor-configs/text-classification-sst2/`: `data/inputs: text_csv` reads
a `text` column from a CSV, `model.tokenizer` tokenises it, and the assessment
runs token attribution end to end. See {doc}`/modules/data/own-vs-built-in`
for the user-facing configuration reference.
