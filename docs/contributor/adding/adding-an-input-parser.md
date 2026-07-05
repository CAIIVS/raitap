---
title: "Adding an input parser"
description: "How to add a new data/inputs variant to RAITAP: implement the InputParser protocol, add an *InputsConfig schema, register it with @input_parser, and select it via defaults: [data/inputs: <name>]. Mirrors the label-parser seam."
myst:
  html_meta:
    "description": "How to add a new data/inputs variant to RAITAP: implement the InputParser protocol, add an *InputsConfig schema, register it with @input_parser, and select it via defaults: [data/inputs: <name>]. Mirrors the label-parser seam."
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
`TextJsonlInputParser`, and `TextDirInputParser` are the three worked
examples; copy whichever is closest to your source format.

## 1. Add an `*InputsConfig` schema

In `src/raitap/configs/schema.py`, subclass `InputsConfig`:

```python
@dataclass
class MyFormatInputsConfig(InputsConfig):
    _target_: str = "MyFormatInputParser"
    source: str = MISSING
    # add only fields this variant uses
```

`_target_` must match the class name from step 2 (resolved against
`raitap.data.input_parsers.` by the factory, see step 4).

## 2. Implement the `InputParser` protocol

`InputParser` (`src/raitap/data/input_parsers/base.py`) is a
`runtime_checkable` `Protocol` with two members:

```python
@runtime_checkable
class InputParser(Protocol):
    supported_modalities: frozenset[InputModality]

    def parse(self, *, source: str, sample_ids: list[str] | None) -> list[str]: ...
```

Write the parser class in `src/raitap/data/input_parsers/<name>.py`,
decorated with `@input_parser`:

```python
from raitap.configs.schema import MyFormatInputsConfig
from raitap.data.input_parsers.registration import input_parser
from raitap.data.types import InputModality


@input_parser(registry_name="my_format", schema=MyFormatInputsConfig)
class MyFormatInputParser:
    supported_modalities = frozenset({InputModality.text})

    def __init__(self, *, source: str) -> None:
        self.source = source

    def parse(self, *, source: str, sample_ids: list[str] | None) -> list[str]:
        # read self.source, return one string per sample
        ...
```

`registry_name` is required; `@input_parser` defaults `extra` to `""`
(no optional-dependency extra needed) unless your parser pulls in one.

## 3. Import in `__init__.py`

Decorators only run when the module is imported. Add the import to
`src/raitap/data/input_parsers/__init__.py` and export the class:

```python
from .my_format import MyFormatInputParser  # pyright: ignore[reportUnusedImport]

__all__ = [..., "MyFormatInputParser"]
```

## 4. Select it from a config

`data/inputs` is a Hydra config-group (`FamilyConfig(group="data/inputs", ...)`
in `registration.py`). Pick the variant with `defaults:`, then set its
fields under `data.inputs:`:

```yaml
defaults:
  - raitap_schema
  - data/inputs: my_format
  - _self_

model:
  tokenizer: some-hf-model-id   # required: selects the text modality

data:
  source: "./data/my_source"
  inputs:
    source: "./data/my_source"
```

`create_input_parser` (`src/raitap/data/input_parsers/factory.py`) resolves
`_target_` against `raitap.data.input_parsers.` at call time, same
dispatch as `create_label_parser` for `data/labels`.

## 5. Add tests

Add tests in `src/raitap/data/tests/`. `test_input_parser_registry.py` and
`test_text_input_parsers.py` cover the registry mechanics and the three
built-in variants; follow their shape for a new variant.

## Worked example: text

The three built-in variants (`src/raitap/data/input_parsers/text_csv.py`,
`text_jsonl.py`, `text_dir.py`) all satisfy `InputModality.text`. A complete
runnable config lives at `contributor-configs/text-classification-sst2/`:
`data/inputs: text_csv` reads a `text` column from a CSV, `model.tokenizer`
tokenises it, and the assessment runs token attribution end to end. See
{doc}`/modules/data/own-vs-built-in` for the user-facing configuration
reference.
