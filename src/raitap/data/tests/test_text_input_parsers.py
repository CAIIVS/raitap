import json
from pathlib import Path

from raitap.data.input_parsers.text_csv import TextCsvInputParser
from raitap.data.input_parsers.text_dir import TextDirInputParser
from raitap.data.input_parsers.text_jsonl import TextJsonlInputParser


def test_csv_reads_text_column(tmp_path: Path) -> None:
    p = tmp_path / "d.csv"
    p.write_text("id,text\n0,hello\n1,world\n")
    assert TextCsvInputParser(text_column="text").parse(source=str(p)) == ["hello", "world"]


def test_jsonl_reads_text_field(tmp_path: Path) -> None:
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps({"text": t}) for t in ["a", "b"]))
    assert TextJsonlInputParser().parse(source=str(p)) == ["a", "b"]


def test_dir_reads_txt_files_sorted(tmp_path: Path) -> None:
    (tmp_path / "1.txt").write_text("one")
    (tmp_path / "2.txt").write_text("two")
    assert TextDirInputParser().parse(source=str(tmp_path)) == ["one", "two"]


def test_dir_reads_md_files(tmp_path: Path) -> None:
    (tmp_path / "1.md").write_text("md-one")
    (tmp_path / "2.txt").write_text("txt-two")
    assert TextDirInputParser().parse(source=str(tmp_path)) == ["md-one", "txt-two"]
