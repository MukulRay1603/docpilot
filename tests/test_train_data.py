"""Fix 1: train.py must accept the dataset build_demo_corpus.py writes."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from train.data_utils import DatasetFormatError, load_splits

ROOT = Path(__file__).resolve().parent.parent


def test_loads_builder_output(tmp_path):
    out = tmp_path / "data"
    out.mkdir()
    # Run the real builder into a temp dir so the repo's data/ is untouched.
    code = (
        "import sys, pathlib; sys.path.insert(0, %r);"
        "import importlib.util as u;"
        "s=u.spec_from_file_location('b', %r); m=u.module_from_spec(s); s.loader.exec_module(m);"
        "m.OUT_DIR=pathlib.Path(%r); m.main()"
    ) % (str(ROOT), str(ROOT / "data" / "build_demo_corpus.py"), str(out))
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
    raw = json.loads((out / "qa_dataset.json").read_text())
    assert "train" not in raw and "data" in raw  # the format that used to KeyError

    train, val = load_splits(out / "qa_dataset.json")
    assert len(train) + len(val) == 96
    assert 5 <= len(val) <= 30
    assert not {r["id"] for r in train} & {r["id"] for r in val}
    a = train[0]["answers"]
    assert isinstance(a["text"], list) and isinstance(a["answer_start"], list)
    ctx = train[0]["context"]
    assert ctx[a["answer_start"][0]:].lower().startswith(a["text"][0][:20].lower())
    # Deterministic
    train2, val2 = load_splits(out / "qa_dataset.json")
    assert [r["id"] for r in val] == [r["id"] for r in val2]


def test_explicit_split_layout(tmp_path):
    art = {"data": [{"title": "t", "paragraphs": [{"context": "abc def", "qas": [
        {"id": "1", "question": "q?", "answers": [{"text": "def", "answer_start": 4}]}]}]}]}
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"train": art, "validation": art}))
    train, val = load_splits(p)
    assert len(train) == len(val) == 1


def test_bad_files_raise(tmp_path):
    with pytest.raises(DatasetFormatError):
        load_splits(tmp_path / "missing.json")
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"version": "x"}))
    with pytest.raises(DatasetFormatError):
        load_splits(p)


def test_train_does_not_create_output_dir_before_success():
    src = (ROOT / "train" / "train.py").read_text()
    main = src[src.index("def main():"):]
    # Dataset is validated first, and the final dir is only created via rename.
    assert main.index("load_squad_json") < main.index("from_pretrained")
    assert "OUTPUT_DIR.mkdir" not in src
    assert "staging.rename(OUTPUT_DIR)" in main
