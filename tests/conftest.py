"""Shared test fixtures for nanochat tests."""

import os
import tempfile

import pytest


@pytest.fixture
def cpp_tokenizer_dir(tmp_path):
    """Create a minimal HuggingFace tokenizer.json for testing.

    Returns a tmp dir containing tokenizer/tokenizer.json that CppTokenizer can load.
    Patches NANOCHAT_BASE_DIR so get_tokenizer() finds it.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer

    # Build a minimal BPE tokenizer with special tokens matching CppTokenizer
    tok = Tokenizer(BPE(unk_token="<UNK>"))
    special_tokens = [
        "<PAD>", "<UNK>", "<BOS>", "<EOS>",
        "<FIM_PREFIX>", "<FIM_MIDDLE>", "<FIM_SUFFIX>",
        "<CODE_START>", "<CODE_END>",
        "<THOUGHT_START>", "<THOUGHT_END>",
        "<QUERY_TOOL>", "<TOOL_RESULT>",
    ]

    trainer = BpeTrainer(
        special_tokens=special_tokens,
        vocab_size=200,
        min_frequency=1,
    )

    # Train on minimal C++ text
    train_file = tmp_path / "train.txt"
    train_file.write_text(
        "int main() { return 0; }\n"
        "void foo(int x) { }\n"
        '#include <iostream>\n'
        'std::string s = "hello";\n'
        "// comment\n"
        "class Foo { public: int bar; };\n"
        "int x = 1;\n"
    )
    tok.train([str(train_file)], trainer)
    tok.add_special_tokens(special_tokens)

    # Save to tokenizer subdir (matching get_tokenizer() expected layout)
    tok_dir = tmp_path / "tokenizer"
    tok_dir.mkdir()
    tok.save(str(tok_dir / "tokenizer.json"))

    return tmp_path


@pytest.fixture
def patched_tokenizer_env(cpp_tokenizer_dir, monkeypatch):
    """Patch environment so get_tokenizer() uses the test tokenizer.json."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(cpp_tokenizer_dir))
    monkeypatch.setenv("NANOCHAT_CPP_TOKENIZER", "1")
