"""
C++ Hybrid Tokenizer: fixed C++ vocab + learned BPE, BERT-style whitespace.
See docs/design/01-tokenizer.md for full design.

Encode works via HuggingFace tokenizers library.
Decode uses custom space reconstruction (BERT-style: insert spaces between word tokens).
"""
import os
import json
from tokenizers import Tokenizer


# Single-char punctuation set
_PUNCT = set("{}()[]<>;:,.+-*/%&|^~!?=#@$_\\\"'")
# Multi-char operators
_MULTI_OPS = {"::", "->", ".*", "->*", "++", "--", "##", "==", "!=",
              "<=", ">=", "<=>", "&&", "||", "<<", ">>",
              "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
              "<<=", ">>=", "...", "//", "/*", "*/",
              "@@", "---", "+++", "a/", "b/"}


class CppTokenizer:
    """Hybrid C++ tokenizer: fixed vocab + learned BPE, BERT-style whitespace."""

    def __init__(self, tokenizer_path: str):
        if os.path.isdir(tokenizer_path):
            tokenizer_path = os.path.join(tokenizer_path, "tokenizer.json")
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        # Build reverse map for decoding
        self._vocab = self._tokenizer.get_vocab()
        self._id_to_token = {v: k for k, v in self._vocab.items()}

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [enc.ids for enc in self._tokenizer.encode_batch(texts)]

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text with BERT-style space reconstruction.

        Heuristic rules for C++ spacing. Output is approximate —
        use clang-format for exact formatting.
        """
        tokens = [self._id_to_token.get(i, "<UNK>") for i in ids]
        if not tokens:
            return ""

        def need_space(prev, curr):
            # Whitespace tokens: never add extra space
            if curr in ("\n", "\n\n") or prev in ("\n", "\n\n"):
                return False
            # Special tokens: no space
            if (curr.startswith("<") and curr.endswith(">") and len(curr) > 1):
                return False
            if (prev.startswith("<") and prev.endswith(">") and len(prev) > 1):
                return False
            # Attach operators (no space either side): :: -> .* ->* ++ -- ##
            if curr in ("::", "->", ".*", "->*", "++", "--", "##"):
                return False
            if prev in ("::", "->", ".*", "->*", "++", "--", "##"):
                return False
            # . always attaches (member access)
            if curr == "." or prev == ".":
                return False
            # No space after ( [ and no space before ) ] ; , .
            if prev in ("(", "["):
                return False
            if curr in (")", "]", ";", ","):
                return False
            # No space between word/keyword and (  e.g. main(, if(, for(
            if curr == "(":
                return False
            # Space before { (block opener)
            if curr == "{":
                return True
            # No space after < and before > (template)
            if prev == "<" or curr == ">":
                return False
            # Binary operators: always space around
            if curr in _MULTI_OPS or prev in _MULTI_OPS:
                return True
            # Single-char operators: = + - * / % & | ^ ~ ! ? < > #
            if curr in "=+-*/%&|^~!?<>#" or prev in "=+-*/%&|^~!?<>#":
                return True
            # Remaining: no space before : (label, access specifier)
            if curr == ":":
                return False
            if prev == ":":
                return True
            # Two word tokens: space
            return True

        parts = [tokens[0]]
        for i in range(1, len(tokens)):
            if need_space(tokens[i - 1], tokens[i]):
                parts.append(" ")
            parts.append(tokens[i])

        return "".join(parts)

    def id_to_token(self, id: int) -> str:
        return self._id_to_token.get(id, "<UNK>")

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def bos_token_id(self) -> int:
        return self._vocab.get("<BOS>", 2)

    @property
    def eos_token_id(self) -> int:
        return self._vocab.get("<EOS>", 3)

    @property
    def pad_token_id(self) -> int:
        return self._vocab.get("<PAD>", 0)
