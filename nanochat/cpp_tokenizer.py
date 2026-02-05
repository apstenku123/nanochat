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

    # --- nanochat-compatible API ---

    def get_bos_token_id(self):
        return self.bos_token_id

    def get_vocab_size(self):
        return self.vocab_size

    def get_special_tokens(self):
        # Return all angle-bracket tokens
        return [t for t in self._vocab if t.startswith("<") and t.endswith(">")]

    def encode_special(self, text):
        # Support both nanochat-style <|bos|> and our <BOS> format
        result = self._vocab.get(text, None)
        if result is None:
            # Map nanochat special tokens to our format
            mapping = {
                "<|bos|>": "<BOS>", "<|eos|>": "<EOS>",
                "<|pad|>": "<PAD>", "<|endoftext|>": "<BOS>",
                # Tool-calling tokens (nanochat-style aliases)
                "<|code_start|>": "<CODE_START>",
                "<|code_end|>": "<CODE_END>",
                "<|thought_start|>": "<THOUGHT_START>",
                "<|thought_end|>": "<THOUGHT_END>",
                "<|query_tool|>": "<QUERY_TOOL>",
                "<|tool_result|>": "<TOOL_RESULT>",
            }
            mapped = mapping.get(text)
            if mapped:
                result = self._vocab.get(mapped, None)
        return result

    def encode(self, text, prepend=None, append=None, num_threads=8):
        """Encode text or list of texts. Compatible with nanochat API."""
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self._tokenizer.encode(text).ids
            if prepend is not None:
                ids.insert(0, prepend_id)
            if append is not None:
                ids.append(append_id)
            return ids
        elif isinstance(text, list):
            results = [enc.ids for enc in self._tokenizer.encode_batch(text)]
            if prepend is not None:
                for row in results:
                    row.insert(0, prepend_id)
            if append is not None:
                for row in results:
                    row.append(append_id)
            return results
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

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

    # Tool-calling special token IDs
    @property
    def code_start_id(self) -> int:
        return self._vocab.get("<CODE_START>", 7)

    @property
    def code_end_id(self) -> int:
        return self._vocab.get("<CODE_END>", 8)

    @property
    def thought_start_id(self) -> int:
        return self._vocab.get("<THOUGHT_START>", 9)

    @property
    def thought_end_id(self) -> int:
        return self._vocab.get("<THOUGHT_END>", 10)

    @property
    def query_tool_id(self) -> int:
        return self._vocab.get("<QUERY_TOOL>", 11)

    @property
    def tool_result_id(self) -> int:
        return self._vocab.get("<TOOL_RESULT>", 19)
