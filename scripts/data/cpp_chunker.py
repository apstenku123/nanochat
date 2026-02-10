"""
Syntax-aware C++ file splitter.

Parses a C++ source file into semantic chunks: preamble (#includes, typedefs),
functions (complete signature + body), class/struct blocks, and top-level code.

Uses brace-depth tracking at the FILE SCOPE level — only splits at depth-0
braces, so class methods, nested blocks, etc. stay together.

Usage as library:
    from scripts.data.cpp_chunker import chunk_file
    chunks = chunk_file(source_code)
    for chunk in chunks:
        print(chunk.kind, len(chunk.text), chunk.name)
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Chunk:
    kind: str        # "preamble", "function", "class", "namespace", "top_level"
    text: str        # full text of this chunk
    name: str = ""   # function/class/namespace name (empty for preamble/top_level)
    start_line: int = 0
    end_line: int = 0


# Lines that belong in preamble
_PREAMBLE_LINE_RE = re.compile(
    r'^\s*(?:'
    r'#\s*(?:include|define|undef|ifdef|ifndef|if|else|elif|endif|pragma|error|warning)\b'
    r'|using\s+'
    r'|typedef\s+'
    r'|extern\s+"C"'
    r'|namespace\s+\w+\s*;'
    r'|class\s+\w+\s*;'
    r'|struct\s+\w+\s*;'
    r'|enum\s+(?:class\s+)?\w+\s*;'
    r'|//.*$'
    r'|\s*$'
    r')$'
)

# Classify what precedes an opening brace at file scope
_CLASS_BEFORE_BRACE = re.compile(
    r'(?:template\s*<[^>]*>\s*)?'
    r'(?:class|struct)\s+'
    r'(?:__declspec\s*\([^)]*\)\s*)?'
    r'(?:__attribute__\s*\(\([^)]*\)\)\s*)?'
    r'(\w+)'                            # class name
    r'(?:\s*:\s*[^{]*)?'               # base classes
    r'\s*$'
)

_NAMESPACE_BEFORE_BRACE = re.compile(
    r'namespace\s+(\w+)\s*$'
)

_ANON_NAMESPACE_BEFORE_BRACE = re.compile(
    r'namespace\s*$'
)

_ENUM_BEFORE_BRACE = re.compile(
    r'enum\s+(?:class\s+)?(\w+)(?:\s*:\s*\w+)?\s*$'
)

# Function: something with parens before the brace that isn't a control flow keyword
_CONTROL_KEYWORDS = {'if', 'else', 'for', 'while', 'do', 'switch', 'try', 'catch'}

_FUNC_BEFORE_BRACE = re.compile(
    r'(\w+)\s*\([^)]*\)\s*'
    r'(?:const|volatile|override|final|noexcept|throw\s*\([^)]*\)|'
    r'->[\w:&*\s<>,]+|\s)*'
    r'\s*$'
)


def _skip_string(code: str, pos: int) -> int:
    """Skip past a string literal starting at pos (which is on the opening quote)."""
    quote = code[pos]
    i = pos + 1
    n = len(code)
    while i < n:
        if code[i] == '\\':
            i += 2
            continue
        if code[i] == quote:
            return i + 1
        i += 1
    return n


def _skip_line_comment(code: str, pos: int) -> int:
    """Skip past a // comment."""
    end = code.find('\n', pos)
    return end + 1 if end >= 0 else len(code)


def _skip_block_comment(code: str, pos: int) -> int:
    """Skip past a /* */ comment."""
    end = code.find('*/', pos + 2)
    return end + 2 if end >= 0 else len(code)


def _find_matching_brace(code: str, open_pos: int) -> Optional[int]:
    """Find closing '}' matching the '{' at open_pos.

    Correctly handles strings, chars, line/block comments, nested braces.
    """
    depth = 1
    i = open_pos + 1
    n = len(code)

    while i < n:
        c = code[i]

        if c == '"' or c == "'":
            i = _skip_string(code, i)
            continue
        if c == '/' and i + 1 < n:
            if code[i + 1] == '/':
                i = _skip_line_comment(code, i)
                continue
            if code[i + 1] == '*':
                i = _skip_block_comment(code, i)
                continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1

    return None


def _line_number(code: str, pos: int) -> int:
    return code[:pos].count('\n') + 1


def _classify_block(prefix: str) -> tuple[str, str]:
    """Given text before an opening '{', classify what kind of block it is.

    Returns (kind, name).
    """
    # Strip trailing whitespace for matching
    text = prefix.rstrip()

    # Namespace
    m = _NAMESPACE_BEFORE_BRACE.search(text)
    if m:
        return "namespace", m.group(1)
    if _ANON_NAMESPACE_BEFORE_BRACE.search(text):
        return "namespace", ""

    # Class/struct
    m = _CLASS_BEFORE_BRACE.search(text)
    if m:
        return "class", m.group(1)

    # Enum
    m = _ENUM_BEFORE_BRACE.search(text)
    if m:
        return "class", m.group(1)  # treat enum as class-like

    # Function (must have parens, must not be control keyword)
    m = _FUNC_BEFORE_BRACE.search(text)
    if m:
        name = m.group(1)
        if name not in _CONTROL_KEYWORDS:
            return "function", name

    # Check for extern "C" { block
    if re.search(r'extern\s+"C"\s*$', text):
        return "namespace", "extern_C"

    return "top_level", ""


def _is_preamble(text: str) -> bool:
    """Check if a block of text is all preamble (includes, macros, comments, blanks)."""
    for line in text.split('\n'):
        if line.strip() and not _PREAMBLE_LINE_RE.match(line):
            return False
    return True


def chunk_file(code: str) -> list[Chunk]:
    """Split a C++ source file into semantic chunks at the top (file) scope.

    Only splits at depth-0 braces — class methods, nested blocks, etc. are
    kept intact within their parent block.

    Returns a list of Chunk objects in source order.
    """
    if not code or not code.strip():
        return []

    chunks = []
    n = len(code)
    pos = 0               # current scan position
    gap_start = 0         # start of text between blocks

    while pos < n:
        c = code[pos]

        # Skip strings
        if c == '"' or c == "'":
            pos = _skip_string(code, pos)
            continue

        # Skip comments
        if c == '/' and pos + 1 < n:
            if code[pos + 1] == '/':
                pos = _skip_line_comment(code, pos)
                continue
            if code[pos + 1] == '*':
                pos = _skip_block_comment(code, pos)
                continue

        # Found top-level opening brace
        if c == '{':
            close_pos = _find_matching_brace(code, pos)
            if close_pos is None:
                # Unmatched — rest is top_level
                break

            # Text before this brace (the "prefix") helps classify the block
            prefix = code[gap_start:pos]

            # Find where the "block start" really begins
            # Look backwards from the brace for the start of the statement
            lines_before = prefix.split('\n')

            # Strip trailing empty lines (handles Allman/C brace style)
            while lines_before and not lines_before[-1].strip():
                lines_before.pop()

            # Walk backwards from end to find where the declaration starts
            # (stop at a blank line)
            sig_start = len(lines_before)
            for j in range(len(lines_before) - 1, -1, -1):
                line = lines_before[j].strip()
                if not line:
                    break
                sig_start = j
            # Everything from gap_start to sig_start is "between blocks" (preamble/top_level)
            between_text = '\n'.join(lines_before[:sig_start]).strip()
            sig_text = '\n'.join(lines_before[sig_start:])

            # Emit the "between" gap if non-empty
            if between_text:
                kind = "preamble" if _is_preamble(between_text) else "top_level"
                chunks.append(Chunk(
                    kind=kind, text=between_text,
                    start_line=_line_number(code, gap_start),
                    end_line=_line_number(code, gap_start + len(between_text)),
                ))

            # Classify the block
            kind, name = _classify_block(sig_text)

            # The full block text = signature + { body }
            block_text = sig_text.strip() + code[pos:close_pos + 1]

            # Include trailing ';' for classes/structs/enums
            after = close_pos + 1
            if kind == "class":
                while after < n and code[after] in ' \t\n':
                    after += 1
                if after < n and code[after] == ';':
                    block_text += ';'
                    after += 1

            # For namespace blocks, recursively chunk the inside
            if kind == "namespace" and name != "extern_C":
                inner = code[pos + 1:close_pos]
                inner_chunks = chunk_file(inner)
                if inner_chunks:
                    ns_open = sig_text.strip() + " {\n"
                    ns_close = f"\n}} // namespace {name}" if name else "\n}"
                    for ic in inner_chunks:
                        ic.text = ns_open + ic.text + ns_close
                        ic.start_line += _line_number(code, pos)
                        ic.end_line += _line_number(code, pos)
                    chunks.extend(inner_chunks)
                else:
                    chunks.append(Chunk(
                        kind=kind, text=block_text, name=name,
                        start_line=_line_number(code, gap_start),
                        end_line=_line_number(code, close_pos),
                    ))
            else:
                chunks.append(Chunk(
                    kind=kind, text=block_text, name=name,
                    start_line=_line_number(code, gap_start),
                    end_line=_line_number(code, close_pos),
                ))

            pos = after
            gap_start = pos
            continue

        pos += 1

    # Remaining text after last block
    remaining = code[gap_start:].strip()
    if remaining:
        kind = "preamble" if _is_preamble(remaining) else "top_level"
        chunks.append(Chunk(
            kind=kind, text=remaining,
            start_line=_line_number(code, gap_start),
            end_line=_line_number(code, n - 1),
        ))

    # Merge consecutive preamble chunks
    merged = []
    for chunk in chunks:
        if merged and merged[-1].kind == "preamble" and chunk.kind == "preamble":
            merged[-1].text = merged[-1].text + "\n\n" + chunk.text
            merged[-1].end_line = chunk.end_line
        else:
            merged.append(chunk)

    return merged


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        demo = '''#include <iostream>
#include <vector>
#include <string>

using namespace std;

// A simple linked list node
struct Node {
    int data;
    Node* next;
    Node(int d) : data(d), next(nullptr) {}
};

class LinkedList {
public:
    Node* head;

    LinkedList() : head(nullptr) {}

    void push(int data) {
        Node* n = new Node(data);
        n->next = head;
        head = n;
    }

    void print() {
        Node* cur = head;
        while (cur) {
            cout << cur->data << " -> ";
            cur = cur->next;
        }
        cout << "null" << endl;
    }
};

namespace utils {

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

} // namespace utils

int main() {
    LinkedList list;
    list.push(3);
    list.push(2);
    list.push(1);
    list.print();
    cout << "5! = " << utils::factorial(5) << endl;
    return 0;
}
'''
        chunks = chunk_file(demo)
        print(f"Total chunks: {len(chunks)}")
        for i, c in enumerate(chunks):
            print(f"\n{'='*60}")
            print(f"Chunk {i}: kind={c.kind}, name={c.name!r}, lines {c.start_line}-{c.end_line}, chars={len(c.text)}")
            print(f"{'='*60}")
            lines = c.text.split('\n')
            if len(lines) <= 10:
                print(c.text)
            else:
                for line in lines[:5]:
                    print(line)
                print(f"  ... ({len(lines) - 7} more lines) ...")
                for line in lines[-2:]:
                    print(line)
    else:
        with open(sys.argv[1]) as f:
            code = f.read()
        chunks = chunk_file(code)
        total_chars = sum(len(c.text) for c in chunks)
        print(f"File: {sys.argv[1]}")
        print(f"Total chars: {len(code):,} → {total_chars:,} in chunks")
        print(f"Chunks: {len(chunks)}")
        for i, c in enumerate(chunks):
            print(f"  [{i}] {c.kind:12s} name={c.name!r:30s} lines={c.start_line}-{c.end_line} chars={len(c.text):,}")
