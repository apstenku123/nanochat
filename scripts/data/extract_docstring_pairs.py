"""
Extract (docstring, function_body) pairs from C++ corpus for structured FIM training.

Finds patterns like:
    /* docstring comment */
    return_type function_name(params) {
        function_body
    }

And extracts them as structured FIM training examples.

Usage:
    python -m scripts.data.extract_docstring_pairs --input data/cpp_combined_10b_v3.jsonl --output data/docstring_pairs.jsonl
"""
import os
import re
import json
import argparse
from pathlib import Path
from typing import Optional


# Regex patterns for C++ function extraction
BLOCK_COMMENT = r'/\*[\s\S]*?\*/'
LINE_COMMENTS = r'(?://[^\n]*\n)+'

# Function signature pattern (simplified but effective)
FUNC_SIGNATURE = r'''
    (?:template\s*<[^>]+>\s*)?                    # optional template
    (?:static\s+|inline\s+|virtual\s+|explicit\s+|constexpr\s+)*  # modifiers
    [\w:]+(?:\s*[*&]+\s*|\s+)                      # return type
    (\w+)\s*                                       # function name
    \([^)]*\)\s*                                   # parameters
    (?:const\s*)?(?:noexcept\s*)?(?:override\s*)?  # qualifiers
    \{                                             # opening brace
'''

# Combined pattern: comment followed by function
DOCSTRING_FUNC_PATTERN = re.compile(
    rf'({BLOCK_COMMENT}|{LINE_COMMENTS})\s*'  # docstring (block or line comments)
    rf'({FUNC_SIGNATURE})',
    re.VERBOSE | re.MULTILINE
)


def extract_function_body(code: str, start_pos: int) -> Optional[tuple[str, int]]:
    """
    Extract function body starting from opening brace position.
    Returns (body_content, end_position) or None if parsing fails.
    """
    depth = 0
    in_string = False
    in_char = False
    escape = False
    body_start = None

    i = start_pos
    while i < len(code):
        c = code[i]

        if escape:
            escape = False
            i += 1
            continue

        if c == '\\':
            escape = True
            i += 1
            continue

        if c == '"' and not in_char:
            in_string = not in_string
        elif c == "'" and not in_string:
            in_char = not in_char

        if not in_string and not in_char:
            if c == '{':
                if depth == 0:
                    body_start = i + 1
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    # Found the closing brace
                    body = code[body_start:i].strip()
                    return body, i

        i += 1

    return None


def is_quality_docstring(comment: str) -> bool:
    """Check if comment looks like a real docstring (not just copyright/license)."""
    comment_lower = comment.lower()

    # Skip license/copyright headers
    skip_keywords = ['copyright', 'license', 'all rights reserved', 'spdx',
                     'permission is hereby granted', 'redistribution']
    if any(kw in comment_lower for kw in skip_keywords):
        return False

    # Must have some descriptive content
    if len(comment.strip()) < 20:
        return False

    # Should describe behavior/functionality
    good_keywords = ['return', 'param', 'check', 'compute', 'calculate', 'find',
                     'get', 'set', 'create', 'delete', 'update', 'process',
                     'convert', 'parse', 'validate', 'initialize', 'handle']
    if any(kw in comment_lower for kw in good_keywords):
        return True

    # Or have reasonable length
    return len(comment.strip()) > 50


def is_quality_function(signature: str, body: str) -> bool:
    """Check if function is worth including in training data."""
    # Skip very short bodies (likely declarations or trivial)
    if len(body.strip()) < 20:
        return False

    # Skip very long bodies (too complex, likely noise)
    if len(body) > 5000:
        return False

    # Skip getter/setter trivia
    trivial_patterns = [
        r'^return\s+\w+_;?\s*$',  # return member_;
        r'^return\s+this->\w+;?\s*$',  # return this->x;
        r'^\w+_\s*=\s*\w+;?\s*$',  # member_ = value;
    ]
    body_stripped = body.strip()
    for pat in trivial_patterns:
        if re.match(pat, body_stripped, re.IGNORECASE):
            return False

    return True


def clean_docstring(comment: str) -> str:
    """Clean up docstring comment."""
    # Remove /* */ markers
    if comment.startswith('/*'):
        comment = comment[2:]
    if comment.endswith('*/'):
        comment = comment[:-2]

    # Remove // prefixes from line comments
    lines = comment.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith('//'):
            line = line[2:].strip()
        cleaned_lines.append(line)

    # Rejoin and clean up
    result = '\n'.join(cleaned_lines).strip()

    # Remove excessive whitespace
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result


def extract_pairs_from_file(code: str, path: str) -> list[dict]:
    """Extract all docstring-function pairs from a single file."""
    pairs = []

    # Find all potential docstring + function patterns
    for match in DOCSTRING_FUNC_PATTERN.finditer(code):
        comment = match.group(1)
        func_match_end = match.end()

        # Find the opening brace position
        brace_pos = code.rfind('{', match.start(), func_match_end)
        if brace_pos == -1:
            continue

        # Extract function body
        result = extract_function_body(code, brace_pos)
        if result is None:
            continue

        body, end_pos = result

        # Quality checks
        if not is_quality_docstring(comment):
            continue
        if not is_quality_function(match.group(0), body):
            continue

        # Extract the full signature (from after comment to opening brace)
        signature_start = match.end(1)  # End of comment
        signature = code[signature_start:brace_pos].strip()

        # Clean up
        docstring = clean_docstring(comment)

        pairs.append({
            'docstring': docstring,
            'signature': signature,
            'body': body,
            'path': path,
        })

    return pairs


def extract_pairs_simple(code: str, path: str) -> list[dict]:
    """
    Simpler extraction: find block comments followed by function-like patterns.
    More robust than regex-based approach.
    """
    pairs = []

    # Find all block comments
    block_pattern = re.compile(r'/\*\*?([\s\S]*?)\*/')

    for match in block_pattern.finditer(code):
        comment_content = match.group(1).strip()
        comment_end = match.end()

        # Skip if not a good docstring
        if not is_quality_docstring('/*' + comment_content + '*/'):
            continue

        # Look for function signature after comment (within 200 chars)
        after_comment = code[comment_end:comment_end + 500]

        # Find opening brace
        brace_match = re.search(r'(\w[\w\s:*&<>,]*\([^)]*\)\s*(?:const\s*)?(?:noexcept\s*)?)\s*\{', after_comment)
        if not brace_match:
            continue

        signature = brace_match.group(1).strip()

        # Extract body
        brace_pos = comment_end + brace_match.end() - 1
        result = extract_function_body(code, brace_pos)
        if result is None:
            continue

        body, _ = result

        if not is_quality_function(signature, body):
            continue

        pairs.append({
            'docstring': comment_content.strip(),
            'signature': signature,
            'body': body,
            'path': path,
        })

    return pairs


def process_corpus(input_path: str, output_path: str, max_pairs: int = -1):
    """Process entire corpus and extract docstring pairs."""
    total_pairs = 0
    total_files = 0

    with open(output_path, 'w') as out_f:
        with open(input_path, 'r') as in_f:
            for line in in_f:
                if not line.strip():
                    continue

                record = json.loads(line)
                code = record.get('text', '')
                path = record.get('path', '')

                # Extract pairs using simple method
                pairs = extract_pairs_simple(code, path)

                for pair in pairs:
                    out_f.write(json.dumps(pair, ensure_ascii=False) + '\n')
                    total_pairs += 1

                    if max_pairs > 0 and total_pairs >= max_pairs:
                        print(f"\nReached max pairs limit: {max_pairs}")
                        print(f"Processed {total_files:,} files, extracted {total_pairs:,} pairs")
                        return

                total_files += 1
                if total_files % 50000 == 0:
                    print(f"  Processed {total_files:,} files, {total_pairs:,} pairs so far")

    print(f"\nDone: {total_files:,} files, {total_pairs:,} docstring-function pairs")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract docstring-function pairs from C++ corpus")
    parser.add_argument('--input', type=str, required=True, help='Input JSONL corpus')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file')
    parser.add_argument('--max-pairs', type=int, default=-1, help='Max pairs to extract (-1 = unlimited)')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / args.input
    output_path = project_root / args.output

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print()

    process_corpus(str(input_path), str(output_path), args.max_pairs)


if __name__ == '__main__':
    main()
