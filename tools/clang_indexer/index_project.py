#!/usr/bin/env python3
"""
Clang-based cross-file dependency indexer for C++ training data preparation.

Uses libclang to parse C++ translation units with full semantic analysis,
building a cross-file call graph and generating bottom-up training documents.

Architecture:
  1. Walk project directory, find all .cpp/.cc/.cxx/.c files
  2. Parse each with libclang (optionally using compile_commands.json)
  3. Extract functions, classes, and cross-file call references
  4. Build global call graph
  5. Topological sort: HAL/system → drivers → subsystems → API
  6. Generate 16K-token training documents with bottom-up dependency ordering

Usage:
  # With compile_commands.json (best quality):
  python index_project.py --project-dir /path/to/project --output chunks.jsonl

  # Without build system (fallback mode):
  python index_project.py --project-dir /path/to/project --output chunks.jsonl --no-compile-db

  # Process multiple projects in parallel:
  python index_project.py --projects-list projects.txt --output chunks.jsonl --workers 48
"""

import argparse
import json
import os
import sys
import hashlib
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

try:
    from clang.cindex import (
        Index, TranslationUnit, CursorKind, Cursor,
        Config as ClangConfig
    )
except ImportError:
    print("ERROR: libclang Python bindings not found.", file=sys.stderr)
    print("Install with: pip install libclang", file=sys.stderr)
    print("Or: sudo apt install python3-clang", file=sys.stderr)
    sys.exit(1)


# C++ source file extensions
CPP_EXTENSIONS = {'.cpp', '.cc', '.cxx', '.c', '.c++', '.cp'}
HEADER_EXTENSIONS = {'.h', '.hpp', '.hxx', '.hh', '.h++', '.inl', '.inc'}

# System/stdlib function prefixes (skip for dependency tracking)
SYSTEM_PREFIXES = (
    'std::', 'boost::', '__builtin', '__', 'operator', 'printf', 'fprintf',
    'sprintf', 'snprintf', 'scanf', 'malloc', 'calloc', 'realloc', 'free',
    'memcpy', 'memmove', 'memset', 'memcmp', 'strlen', 'strcpy', 'strcat',
    'strcmp', 'fopen', 'fclose', 'fread', 'fwrite', 'exit', 'abort',
    'assert', 'pthread_', 'EXPECT_', 'ASSERT_', 'TEST',
)


class FunctionDef:
    """A function definition with its source location and call references."""
    __slots__ = ['name', 'qualified_name', 'file', 'line', 'text', 'callees',
                 'dep_level', 'is_definition']

    def __init__(self, name: str, qualified_name: str, file: str, line: int,
                 text: str, callees: list, is_definition: bool = True):
        self.name = name
        self.qualified_name = qualified_name
        self.file = file
        self.line = line
        self.text = text
        self.callees = callees  # list of qualified names called
        self.dep_level = 0
        self.is_definition = is_definition

    def to_dict(self) -> dict:
        """Serialize for multiprocessing IPC."""
        return {
            'name': self.name, 'qualified_name': self.qualified_name,
            'file': self.file, 'line': self.line, 'text': self.text,
            'callees': self.callees, 'is_definition': self.is_definition,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'FunctionDef':
        return cls(**d)


class ProjectIndex:
    """Cross-file function index for a single project."""

    def __init__(self):
        # qualified_name -> FunctionDef (definitions only)
        self.functions: dict[str, FunctionDef] = {}
        # file -> list of function qualified_names defined there
        self.file_functions: dict[str, list[str]] = defaultdict(list)
        # file -> preamble text (includes, typedefs, forward decls)
        self.file_preambles: dict[str, str] = {}
        # qualified_name -> list of qualified_names that call it
        self.callers: dict[str, list[str]] = defaultdict(list)

    def add_function(self, func: FunctionDef):
        """Add a function definition to the index."""
        key = func.qualified_name
        if key in self.functions and self.functions[key].is_definition:
            return  # don't overwrite definitions with declarations
        self.functions[key] = func
        if func.is_definition:
            self.file_functions[func.file].append(key)

    def build_reverse_edges(self):
        """Build caller -> callee reverse edges for dep level computation."""
        self.callers.clear()
        for qname, func in self.functions.items():
            for callee in func.callees:
                if callee in self.functions:
                    self.callers[callee].append(qname)

    def compute_dep_levels(self):
        """Compute dependency levels via BFS from leaves."""
        # Find leaves: functions with no callees in the index
        in_degree = {}
        for qname, func in self.functions.items():
            local_callees = [c for c in func.callees if c in self.functions and c != qname]
            in_degree[qname] = len(local_callees)

        queue = deque()
        for qname, deg in in_degree.items():
            if deg == 0:
                self.functions[qname].dep_level = 0
                queue.append(qname)

        self.build_reverse_edges()

        while queue:
            qname = queue.popleft()
            level = self.functions[qname].dep_level
            for caller_name in self.callers.get(qname, []):
                new_level = level + 1
                if new_level > self.functions[caller_name].dep_level:
                    self.functions[caller_name].dep_level = new_level
                in_degree[caller_name] -= 1
                if in_degree[caller_name] == 0:
                    queue.append(caller_name)

        # Handle cycles
        max_level = max((f.dep_level for f in self.functions.values()), default=0)
        for qname, deg in in_degree.items():
            if deg > 0:
                self.functions[qname].dep_level = max_level + 1

    def stats(self) -> dict:
        """Return index statistics."""
        return {
            'total_functions': len(self.functions),
            'total_files': len(self.file_functions),
            'definitions': sum(1 for f in self.functions.values() if f.is_definition),
            'max_dep_level': max((f.dep_level for f in self.functions.values()), default=0),
        }


def is_system_function(name: str) -> bool:
    """Check if a function name looks like a system/stdlib function."""
    return any(name.startswith(p) for p in SYSTEM_PREFIXES)


def get_function_text(cursor: Cursor, tu: TranslationUnit) -> str:
    """Extract the source text for a cursor's extent."""
    extent = cursor.extent
    start = extent.start
    end = extent.end

    try:
        filename = start.file.name if start.file else None
        if not filename or not os.path.exists(filename):
            return ""

        with open(filename, 'r', errors='replace') as f:
            content = f.read()

        # Convert offsets
        start_offset = start.offset
        end_offset = end.offset
        if start_offset < len(content) and end_offset <= len(content):
            return content[start_offset:end_offset]
    except Exception:
        pass
    return ""


def extract_callees(cursor: Cursor) -> list[str]:
    """Extract all function call references from a cursor's children."""
    callees = set()

    def walk(node: Cursor):
        if node.kind == CursorKind.CALL_EXPR:
            ref = node.referenced
            if ref and ref.spelling:
                # Get fully qualified name
                qname = get_qualified_name(ref)
                if qname and not is_system_function(qname):
                    callees.add(qname)
        for child in node.get_children():
            walk(child)

    walk(cursor)
    return list(callees)


def get_qualified_name(cursor: Cursor) -> str:
    """Get the fully qualified name of a cursor (namespace::class::func)."""
    parts = []
    c = cursor
    while c and c.kind != CursorKind.TRANSLATION_UNIT:
        if c.spelling:
            parts.append(c.spelling)
        c = c.semantic_parent
    parts.reverse()
    return '::'.join(parts)


def extract_preamble(tu: TranslationUnit, filename: str) -> str:
    """Extract #include directives and forward declarations from a file."""
    preamble_parts = []
    for cursor in tu.cursor.get_children():
        if cursor.location.file and cursor.location.file.name != filename:
            continue
        if cursor.kind in (CursorKind.INCLUSION_DIRECTIVE,
                           CursorKind.USING_DIRECTIVE,
                           CursorKind.USING_DECLARATION,
                           CursorKind.TYPEDEF_DECL,
                           CursorKind.TYPE_ALIAS_DECL,
                           CursorKind.NAMESPACE_ALIAS):
            text = get_function_text(cursor, tu)
            if text:
                preamble_parts.append(text)
    return '\n'.join(preamble_parts)


FUNCTION_KINDS = {
    CursorKind.FUNCTION_DECL,
    CursorKind.CXX_METHOD,
    CursorKind.FUNCTION_TEMPLATE,
    CursorKind.CONSTRUCTOR,
    CursorKind.DESTRUCTOR,
    CursorKind.CONVERSION_FUNCTION,
}

CONTAINER_KINDS = {
    CursorKind.NAMESPACE,
    CursorKind.CLASS_DECL,
    CursorKind.STRUCT_DECL,
    CursorKind.CLASS_TEMPLATE,
    CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
}


def parse_translation_unit(
    filepath: str,
    index: Index,
    compile_args: list[str],
    project_dir: str,
) -> list[FunctionDef]:
    """Parse a single C++ file and extract function definitions with callees."""
    functions = []

    try:
        tu = index.parse(
            filepath,
            args=compile_args,
            options=(
                TranslationUnit.PARSE_INCOMPLETE |
                TranslationUnit.PARSE_PRECOMPILED_PREAMBLE
            ),
        )
    except Exception as e:
        print(f"  WARN: Failed to parse {filepath}: {e}", file=sys.stderr)
        return functions

    rel_path = os.path.relpath(filepath, project_dir)

    def visit(cursor):
        """Recursively visit cursors, descending into namespaces and classes."""
        if not cursor.location.file:
            return
        if cursor.location.file.name != filepath:
            return

        if cursor.kind in FUNCTION_KINDS and cursor.is_definition():
            text = get_function_text(cursor, tu)
            if text and len(text) >= 20:
                callees = extract_callees(cursor)
                qname = get_qualified_name(cursor)
                functions.append(FunctionDef(
                    name=cursor.spelling,
                    qualified_name=qname,
                    file=rel_path,
                    line=cursor.location.line,
                    text=text,
                    callees=callees,
                    is_definition=True,
                ))

        elif cursor.kind in CONTAINER_KINDS:
            # Recurse into namespaces, classes, structs
            for child in cursor.get_children():
                visit(child)

    for cursor in tu.cursor.get_children():
        visit(cursor)

    return functions


def find_cpp_files(project_dir: str) -> list[str]:
    """Find all C/C++ source files in a directory."""
    files = []
    for root, _, filenames in os.walk(project_dir):
        # Skip common non-source directories
        skip_dirs = {'.git', 'build', 'cmake-build', '__pycache__', 'node_modules',
                     '.vs', '.vscode', 'third_party', 'external', 'deps', 'vendor'}
        if any(d in root.split(os.sep) for d in skip_dirs):
            continue
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in CPP_EXTENSIONS:
                filepath = os.path.join(root, fname)
                # Skip very large files
                try:
                    if os.path.getsize(filepath) > 500_000:
                        continue
                except OSError:
                    continue
                files.append(filepath)
    return files


def load_compile_commands(project_dir: str) -> Optional[dict]:
    """Load compile_commands.json if available."""
    cc_path = os.path.join(project_dir, 'compile_commands.json')
    if os.path.exists(cc_path):
        with open(cc_path) as f:
            commands = json.load(f)
        # Build file -> args map
        file_args = {}
        for entry in commands:
            filepath = entry.get('file', '')
            if not os.path.isabs(filepath):
                filepath = os.path.join(entry.get('directory', ''), filepath)
            filepath = os.path.normpath(filepath)
            cmd = entry.get('command', '') or ' '.join(entry.get('arguments', []))
            # Extract compiler flags (skip compiler name and file)
            args = cmd.split()
            flags = []
            skip_next = False
            for arg in args[1:]:  # skip compiler
                if skip_next:
                    skip_next = False
                    continue
                if arg in ('-o', '-MF', '-MQ', '-MT'):
                    skip_next = True
                    continue
                if arg.startswith('-o') or arg == filepath or arg.endswith('.o'):
                    continue
                if arg in ('-c', '-S'):
                    continue
                flags.append(arg)
            file_args[filepath] = flags
        return file_args
    return None


def get_default_compile_args(project_dir: str) -> list[str]:
    """Generate default compile args for projects without compile_commands.json."""
    include_dirs = set()
    # Find common include directories
    for candidate in ['include', 'src', 'lib', 'source', '.']:
        d = os.path.join(project_dir, candidate)
        if os.path.isdir(d):
            include_dirs.add(d)

    args = [
        '-std=c++17',
        '-fsyntax-only',
        '-Wno-everything',  # suppress all warnings for speed
        f'-I{project_dir}',
    ]
    for d in include_dirs:
        args.append(f'-I{d}')

    return args


def estimate_tokens(text: str) -> int:
    """Estimate token count (~4 bytes per token for code)."""
    return max(1, len(text) // 4)


def collect_transitive_deps(
    root_qname: str,
    index: ProjectIndex,
    max_depth: int = 5,
) -> list[str]:
    """BFS to collect transitive dependencies of a function."""
    visited = {root_qname}
    queue = deque([(root_qname, 0)])
    deps = []

    while queue:
        qname, depth = queue.popleft()
        if depth >= max_depth:
            continue
        func = index.functions.get(qname)
        if not func:
            continue
        for callee in func.callees:
            if callee not in visited and callee in index.functions:
                visited.add(callee)
                deps.append(callee)
                queue.append((callee, depth + 1))

    return deps


def build_training_documents(
    index: ProjectIndex,
    max_tokens: int = 16384,
    max_dep_depth: int = 5,
) -> list[str]:
    """Build training documents with bottom-up dependency ordering."""
    documents = []
    seen_hashes = set()

    index.compute_dep_levels()

    for qname, func in index.functions.items():
        if not func.is_definition:
            continue

        # Collect transitive deps
        dep_qnames = collect_transitive_deps(qname, index, max_dep_depth)

        # Sort by dep_level (leaves/most foundational first)
        dep_funcs = []
        for dq in dep_qnames:
            df = index.functions.get(dq)
            if df and df.is_definition and df.text:
                dep_funcs.append(df)
        dep_funcs.sort(key=lambda f: f.dep_level)

        # Build document
        parts = []

        # Add preamble from root function's file
        preamble = index.file_preambles.get(func.file, '')
        if preamble:
            parts.append(preamble)

        # Add deps (leaves first = most foundational)
        for df in dep_funcs:
            parts.append(df.text)

        # Add root function last
        parts.append(func.text)

        doc = '\n\n'.join(parts)
        tokens = estimate_tokens(doc)

        # Token budget management
        if tokens > max_tokens * 2 and dep_funcs:
            # Too big: trim deps from highest dep_level first
            while tokens > max_tokens * 2 and dep_funcs:
                dep_funcs.pop()  # remove highest-level dep
                parts = []
                if preamble:
                    parts.append(preamble)
                for df in dep_funcs:
                    parts.append(df.text)
                parts.append(func.text)
                doc = '\n\n'.join(parts)
                tokens = estimate_tokens(doc)

        if tokens < 20:
            continue

        # Deduplicate
        doc_hash = hashlib.md5(doc.encode()).hexdigest()
        if doc_hash in seen_hashes:
            continue
        seen_hashes.add(doc_hash)
        documents.append(doc)

    return documents


def _parse_file_batch(args_tuple):
    """Worker function for parallel parsing. Each worker creates its own Index."""
    filepaths, compile_db, default_args, project_dir = args_tuple
    clang_index = Index.create()
    results = []
    errors = 0
    for filepath in filepaths:
        if compile_db and filepath in compile_db:
            args = compile_db[filepath]
        else:
            args = default_args
        try:
            functions = parse_translation_unit(filepath, clang_index, args, project_dir)
            results.extend(f.to_dict() for f in functions)
        except Exception:
            errors += 1
    return results, len(filepaths), errors


def process_project(
    project_dir: str,
    max_tokens: int = 16384,
    max_dep_depth: int = 5,
    parse_workers: int = 1,
) -> list[str]:
    """Process a single project: parse all files, build index, generate docs."""
    project_dir = os.path.abspath(project_dir)
    project_name = os.path.basename(project_dir)

    print(f"\n--- Processing project: {project_name} ---", file=sys.stderr)

    # Find source files
    cpp_files = find_cpp_files(project_dir)
    print(f"  Found {len(cpp_files)} C/C++ source files", file=sys.stderr)

    if not cpp_files:
        return []

    # Load or generate compile commands
    compile_db = load_compile_commands(project_dir)
    default_args = get_default_compile_args(project_dir)

    # Parse all files and build index
    index_obj = ProjectIndex()

    # Use parallel parsing for large projects
    effective_workers = min(parse_workers, max(1, len(cpp_files) // 100))
    if effective_workers > 1 and len(cpp_files) > 200:
        print(f"  Using {effective_workers} parse workers", file=sys.stderr)
        chunk_size = max(50, len(cpp_files) // effective_workers)
        batches = []
        for i in range(0, len(cpp_files), chunk_size):
            batch = cpp_files[i:i + chunk_size]
            batches.append((batch, compile_db, default_args, project_dir))

        total_parsed = 0
        total_errors = 0
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            for func_dicts, parsed_count, error_count in executor.map(_parse_file_batch, batches):
                for d in func_dicts:
                    index_obj.add_function(FunctionDef.from_dict(d))
                total_parsed += parsed_count
                total_errors += error_count
                print(f"  Parsed {total_parsed}/{len(cpp_files)} files, "
                      f"{len(index_obj.functions)} functions", file=sys.stderr)

        print(f"  Parsed {total_parsed} files ({total_errors} errors), "
              f"{len(index_obj.functions)} functions indexed", file=sys.stderr)
    else:
        # Sequential for small projects
        clang_index = Index.create()
        parsed = 0
        errors = 0
        for filepath in cpp_files:
            if compile_db and filepath in compile_db:
                args = compile_db[filepath]
            else:
                args = default_args
            try:
                functions = parse_translation_unit(filepath, clang_index, args, project_dir)
                for func in functions:
                    index_obj.add_function(func)
                parsed += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  ERROR parsing {filepath}: {e}", file=sys.stderr)
            if parsed % 500 == 0 and parsed > 0:
                print(f"  Parsed {parsed}/{len(cpp_files)} files, "
                      f"{len(index_obj.functions)} functions", file=sys.stderr)
        print(f"  Parsed {parsed} files ({errors} errors), "
              f"{len(index_obj.functions)} functions indexed", file=sys.stderr)

    # Build training documents
    documents = build_training_documents(index_obj, max_tokens, max_dep_depth)
    print(f"  Generated {len(documents)} training documents", file=sys.stderr)

    stats = index_obj.stats()
    print(f"  Index stats: {stats}", file=sys.stderr)

    return documents


def main():
    parser = argparse.ArgumentParser(
        description='Clang-based cross-file C++ dependency indexer')
    parser.add_argument('--project-dir', type=str,
                        help='Single project directory to process')
    parser.add_argument('--projects-list', type=str,
                        help='File listing project directories (one per line)')
    parser.add_argument('--projects-dir', type=str,
                        help='Directory containing multiple project subdirectories')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSONL file path')
    parser.add_argument('--max-tokens', type=int, default=16384,
                        help='Max tokens per training document (default: 16384)')
    parser.add_argument('--max-dep-depth', type=int, default=5,
                        help='Max dependency resolution depth (default: 5)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers for multi-project mode')
    parser.add_argument('--parse-workers', type=int, default=8,
                        help='Number of parallel parse workers within each project (default: 8)')
    parser.add_argument('--libclang-path', type=str, default=None,
                        help='Path to libclang.so (auto-detected if not set)')

    args = parser.parse_args()

    # Set libclang path if specified
    if args.libclang_path:
        ClangConfig.set_library_file(args.libclang_path)

    # Collect project directories
    project_dirs = []
    if args.project_dir:
        project_dirs.append(args.project_dir)
    elif args.projects_list:
        with open(args.projects_list) as f:
            project_dirs = [line.strip() for line in f if line.strip()]
    elif args.projects_dir:
        for entry in sorted(os.listdir(args.projects_dir)):
            full = os.path.join(args.projects_dir, entry)
            if os.path.isdir(full):
                project_dirs.append(full)
    else:
        parser.error("Provide --project-dir, --projects-list, or --projects-dir")

    print(f"Processing {len(project_dirs)} project(s)", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)
    print(f"Max tokens: {args.max_tokens}", file=sys.stderr)

    total_docs = 0
    seen_hashes = set()

    with open(args.output, 'w') as out:
        if args.workers > 1 and len(project_dirs) > 1:
            # Multi-project parallel mode
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        process_project, pd, args.max_tokens, args.max_dep_depth,
                        args.parse_workers
                    ): pd
                    for pd in project_dirs
                }
                for future in as_completed(futures):
                    pd = futures[future]
                    try:
                        docs = future.result()
                        for doc in docs:
                            doc_hash = hashlib.md5(doc.encode()).hexdigest()
                            if doc_hash in seen_hashes:
                                continue
                            seen_hashes.add(doc_hash)
                            json.dump({'text': doc}, out)
                            out.write('\n')
                            total_docs += 1
                    except Exception as e:
                        print(f"ERROR processing {pd}: {e}", file=sys.stderr)
        else:
            # Sequential mode
            for pd in project_dirs:
                docs = process_project(pd, args.max_tokens, args.max_dep_depth,
                                       args.parse_workers)
                for doc in docs:
                    doc_hash = hashlib.md5(doc.encode()).hexdigest()
                    if doc_hash in seen_hashes:
                        continue
                    seen_hashes.add(doc_hash)
                    json.dump({'text': doc}, out)
                    out.write('\n')
                    total_docs += 1

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Total documents: {total_docs}", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == '__main__':
    main()
