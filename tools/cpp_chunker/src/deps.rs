//! Dependency-aware chunking: extract call graphs, topologically sort functions,
//! and build self-contained training documents with dependencies included.
//!
//! Key idea: for each C++ source file, extract which functions call which other
//! functions (within the same file). Then order them bottom-up:
//!   Level 0 = leaf functions (only call system/external/unknown functions)
//!   Level 1 = functions that call only Level 0 locals
//!   Level N = functions that call Level N-1 locals
//!
//! Training documents are built by including a function + all its transitive
//! local dependencies, ordered from leaves to root. This gives the model
//! self-contained code where every local function it sees is already defined.

use std::collections::{HashMap, HashSet, VecDeque};
use tree_sitter::{Node, Parser};

use crate::chunker::{chunk_file, Chunk, ChunkKind};

/// A function with its extracted call information.
#[derive(Debug, Clone)]
pub struct FunctionInfo {
    pub name: String,
    pub qualified_name: String,
    pub text: String,
    pub start_line: usize,
    pub end_line: usize,
    /// Names of functions called within this function's body.
    pub callees: Vec<String>,
    /// Whether this function only has system/external dependencies.
    pub is_leaf: bool,
    /// Dependency level (0 = leaf, higher = depends on lower levels).
    pub dep_level: u32,
}

/// Include directive information.
#[derive(Debug, Clone)]
pub struct IncludeInfo {
    pub path: String,
    pub is_system: bool,
}

/// Result of dependency analysis on a single file.
#[derive(Debug)]
pub struct FileDepInfo {
    pub includes: Vec<IncludeInfo>,
    pub preamble: String,
    pub functions: Vec<FunctionInfo>,
    pub classes: Vec<Chunk>,
    pub others: Vec<Chunk>,
    /// Functions ordered by dependency level (leaves first).
    pub topo_order: Vec<usize>,
}

/// Known C/C++ standard library namespaces and common prefixes.
const SYSTEM_PREFIXES: &[&str] = &[
    "std::", "boost::", "__", "_", "operator", "printf", "fprintf", "sprintf",
    "snprintf", "scanf", "sscanf", "malloc", "calloc", "realloc", "free",
    "memcpy", "memmove", "memset", "memcmp", "strlen", "strcpy", "strcat",
    "strcmp", "strncpy", "strncat", "strncmp", "strstr", "strchr", "strrchr",
    "atoi", "atof", "atol", "strtol", "strtod", "strtoul",
    "fopen", "fclose", "fread", "fwrite", "fgets", "fputs", "fseek", "ftell",
    "exit", "abort", "atexit", "system", "getenv",
    "assert", "static_assert",
    "sin", "cos", "tan", "sqrt", "pow", "exp", "log", "log2", "log10",
    "abs", "fabs", "ceil", "floor", "round",
    "rand", "srand", "time", "clock",
    "pthread_", "mutex_", "signal",
    "EXPECT_", "ASSERT_", "TEST", "TEST_F", "TYPED_TEST",
];

/// Check if a callee name looks like a system/external function.
fn is_system_call(name: &str) -> bool {
    // Empty or too short
    if name.is_empty() {
        return true;
    }
    // Starts with known system prefix
    for prefix in SYSTEM_PREFIXES {
        if name.starts_with(prefix) {
            return true;
        }
    }
    // Looks like a constructor/type cast (starts with uppercase, common pattern)
    // e.g., String(...), Vector3(...), shared_ptr<...>(...)
    // We can't be sure without type info, so don't classify these as system
    false
}

/// Extract function call names from a tree-sitter AST node (recursively).
fn extract_calls(node: &Node, source: &str, calls: &mut Vec<String>) {
    if node.kind() == "call_expression" {
        if let Some(func_node) = node.child_by_field_name("function") {
            let callee = extract_callee_name(&func_node, source);
            if !callee.is_empty() {
                calls.push(callee);
            }
        }
    }

    // Recurse into children
    let cursor = &mut node.walk();
    for child in node.children(cursor) {
        extract_calls(&child, source, calls);
    }
}

/// Extract the callee name from a call expression's function node.
fn extract_callee_name(node: &Node, source: &str) -> String {
    match node.kind() {
        "identifier" => source[node.byte_range()].to_string(),
        "qualified_identifier" | "template_function" | "dependent_name" => {
            source[node.byte_range()].to_string()
        }
        "field_expression" => {
            // obj.method() or obj->method() — extract the method name
            if let Some(field) = node.child_by_field_name("field") {
                return source[field.byte_range()].to_string();
            }
            source[node.byte_range()].to_string()
        }
        "scoped_identifier" => source[node.byte_range()].to_string(),
        "parenthesized_expression" => {
            // (*func_ptr)() — try to extract inner
            let cursor = &mut node.walk();
            for child in node.children(cursor) {
                let name = extract_callee_name(&child, source);
                if !name.is_empty() {
                    return name;
                }
            }
            String::new()
        }
        _ => {
            // For complex expressions, try to get any identifier
            source[node.byte_range()].to_string()
        }
    }
}

/// Extract #include directives from AST root.
fn extract_includes(root: &Node, source: &str) -> Vec<IncludeInfo> {
    let mut includes = Vec::new();
    let cursor = &mut root.walk();

    for child in root.children(cursor) {
        if child.kind() == "preproc_include" {
            if let Some(path_node) = child.child_by_field_name("path") {
                let path_text = source[path_node.byte_range()].to_string();
                let is_system = path_node.kind() == "system_lib_string";
                // Strip quotes/brackets
                let path = path_text
                    .trim_start_matches('<')
                    .trim_end_matches('>')
                    .trim_start_matches('"')
                    .trim_end_matches('"')
                    .to_string();
                includes.push(IncludeInfo { path, is_system });
            }
        }
    }

    includes
}

/// Normalize a callee name for matching against function definitions.
/// Strips namespace qualifiers, template args, etc.
fn normalize_name(name: &str) -> String {
    // Strip template arguments: foo<int> -> foo
    let without_templates = if let Some(pos) = name.find('<') {
        &name[..pos]
    } else {
        name
    };
    // Get the last component after ::
    let base = without_templates.rsplit("::").next().unwrap_or(without_templates);
    base.to_string()
}

/// Analyze dependencies within a single source file.
pub fn analyze_file(parser: &mut Parser, source: &str) -> FileDepInfo {
    let tree = match parser.parse(source, None) {
        Some(t) => t,
        None => {
            return FileDepInfo {
                includes: vec![],
                preamble: String::new(),
                functions: vec![],
                classes: vec![],
                others: vec![],
                topo_order: vec![],
            };
        }
    };

    let root = tree.root_node();

    // 1. Extract includes
    let includes = extract_includes(&root, source);

    // 2. Get semantic chunks from existing chunker
    let chunks = chunk_file(parser, source);

    // 3. Separate preamble, functions, classes, others
    let mut preamble_parts: Vec<&str> = Vec::new();
    let mut functions: Vec<FunctionInfo> = Vec::new();
    let mut classes: Vec<Chunk> = Vec::new();
    let mut others: Vec<Chunk> = Vec::new();

    for (ci, chunk) in chunks.iter().enumerate() {
        match chunk.kind {
            ChunkKind::Preamble => preamble_parts.push(&chunk.text),
            ChunkKind::Function => {
                // Skip extremely large functions to avoid pathological parsing
                if chunk.text.len() > 200_000 {
                    functions.push(FunctionInfo {
                        name: chunk.name.clone(),
                        qualified_name: chunk.name.clone(),
                        text: chunk.text.clone(),
                        start_line: chunk.start_line,
                        end_line: chunk.end_line,
                        callees: vec![],
                        is_leaf: true,
                        dep_level: 0,
                    });
                    continue;
                }
                // Re-parse this function to extract calls
                if std::env::var("DEBUG_CHUNKS").is_ok() {
                    let preview: String = chunk.text.chars().take(80).collect();
                    eprintln!("  [DEBUG] chunk {}: func '{}' len={} preview='{}'",
                        ci, chunk.name, chunk.text.len(), preview.replace('\n', "\\n"));
                }
                let func_calls = extract_calls_from_text(parser, &chunk.text);
                functions.push(FunctionInfo {
                    name: chunk.name.clone(),
                    qualified_name: chunk.name.clone(),
                    text: chunk.text.clone(),
                    start_line: chunk.start_line,
                    end_line: chunk.end_line,
                    callees: func_calls,
                    is_leaf: false,
                    dep_level: 0,
                });
            }
            ChunkKind::Class => classes.push(chunk.clone()),
            _ => others.push(chunk.clone()),
        }
    }

    let preamble = preamble_parts.join("\n\n");

    if std::env::var("DEBUG_CHUNKS").is_ok() {
        eprintln!("  [DEBUG] {} functions extracted, computing deps...", functions.len());
    }

    // 4. Build function name set (what's defined in this file)
    let local_names: HashSet<String> = functions
        .iter()
        .map(|f| normalize_name(&f.name))
        .filter(|n| !n.is_empty())
        .collect();

    // 5. Classify each function's callees as local vs external
    for func in &mut functions {
        let has_local_dep = func.callees.iter().any(|c| {
            let norm = normalize_name(c);
            local_names.contains(&norm) && norm != normalize_name(&func.name)
        });
        func.is_leaf = !has_local_dep;
    }

    // 6. Compute dependency levels via BFS
    if std::env::var("DEBUG_CHUNKS").is_ok() {
        for (i, f) in functions.iter().enumerate() {
            eprintln!("  [DEBUG] func[{}] '{}': {} callees: {:?}",
                i, f.name, f.callees.len(), &f.callees);
        }
        eprintln!("  [DEBUG] Computing dep levels...");
    }
    compute_dep_levels(&mut functions, &local_names);
    if std::env::var("DEBUG_CHUNKS").is_ok() {
        eprintln!("  [DEBUG] Dep levels computed.");
    }

    // 7. Build topological order (lowest dep_level first)
    let mut topo_order: Vec<usize> = (0..functions.len()).collect();
    topo_order.sort_by_key(|&i| functions[i].dep_level);

    FileDepInfo {
        includes,
        preamble,
        functions,
        classes,
        others,
        topo_order,
    }
}

/// Extract call expressions from a function text by re-parsing.
fn extract_calls_from_text(parser: &mut Parser, func_text: &str) -> Vec<String> {
    // Parse the function text directly — it's already a complete function definition
    let tree = match parser.parse(func_text, None) {
        Some(t) => t,
        None => return vec![],
    };

    let root = tree.root_node();
    let mut calls = Vec::new();
    extract_calls(&root, func_text, &mut calls);

    // Filter out __dummy names and deduplicate
    calls.retain(|c| !c.starts_with("__dummy"));
    calls.sort();
    calls.dedup();
    calls
}

/// Compute dependency levels for all functions using iterative BFS.
fn compute_dep_levels(functions: &mut Vec<FunctionInfo>, local_names: &HashSet<String>) {
    if functions.is_empty() {
        return;
    }

    // Build name -> index map
    let name_to_idx: HashMap<String, usize> = functions
        .iter()
        .enumerate()
        .filter(|(_, f)| !f.name.is_empty())
        .map(|(i, f)| (normalize_name(&f.name), i))
        .collect();

    // Build adjacency: func_idx -> set of local callee indices
    let edges: Vec<Vec<usize>> = functions
        .iter()
        .map(|f| {
            f.callees
                .iter()
                .filter_map(|c| {
                    let norm = normalize_name(c);
                    if !is_system_call(c) && local_names.contains(&norm) {
                        name_to_idx.get(&norm).copied()
                    } else {
                        None
                    }
                })
                .filter(|&idx| idx != name_to_idx.get(&normalize_name(&f.name)).copied().unwrap_or(usize::MAX))
                .collect()
        })
        .collect();

    // BFS from leaves (nodes with no local callees)
    let mut in_degree: Vec<usize> = vec![0; functions.len()];
    // in_degree[i] = number of local functions that i calls (not reverse!)
    // We want level 0 = functions that call NO local functions
    for (i, deps) in edges.iter().enumerate() {
        // Remove self-references
        let unique_deps: HashSet<usize> = deps.iter().copied().collect();
        in_degree[i] = unique_deps.len();
    }

    let mut queue: VecDeque<usize> = VecDeque::new();
    for i in 0..functions.len() {
        if in_degree[i] == 0 {
            functions[i].dep_level = 0;
            functions[i].is_leaf = true;
            queue.push_back(i);
        }
    }

    // Reverse edges: who calls me?
    let mut reverse_edges: Vec<Vec<usize>> = vec![vec![]; functions.len()];
    for (caller, callees) in edges.iter().enumerate() {
        for &callee in callees {
            reverse_edges[callee].push(caller);
        }
    }

    // Process in BFS order
    while let Some(idx) = queue.pop_front() {
        let level = functions[idx].dep_level;
        for &caller in &reverse_edges[idx] {
            // The caller's level must be at least max(callee levels) + 1
            let new_level = level + 1;
            if new_level > functions[caller].dep_level {
                functions[caller].dep_level = new_level;
            }
            in_degree[caller] = in_degree[caller].saturating_sub(1);
            if in_degree[caller] == 0 {
                queue.push_back(caller);
            }
        }
    }

    // Handle cycles (mutual recursion): assign max level seen + 1
    let max_level = functions.iter().map(|f| f.dep_level).max().unwrap_or(0);
    for (i, func) in functions.iter_mut().enumerate() {
        if in_degree[i] > 0 {
            // Part of a cycle
            func.dep_level = max_level + 1;
        }
    }
}

/// Build self-contained training documents from dependency analysis.
/// Each document includes a function and all its transitive local dependencies,
/// ordered from leaves (level 0) to the root function.
pub fn build_dep_aware_documents(
    dep_info: &FileDepInfo,
    max_tokens: usize,
) -> Vec<String> {
    let estimate_tokens = |text: &str| -> usize { std::cmp::max(1, text.len() / 4) };

    if dep_info.functions.is_empty() && dep_info.classes.is_empty() {
        if !dep_info.preamble.is_empty() && estimate_tokens(&dep_info.preamble) >= 20 {
            return vec![dep_info.preamble.clone()];
        }
        return vec![];
    }

    let mut documents: Vec<String> = Vec::new();

    // Build name -> index map
    let name_to_idx: HashMap<String, usize> = dep_info
        .functions
        .iter()
        .enumerate()
        .filter(|(_, f)| !f.name.is_empty())
        .map(|(i, f)| (normalize_name(&f.name), i))
        .collect();

    // Check if full file fits in max_tokens
    let all_text = build_full_file_text(dep_info);
    if estimate_tokens(&all_text) <= max_tokens && !all_text.is_empty() {
        return vec![all_text];
    }

    // For each function, collect its transitive dependencies
    for (root_idx, func) in dep_info.functions.iter().enumerate() {
        let mut dep_chain = collect_transitive_deps(
            root_idx,
            &dep_info.functions,
            &name_to_idx,
        );

        // Sort by dep_level (leaves first, root last)
        dep_chain.sort_by_key(|&idx| dep_info.functions[idx].dep_level);

        // Build document: preamble + deps (leaves first) + root function
        let mut parts: Vec<&str> = Vec::new();
        if !dep_info.preamble.is_empty() {
            parts.push(&dep_info.preamble);
        }

        // Add comments showing dependency structure
        for &idx in &dep_chain {
            parts.push(&dep_info.functions[idx].text);
        }

        let doc = parts.join("\n\n");

        if estimate_tokens(&doc) <= max_tokens * 2 {
            if doc.len() >= 50 {
                documents.push(doc);
            }
        } else {
            // Too big — just emit the root function with preamble
            let simple = if !dep_info.preamble.is_empty() {
                format!("{}\n\n{}", dep_info.preamble, func.text)
            } else {
                func.text.clone()
            };
            if simple.len() >= 50 {
                documents.push(simple);
            }
        }
    }

    // Emit classes with preamble
    for cls in &dep_info.classes {
        let doc = if !dep_info.preamble.is_empty() {
            format!("{}\n\n{}", dep_info.preamble, cls.text)
        } else {
            cls.text.clone()
        };
        if estimate_tokens(&doc) <= max_tokens * 2 && doc.len() >= 50 {
            documents.push(doc);
        } else if doc.len() >= 50 {
            documents.push(cls.text.clone());
        }
    }

    // Emit level-ordered document: all functions sorted by level
    if dep_info.functions.len() >= 2 {
        let mut level_parts: Vec<&str> = Vec::new();
        if !dep_info.preamble.is_empty() {
            level_parts.push(&dep_info.preamble);
        }
        for &idx in &dep_info.topo_order {
            level_parts.push(&dep_info.functions[idx].text);
        }
        let level_doc = level_parts.join("\n\n");
        if estimate_tokens(&level_doc) <= max_tokens && level_doc.len() >= 50 {
            documents.push(level_doc);
        }
    }

    documents
}

/// Collect transitive local dependencies for a function (BFS).
fn collect_transitive_deps(
    root_idx: usize,
    functions: &[FunctionInfo],
    name_to_idx: &HashMap<String, usize>,
) -> Vec<usize> {
    let mut visited: HashSet<usize> = HashSet::new();
    let mut queue: VecDeque<usize> = VecDeque::new();
    queue.push_back(root_idx);
    visited.insert(root_idx);

    while let Some(idx) = queue.pop_front() {
        for callee_name in &functions[idx].callees {
            let norm = normalize_name(callee_name);
            if let Some(&callee_idx) = name_to_idx.get(&norm) {
                if callee_idx != root_idx && visited.insert(callee_idx) {
                    queue.push_back(callee_idx);
                }
            }
        }
    }

    visited.into_iter().collect()
}

fn build_full_file_text(dep_info: &FileDepInfo) -> String {
    let mut parts: Vec<&str> = Vec::new();
    if !dep_info.preamble.is_empty() {
        parts.push(&dep_info.preamble);
    }
    // Functions in topo order
    for &idx in &dep_info.topo_order {
        parts.push(&dep_info.functions[idx].text);
    }
    for cls in &dep_info.classes {
        parts.push(&cls.text);
    }
    for other in &dep_info.others {
        if other.text.trim().len() >= 20 {
            parts.push(&other.text);
        }
    }
    parts.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_parser() -> Parser {
        let mut parser = Parser::new();
        let lang = tree_sitter_cpp::LANGUAGE;
        parser.set_language(&lang.into()).unwrap();
        parser
    }

    #[test]
    fn test_leaf_function_detection() {
        let mut parser = make_parser();
        let source = r#"
#include <iostream>

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    int result = 0;
    for (int i = 0; i < b; i++) {
        result = add(result, a);
    }
    return result;
}

int power(int base, int exp) {
    int result = 1;
    for (int i = 0; i < exp; i++) {
        result = multiply(result, base);
    }
    return result;
}
"#;
        let dep_info = analyze_file(&mut parser, source);

        assert_eq!(dep_info.functions.len(), 3, "Should find 3 functions");

        // add is a leaf (no local calls)
        let add_fn = dep_info.functions.iter().find(|f| f.name == "add").unwrap();
        assert_eq!(add_fn.dep_level, 0, "add should be level 0 (leaf)");

        // multiply calls add (level 1)
        let mult_fn = dep_info.functions.iter().find(|f| f.name == "multiply").unwrap();
        assert_eq!(mult_fn.dep_level, 1, "multiply should be level 1");

        // power calls multiply (level 2)
        let pow_fn = dep_info.functions.iter().find(|f| f.name == "power").unwrap();
        assert_eq!(pow_fn.dep_level, 2, "power should be level 2");

        // Topo order should be: add, multiply, power
        let names: Vec<&str> = dep_info
            .topo_order
            .iter()
            .map(|&i| dep_info.functions[i].name.as_str())
            .collect();
        assert_eq!(names, vec!["add", "multiply", "power"]);
    }

    #[test]
    fn test_system_calls_are_external() {
        assert!(is_system_call("std::cout"));
        assert!(is_system_call("std::vector"));
        assert!(is_system_call("printf"));
        assert!(is_system_call("malloc"));
        assert!(is_system_call("boost::asio"));
        assert!(!is_system_call("myFunction"));
        assert!(!is_system_call("calculate"));
    }

    #[test]
    fn test_dep_aware_documents() {
        let mut parser = make_parser();
        let source = r#"
#include <cstdio>

int helper(int x) {
    return x * 2;
}

int process(int x) {
    return helper(x) + 1;
}

int main() {
    printf("%d\n", process(5));
    return 0;
}
"#;
        let dep_info = analyze_file(&mut parser, source);
        let docs = build_dep_aware_documents(&dep_info, 4096);

        // Should have documents — at least one per function
        assert!(!docs.is_empty(), "Should produce documents");

        // The document for 'process' should include 'helper' before it
        let process_doc = docs.iter().find(|d| d.contains("process") && d.contains("helper"));
        assert!(
            process_doc.is_some(),
            "Should have a document with process and its dependency helper"
        );

        // In the process doc, helper should appear before process
        if let Some(doc) = process_doc {
            let helper_pos = doc.find("int helper").unwrap_or(usize::MAX);
            let process_pos = doc.find("int process").unwrap_or(0);
            assert!(
                helper_pos < process_pos,
                "helper should appear before process (bottom-up ordering)"
            );
        }
    }

    #[test]
    fn test_include_extraction() {
        let mut parser = make_parser();
        let source = r#"
#include <iostream>
#include <vector>
#include "myheader.h"
#include "utils/helpers.h"

int main() { return 0; }
"#;
        let dep_info = analyze_file(&mut parser, source);

        assert_eq!(dep_info.includes.len(), 4);
        assert!(dep_info.includes[0].is_system);
        assert!(dep_info.includes[1].is_system);
        assert!(!dep_info.includes[2].is_system);
        assert!(!dep_info.includes[3].is_system);
        assert_eq!(dep_info.includes[0].path, "iostream");
        assert_eq!(dep_info.includes[2].path, "myheader.h");
    }

    #[test]
    fn test_mutual_recursion_handled() {
        let mut parser = make_parser();
        let source = r#"
bool is_even(int n);
bool is_odd(int n);

bool is_even(int n) {
    if (n == 0) return true;
    return is_odd(n - 1);
}

bool is_odd(int n) {
    if (n == 0) return false;
    return is_even(n - 1);
}
"#;
        let dep_info = analyze_file(&mut parser, source);

        // Both should be in a cycle — assigned max_level + 1
        let even_fn = dep_info.functions.iter().find(|f| f.name == "is_even");
        let odd_fn = dep_info.functions.iter().find(|f| f.name == "is_odd");

        assert!(even_fn.is_some(), "Should find is_even");
        assert!(odd_fn.is_some(), "Should find is_odd");

        // Both in cycle = same level
        assert_eq!(
            even_fn.unwrap().dep_level,
            odd_fn.unwrap().dep_level,
            "Mutually recursive functions should have same level"
        );
    }
}
