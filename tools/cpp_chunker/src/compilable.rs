//! Compilable chunk generation.
//!
//! Produces training documents structured as near-compilable C++ units:
//!   1. Preamble (#includes, using directives, forward declarations)
//!   2. Type definitions (structs, classes, enums) in dependency order
//!   3. Functions in bottom-up call order (leaf callees first, root callers last)
//!
//! This ensures the model sees every definition before its usage.

use std::collections::{HashMap, HashSet};

use crate::chunker::Chunk;
use crate::deps::{self, FileDepInfo, normalize_name};
use crate::global_index::GlobalIndex;

fn estimate_tokens(text: &str) -> usize {
    std::cmp::max(1, text.len() / 4)
}

/// Check if `word` appears as a whole word in `text` (not as substring of identifier).
fn contains_word(text: &str, word: &str) -> bool {
    if word.len() < 2 || word.len() > text.len() {
        return false;
    }
    let bytes = text.as_bytes();
    let wlen = word.len();
    let mut start = 0;
    while start + wlen <= bytes.len() {
        match text[start..].find(word) {
            Some(pos) => {
                let abs = start + pos;
                let before_ok = abs == 0 || {
                    let c = bytes[abs - 1];
                    !c.is_ascii_alphanumeric() && c != b'_'
                };
                let after_pos = abs + wlen;
                let after_ok = after_pos >= bytes.len() || {
                    let c = bytes[after_pos];
                    !c.is_ascii_alphanumeric() && c != b'_'
                };
                if before_ok && after_ok {
                    return true;
                }
                start = abs + 1;
            }
            None => break,
        }
    }
    false
}

/// Topologically sort types by inter-type references (dependency-free types first).
fn topo_sort_types(types: &[&Chunk], type_name_idx: &HashMap<&str, usize>) -> Vec<usize> {
    let n = types.len();
    if n == 0 {
        return vec![];
    }

    let mut in_degree = vec![0usize; n];
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (i, ty) in types.iter().enumerate() {
        for (&name, &j) in type_name_idx {
            if j != i && contains_word(&ty.text, name) {
                in_degree[i] += 1;
                dependents[j].push(i);
            }
        }
    }

    // Kahn's algorithm
    let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut order = Vec::with_capacity(n);

    while let Some(idx) = queue.pop() {
        order.push(idx);
        for &dep in &dependents[idx] {
            in_degree[dep] -= 1;
            if in_degree[dep] == 0 {
                queue.push(dep);
            }
        }
    }

    // Remaining (cycles) go at end
    for i in 0..n {
        if !order.contains(&i) {
            order.push(i);
        }
    }

    order
}

/// Collect type indices transitively needed by a set of functions.
fn collect_needed_types(
    func_indices: &[usize],
    dep_info: &FileDepInfo,
    type_name_idx: &HashMap<&str, usize>,
    extra_texts: &[&str],
) -> HashSet<usize> {
    let mut needed: HashSet<usize> = HashSet::new();

    // Types referenced by local functions
    for &fi in func_indices {
        for (&name, &ti) in type_name_idx {
            if contains_word(&dep_info.functions[fi].text, name) {
                needed.insert(ti);
            }
        }
    }

    // Types referenced by cross-file functions
    for text in extra_texts {
        for (&name, &ti) in type_name_idx {
            if contains_word(text, name) {
                needed.insert(ti);
            }
        }
    }

    // Transitively resolve type-on-type dependencies
    loop {
        let before = needed.len();
        let current: Vec<usize> = needed.iter().copied().collect();
        for &idx in &current {
            for (&name, &dep_idx) in type_name_idx {
                if dep_idx != idx
                    && !needed.contains(&dep_idx)
                    && contains_word(&dep_info.classes[idx].text, name)
                {
                    needed.insert(dep_idx);
                }
            }
        }
        if needed.len() == before {
            break;
        }
    }

    needed
}

/// Build compilable training documents from file dependency analysis.
///
/// Document structure: preamble → type definitions (topo order) → functions (bottom-up).
/// Falls back to progressively simpler structures when content exceeds token budget.
pub fn build_compilable_documents(
    dep_info: &FileDepInfo,
    global: &GlobalIndex,
    max_tokens: usize,
    max_cross_depth: usize,
) -> Vec<String> {
    if dep_info.functions.is_empty() && dep_info.classes.is_empty() {
        if !dep_info.preamble.is_empty() && estimate_tokens(&dep_info.preamble) >= 20 {
            return vec![dep_info.preamble.clone()];
        }
        return vec![];
    }

    // Type name -> index in dep_info.classes
    let type_name_idx: HashMap<&str, usize> = dep_info
        .classes
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.name.is_empty())
        .map(|(i, c)| (c.name.as_str(), i))
        .collect();

    // Topo-sorted type indices
    let type_refs: Vec<&Chunk> = dep_info.classes.iter().collect();
    let type_topo = topo_sort_types(&type_refs, &type_name_idx);

    // Local function name maps
    let local_names: HashSet<String> = dep_info
        .functions
        .iter()
        .map(|f| normalize_name(&f.name))
        .filter(|n| !n.is_empty())
        .collect();

    let name_to_idx: HashMap<String, usize> = dep_info
        .functions
        .iter()
        .enumerate()
        .filter(|(_, f)| !f.name.is_empty())
        .map(|(i, f)| (normalize_name(&f.name), i))
        .collect();

    // --- Strategy 1: Whole file as one compilable document ---
    {
        let mut parts: Vec<&str> = Vec::new();
        if !dep_info.preamble.is_empty() {
            parts.push(&dep_info.preamble);
        }
        for &ti in &type_topo {
            parts.push(&dep_info.classes[ti].text);
        }
        for &fi in &dep_info.topo_order {
            parts.push(&dep_info.functions[fi].text);
        }
        for other in &dep_info.others {
            if other.text.trim().len() >= 20 {
                parts.push(&other.text);
            }
        }
        let full = parts.join("\n\n");
        if estimate_tokens(&full) <= max_tokens && full.len() >= 50 {
            return vec![full];
        }
    }

    // --- Strategy 2: Per-function compilable documents ---
    let mut documents: Vec<String> = Vec::new();

    for (root_idx, func) in dep_info.functions.iter().enumerate() {
        // Collect transitive local deps (includes root_idx)
        let local_deps = deps::collect_transitive_deps(
            root_idx,
            &dep_info.functions,
            &name_to_idx,
        );

        // Gather all callees for cross-file resolution
        let mut all_callees: Vec<String> = Vec::new();
        for &di in &local_deps {
            all_callees.extend(dep_info.functions[di].callees.iter().cloned());
        }
        all_callees.sort();
        all_callees.dedup();

        // Resolve cross-file function deps
        let cross_deps = deps::resolve_cross_file_deps(
            &all_callees,
            &local_names,
            global,
            max_cross_depth,
        );
        let cross_texts: Vec<&str> = cross_deps.iter().map(|(_, f)| f.text.as_str()).collect();

        // Find needed local types
        let needed_types = collect_needed_types(
            &local_deps,
            dep_info,
            &type_name_idx,
            &cross_texts,
        );

        // Sort cross-file deps: deepest (most foundational) first
        let mut sorted_cross: Vec<_> = cross_deps.iter().collect();
        sorted_cross.sort_by(|a, b| b.0.cmp(&a.0));

        // Sort local deps: bottom-up (lowest dep_level first)
        let mut sorted_local: Vec<usize> = local_deps;
        sorted_local.sort_by_key(|&i| dep_info.functions[i].dep_level);

        // --- Try full document with cross-file deps ---
        {
            let mut parts: Vec<&str> = Vec::new();
            if !dep_info.preamble.is_empty() {
                parts.push(&dep_info.preamble);
            }
            for &ti in &type_topo {
                if needed_types.contains(&ti) {
                    parts.push(&dep_info.classes[ti].text);
                }
            }
            for (_, cf) in &sorted_cross {
                parts.push(&cf.text);
            }
            for &i in &sorted_local {
                parts.push(&dep_info.functions[i].text);
            }
            let doc = parts.join("\n\n");
            if estimate_tokens(&doc) <= max_tokens && doc.len() >= 50 {
                documents.push(doc);
                continue;
            }
        }

        // --- Try without cross-file deps ---
        if !sorted_cross.is_empty() {
            let mut parts: Vec<&str> = Vec::new();
            if !dep_info.preamble.is_empty() {
                parts.push(&dep_info.preamble);
            }
            for &ti in &type_topo {
                if needed_types.contains(&ti) {
                    parts.push(&dep_info.classes[ti].text);
                }
            }
            for &i in &sorted_local {
                parts.push(&dep_info.functions[i].text);
            }
            let doc = parts.join("\n\n");
            if estimate_tokens(&doc) <= max_tokens && doc.len() >= 50 {
                documents.push(doc);
                continue;
            }
        }

        // --- Fallback: preamble + direct types + root function ---
        let mut parts: Vec<&str> = Vec::new();
        if !dep_info.preamble.is_empty() {
            parts.push(&dep_info.preamble);
        }
        for (&name, &ti) in &type_name_idx {
            if contains_word(&func.text, name) {
                parts.push(&dep_info.classes[ti].text);
            }
        }
        parts.push(&func.text);
        let doc = parts.join("\n\n");
        if doc.len() >= 50 {
            documents.push(doc);
        }
    }

    // Emit standalone type documents (classes with preamble)
    for cls in &dep_info.classes {
        let doc = if !dep_info.preamble.is_empty() {
            format!("{}\n\n{}", dep_info.preamble, cls.text)
        } else {
            cls.text.clone()
        };
        if estimate_tokens(&doc) <= max_tokens && doc.len() >= 50 {
            documents.push(doc);
        }
    }

    documents
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deps::analyze_file;
    use tree_sitter::Parser;

    fn make_parser() -> Parser {
        let mut parser = Parser::new();
        let lang = tree_sitter_cpp::LANGUAGE;
        parser.set_language(&lang.into()).unwrap();
        parser
    }

    #[test]
    fn test_contains_word() {
        assert!(contains_word("void foo(Point p)", "Point"));
        assert!(contains_word("Point p;", "Point"));
        assert!(!contains_word("PointCloud p;", "Point"));
        assert!(!contains_word("checkpoint p;", "Point"));
        assert!(contains_word("const Point& ref", "Point"));
        assert!(!contains_word("", "Point"));
    }

    #[test]
    fn test_compilable_type_before_function() {
        let mut parser = make_parser();
        let source = r#"
#include <cstdio>

struct Point {
    double x, y;
};

double distance(const Point& a, const Point& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return dx * dx + dy * dy;
}

void print_point(const Point& p) {
    printf("(%f, %f)\n", p.x, p.y);
}
"#;
        let dep_info = analyze_file(&mut parser, source);
        let global = GlobalIndex::new();
        let docs = build_compilable_documents(&dep_info, &global, 4096, 3);

        assert!(!docs.is_empty(), "Should produce documents");

        // In every document containing Point and a function,
        // Point struct should appear before the function
        for doc in &docs {
            if doc.contains("struct Point") && doc.contains("double distance") {
                let type_pos = doc.find("struct Point").unwrap();
                let func_pos = doc.find("double distance").unwrap();
                assert!(
                    type_pos < func_pos,
                    "Type definition should appear before function"
                );
            }
        }
    }

    #[test]
    fn test_compilable_bottom_up_order() {
        let mut parser = make_parser();
        let source = r#"
#include <cstdio>

int leaf(int x) {
    return x * 2;
}

int middle(int x) {
    return leaf(x) + 1;
}

int root(int x) {
    return middle(x) * 3;
}
"#;
        let dep_info = analyze_file(&mut parser, source);
        let global = GlobalIndex::new();
        let docs = build_compilable_documents(&dep_info, &global, 4096, 3);

        assert!(!docs.is_empty());

        // Find the document containing all three functions
        let full_doc = docs.iter().find(|d| {
            d.contains("int leaf") && d.contains("int middle") && d.contains("int root")
        });
        assert!(full_doc.is_some(), "Should have doc with all 3 functions");

        let doc = full_doc.unwrap();
        let leaf_pos = doc.find("int leaf").unwrap();
        let middle_pos = doc.find("int middle").unwrap();
        let root_pos = doc.find("int root").unwrap();

        assert!(leaf_pos < middle_pos, "leaf should come before middle");
        assert!(middle_pos < root_pos, "middle should come before root");
    }

    #[test]
    fn test_compilable_type_dependency_order() {
        let mut parser = make_parser();
        let source = r#"
struct Base {
    int value;
};

struct Derived {
    Base base;
    int extra;
};

void use_derived(Derived& d) {
    d.base.value = 42;
}
"#;
        let dep_info = analyze_file(&mut parser, source);
        let global = GlobalIndex::new();
        let docs = build_compilable_documents(&dep_info, &global, 4096, 3);

        assert!(!docs.is_empty());

        // Find doc with both types
        let doc = docs.iter().find(|d| {
            d.contains("struct Base") && d.contains("struct Derived")
        });
        if let Some(doc) = doc {
            let base_pos = doc.find("struct Base").unwrap();
            let derived_pos = doc.find("struct Derived").unwrap();
            assert!(
                base_pos < derived_pos,
                "Base should appear before Derived (dependency order)"
            );
        }
    }
}
