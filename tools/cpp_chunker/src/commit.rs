//! Commit-mode: process git commit diffs into training documents with
//! function/class chain extraction.
//!
//! Input: JSONL with {old_content, new_content, diff, subject, body, filepath, repo}
//! Output: Training documents with:
//!   Format A (chain): docstring → pre-version function chain → post-version function chain
//!   Format B (diff):  docstring → context function chain → applicable git diff

use std::collections::HashSet;
use tree_sitter::Parser;
use regex::Regex;

use crate::deps::{self, FileDepInfo, FunctionInfo, normalize_name};

/// A single commit record from the input JSONL.
#[derive(serde::Deserialize)]
pub struct CommitRecord {
    pub old_content: String,
    pub new_content: String,
    pub diff: String,
    pub subject: String,
    #[serde(default)]
    pub body: String,
    pub filepath: String,
    pub repo: String,
}

/// Parsed hunk range from a unified diff.
struct HunkRange {
    old_start: usize,
    old_count: usize,
    new_start: usize,
    new_count: usize,
}

/// Parse unified diff to extract hunk line ranges.
fn parse_hunk_ranges(diff: &str) -> Vec<HunkRange> {
    let re = Regex::new(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@").unwrap();
    re.captures_iter(diff)
        .map(|cap| HunkRange {
            old_start: cap[1].parse().unwrap_or(0),
            old_count: cap.get(2).map(|m| m.as_str().parse().unwrap_or(1)).unwrap_or(1),
            new_start: cap[3].parse().unwrap_or(0),
            new_count: cap.get(4).map(|m| m.as_str().parse().unwrap_or(1)).unwrap_or(1),
        })
        .collect()
}

/// Get set of changed line numbers from hunk ranges.
fn changed_lines(hunks: &[HunkRange], use_old: bool) -> HashSet<usize> {
    let mut lines = HashSet::new();
    for h in hunks {
        if use_old {
            for l in h.old_start..h.old_start + h.old_count.max(1) {
                lines.insert(l);
            }
        } else {
            for l in h.new_start..h.new_start + h.new_count.max(1) {
                lines.insert(l);
            }
        }
    }
    lines
}

/// Find which functions in a FileDepInfo overlap with the given line numbers.
/// Returns indices into dep_info.functions.
fn find_changed_functions(dep_info: &FileDepInfo, lines: &HashSet<usize>) -> Vec<usize> {
    dep_info.functions.iter().enumerate()
        .filter(|(_, f)| {
            // 1-indexed lines from tree-sitter
            (f.start_line..=f.end_line).any(|l| lines.contains(&l))
        })
        .map(|(i, _)| i)
        .collect()
}

/// Extract the function chain for changed functions: the changed functions
/// plus their transitive local dependencies, ordered leaves-first.
fn extract_function_chain<'a>(
    dep_info: &'a FileDepInfo,
    changed_indices: &[usize],
) -> Vec<&'a FunctionInfo> {
    if changed_indices.is_empty() {
        return vec![];
    }

    // Build name→index map
    let name_to_idx: std::collections::HashMap<String, usize> = dep_info.functions.iter()
        .enumerate()
        .filter(|(_, f)| !f.name.is_empty())
        .map(|(i, f)| (normalize_name(&f.name), i))
        .collect();

    // Collect all transitive deps for each changed function
    let mut all_indices: HashSet<usize> = HashSet::new();
    for &idx in changed_indices {
        let deps = deps::collect_transitive_deps(idx, &dep_info.functions, &name_to_idx);
        all_indices.extend(deps);
    }

    // Sort by dep_level (leaves first, changed functions last)
    let mut sorted: Vec<usize> = all_indices.into_iter().collect();
    sorted.sort_by_key(|&i| dep_info.functions[i].dep_level);

    sorted.iter().map(|&i| &dep_info.functions[i]).collect()
}

/// Find which classes in a FileDepInfo overlap with the given line numbers.
fn find_changed_classes<'a>(dep_info: &'a FileDepInfo, lines: &HashSet<usize>) -> Vec<&'a crate::chunker::Chunk> {
    dep_info.classes.iter()
        .filter(|c| (c.start_line..=c.end_line).any(|l| lines.contains(&l)))
        .collect()
}

/// Build C++ docstring comment from commit info.
fn build_docstring(record: &CommitRecord) -> String {
    let mut lines = vec![
        "/**".to_string(),
        format!(" * @brief {}", record.subject),
        " *".to_string(),
        format!(" * Repository: {}", record.repo),
        format!(" * File: {}", record.filepath),
    ];
    if !record.body.is_empty() {
        lines.push(" *".to_string());
        for body_line in record.body.lines().take(8) {
            lines.push(format!(" * {}", body_line));
        }
    }
    lines.push(" */".to_string());
    lines.join("\n")
}

/// Format A: docstring → pre-version function/class chain → post-version chain.
fn format_chain_document(
    record: &CommitRecord,
    old_dep: &FileDepInfo,
    new_dep: &FileDepInfo,
    hunks: &[HunkRange],
) -> Option<String> {
    let old_lines = changed_lines(hunks, true);
    let new_lines = changed_lines(hunks, false);

    // Find changed functions in old and new versions
    let old_changed = find_changed_functions(old_dep, &old_lines);
    let new_changed = find_changed_functions(new_dep, &new_lines);

    // Need at least one changed function in either version
    if old_changed.is_empty() && new_changed.is_empty() {
        return None;
    }

    // Extract chains
    let old_chain = extract_function_chain(old_dep, &old_changed);
    let new_chain = extract_function_chain(new_dep, &new_changed);

    // Find changed classes
    let old_classes = find_changed_classes(old_dep, &old_lines);
    let new_classes = find_changed_classes(new_dep, &new_lines);

    // Build pre-version
    let mut pre_parts: Vec<String> = Vec::new();
    if !old_dep.preamble.is_empty() && old_chain.len() + old_classes.len() > 0 {
        // Only include minimal preamble (first 20 lines max)
        let short_preamble: String = old_dep.preamble.lines().take(20).collect::<Vec<_>>().join("\n");
        if !short_preamble.is_empty() {
            pre_parts.push(short_preamble);
        }
    }
    for cls in &old_classes {
        pre_parts.push(cls.text.clone());
    }
    for func in &old_chain {
        pre_parts.push(func.text.clone());
    }

    // Build post-version
    let mut post_parts: Vec<String> = Vec::new();
    if !new_dep.preamble.is_empty() && new_chain.len() + new_classes.len() > 0 {
        let short_preamble: String = new_dep.preamble.lines().take(20).collect::<Vec<_>>().join("\n");
        if !short_preamble.is_empty() {
            post_parts.push(short_preamble);
        }
    }
    for cls in &new_classes {
        post_parts.push(cls.text.clone());
    }
    for func in &new_chain {
        post_parts.push(func.text.clone());
    }

    if pre_parts.is_empty() && post_parts.is_empty() {
        return None;
    }

    let docstring = build_docstring(record);
    let pre_code = pre_parts.join("\n\n");
    let post_code = post_parts.join("\n\n");

    Some(format!(
        "{}\n\n// Pre-version: code to be changed\n{}\n\n/**\n * @brief Fix applied: {}\n */\n\n// Post-version: fixed code\n{}",
        docstring, pre_code, record.subject, post_code
    ))
}

/// Format B: docstring → context function chain → applicable diff.
fn format_diff_document(
    record: &CommitRecord,
    old_dep: &FileDepInfo,
    hunks: &[HunkRange],
) -> Option<String> {
    let old_lines = changed_lines(hunks, true);
    let old_changed = find_changed_functions(old_dep, &old_lines);
    let old_classes = find_changed_classes(old_dep, &old_lines);

    let docstring = build_docstring(record);

    let mut parts = vec![docstring];

    // Add context chain if we found changed functions/classes
    if !old_changed.is_empty() || !old_classes.is_empty() {
        let chain = extract_function_chain(old_dep, &old_changed);
        if !chain.is_empty() || !old_classes.is_empty() {
            parts.push(format!("\n// Context: affected code in {}", record.filepath));
            for cls in &old_classes {
                parts.push(cls.text.clone());
            }
            for func in &chain {
                parts.push(func.text.clone());
            }
        }
    }

    parts.push(format!("\n// Applied fix ({}):", record.filepath));
    parts.push(record.diff.clone());

    Some(parts.join("\n"))
}

/// Process a batch of commit records into training documents.
pub fn process_commit_batch(
    records: &[CommitRecord],
    max_tokens: usize,
    max_file_bytes: usize,
    format: &str,  // "chain", "diff", or "both"
) -> Vec<String> {
    let mut parser = Parser::new();
    let lang = tree_sitter_cpp::LANGUAGE;
    parser.set_language(&lang.into()).expect("Failed to set C++ language");

    let mut documents = Vec::new();
    let estimate_tokens = |text: &str| -> usize { std::cmp::max(1, text.len() / 4) };

    for record in records {
        // Skip oversized files
        if record.old_content.len() > max_file_bytes || record.new_content.len() > max_file_bytes {
            continue;
        }
        if record.old_content.len() < 50 || record.new_content.len() < 50 {
            continue;
        }

        // Parse diff hunks
        let hunks = parse_hunk_ranges(&record.diff);
        if hunks.is_empty() {
            continue;
        }

        // Analyze both versions with tree-sitter
        let old_dep = deps::analyze_file(&mut parser, &record.old_content);
        let new_dep = deps::analyze_file(&mut parser, &record.new_content);

        // Format A: chain document
        if format == "chain" || format == "both" {
            if let Some(doc) = format_chain_document(record, &old_dep, &new_dep, &hunks) {
                let tokens = estimate_tokens(&doc);
                if tokens <= max_tokens && doc.len() >= 100 {
                    documents.push(doc);
                }
            }
        }

        // Format B: diff document
        if format == "diff" || format == "both" {
            if let Some(doc) = format_diff_document(record, &old_dep, &hunks) {
                let tokens = estimate_tokens(&doc);
                if tokens <= max_tokens && doc.len() >= 100 {
                    documents.push(doc);
                }
            }
        }
    }

    documents
}
