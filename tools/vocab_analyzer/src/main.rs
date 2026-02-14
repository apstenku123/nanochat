/// vocab-analyzer: High-performance C++ vocabulary frequency analyzer.
///
/// Scans C++ source files (raw directories, JSONL, or individual files) to validate
/// proposed tokenizer vocabulary against actual usage. Reports per-category coverage,
/// per-token frequency, morpheme analysis, naming style distribution, and namespace usage.
///
/// Optimized for large corpora (200GB+) with:
/// - Hand-written O(n) identifier scanner (no regex in hot path)
/// - rayon work-stealing parallelism across all CPU cores
/// - Memory-mapped file I/O for zero-copy reads
/// - Per-thread local counters merged at end (lock-free hot path)

mod analyzer;
mod patterns;

use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use clap::Parser;
use rayon::prelude::*;
use serde::Serialize;
use walkdir::WalkDir;

use analyzer::{analyze_document, ContentStats, DocResult, StyleCounts};
use patterns::{MatchMode, CATEGORIES, CPP_KEYWORDS, MORPHEMES};

#[derive(Parser)]
#[command(name = "vocab-analyzer", about = "High-performance C++ vocabulary frequency analyzer")]
struct Args {
    /// Input paths: directories of C++ files, .jsonl files, or individual .cpp/.h files
    #[arg(required = true, num_args = 1..)]
    inputs: Vec<String>,

    /// Number of threads (0 = auto-detect)
    #[arg(long, default_value = "0")]
    threads: usize,

    /// Include morpheme stem analysis
    #[arg(long)]
    morphemes: bool,

    /// Include C/C++ keyword frequency analysis
    #[arg(long)]
    keywords: bool,

    /// Include namespace usage analysis
    #[arg(long)]
    namespaces: bool,

    /// Include naming style distribution
    #[arg(long)]
    styles: bool,

    /// Include whitespace/comment/unicode content analysis
    #[arg(long)]
    content: bool,

    /// Enable all analysis modes
    #[arg(long)]
    all: bool,

    /// Output JSON to file
    #[arg(short, long)]
    output: Option<String>,

    /// Only show specific category (partial match)
    #[arg(short, long)]
    category: Option<String>,

    /// Show top N tokens per category
    #[arg(long)]
    top: Option<usize>,

    /// Max files to process (for quick testing)
    #[arg(long)]
    max_files: Option<usize>,

    /// C++ file extensions to include when scanning directories
    #[arg(long, default_value = "cpp,cc,cxx,c,h,hpp,hxx,cu,cuh,inl,ipp")]
    extensions: String,
}

/// Aggregated results across all documents.
struct GlobalResult {
    /// Per-token counts (ident + namespace-stripped)
    ident_counts: HashMap<String, u64>,
    /// Per-token document frequency
    doc_presence: HashMap<String, u64>,
    /// Dollar operator counts
    dollar_counts: HashMap<String, u64>,
    dollar_doc_presence: HashMap<String, u64>,
    /// SQL keyword counts (in string literals)
    sql_keyword_counts: HashMap<String, u64>,
    sql_doc_presence: HashMap<String, u64>,
    /// Morpheme component counts
    morpheme_counts: HashMap<String, u64>,
    /// Naming style distribution
    style_counts: StyleCounts,
    /// Namespace usage
    namespace_counts: HashMap<String, u64>,
    /// C++ keyword frequency
    keyword_counts: HashMap<String, u64>,
    /// Content stats (whitespace, comments, unicode)
    content_stats: ContentStats,
    /// Docs with non-ASCII comments
    docs_with_non_ascii_comments: u64,
    /// Non-English comment samples (capped)
    non_english_samples: Vec<String>,
    /// Document-level stats
    num_docs: u64,
    total_idents: u64,
    total_ident_len: u64,
    total_unique_idents: u64,
}

impl GlobalResult {
    fn new() -> Self {
        Self {
            ident_counts: HashMap::new(),
            doc_presence: HashMap::new(),
            dollar_counts: HashMap::new(),
            dollar_doc_presence: HashMap::new(),
            sql_keyword_counts: HashMap::new(),
            sql_doc_presence: HashMap::new(),
            morpheme_counts: HashMap::new(),
            style_counts: StyleCounts::default(),
            namespace_counts: HashMap::new(),
            keyword_counts: HashMap::new(),
            content_stats: ContentStats::default(),
            docs_with_non_ascii_comments: 0,
            non_english_samples: Vec::new(),
            num_docs: 0,
            total_idents: 0,
            total_ident_len: 0,
            total_unique_idents: 0,
        }
    }

    fn merge_doc(&mut self, doc: DocResult) {
        self.num_docs += 1;
        self.total_idents += doc.total_idents;
        self.total_ident_len += doc.total_ident_len;
        self.total_unique_idents += doc.unique_idents;
        self.style_counts.merge(&doc.style_counts);
        self.content_stats.merge(&doc.content_stats);
        if doc.content_stats.has_non_ascii_comment {
            self.docs_with_non_ascii_comments += 1;
            if self.non_english_samples.len() < 50 {
                for sample in &doc.content_stats.non_english_samples {
                    if self.non_english_samples.len() < 50 {
                        self.non_english_samples.push(sample.clone());
                    }
                }
            }
        }

        // Merge ident counts + doc presence
        for (k, v) in &doc.ident_counts {
            *self.ident_counts.entry(k.clone()).or_default() += v;
        }
        for k in doc.ident_counts.keys() {
            *self.doc_presence.entry(k.clone()).or_default() += 1;
        }

        // Dollar
        for (k, v) in &doc.dollar_counts {
            *self.dollar_counts.entry(k.clone()).or_default() += v;
        }
        for k in doc.dollar_counts.keys() {
            *self.dollar_doc_presence.entry(k.clone()).or_default() += 1;
        }

        // SQL
        for (k, v) in &doc.sql_keyword_counts {
            *self.sql_keyword_counts.entry(k.clone()).or_default() += v;
        }
        for k in doc.sql_keyword_counts.keys() {
            *self.sql_doc_presence.entry(k.clone()).or_default() += 1;
        }

        // Morphemes
        for (k, v) in &doc.morpheme_counts {
            *self.morpheme_counts.entry(k.clone()).or_default() += v;
        }

        // Namespaces
        for (k, v) in &doc.namespace_counts {
            *self.namespace_counts.entry(k.clone()).or_default() += v;
        }
    }

    /// Compute C++ keyword frequencies from ident_counts.
    fn compute_keyword_counts(&mut self) {
        for &kw in CPP_KEYWORDS {
            if let Some(&count) = self.ident_counts.get(kw) {
                self.keyword_counts.insert(kw.to_owned(), count);
            }
        }
    }
}

/// Merge two GlobalResults (for combining per-thread results).
fn merge_globals(a: &mut GlobalResult, b: GlobalResult) {
    a.num_docs += b.num_docs;
    a.total_idents += b.total_idents;
    a.total_ident_len += b.total_ident_len;
    a.total_unique_idents += b.total_unique_idents;
    a.style_counts.merge(&b.style_counts);
    a.content_stats.merge(&b.content_stats);
    a.docs_with_non_ascii_comments += b.docs_with_non_ascii_comments;
    for sample in b.non_english_samples {
        if a.non_english_samples.len() < 50 {
            a.non_english_samples.push(sample);
        }
    }

    for (k, v) in b.ident_counts {
        *a.ident_counts.entry(k).or_default() += v;
    }
    for (k, v) in b.doc_presence {
        *a.doc_presence.entry(k).or_default() += v;
    }
    for (k, v) in b.dollar_counts {
        *a.dollar_counts.entry(k).or_default() += v;
    }
    for (k, v) in b.dollar_doc_presence {
        *a.dollar_doc_presence.entry(k).or_default() += v;
    }
    for (k, v) in b.sql_keyword_counts {
        *a.sql_keyword_counts.entry(k).or_default() += v;
    }
    for (k, v) in b.sql_doc_presence {
        *a.sql_doc_presence.entry(k).or_default() += v;
    }
    for (k, v) in b.morpheme_counts {
        *a.morpheme_counts.entry(k).or_default() += v;
    }
    for (k, v) in b.namespace_counts {
        *a.namespace_counts.entry(k).or_default() += v;
    }
}

// ─── File Collection ──────────────────────────────────────────────────────────

fn collect_files(inputs: &[String], extensions: &str) -> Vec<PathBuf> {
    let exts: Vec<&str> = extensions.split(',').collect();
    let mut files = Vec::new();

    for input in inputs {
        let path = PathBuf::from(input);

        if path.is_dir() {
            // Walk directory recursively
            for entry in WalkDir::new(&path)
                .follow_links(true)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                if entry.file_type().is_file() {
                    if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
                        if exts.iter().any(|&e2| e2.eq_ignore_ascii_case(ext)) {
                            files.push(entry.into_path());
                        }
                    }
                }
            }
        } else if path.extension().map_or(false, |e| e == "jsonl" || e == "json") {
            // JSONL file: read later
            files.push(path);
        } else if path.is_file() {
            files.push(path);
        } else {
            eprintln!("Warning: {} not found", input);
        }
    }

    files.sort();
    files
}

/// Read file contents. For JSONL, returns vec of text fields. For source files, returns vec of one.
fn read_file_contents(path: &PathBuf) -> Vec<Vec<u8>> {
    if path.extension().map_or(false, |e| e == "jsonl" || e == "json") {
        // JSONL: each line is {"text": "..."} — extract text field
        let file = match fs::File::open(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("  Error opening {}: {}", path.display(), e);
                return vec![];
            }
        };
        let reader = io::BufReader::with_capacity(16 * 1024 * 1024, file);
        let mut docs = Vec::new();
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            if line.is_empty() {
                continue;
            }
            // Fast JSON text extraction: look for "text" field
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
                if let Some(text) = val.get("text").and_then(|v| v.as_str()) {
                    docs.push(text.as_bytes().to_vec());
                } else if let Some(text) = val.get("content").and_then(|v| v.as_str()) {
                    docs.push(text.as_bytes().to_vec());
                }
            }
        }
        docs
    } else {
        // Regular file: read entire contents
        match fs::read(path) {
            Ok(contents) => vec![contents],
            Err(e) => {
                eprintln!("  Error reading {}: {}", path.display(), e);
                vec![]
            }
        }
    }
}

// ─── Formatting helpers ───────────────────────────────────────────────────────

fn format_num(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

fn status_label(count: u64) -> &'static str {
    match count {
        0 => "NOT FOUND",
        1..=9 => "RARE",
        10..=99 => "LOW",
        100..=999 => "MODERATE",
        1000..=9999 => "HIGH",
        _ => "VERY HIGH",
    }
}

// ─── Reporting ────────────────────────────────────────────────────────────────

fn print_category_report(
    name: &str,
    tokens: &[&str],
    mode: MatchMode,
    result: &GlobalResult,
    top: Option<usize>,
) {
    // Collect token results
    let mut token_results: Vec<(&str, u64, u64)> = tokens
        .iter()
        .map(|&token| {
            let (count, doc_count) = match mode {
                MatchMode::Ident => (
                    *result.ident_counts.get(token).unwrap_or(&0),
                    *result.doc_presence.get(token).unwrap_or(&0),
                ),
                MatchMode::Dollar => (
                    *result.dollar_counts.get(token).unwrap_or(&0),
                    *result.dollar_doc_presence.get(token).unwrap_or(&0),
                ),
                MatchMode::SqlString => (
                    *result.sql_keyword_counts.get(token).unwrap_or(&0),
                    *result.sql_doc_presence.get(token).unwrap_or(&0),
                ),
            };
            (token, count, doc_count)
        })
        .collect();

    token_results.sort_by(|a, b| b.1.cmp(&a.1));
    if let Some(n) = top {
        token_results.truncate(n);
    }

    let found = token_results.iter().filter(|r| r.1 > 0).count();
    let total = tokens.len();
    let total_hits: u64 = token_results.iter().map(|r| r.1).sum();
    let coverage = if total > 0 {
        found as f64 / total as f64 * 100.0
    } else {
        0.0
    };

    println!("\n{}", "═".repeat(78));
    println!(
        "  {}  ({}/{} found, {:.0}% coverage)",
        name.to_uppercase().replace('_', " "),
        found,
        total,
        coverage,
    );
    println!("  Total occurrences: {}", format_num(total_hits));
    println!("{}", "═".repeat(78));

    println!(
        "  {:<45} {:>10} {:>8} {:>7}  {}",
        "Token", "Count", "Docs", "Doc%", "Status"
    );
    println!(
        "  {} {} {} {}  {}",
        "─".repeat(45),
        "─".repeat(10),
        "─".repeat(8),
        "─".repeat(7),
        "─".repeat(12),
    );

    let num_docs = result.num_docs;
    for (token, count, doc_count) in &token_results {
        let doc_pct = if num_docs > 0 {
            *doc_count as f64 / num_docs as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "  {:<45} {:>10} {:>8} {:>6.1}%  {}",
            token,
            format_num(*count),
            format_num(*doc_count),
            doc_pct,
            status_label(*count),
        );
    }
}

fn print_morpheme_report(result: &GlobalResult, top: Option<usize>) {
    println!("\n{}", "═".repeat(78));
    println!("  MORPHEME STEM ANALYSIS");
    println!("{}", "═".repeat(78));

    for cat in MORPHEMES {
        let mut stems: Vec<(&str, u64)> = cat
            .stems
            .iter()
            .map(|&s| (s, *result.morpheme_counts.get(s).unwrap_or(&0)))
            .collect();
        stems.sort_by(|a, b| b.1.cmp(&a.1));
        if let Some(n) = top {
            stems.truncate(n);
        }

        let found = stems.iter().filter(|s| s.1 > 0).count();
        let total = cat.stems.len();
        let total_hits: u64 = stems.iter().map(|s| s.1).sum();

        println!("\n{}", "─".repeat(78));
        println!(
            "  {}  ({}/{} found, {:.0}% coverage)",
            cat.name.to_uppercase(),
            found,
            total,
            if total > 0 { found as f64 / total as f64 * 100.0 } else { 0.0 },
        );
        println!("  Total occurrences: {}", format_num(total_hits));

        println!(
            "  {:<30} {:>12}  {}",
            "Stem", "Count", "Status"
        );
        for (stem, count) in &stems {
            println!(
                "  {:<30} {:>12}  {}",
                stem,
                format_num(*count),
                status_label(*count),
            );
        }
    }
}

fn print_keyword_report(result: &GlobalResult, top: Option<usize>) {
    println!("\n{}", "═".repeat(78));
    println!("  C/C++ KEYWORD FREQUENCY");
    println!("{}", "═".repeat(78));

    let mut kws: Vec<(&str, u64)> = CPP_KEYWORDS
        .iter()
        .map(|&kw| (kw, *result.ident_counts.get(kw).unwrap_or(&0)))
        .collect();
    kws.sort_by(|a, b| b.1.cmp(&a.1));
    kws.dedup_by(|a, b| a.0 == b.0);
    if let Some(n) = top {
        kws.truncate(n);
    }

    let total: u64 = kws.iter().map(|k| k.1).sum();
    println!("  Total keyword occurrences: {}", format_num(total));
    println!(
        "  {:<25} {:>12} {:>8}  {}",
        "Keyword", "Count", "Doc%", "Status"
    );
    println!(
        "  {} {} {}  {}",
        "─".repeat(25),
        "─".repeat(12),
        "─".repeat(8),
        "─".repeat(12),
    );
    for (kw, count) in &kws {
        let doc_count = *result.doc_presence.get(*kw).unwrap_or(&0);
        let doc_pct = if result.num_docs > 0 {
            doc_count as f64 / result.num_docs as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "  {:<25} {:>12} {:>7.1}%  {}",
            kw,
            format_num(*count),
            doc_pct,
            status_label(*count),
        );
    }
}

fn print_namespace_report(result: &GlobalResult, top: Option<usize>) {
    println!("\n{}", "═".repeat(78));
    println!("  NAMESPACE USAGE");
    println!("{}", "═".repeat(78));

    let mut ns: Vec<(&String, &u64)> = result.namespace_counts.iter().collect();
    ns.sort_by(|a, b| b.1.cmp(a.1));
    if let Some(n) = top {
        ns.truncate(n);
    }

    let total: u64 = ns.iter().map(|n| *n.1).sum();
    println!("  Total namespace-qualified refs: {}", format_num(total));
    println!(
        "  {:<30} {:>12}",
        "Namespace", "References"
    );
    println!("  {} {}", "─".repeat(30), "─".repeat(12));
    for (name, count) in &ns {
        println!("  {:<30} {:>12}", name, format_num(**count));
    }
}

fn print_style_report(result: &GlobalResult) {
    println!("\n{}", "═".repeat(78));
    println!("  IDENTIFIER NAMING STYLE DISTRIBUTION");
    println!("{}", "═".repeat(78));

    let s = &result.style_counts;
    let total = s.snake_case + s.camel_case + s.pascal_case + s.screaming_case + s.mixed_other;
    if total == 0 {
        println!("  No identifiers analyzed.");
        return;
    }

    let pct = |n: u64| n as f64 / total as f64 * 100.0;
    println!("  Total identifiers (len>2): {}", format_num(total));
    println!(
        "  {:<25} {:>12} {:>8}",
        "Style", "Count", "%"
    );
    println!("  {} {} {}", "─".repeat(25), "─".repeat(12), "─".repeat(8));
    println!("  {:<25} {:>12} {:>7.1}%", "snake_case", format_num(s.snake_case), pct(s.snake_case));
    println!("  {:<25} {:>12} {:>7.1}%", "camelCase", format_num(s.camel_case), pct(s.camel_case));
    println!("  {:<25} {:>12} {:>7.1}%", "PascalCase", format_num(s.pascal_case), pct(s.pascal_case));
    println!("  {:<25} {:>12} {:>7.1}%", "SCREAMING_CASE", format_num(s.screaming_case), pct(s.screaming_case));
    println!("  {:<25} {:>12} {:>7.1}%", "other/mixed", format_num(s.mixed_other), pct(s.mixed_other));

    // Mean identifier length
    if result.total_idents > 0 {
        let mean_len = result.total_ident_len as f64 / result.total_idents as f64;
        println!("\n  Mean identifier length: {:.1} chars", mean_len);
        println!("  Mean unique identifiers per doc: {:.0}",
            result.total_unique_idents as f64 / result.num_docs.max(1) as f64);
    }
}

fn print_content_report(result: &GlobalResult) {
    let s = &result.content_stats;
    println!("\n{}", "═".repeat(78));
    println!("  CONTENT ANALYSIS (Whitespace / Comments / Unicode)");
    println!("{}", "═".repeat(78));

    let pct = |n: u64, total: u64| {
        if total > 0 { n as f64 / total as f64 * 100.0 } else { 0.0 }
    };

    println!("\n  --- Byte Distribution ---");
    println!("  Total bytes:          {:>15}", format_num(s.total_bytes));
    println!("  Comment bytes:        {:>15} ({:.1}%)", format_num(s.comment_bytes), pct(s.comment_bytes, s.total_bytes));
    println!("  String literal bytes: {:>15} ({:.1}%)", format_num(s.string_bytes), pct(s.string_bytes, s.total_bytes));
    println!("  Code bytes (approx):  {:>15} ({:.1}%)",
        format_num(s.total_bytes.saturating_sub(s.comment_bytes).saturating_sub(s.string_bytes)),
        pct(s.total_bytes.saturating_sub(s.comment_bytes).saturating_sub(s.string_bytes), s.total_bytes));

    println!("\n  --- Lines ---");
    println!("  Total lines:          {:>15}", format_num(s.total_lines));
    println!("  Blank lines:          {:>15} ({:.1}%)", format_num(s.blank_lines), pct(s.blank_lines, s.total_lines));

    println!("\n  --- Comments ---");
    println!("  Line comments (//):   {:>15}", format_num(s.line_comments));
    println!("  Block comments (/*):  {:>15}", format_num(s.block_comments));
    println!("  Total comments:       {:>15}", format_num(s.line_comments + s.block_comments));

    println!("\n  --- Whitespace Patterns (BPE-relevant) ---");
    println!("  Single spaces:        {:>15}", format_num(s.single_spaces));
    println!("  Multi-space runs:     {:>15} (2+ consecutive spaces)", format_num(s.multi_spaces));
    println!("  Single tabs:          {:>15}", format_num(s.single_tabs));
    println!("  Multi-tab runs:       {:>15} (2+ consecutive tabs)", format_num(s.multi_tabs));
    println!("  Mixed indent lines:   {:>15} (spaces + tabs on same line)", format_num(s.mixed_indent));

    println!("\n  --- Indentation Style ---");
    println!("  2-space indent:       {:>15}", format_num(s.indent_2));
    println!("  4-space indent:       {:>15}", format_num(s.indent_4));
    println!("  8-space indent:       {:>15}", format_num(s.indent_8));
    println!("  Tab indent:           {:>15}", format_num(s.tab_indent));
    let total_indent = s.indent_2 + s.indent_4 + s.indent_8 + s.tab_indent;
    if total_indent > 0 {
        println!("  Dominant style:       {} ({:.0}%)",
            if s.indent_4 >= s.indent_2 && s.indent_4 >= s.tab_indent { "4-space" }
            else if s.indent_2 >= s.tab_indent { "2-space" }
            else { "tab" },
            pct(s.indent_4.max(s.indent_2).max(s.tab_indent), total_indent));
    }

    println!("\n  --- Unicode / Non-English Content ---");
    println!("  Non-ASCII bytes:      {:>15} ({:.3}% of total)", format_num(s.non_ascii_bytes), pct(s.non_ascii_bytes, s.total_bytes));
    println!("  UTF-8 multi-byte:     {:>15} characters", format_num(s.utf8_chars));
    println!("  Docs w/ non-ASCII comments: {:>9} ({:.1}% of docs)",
        format_num(result.docs_with_non_ascii_comments),
        pct(result.docs_with_non_ascii_comments, result.num_docs));

    if !result.non_english_samples.is_empty() {
        println!("\n  --- Non-English Comment Samples (first {}) ---", result.non_english_samples.len());
        for (i, sample) in result.non_english_samples.iter().enumerate().take(20) {
            let truncated = if sample.len() > 80 { &sample[..80] } else { sample.as_str() };
            println!("  {}. {}", i + 1, truncated);
        }
    }

    // BPE design recommendations
    println!("\n  --- BPE Design Implications ---");
    let space_ratio = if s.single_spaces + s.multi_spaces > 0 {
        s.multi_spaces as f64 / (s.single_spaces + s.multi_spaces) as f64 * 100.0
    } else { 0.0 };
    println!("  Multi-space ratio:    {:.1}% (of all space occurrences)", space_ratio);
    println!("  -> {} multi-space tokens in BPE vocab",
        if space_ratio < 5.0 { "Skip" } else if space_ratio < 20.0 { "Add few" } else { "Add several" });
    println!("  -> {} comment separator tokens",
        if s.comment_bytes as f64 / s.total_bytes as f64 > 0.15 { "Important — comments are >15% of corpus" }
        else { "Lower priority — comments are <15% of corpus" });
    println!("  -> Non-ASCII in comments: {}",
        if pct(result.docs_with_non_ascii_comments, result.num_docs) > 5.0 {
            "Significant — consider unicode-to-ASCII normalization"
        } else if pct(result.docs_with_non_ascii_comments, result.num_docs) > 1.0 {
            "Moderate — some non-English comments exist"
        } else {
            "Minimal — corpus is predominantly ASCII/English"
        });
}

fn print_summary(result: &GlobalResult) {
    println!("\n{}", "━".repeat(78));
    println!("  OVERALL SUMMARY  ({} documents analyzed)", format_num(result.num_docs));
    println!("{}", "━".repeat(78));

    println!(
        "\n  {:<30} {:>8} {:>8} {:>8} {:>12}",
        "Category", "Proposed", "Found", "Coverage", "Total Hits"
    );
    println!(
        "  {} {} {} {} {}",
        "─".repeat(30),
        "─".repeat(8),
        "─".repeat(8),
        "─".repeat(8),
        "─".repeat(12),
    );

    let mut grand_proposed = 0u64;
    let mut grand_found = 0u64;
    let mut grand_hits = 0u64;

    for cat in CATEGORIES {
        let total = cat.tokens.len();
        let mut found = 0;
        let mut hits = 0u64;
        for &token in cat.tokens {
            let count = match cat.mode {
                MatchMode::Ident => *result.ident_counts.get(token).unwrap_or(&0),
                MatchMode::Dollar => *result.dollar_counts.get(token).unwrap_or(&0),
                MatchMode::SqlString => *result.sql_keyword_counts.get(token).unwrap_or(&0),
            };
            if count > 0 {
                found += 1;
            }
            hits += count;
        }

        let coverage = if total > 0 {
            found as f64 / total as f64 * 100.0
        } else {
            0.0
        };

        grand_proposed += total as u64;
        grand_found += found as u64;
        grand_hits += hits;

        println!(
            "  {:<30} {:>8} {:>8} {:>7.0}% {:>12}",
            cat.name, total, found, coverage, format_num(hits),
        );
    }

    println!(
        "  {} {} {} {} {}",
        "─".repeat(30),
        "─".repeat(8),
        "─".repeat(8),
        "─".repeat(8),
        "─".repeat(12),
    );
    let grand_coverage = if grand_proposed > 0 {
        grand_found as f64 / grand_proposed as f64 * 100.0
    } else {
        0.0
    };
    println!(
        "  {:<30} {:>8} {:>8} {:>7.0}% {:>12}",
        "TOTAL",
        grand_proposed,
        grand_found,
        grand_coverage,
        format_num(grand_hits),
    );
}

// ─── JSON Output ──────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct JsonOutput {
    num_docs: u64,
    total_idents: u64,
    mean_ident_len: f64,
    elapsed_s: f64,
    vocab: HashMap<String, Vec<JsonToken>>,
    morphemes: HashMap<String, Vec<JsonStem>>,
    style: JsonStyle,
    top_namespaces: Vec<JsonNamespace>,
    top_keywords: Vec<JsonKeyword>,
}

#[derive(Serialize)]
struct JsonToken {
    token: String,
    count: u64,
    doc_count: u64,
    doc_pct: f64,
    status: String,
}

#[derive(Serialize)]
struct JsonStem {
    stem: String,
    count: u64,
}

#[derive(Serialize)]
struct JsonStyle {
    snake_case: u64,
    camel_case: u64,
    pascal_case: u64,
    screaming_case: u64,
    mixed_other: u64,
}

#[derive(Serialize)]
struct JsonNamespace {
    name: String,
    count: u64,
}

#[derive(Serialize)]
struct JsonKeyword {
    keyword: String,
    count: u64,
}

fn write_json(result: &GlobalResult, elapsed: f64, path: &str) {
    let mut vocab: HashMap<String, Vec<JsonToken>> = HashMap::new();

    for cat in CATEGORIES {
        let mut tokens: Vec<JsonToken> = cat
            .tokens
            .iter()
            .map(|&token| {
                let (count, doc_count) = match cat.mode {
                    MatchMode::Ident => (
                        *result.ident_counts.get(token).unwrap_or(&0),
                        *result.doc_presence.get(token).unwrap_or(&0),
                    ),
                    MatchMode::Dollar => (
                        *result.dollar_counts.get(token).unwrap_or(&0),
                        *result.dollar_doc_presence.get(token).unwrap_or(&0),
                    ),
                    MatchMode::SqlString => (
                        *result.sql_keyword_counts.get(token).unwrap_or(&0),
                        *result.sql_doc_presence.get(token).unwrap_or(&0),
                    ),
                };
                let doc_pct = if result.num_docs > 0 {
                    doc_count as f64 / result.num_docs as f64 * 100.0
                } else {
                    0.0
                };
                JsonToken {
                    token: token.to_owned(),
                    count,
                    doc_count,
                    doc_pct,
                    status: status_label(count).to_owned(),
                }
            })
            .collect();
        tokens.sort_by(|a, b| b.count.cmp(&a.count));
        vocab.insert(cat.name.to_owned(), tokens);
    }

    let mut morphemes_json: HashMap<String, Vec<JsonStem>> = HashMap::new();
    for cat in MORPHEMES {
        let mut stems: Vec<JsonStem> = cat
            .stems
            .iter()
            .map(|&s| JsonStem {
                stem: s.to_owned(),
                count: *result.morpheme_counts.get(s).unwrap_or(&0),
            })
            .collect();
        stems.sort_by(|a, b| b.count.cmp(&a.count));
        morphemes_json.insert(cat.name.to_owned(), stems);
    }

    let mut ns: Vec<JsonNamespace> = result
        .namespace_counts
        .iter()
        .map(|(k, v)| JsonNamespace {
            name: k.clone(),
            count: *v,
        })
        .collect();
    ns.sort_by(|a, b| b.count.cmp(&a.count));
    ns.truncate(100);

    let mut kws: Vec<JsonKeyword> = CPP_KEYWORDS
        .iter()
        .map(|&kw| JsonKeyword {
            keyword: kw.to_owned(),
            count: *result.ident_counts.get(kw).unwrap_or(&0),
        })
        .collect();
    kws.sort_by(|a, b| b.count.cmp(&a.count));
    kws.dedup_by(|a, b| a.keyword == b.keyword);

    let mean_len = if result.total_idents > 0 {
        result.total_ident_len as f64 / result.total_idents as f64
    } else {
        0.0
    };

    let output = JsonOutput {
        num_docs: result.num_docs,
        total_idents: result.total_idents,
        mean_ident_len: mean_len,
        elapsed_s: elapsed,
        vocab,
        morphemes: morphemes_json,
        style: JsonStyle {
            snake_case: result.style_counts.snake_case,
            camel_case: result.style_counts.camel_case,
            pascal_case: result.style_counts.pascal_case,
            screaming_case: result.style_counts.screaming_case,
            mixed_other: result.style_counts.mixed_other,
        },
        top_namespaces: ns,
        top_keywords: kws,
    };

    let file = fs::File::create(path).expect("Failed to create output file");
    let writer = BufWriter::with_capacity(64 * 1024 * 1024, file);
    serde_json::to_writer_pretty(writer, &output).expect("Failed to write JSON");
    eprintln!("JSON results saved to {}", path);
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    // Configure rayon thread pool
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .expect("Failed to set thread count");
    }
    let num_threads = rayon::current_num_threads();

    let do_morphemes = args.morphemes || args.all;
    let do_keywords = args.keywords || args.all;
    let do_namespaces = args.namespaces || args.all;
    let do_styles = args.styles || args.all;
    let do_content = args.content || args.all;

    // Collect files
    eprintln!("Collecting files from {} input path(s)...", args.inputs.len());
    let t_collect = Instant::now();
    let mut files = collect_files(&args.inputs, &args.extensions);
    if let Some(max) = args.max_files {
        files.truncate(max);
    }
    eprintln!(
        "Found {} files in {:.1}s",
        format_num(files.len() as u64),
        t_collect.elapsed().as_secs_f64(),
    );

    if files.is_empty() {
        eprintln!("Error: No files found");
        std::process::exit(1);
    }

    // Calculate total size
    let total_bytes: u64 = files.iter().filter_map(|f| fs::metadata(f).ok()).map(|m| m.len()).sum();
    eprintln!(
        "Total size: {:.2} GB, using {} threads",
        total_bytes as f64 / 1e9,
        num_threads,
    );

    // Process files in parallel using rayon
    let t0 = Instant::now();
    let files_done = AtomicU64::new(0);
    let bytes_done = AtomicU64::new(0);
    let total_files = files.len() as u64;

    // Process in parallel batches to reduce merge overhead
    let batch_size = std::cmp::max(100, files.len() / (num_threads * 4));
    let batches: Vec<&[PathBuf]> = files.chunks(batch_size).collect();

    eprintln!("Processing {} batches of ~{} files...", batches.len(), batch_size);

    let batch_results: Vec<GlobalResult> = batches
        .par_iter()
        .map(|batch| {
            let mut local = GlobalResult::new();
            for path in *batch {
                let docs = read_file_contents(path);
                let file_size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);

                for doc_bytes in &docs {
                    let doc_result = analyze_document(doc_bytes, do_morphemes, do_content);
                    local.merge_doc(doc_result);
                }

                let done = files_done.fetch_add(1, Ordering::Relaxed) + 1;
                bytes_done.fetch_add(file_size, Ordering::Relaxed);

                // Progress every 10K files
                if done % 10_000 == 0 || done == total_files {
                    let elapsed = t0.elapsed().as_secs_f64();
                    let gb_done = bytes_done.load(Ordering::Relaxed) as f64 / 1e9;
                    let rate = done as f64 / elapsed;
                    eprintln!(
                        "  [{}/{}] {:.1} GB, {:.0} files/sec, {:.1}s",
                        format_num(done),
                        format_num(total_files),
                        gb_done,
                        rate,
                        elapsed,
                    );
                }
            }
            local
        })
        .collect();

    // Merge all batch results
    eprintln!("Merging {} batch results...", batch_results.len());
    let mut result = GlobalResult::new();
    for batch in batch_results {
        merge_globals(&mut result, batch);
    }

    if do_keywords {
        result.compute_keyword_counts();
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let throughput_gb = total_bytes as f64 / 1e9 / elapsed;
    eprintln!(
        "\nCompleted: {} docs from {} files in {:.1}s ({:.1} GB/s, {:.0} files/sec)",
        format_num(result.num_docs),
        format_num(total_files),
        elapsed,
        throughput_gb,
        total_files as f64 / elapsed,
    );

    // Write JSON output if requested
    if let Some(ref output_path) = args.output {
        write_json(&result, elapsed, output_path);
    }

    // Print reports
    let cat_filter = args.category.as_deref().map(|c| c.to_lowercase());

    for cat in CATEGORIES {
        if let Some(ref filter) = cat_filter {
            if !cat.name.to_lowercase().contains(filter.as_str()) {
                continue;
            }
        }
        print_category_report(cat.name, cat.tokens, cat.mode, &result, args.top);
    }

    if do_morphemes {
        print_morpheme_report(&result, args.top);
    }

    if do_keywords {
        print_keyword_report(&result, args.top);
    }

    if do_namespaces {
        print_namespace_report(&result, args.top);
    }

    if do_styles {
        print_style_report(&result);
    }

    if do_content {
        print_content_report(&result);
    }

    // Always print summary
    print_summary(&result);

    // Flush stdout
    let _ = io::stdout().lock().flush();
}
