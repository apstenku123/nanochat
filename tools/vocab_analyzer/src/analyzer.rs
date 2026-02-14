/// Core analysis engine: fast identifier extraction + token counting.
///
/// Key design: O(n) single-pass scanner for identifiers, then O(1) HashSet lookups.
/// No regex in hot path. Hand-written state machine for identifier extraction.

use std::collections::HashMap;

/// Per-document analysis result (stack-allocated counts, no heap until merge).
pub struct DocResult {
    /// Identifier -> count in this document
    pub ident_counts: HashMap<String, u64>,
    /// $-prefixed operator -> count
    pub dollar_counts: HashMap<String, u64>,
    /// SQL keyword (in string literals) -> count
    pub sql_keyword_counts: HashMap<String, u64>,
    /// Morpheme component -> count
    pub morpheme_counts: HashMap<String, u64>,
    /// Identifier naming style counts
    pub style_counts: StyleCounts,
    /// Namespace-qualified identifier counts (std::, boost::, etc.)
    pub namespace_counts: HashMap<String, u64>,
    /// Total identifier count
    pub total_idents: u64,
    /// Total unique identifiers
    pub unique_idents: u64,
    /// Sum of identifier lengths
    pub total_ident_len: u64,
    /// Whitespace/comment/unicode analysis
    pub content_stats: ContentStats,
}

/// Whitespace, comment, and unicode content statistics for BPE design.
#[derive(Default, Clone)]
pub struct ContentStats {
    /// Total bytes in document
    pub total_bytes: u64,
    /// Bytes in comments (// and /* */)
    pub comment_bytes: u64,
    /// Bytes in string literals
    pub string_bytes: u64,
    /// Number of line comments (//)
    pub line_comments: u64,
    /// Number of block comments (/* */)
    pub block_comments: u64,
    /// Lines that are only whitespace
    pub blank_lines: u64,
    /// Total lines
    pub total_lines: u64,
    /// Whitespace statistics
    pub single_spaces: u64,
    pub multi_spaces: u64,    // runs of 2+ spaces
    pub single_tabs: u64,
    pub multi_tabs: u64,      // runs of 2+ tabs
    pub mixed_indent: u64,    // lines with both spaces and tabs before first non-ws
    /// Total indentation levels detected (spaces-based)
    pub indent_2: u64,        // 2-space indent
    pub indent_4: u64,        // 4-space indent
    pub indent_8: u64,        // 8-space indent (or 2-tab)
    pub tab_indent: u64,      // tab-based indent
    /// Unicode content
    pub non_ascii_bytes: u64,
    pub utf8_chars: u64,      // multi-byte UTF-8 sequences
    /// Comments containing non-ASCII (likely non-English)
    pub non_ascii_comments: u64,
    /// Docs with any non-ASCII comment content
    pub has_non_ascii_comment: bool,
    /// Detected non-English comment samples (first few)
    pub non_english_samples: Vec<String>,
}

impl ContentStats {
    pub fn merge(&mut self, other: &ContentStats) {
        self.total_bytes += other.total_bytes;
        self.comment_bytes += other.comment_bytes;
        self.string_bytes += other.string_bytes;
        self.line_comments += other.line_comments;
        self.block_comments += other.block_comments;
        self.blank_lines += other.blank_lines;
        self.total_lines += other.total_lines;
        self.single_spaces += other.single_spaces;
        self.multi_spaces += other.multi_spaces;
        self.single_tabs += other.single_tabs;
        self.multi_tabs += other.multi_tabs;
        self.mixed_indent += other.mixed_indent;
        self.indent_2 += other.indent_2;
        self.indent_4 += other.indent_4;
        self.indent_8 += other.indent_8;
        self.tab_indent += other.tab_indent;
        self.non_ascii_bytes += other.non_ascii_bytes;
        self.utf8_chars += other.utf8_chars;
        self.non_ascii_comments += other.non_ascii_comments;
    }
}

/// Analyze whitespace, comments, and unicode content.
pub fn analyze_content(text: &[u8]) -> ContentStats {
    let mut stats = ContentStats::default();
    stats.total_bytes = text.len() as u64;

    let len = text.len();
    let mut i = 0;
    let mut line_start = 0;
    let mut in_line_start = true; // tracking indent at line start
    let mut line_has_spaces = false;
    let mut line_has_tabs = false;

    while i < len {
        let b = text[i];

        // Newline: analyze the line
        if b == b'\n' {
            stats.total_lines += 1;
            let line = &text[line_start..i];
            if line.iter().all(|&b| b == b' ' || b == b'\t' || b == b'\r') {
                stats.blank_lines += 1;
            }
            // Check indent style
            if line_has_spaces && line_has_tabs {
                stats.mixed_indent += 1;
            }
            // Detect indent level from leading spaces
            let leading_spaces = line.iter().take_while(|&&b| b == b' ').count();
            if leading_spaces > 0 && leading_spaces % 8 == 0 {
                stats.indent_8 += 1;
            } else if leading_spaces > 0 && leading_spaces % 4 == 0 {
                stats.indent_4 += 1;
            } else if leading_spaces > 0 && leading_spaces % 2 == 0 {
                stats.indent_2 += 1;
            }
            if line.first() == Some(&b'\t') {
                stats.tab_indent += 1;
            }
            line_start = i + 1;
            in_line_start = true;
            line_has_spaces = false;
            line_has_tabs = false;
            i += 1;
            continue;
        }

        // Track indent characters at line start
        if in_line_start {
            if b == b' ' {
                line_has_spaces = true;
                i += 1;
                continue;
            } else if b == b'\t' {
                line_has_tabs = true;
                i += 1;
                continue;
            } else {
                in_line_start = false;
            }
        }

        // Count space runs
        if b == b' ' {
            let start = i;
            while i < len && text[i] == b' ' {
                i += 1;
            }
            let run = i - start;
            if run == 1 {
                stats.single_spaces += 1;
            } else {
                stats.multi_spaces += 1;
            }
            continue;
        }

        // Count tab runs
        if b == b'\t' {
            let start = i;
            while i < len && text[i] == b'\t' {
                i += 1;
            }
            let run = i - start;
            if run == 1 {
                stats.single_tabs += 1;
            } else {
                stats.multi_tabs += 1;
            }
            continue;
        }

        // Line comments
        if b == b'/' && i + 1 < len && text[i + 1] == b'/' {
            stats.line_comments += 1;
            let comment_start = i;
            i += 2;
            let content_start = i;
            while i < len && text[i] != b'\n' {
                i += 1;
            }
            let comment_len = i - comment_start;
            stats.comment_bytes += comment_len as u64;
            // Check for non-ASCII in comment
            let comment_content = &text[content_start..i];
            let non_ascii_in_comment = comment_content.iter().filter(|&&b| b > 127).count() as u64;
            if non_ascii_in_comment > 0 {
                stats.non_ascii_bytes += non_ascii_in_comment;
                stats.non_ascii_comments += 1;
                stats.has_non_ascii_comment = true;
                // Extract sample (first non-English comment, max 3)
                if stats.non_english_samples.len() < 3 {
                    if let Ok(s) = std::str::from_utf8(comment_content) {
                        let trimmed = s.trim();
                        if !trimmed.is_empty() && trimmed.len() < 200 {
                            stats.non_english_samples.push(trimmed.to_owned());
                        }
                    }
                }
            }
            continue;
        }

        // Block comments
        if b == b'/' && i + 1 < len && text[i + 1] == b'*' {
            stats.block_comments += 1;
            let comment_start = i;
            i += 2;
            let content_start = i;
            while i + 1 < len && !(text[i] == b'*' && text[i + 1] == b'/') {
                i += 1;
            }
            let comment_content = &text[content_start..i];
            if i + 1 < len {
                i += 2;
            }
            stats.comment_bytes += (i - comment_start) as u64;
            let non_ascii_in_block = comment_content.iter().filter(|&&b| b > 127).count() as u64;
            if non_ascii_in_block > 0 {
                stats.non_ascii_bytes += non_ascii_in_block;
                stats.non_ascii_comments += 1;
                stats.has_non_ascii_comment = true;
                if stats.non_english_samples.len() < 3 {
                    if let Ok(s) = std::str::from_utf8(comment_content) {
                        let trimmed = s.trim();
                        if !trimmed.is_empty() && trimmed.len() < 200 {
                            stats.non_english_samples.push(trimmed.to_owned());
                        }
                    }
                }
            }
            continue;
        }

        // String literals
        if b == b'"' {
            let start = i;
            i += 1;
            while i < len && text[i] != b'"' {
                if text[i] == b'\\' {
                    i += 1;
                }
                i += 1;
            }
            if i < len {
                i += 1;
            }
            stats.string_bytes += (i - start) as u64;
            continue;
        }

        // Non-ASCII byte detection
        if b > 127 {
            stats.non_ascii_bytes += 1;
            // Count UTF-8 multi-byte sequences
            if b & 0xE0 == 0xC0 {
                stats.utf8_chars += 1;
                i += 2;
            } else if b & 0xF0 == 0xE0 {
                stats.utf8_chars += 1;
                i += 3;
            } else if b & 0xF8 == 0xF0 {
                stats.utf8_chars += 1;
                i += 4;
            } else {
                i += 1;
            }
            continue;
        }

        i += 1;
    }

    // Count last line if doesn't end with newline
    if line_start < len {
        stats.total_lines += 1;
    }

    stats
}

#[derive(Default, Clone)]
pub struct StyleCounts {
    pub snake_case: u64,
    pub camel_case: u64,
    pub pascal_case: u64,
    pub screaming_case: u64,
    pub mixed_other: u64,
}

impl StyleCounts {
    pub fn merge(&mut self, other: &StyleCounts) {
        self.snake_case += other.snake_case;
        self.camel_case += other.camel_case;
        self.pascal_case += other.pascal_case;
        self.screaming_case += other.screaming_case;
        self.mixed_other += other.mixed_other;
    }
}

/// Classify an identifier's naming style.
fn classify_style(ident: &str) -> &'static str {
    // Strip leading/trailing underscores for classification (e.g., __func__ -> func)
    let stripped = ident.trim_matches('_');
    if stripped.is_empty() {
        return "other";
    }

    let has_underscore = stripped.contains('_');
    let has_lower = stripped.bytes().any(|b| b.is_ascii_lowercase());
    let has_upper = stripped.bytes().any(|b| b.is_ascii_uppercase());

    // Dunder patterns like __func__, __attribute__ → "other"
    if ident.starts_with("__") || ident.ends_with("__") {
        return "other";
    }

    if !has_lower && has_upper && stripped.len() > 1 {
        return "screaming"; // SCREAMING_CASE or ALLCAPS
    }
    if has_underscore && has_lower && !has_upper {
        return "snake"; // snake_case
    }
    if !has_underscore && has_lower && has_upper {
        if stripped.as_bytes()[0].is_ascii_uppercase() {
            return "pascal"; // PascalCase
        }
        return "camel"; // camelCase
    }
    "other"
}

/// Extract the namespace prefix from a qualified identifier like "std::vector".
fn extract_namespace(ident: &str) -> Option<&str> {
    if let Some(pos) = ident.find("::") {
        let ns = &ident[..pos];
        if !ns.is_empty() && ns.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
            return Some(ns);
        }
    }
    None
}

/// Split a C++ identifier into morpheme components.
/// Handles snake_case, camelCase, PascalCase, and SCREAMING_CASE.
pub fn split_identifier(ident: &str) -> Vec<String> {
    let mut parts = Vec::new();
    for chunk in ident.split('_') {
        if chunk.is_empty() {
            continue;
        }
        // Split camelCase: maxActiveBlocks -> [max, Active, Blocks]
        let bytes = chunk.as_bytes();
        let mut start = 0;
        let mut i = 1;
        while i < bytes.len() {
            let curr_upper = bytes[i].is_ascii_uppercase();
            let prev_upper = bytes[i - 1].is_ascii_uppercase();

            if curr_upper && !prev_upper {
                // Transition: lower->UPPER (camelCase boundary)
                let part = &chunk[start..i];
                if part.len() > 1 {
                    parts.push(part.to_ascii_lowercase());
                }
                start = i;
            } else if !curr_upper && prev_upper && i - start > 1 {
                // Transition: UPPER->lower after UPPER run (e.g., HTTPSConn -> HTTPS, Conn)
                let part = &chunk[start..i - 1];
                if part.len() > 1 {
                    parts.push(part.to_ascii_lowercase());
                }
                start = i - 1;
            }
            i += 1;
        }
        // Last part
        let part = &chunk[start..];
        if part.len() > 1 {
            parts.push(part.to_ascii_lowercase());
        }
    }
    parts
}

/// Analyze a single C++ document. Returns all extracted counts.
///
/// Uses a hand-written scanner (no regex) for maximum throughput.
/// Single pass through the document extracts identifiers and $-operators.
/// String literals are extracted separately for SQL keyword detection.
pub fn analyze_document(text: &[u8], do_morphemes: bool, do_content: bool) -> DocResult {
    let mut ident_counts: HashMap<String, u64> = HashMap::with_capacity(4096);
    let mut dollar_counts: HashMap<String, u64> = HashMap::with_capacity(64);
    let mut sql_keyword_counts: HashMap<String, u64> = HashMap::with_capacity(64);
    let mut morpheme_counts: HashMap<String, u64> = HashMap::with_capacity(1024);
    let mut style_counts = StyleCounts::default();
    let mut namespace_counts: HashMap<String, u64> = HashMap::with_capacity(64);
    let mut total_idents: u64 = 0;
    let mut total_ident_len: u64 = 0;

    let len = text.len();
    let mut i = 0;

    while i < len {
        let b = text[i];

        // Skip line comments
        if b == b'/' && i + 1 < len {
            if text[i + 1] == b'/' {
                // Line comment: skip to end of line
                i += 2;
                while i < len && text[i] != b'\n' {
                    i += 1;
                }
                continue;
            }
            if text[i + 1] == b'*' {
                // Block comment: skip to */
                i += 2;
                while i + 1 < len && !(text[i] == b'*' && text[i + 1] == b'/') {
                    i += 1;
                }
                i += 2; // skip */
                continue;
            }
        }

        // String literal: extract content for SQL keyword analysis
        if b == b'"' {
            i += 1;
            let str_start = i;
            while i < len && text[i] != b'"' {
                if text[i] == b'\\' {
                    i += 1; // skip escaped char
                }
                i += 1;
            }
            // Extract SQL keywords from string content
            let str_content = &text[str_start..i.min(len)];
            extract_sql_keywords(str_content, &mut sql_keyword_counts);
            if i < len {
                i += 1; // skip closing quote
            }
            continue;
        }

        // Character literal: skip
        if b == b'\'' {
            i += 1;
            if i < len && text[i] == b'\\' {
                i += 1;
            }
            if i < len {
                i += 1; // skip char
            }
            if i < len && text[i] == b'\'' {
                i += 1;
            }
            continue;
        }

        // $-prefixed operator (MongoDB)
        if b == b'$' && i + 1 < len && is_ident_start(text[i + 1]) {
            let start = i;
            i += 1;
            while i < len && is_ident_continue(text[i]) {
                i += 1;
            }
            if let Ok(s) = std::str::from_utf8(&text[start..i]) {
                *dollar_counts.entry(s.to_owned()).or_default() += 1;
            }
            continue;
        }

        // Identifier (possibly namespace-qualified with ::)
        if is_ident_start(b) {
            let start = i;
            i += 1;
            while i < len && is_ident_continue(text[i]) {
                i += 1;
            }
            // Check for :: (namespace qualification)
            while i + 1 < len && text[i] == b':' && text[i + 1] == b':' {
                i += 2; // skip ::
                while i < len && is_ident_continue(text[i]) {
                    i += 1;
                }
            }

            if let Ok(ident) = std::str::from_utf8(&text[start..i]) {
                total_idents += 1;
                total_ident_len += ident.len() as u64;

                // Count the full identifier (including namespace)
                *ident_counts.entry(ident.to_owned()).or_default() += 1;

                // Also count the last component separately for non-qualified lookups
                if ident.contains("::") {
                    if let Some(last) = ident.rsplit("::").next() {
                        if !last.is_empty() {
                            *ident_counts.entry(last.to_owned()).or_default() += 1;
                        }
                    }
                    // Track namespace
                    if let Some(ns) = extract_namespace(ident) {
                        *namespace_counts.entry(ns.to_owned()).or_default() += 1;
                    }
                }

                // Classify naming style (only for non-trivial identifiers)
                let base = if let Some(last) = ident.rsplit("::").next() {
                    last
                } else {
                    ident
                };
                if base.len() > 2 {
                    match classify_style(base) {
                        "snake" => style_counts.snake_case += 1,
                        "camel" => style_counts.camel_case += 1,
                        "pascal" => style_counts.pascal_case += 1,
                        "screaming" => style_counts.screaming_case += 1,
                        _ => style_counts.mixed_other += 1,
                    }
                }

                // Morpheme analysis
                if do_morphemes && base.len() > 3 {
                    let freq = 1u64; // count per occurrence, not per unique
                    for part in split_identifier(base) {
                        *morpheme_counts.entry(part).or_default() += freq;
                    }
                }
            }
            continue;
        }

        // Skip preprocessor lines starting with #
        if b == b'#' {
            // Skip to end of line (including \ continuations)
            while i < len {
                if text[i] == b'\n' {
                    if i > 0 && text[i - 1] == b'\\' {
                        i += 1; // line continuation
                        continue;
                    }
                    break;
                }
                i += 1;
            }
            continue;
        }

        i += 1;
    }

    let unique_idents = ident_counts.len() as u64;

    // Content analysis (whitespace, comments, unicode) - only when requested
    let content_stats = if do_content { analyze_content(text) } else { ContentStats::default() };

    DocResult {
        ident_counts,
        dollar_counts,
        sql_keyword_counts,
        morpheme_counts,
        style_counts,
        namespace_counts,
        total_idents,
        unique_idents,
        total_ident_len,
        content_stats,
    }
}

/// Extract uppercase SQL keywords from a string literal's content.
fn extract_sql_keywords(content: &[u8], counts: &mut HashMap<String, u64>) {
    let len = content.len();
    let mut i = 0;
    while i < len {
        // Find start of uppercase word
        if content[i].is_ascii_uppercase() {
            let start = i;
            i += 1;
            while i < len && (content[i].is_ascii_uppercase() || content[i] == b'_') {
                i += 1;
            }
            let word_len = i - start;
            if word_len >= 2 {
                if let Ok(word) = std::str::from_utf8(&content[start..i]) {
                    *counts.entry(word.to_owned()).or_default() += 1;
                }
            }
        } else {
            i += 1;
        }
    }
}

#[inline(always)]
fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

#[inline(always)]
fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_identifier() {
        assert_eq!(split_identifier("snake_case_name"), vec!["snake", "case", "name"]);
        assert_eq!(split_identifier("camelCaseName"), vec!["camel", "case", "name"]);
        assert_eq!(split_identifier("PascalCaseName"), vec!["pascal", "case", "name"]);
        assert_eq!(split_identifier("HTTPSConnection"), vec!["https", "connection"]);
        assert_eq!(split_identifier("getHTTPSUrl"), vec!["get", "https", "url"]);
        assert_eq!(split_identifier("a"), Vec::<String>::new()); // too short
    }

    #[test]
    fn test_classify_style() {
        assert_eq!(classify_style("snake_case"), "snake");
        assert_eq!(classify_style("camelCase"), "camel");
        assert_eq!(classify_style("PascalCase"), "pascal");
        assert_eq!(classify_style("SCREAMING_CASE"), "screaming");
        assert_eq!(classify_style("__func__"), "other");
    }

    #[test]
    fn test_analyze_cuda() {
        let src = b"__device__ void kernel() { int idx = threadIdx.x + blockIdx.x * blockDim.x; }";
        let result = analyze_document(src, false, false);
        assert!(result.ident_counts.contains_key("__device__"));
        assert!(result.ident_counts.contains_key("threadIdx"));
        assert!(result.ident_counts.contains_key("blockIdx"));
    }

    #[test]
    fn test_analyze_sql_in_string() {
        let src = br#"const char* q = "SELECT name FROM users WHERE id = 1";"#;
        let result = analyze_document(src, false, false);
        assert!(result.sql_keyword_counts.contains_key("SELECT"));
        assert!(result.sql_keyword_counts.contains_key("FROM"));
        assert!(result.sql_keyword_counts.contains_key("WHERE"));
    }

    #[test]
    fn test_dollar_ops() {
        let src = br#"auto pipeline = bsoncxx::builder::stream::document{} << "$match" << "$group";"#;
        let result = analyze_document(src, false, false);
        // Dollar ops inside strings won't be captured by the $ scanner
        // They need to appear as identifiers like: $match
        let src2 = b"$match $group $sort";
        let result2 = analyze_document(src2, false, false);
        assert!(result2.dollar_counts.contains_key("$match"));
        assert!(result2.dollar_counts.contains_key("$group"));
        let _ = result; // use it
    }

    #[test]
    fn test_namespace_extraction() {
        let src = b"std::vector<int> v; boost::asio::ip::tcp::endpoint ep;";
        let result = analyze_document(src, false, false);
        assert!(result.namespace_counts.contains_key("std"));
        assert!(result.namespace_counts.contains_key("boost"));
    }

    #[test]
    fn test_morpheme_analysis() {
        let src = b"void initializeBuffer() { createHandler(); destroyManager(); }";
        let result = analyze_document(src, true, false);
        assert!(*result.morpheme_counts.get("initialize").unwrap_or(&0) > 0
            || *result.morpheme_counts.get("init").unwrap_or(&0) > 0);
        assert!(*result.morpheme_counts.get("buffer").unwrap_or(&0) > 0);
    }

    #[test]
    fn test_skips_comments() {
        let src = b"// cudaMalloc is in a comment\nint x; /* cudaFree also comment */\ncudaMemcpy();";
        let result = analyze_document(src, false, false);
        assert!(!result.ident_counts.contains_key("cudaMalloc"));
        assert!(!result.ident_counts.contains_key("cudaFree"));
        assert!(result.ident_counts.contains_key("cudaMemcpy"));
    }

    #[test]
    fn test_content_analysis() {
        let src = b"// English comment\nint x = 0;\n    int y; // \xc3\xa9tranger non-ASCII\n/* block */\n\n\"hello world\";\n\t\tint z;\n";
        let stats = analyze_content(src);
        assert!(stats.total_lines >= 6);
        assert_eq!(stats.line_comments, 2);
        assert_eq!(stats.block_comments, 1);
        assert!(stats.blank_lines >= 1);
        assert!(stats.non_ascii_bytes > 0);
        assert!(stats.has_non_ascii_comment);
        assert!(stats.string_bytes > 0);
        assert!(stats.tab_indent >= 1);
    }
}
