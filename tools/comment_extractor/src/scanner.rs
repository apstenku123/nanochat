use crate::classify::classify_comment;
use crate::models::CommentRecord;

/// Scan a C++ source file for comments containing non-ASCII bytes.
/// Returns only comments that have at least one non-ASCII byte.
///
/// Handles:
/// - `//` line comments
/// - `/* */` block comments
/// - Skips `R"delim(...)delim"` raw string literals
/// - Skips `"..."` string literals and `'...'` char literals
pub fn scan_file(
    source: &[u8],
    file_path: &str,
    relative_id_prefix: &str,
) -> Vec<CommentRecord> {
    let mut comments = Vec::new();
    let len = source.len();
    let mut i = 0;

    while i < len {
        let b = source[i];

        // Line comment
        if b == b'/' && i + 1 < len && source[i + 1] == b'/' {
            let start = i;
            i += 2; // skip //
            let content_start = i;
            while i < len && source[i] != b'\n' {
                i += 1;
            }
            let content_end = i;
            // Check for non-ASCII
            let content_bytes = &source[content_start..content_end];
            let non_ascii = content_bytes.iter().filter(|&&b| b > 127).count();
            if non_ascii > 0 {
                if let Ok(original) = std::str::from_utf8(&source[start..content_end]) {
                    if let Ok(content) = std::str::from_utf8(content_bytes) {
                        let content_len = content.len().max(1);
                        let ratio = non_ascii as f64 / content_len as f64;
                        let (classification, ascii_replacement) = classify_comment(content);

                        comments.push(CommentRecord {
                            id: format!("{}:{}", relative_id_prefix, start),
                            file_path: file_path.to_string(),
                            byte_start: start,
                            byte_end: content_end,
                            comment_type: "line".to_string(),
                            original_text: original.to_string(),
                            content: content.to_string(),
                            non_ascii_bytes: non_ascii,
                            non_ascii_ratio: ratio,
                            classification: classification.to_string(),
                            ascii_replacement,
                        });
                    }
                }
            }
            continue;
        }

        // Block comment
        if b == b'/' && i + 1 < len && source[i + 1] == b'*' {
            let start = i;
            i += 2; // skip /*
            let content_start = i;
            while i + 1 < len && !(source[i] == b'*' && source[i + 1] == b'/') {
                i += 1;
            }
            let content_end = i;
            if i + 1 < len {
                i += 2; // skip */
            }
            let end = i;

            let content_bytes = &source[content_start..content_end];
            let non_ascii = content_bytes.iter().filter(|&&b| b > 127).count();
            if non_ascii > 0 {
                if let Ok(original) = std::str::from_utf8(&source[start..end]) {
                    if let Ok(content) = std::str::from_utf8(content_bytes) {
                        let content_len = content.len().max(1);
                        let ratio = non_ascii as f64 / content_len as f64;
                        let (classification, ascii_replacement) = classify_comment(content);

                        comments.push(CommentRecord {
                            id: format!("{}:{}", relative_id_prefix, start),
                            file_path: file_path.to_string(),
                            byte_start: start,
                            byte_end: end,
                            comment_type: "block".to_string(),
                            original_text: original.to_string(),
                            content: content.to_string(),
                            non_ascii_bytes: non_ascii,
                            non_ascii_ratio: ratio,
                            classification: classification.to_string(),
                            ascii_replacement,
                        });
                    }
                }
            }
            continue;
        }

        // Raw string literal: R"delim(...)delim"
        if b == b'R' && i + 1 < len && source[i + 1] == b'"' {
            i += 2; // skip R"
            // Extract delimiter (everything until '(')
            let delim_start = i;
            while i < len && source[i] != b'(' {
                i += 1;
            }
            let delimiter = &source[delim_start..i];
            if i < len {
                i += 1; // skip (
            }
            // Find matching )delimiter"
            'raw_search: while i < len {
                if source[i] == b')' {
                    let after_paren = i + 1;
                    if after_paren + delimiter.len() < len
                        && &source[after_paren..after_paren + delimiter.len()] == delimiter
                        && source[after_paren + delimiter.len()] == b'"'
                    {
                        i = after_paren + delimiter.len() + 1;
                        break 'raw_search;
                    }
                }
                i += 1;
            }
            continue;
        }

        // String literal
        if b == b'"' {
            i += 1;
            while i < len && source[i] != b'"' {
                if source[i] == b'\\' && i + 1 < len {
                    i += 1; // skip escaped char
                }
                i += 1;
            }
            if i < len {
                i += 1; // skip closing "
            }
            continue;
        }

        // Char literal
        if b == b'\'' {
            i += 1;
            if i < len && source[i] == b'\\' && i + 1 < len {
                i += 1; // skip escape
            }
            if i < len {
                i += 1; // skip char
            }
            if i < len && source[i] == b'\'' {
                i += 1; // skip closing '
            }
            continue;
        }

        i += 1;
    }

    comments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_comment_with_unicode() {
        let src = b"int x; // \xc3\xa9tranger\nint y;";
        let comments = scan_file(src, "/test.cpp", "test.cpp");
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].comment_type, "line");
        assert_eq!(comments[0].byte_start, 7);
        assert!(comments[0].byte_end > comments[0].byte_start);
    }

    #[test]
    fn test_block_comment_with_unicode() {
        let src = "int x; /* \u{00D7} multiply */ int y;".as_bytes();
        let comments = scan_file(src, "/test.cpp", "test.cpp");
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].comment_type, "block");
        assert_eq!(comments[0].classification, "symbol_replace");
    }

    #[test]
    fn test_raw_string_not_scanned() {
        // The // inside the raw string should NOT be detected as a comment
        let src = b"auto s = R\"SQL(\n// not a comment\n)SQL\";\n// \xc3\xa9 real comment\n";
        let comments = scan_file(src, "/test.cpp", "test.cpp");
        assert_eq!(comments.len(), 1);
        assert!(comments[0].content.contains("\u{00e9}"));
    }

    #[test]
    fn test_string_literal_skipped() {
        // The // inside the string should NOT be detected as a comment
        let src = b"auto s = \"// not a comment \\xc3\\xa9\";\n// \xc3\xa9 real\n";
        let comments = scan_file(src, "/test.cpp", "test.cpp");
        assert_eq!(comments.len(), 1);
    }

    #[test]
    fn test_byte_offset_precision() {
        let src = "int x = 0; // \u{2013} dash\n".as_bytes();
        let comments = scan_file(src, "/test.cpp", "test.cpp");
        assert_eq!(comments.len(), 1);
        let c = &comments[0];
        // Verify the original text matches the bytes at the recorded offsets
        let extracted = std::str::from_utf8(&src[c.byte_start..c.byte_end]).unwrap();
        assert_eq!(extracted, c.original_text);
    }

    #[test]
    fn test_classify_cjk_as_translate() {
        // Japanese katakana テスト
        let src = "// \u{30C6}\u{30B9}\u{30C8}\n".as_bytes();
        let comments = scan_file(src, "/test.cpp", "test.cpp");
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].classification, "translate");
    }

    #[test]
    fn test_classify_symbol_as_replace() {
        // En-dash in English comment: 0–100
        let src = "// range 0\u{2013}100\n".as_bytes();
        let comments = scan_file(src, "/test.cpp", "test.cpp");
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].classification, "symbol_replace");
        assert!(comments[0].ascii_replacement.as_ref().unwrap().contains("0-100"));
    }

    #[test]
    fn test_no_false_positives() {
        let src = b"int x = 42; // plain ascii comment\nvoid foo() {}\n";
        let comments = scan_file(src, "/test.cpp", "test.cpp");
        assert_eq!(comments.len(), 0); // No non-ASCII -> no records
    }

    #[test]
    fn test_multiple_comments() {
        let src = "// \u{2013} first\n// normal\n// \u{2013} second\n".as_bytes();
        let comments = scan_file(src, "/test.cpp", "test.cpp");
        assert_eq!(comments.len(), 2); // Only non-ASCII ones
    }
}
