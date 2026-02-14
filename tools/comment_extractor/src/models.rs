use serde::{Deserialize, Serialize};

/// A single extracted comment with byte offsets for precise replacement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommentRecord {
    /// Unique ID: relative_path:byte_start
    pub id: String,
    /// Absolute file path
    pub file_path: String,
    /// Byte offset of comment start (inclusive, points to // or /*)
    pub byte_start: usize,
    /// Byte offset of comment end (exclusive)
    pub byte_end: usize,
    /// "line" or "block"
    pub comment_type: String,
    /// Full original text including comment markers
    pub original_text: String,
    /// Content only (without // or /* */)
    pub content: String,
    /// Number of non-ASCII bytes in content
    pub non_ascii_bytes: usize,
    /// Ratio of non-ASCII bytes to total content bytes
    pub non_ascii_ratio: f64,
    /// "symbol_replace" or "translate"
    pub classification: String,
    /// For symbol_replace: the ASCII replacement text (content only, no markers)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ascii_replacement: Option<String>,
}
