/// Unicode-to-ASCII substitution map for common typographic characters.
/// Returns None if the character has no simple ASCII equivalent.
pub fn unicode_to_ascii(c: char) -> Option<&'static str> {
    match c {
        // Dashes
        '\u{2013}' => Some("-"),        // en-dash –
        '\u{2014}' => Some("--"),       // em-dash —
        '\u{2015}' => Some("--"),       // horizontal bar ―
        '\u{2212}' => Some("-"),        // minus sign −

        // Quotes
        '\u{2018}' => Some("'"),        // left single quote '
        '\u{2019}' => Some("'"),        // right single quote '
        '\u{201A}' => Some(","),        // single low-9 quote ‚
        '\u{201C}' => Some("\""),       // left double quote "
        '\u{201D}' => Some("\""),       // right double quote "
        '\u{201E}' => Some("\""),       // double low-9 quote „
        '\u{00AB}' => Some("\""),       // left guillemet «
        '\u{00BB}' => Some("\""),       // right guillemet »

        // Dots and ellipsis
        '\u{2026}' => Some("..."),      // ellipsis …
        '\u{00B7}' => Some("*"),        // middle dot ·
        '\u{2022}' => Some("*"),        // bullet •
        '\u{2023}' => Some(">"),        // triangular bullet ‣
        '\u{25CF}' => Some("*"),        // black circle ●
        '\u{25CB}' => Some("o"),        // white circle ○

        // Math symbols
        '\u{00D7}' => Some("x"),        // multiplication ×
        '\u{00F7}' => Some("/"),        // division ÷
        '\u{00B2}' => Some("^2"),       // superscript 2 ²
        '\u{00B3}' => Some("^3"),       // superscript 3 ³
        '\u{00B9}' => Some("^1"),       // superscript 1 ¹
        '\u{2260}' => Some("!="),       // not equal ≠
        '\u{2264}' => Some("<="),       // less or equal ≤
        '\u{2265}' => Some(">="),       // greater or equal ≥
        '\u{221E}' => Some("inf"),      // infinity ∞
        '\u{00B1}' => Some("+/-"),      // plus-minus ±
        '\u{2248}' => Some("~="),       // approximately ≈
        '\u{2192}' => Some("->"),       // right arrow →
        '\u{2190}' => Some("<-"),       // left arrow ←
        '\u{2191}' => Some("^"),        // up arrow ↑
        '\u{2193}' => Some("v"),        // down arrow ↓
        '\u{21D2}' => Some("=>"),       // double right arrow ⇒

        // Japanese/CJK punctuation
        '\u{3002}' => Some("."),        // ideographic period 。
        '\u{FF0E}' => Some("."),        // fullwidth period ．
        '\u{FF61}' => Some("."),        // halfwidth period ｡
        '\u{3001}' => Some(","),        // ideographic comma 、
        '\u{FF0C}' => Some(","),        // fullwidth comma ，
        '\u{FF1A}' => Some(":"),        // fullwidth colon ：
        '\u{FF1B}' => Some(";"),        // fullwidth semicolon ；
        '\u{FF01}' => Some("!"),        // fullwidth exclamation ！
        '\u{FF1F}' => Some("?"),        // fullwidth question ？
        '\u{FF08}' => Some("("),        // fullwidth left paren （
        '\u{FF09}' => Some(")"),        // fullwidth right paren ）
        '\u{FF3B}' => Some("["),        // fullwidth left bracket ［
        '\u{FF3D}' => Some("]"),        // fullwidth right bracket ］
        '\u{3010}' => Some("["),        // left black lenticular bracket 【
        '\u{3011}' => Some("]"),        // right black lenticular bracket 】

        // Spaces
        '\u{00A0}' => Some(" "),        // non-breaking space
        '\u{2002}' => Some(" "),        // en space
        '\u{2003}' => Some(" "),        // em space
        '\u{2009}' => Some(" "),        // thin space
        '\u{200B}' => Some(""),         // zero-width space
        '\u{FEFF}' => Some(""),         // BOM / zero-width no-break space

        // Copyright and trademark
        '\u{00A9}' => Some("(c)"),      // copyright ©
        '\u{00AE}' => Some("(R)"),      // registered ®
        '\u{2122}' => Some("(TM)"),     // trademark ™

        // Misc
        '\u{00B0}' => Some("deg"),      // degree °
        '\u{00B5}' => Some("u"),        // micro µ

        _ => None,
    }
}

/// Try to produce an ASCII replacement for a comment's content.
/// Returns Some(replacement) if all non-ASCII chars can be substituted.
/// Returns None if any non-ASCII char has no known ASCII mapping (needs translation).
pub fn try_ascii_replace(content: &str) -> Option<String> {
    let mut result = String::with_capacity(content.len());
    for c in content.chars() {
        if c.is_ascii() {
            result.push(c);
        } else if let Some(replacement) = unicode_to_ascii(c) {
            result.push_str(replacement);
        } else {
            return None; // Unknown non-ASCII char, needs translation
        }
    }
    Some(result)
}

/// Classify a comment: "symbol_replace" if all non-ASCII can be mapped to ASCII,
/// "translate" if it contains characters that need actual translation.
pub fn classify_comment(content: &str) -> (&'static str, Option<String>) {
    let non_ascii_count = content.chars().filter(|c| !c.is_ascii()).count();
    if non_ascii_count == 0 {
        return ("symbol_replace", Some(content.to_string()));
    }

    // Try ASCII replacement first
    if let Some(replacement) = try_ascii_replace(content) {
        return ("symbol_replace", Some(replacement));
    }

    ("translate", None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_replace() {
        let (class, repl) = classify_comment(" 2^32 \u{2013} 1");
        assert_eq!(class, "symbol_replace");
        assert_eq!(repl.unwrap(), " 2^32 - 1");
    }

    #[test]
    fn test_smart_quotes() {
        let (class, repl) = classify_comment(" it\u{2019}s a test");
        assert_eq!(class, "symbol_replace");
        assert_eq!(repl.unwrap(), " it's a test");
    }

    #[test]
    fn test_japanese_needs_translate() {
        let (class, _) = classify_comment(" \u{30C6}\u{30B9}\u{30C8}"); // テスト
        assert_eq!(class, "translate");
    }

    #[test]
    fn test_chinese_needs_translate() {
        let (class, _) = classify_comment(" \u{6D4B}\u{8BD5}"); // 测试
        assert_eq!(class, "translate");
    }

    #[test]
    fn test_math_symbols() {
        let (class, repl) = classify_comment(" x \u{00D7} y \u{2264} z");
        assert_eq!(class, "symbol_replace");
        assert_eq!(repl.unwrap(), " x x y <= z");
    }

    #[test]
    fn test_pure_ascii() {
        let (class, repl) = classify_comment(" normal comment");
        assert_eq!(class, "symbol_replace");
        assert_eq!(repl.unwrap(), " normal comment");
    }

    #[test]
    fn test_fullwidth_punctuation() {
        let (class, repl) = classify_comment("\u{FF08}test\u{FF09}");
        assert_eq!(class, "symbol_replace");
        assert_eq!(repl.unwrap(), "(test)");
    }
}
