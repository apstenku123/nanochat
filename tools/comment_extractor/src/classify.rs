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
        '\u{00A7}' => Some("S"),        // section §
        '\u{00B6}' => Some("P"),        // pilcrow ¶
        '\u{00AC}' => Some("!"),        // not ¬
        '\u{00BC}' => Some("1/4"),      // ¼
        '\u{00BD}' => Some("1/2"),      // ½
        '\u{00BE}' => Some("3/4"),      // ¾
        '\u{2030}' => Some("%%"),       // per mille ‰
        '\u{20AC}' => Some("EUR"),      // euro €
        '\u{00A3}' => Some("GBP"),      // pound £
        '\u{00A5}' => Some("JPY"),      // yen ¥

        // Latin diacritics — transliterate to ASCII
        '\u{00C0}' => Some("A"),        // À
        '\u{00C1}' => Some("A"),        // Á
        '\u{00C2}' => Some("A"),        // Â
        '\u{00C3}' => Some("A"),        // Ã
        '\u{00C4}' => Some("A"),        // Ä
        '\u{00C5}' => Some("A"),        // Å
        '\u{00C6}' => Some("AE"),       // Æ
        '\u{00C7}' => Some("C"),        // Ç
        '\u{00C8}' => Some("E"),        // È
        '\u{00C9}' => Some("E"),        // É
        '\u{00CA}' => Some("E"),        // Ê
        '\u{00CB}' => Some("E"),        // Ë
        '\u{00CC}' => Some("I"),        // Ì
        '\u{00CD}' => Some("I"),        // Í
        '\u{00CE}' => Some("I"),        // Î
        '\u{00CF}' => Some("I"),        // Ï
        '\u{00D0}' => Some("D"),        // Ð
        '\u{00D1}' => Some("N"),        // Ñ
        '\u{00D2}' => Some("O"),        // Ò
        '\u{00D3}' => Some("O"),        // Ó
        '\u{00D4}' => Some("O"),        // Ô
        '\u{00D5}' => Some("O"),        // Õ
        '\u{00D6}' => Some("O"),        // Ö
        '\u{00D8}' => Some("O"),        // Ø
        '\u{00D9}' => Some("U"),        // Ù
        '\u{00DA}' => Some("U"),        // Ú
        '\u{00DB}' => Some("U"),        // Û
        '\u{00DC}' => Some("U"),        // Ü
        '\u{00DD}' => Some("Y"),        // Ý
        '\u{00DE}' => Some("Th"),       // Þ
        '\u{00DF}' => Some("ss"),       // ß
        '\u{00E0}' => Some("a"),        // à
        '\u{00E1}' => Some("a"),        // á
        '\u{00E2}' => Some("a"),        // â
        '\u{00E3}' => Some("a"),        // ã
        '\u{00E4}' => Some("a"),        // ä
        '\u{00E5}' => Some("a"),        // å
        '\u{00E6}' => Some("ae"),       // æ
        '\u{00E7}' => Some("c"),        // ç
        '\u{00E8}' => Some("e"),        // è
        '\u{00E9}' => Some("e"),        // é
        '\u{00EA}' => Some("e"),        // ê
        '\u{00EB}' => Some("e"),        // ë
        '\u{00EC}' => Some("i"),        // ì
        '\u{00ED}' => Some("i"),        // í
        '\u{00EE}' => Some("i"),        // î
        '\u{00EF}' => Some("i"),        // ï
        '\u{00F0}' => Some("d"),        // ð
        '\u{00F1}' => Some("n"),        // ñ
        '\u{00F2}' => Some("o"),        // ò
        '\u{00F3}' => Some("o"),        // ó
        '\u{00F4}' => Some("o"),        // ô
        '\u{00F5}' => Some("o"),        // õ
        '\u{00F6}' => Some("o"),        // ö
        '\u{00F8}' => Some("o"),        // ø
        '\u{00F9}' => Some("u"),        // ù
        '\u{00FA}' => Some("u"),        // ú
        '\u{00FB}' => Some("u"),        // û
        '\u{00FC}' => Some("u"),        // ü
        '\u{00FD}' => Some("y"),        // ý
        '\u{00FE}' => Some("th"),       // þ
        '\u{00FF}' => Some("y"),        // ÿ

        // Extended Latin (common in European names)
        '\u{0100}' => Some("A"),        // Ā
        '\u{0101}' => Some("a"),        // ā
        '\u{0102}' => Some("A"),        // Ă
        '\u{0103}' => Some("a"),        // ă
        '\u{0104}' => Some("A"),        // Ą
        '\u{0105}' => Some("a"),        // ą
        '\u{0106}' => Some("C"),        // Ć
        '\u{0107}' => Some("c"),        // ć
        '\u{010C}' => Some("C"),        // Č
        '\u{010D}' => Some("c"),        // č
        '\u{010E}' => Some("D"),        // Ď
        '\u{010F}' => Some("d"),        // ď
        '\u{0110}' => Some("D"),        // Đ
        '\u{0111}' => Some("d"),        // đ
        '\u{0112}' => Some("E"),        // Ē
        '\u{0113}' => Some("e"),        // ē
        '\u{0118}' => Some("E"),        // Ę
        '\u{0119}' => Some("e"),        // ę
        '\u{011A}' => Some("E"),        // Ě
        '\u{011B}' => Some("e"),        // ě
        '\u{011E}' => Some("G"),        // Ğ
        '\u{011F}' => Some("g"),        // ğ
        '\u{0130}' => Some("I"),        // İ
        '\u{0131}' => Some("i"),        // ı
        '\u{0141}' => Some("L"),        // Ł
        '\u{0142}' => Some("l"),        // ł
        '\u{0143}' => Some("N"),        // Ń
        '\u{0144}' => Some("n"),        // ń
        '\u{0147}' => Some("N"),        // Ň
        '\u{0148}' => Some("n"),        // ň
        '\u{0150}' => Some("O"),        // Ő
        '\u{0151}' => Some("o"),        // ő
        '\u{0152}' => Some("OE"),       // Œ
        '\u{0153}' => Some("oe"),       // œ
        '\u{0158}' => Some("R"),        // Ř
        '\u{0159}' => Some("r"),        // ř
        '\u{015A}' => Some("S"),        // Ś
        '\u{015B}' => Some("s"),        // ś
        '\u{015E}' => Some("S"),        // Ş
        '\u{015F}' => Some("s"),        // ş
        '\u{0160}' => Some("S"),        // Š
        '\u{0161}' => Some("s"),        // š
        '\u{0162}' => Some("T"),        // Ţ
        '\u{0163}' => Some("t"),        // ţ
        '\u{0164}' => Some("T"),        // Ť
        '\u{0165}' => Some("t"),        // ť
        '\u{016E}' => Some("U"),        // Ů
        '\u{016F}' => Some("u"),        // ů
        '\u{0170}' => Some("U"),        // Ű
        '\u{0171}' => Some("u"),        // ű
        '\u{017A}' => Some("z"),        // ź
        '\u{017B}' => Some("Z"),        // Ż
        '\u{017C}' => Some("z"),        // ż
        '\u{017D}' => Some("Z"),        // Ž
        '\u{017E}' => Some("z"),        // ž
        '\u{0179}' => Some("Z"),        // Ź

        // Box drawing (common in ASCII art diagrams)
        '\u{2500}' => Some("-"),        // ─ horizontal
        '\u{2502}' => Some("|"),        // │ vertical
        '\u{250C}' => Some("+"),        // ┌ top-left
        '\u{2510}' => Some("+"),        // ┐ top-right
        '\u{2514}' => Some("+"),        // └ bottom-left
        '\u{2518}' => Some("+"),        // ┘ bottom-right
        '\u{251C}' => Some("+"),        // ├ left tee
        '\u{2524}' => Some("+"),        // ┤ right tee
        '\u{252C}' => Some("+"),        // ┬ top tee
        '\u{2534}' => Some("+"),        // ┴ bottom tee
        '\u{253C}' => Some("+"),        // ┼ cross
        '\u{2550}' => Some("="),        // ═ double horizontal
        '\u{2551}' => Some("|"),        // ║ double vertical

        // Math set/logic symbols
        '\u{2208}' => Some("in"),       // ∈ element of
        '\u{2209}' => Some("!in"),      // ∉ not element of
        '\u{2282}' => Some("C"),        // ⊂ subset
        '\u{2286}' => Some("C="),       // ⊆ subset or equal
        '\u{2229}' => Some("&"),        // ∩ intersection
        '\u{222A}' => Some("|"),        // ∪ union
        '\u{2200}' => Some("forall"),   // ∀ for all
        '\u{2203}' => Some("exists"),   // ∃ there exists
        '\u{2227}' => Some("&&"),       // ∧ logical and
        '\u{2228}' => Some("||"),       // ∨ logical or
        '\u{230A}' => Some("floor("),   // ⌊ left floor
        '\u{230B}' => Some(")"),        // ⌋ right floor
        '\u{2308}' => Some("ceil("),    // ⌈ left ceiling
        '\u{2309}' => Some(")"),        // ⌉ right ceiling
        '\u{211D}' => Some("R"),        // ℝ real numbers
        '\u{2124}' => Some("Z"),        // ℤ integers
        '\u{2115}' => Some("N"),        // ℕ natural numbers
        '\u{2102}' => Some("C"),        // ℂ complex numbers
        '\u{1D53D}' => Some("F"),       // 𝔽 math double-struck F (finite field)
        '\u{2205}' => Some("{}"),       // ∅ empty set
        '\u{221A}' => Some("sqrt"),     // √ square root
        '\u{2211}' => Some("sum"),      // ∑ summation
        '\u{220F}' => Some("prod"),     // ∏ product
        '\u{222B}' => Some("int"),      // ∫ integral
        '\u{2202}' => Some("d"),        // ∂ partial derivative
        '\u{2207}' => Some("nabla"),    // ∇ nabla/del
        '\u{2297}' => Some("(x)"),      // ⊗ tensor product
        '\u{2295}' => Some("(+)"),      // ⊕ direct sum/xor

        // Additional misc symbols
        '\u{FFFD}' => Some("?"),        // � replacement character
        '\u{202F}' => Some(" "),        // narrow no-break space
        '\u{2010}' => Some("-"),        // ‐ hyphen
        '\u{2011}' => Some("-"),        // non-breaking hyphen
        '\u{00A6}' => Some("|"),        // ¦ broken bar
        '\u{00A1}' => Some("!"),        // ¡ inverted exclamation
        '\u{00BF}' => Some("?"),        // ¿ inverted question
        '\u{00AF}' => Some("-"),        // ¯ macron
        '\u{00B4}' => Some("'"),        // ´ acute accent
        '\u{23BD}' => Some("_"),        // ⎽ horizontal scan line
        '\u{00B8}' => Some(","),        // ¸ cedilla
        '\u{02C6}' => Some("^"),        // ˆ modifier circumflex
        '\u{02DC}' => Some("~"),        // ˜ small tilde
        '\u{200C}' => Some(""),         // zero-width non-joiner
        '\u{200D}' => Some(""),         // zero-width joiner
        '\u{2039}' => Some("<"),        // ‹ single left guillemet
        '\u{203A}' => Some(">"),        // › single right guillemet

        // Superscript/subscript digits
        '\u{2070}' => Some("^0"),       // ⁰
        '\u{2074}' => Some("^4"),       // ⁴
        '\u{2075}' => Some("^5"),       // ⁵
        '\u{2076}' => Some("^6"),       // ⁶
        '\u{2077}' => Some("^7"),       // ⁷
        '\u{2078}' => Some("^8"),       // ⁸
        '\u{2079}' => Some("^9"),       // ⁹
        '\u{207F}' => Some("^n"),       // ⁿ
        '\u{2080}' => Some("_0"),       // ₀
        '\u{2081}' => Some("_1"),       // ₁
        '\u{2082}' => Some("_2"),       // ₂
        '\u{2083}' => Some("_3"),       // ₃
        '\u{2084}' => Some("_4"),       // ₄

        // Greek letters (common in math/science comments)
        '\u{0391}' => Some("Alpha"),    // Α
        '\u{0392}' => Some("Beta"),     // Β
        '\u{0393}' => Some("Gamma"),    // Γ
        '\u{0394}' => Some("Delta"),    // Δ
        '\u{0395}' => Some("Epsilon"),  // Ε
        '\u{0396}' => Some("Zeta"),     // Ζ
        '\u{0397}' => Some("Eta"),      // Η
        '\u{0398}' => Some("Theta"),    // Θ
        '\u{0399}' => Some("Iota"),     // Ι
        '\u{039A}' => Some("Kappa"),    // Κ
        '\u{039B}' => Some("Lambda"),   // Λ
        '\u{039C}' => Some("Mu"),       // Μ
        '\u{039D}' => Some("Nu"),       // Ν
        '\u{039E}' => Some("Xi"),       // Ξ
        '\u{039F}' => Some("Omicron"),  // Ο
        '\u{03A0}' => Some("Pi"),       // Π
        '\u{03A1}' => Some("Rho"),      // Ρ
        '\u{03A3}' => Some("Sigma"),    // Σ
        '\u{03A4}' => Some("Tau"),      // Τ
        '\u{03A5}' => Some("Upsilon"),  // Υ
        '\u{03A6}' => Some("Phi"),      // Φ
        '\u{03A7}' => Some("Chi"),      // Χ
        '\u{03A8}' => Some("Psi"),      // Ψ
        '\u{03A9}' => Some("Omega"),    // Ω
        '\u{03B1}' => Some("alpha"),    // α
        '\u{03B2}' => Some("beta"),     // β
        '\u{03B3}' => Some("gamma"),    // γ
        '\u{03B4}' => Some("delta"),    // δ
        '\u{03B5}' => Some("epsilon"),  // ε
        '\u{03B6}' => Some("zeta"),     // ζ
        '\u{03B7}' => Some("eta"),      // η
        '\u{03B8}' => Some("theta"),    // θ
        '\u{03B9}' => Some("iota"),     // ι
        '\u{03BA}' => Some("kappa"),    // κ
        '\u{03BB}' => Some("lambda"),   // λ
        '\u{03BC}' => Some("mu"),       // μ
        '\u{03BD}' => Some("nu"),       // ν
        '\u{03BE}' => Some("xi"),       // ξ
        '\u{03BF}' => Some("omicron"),  // ο
        '\u{03C0}' => Some("pi"),       // π
        '\u{03C1}' => Some("rho"),      // ρ
        '\u{03C2}' => Some("sigma"),    // ς (final)
        '\u{03C3}' => Some("sigma"),    // σ
        '\u{03C4}' => Some("tau"),      // τ
        '\u{03C5}' => Some("upsilon"),  // υ
        '\u{03C6}' => Some("phi"),      // φ
        '\u{03C7}' => Some("chi"),      // χ
        '\u{03C8}' => Some("psi"),      // ψ
        '\u{03C9}' => Some("omega"),    // ω

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

    #[test]
    fn test_accented_latin_names() {
        // "Author: Morné Chamberlain" — should be symbol_replace, not translate
        let (class, repl) = classify_comment(" Author: Morn\u{00E9} Chamberlain");
        assert_eq!(class, "symbol_replace");
        assert_eq!(repl.unwrap(), " Author: Morne Chamberlain");
    }

    #[test]
    fn test_mixed_latin_diacritics() {
        // Polish name: Michał Łukaszewski
        let (class, repl) = classify_comment(" Micha\u{0142} \u{0141}ukaszewski");
        assert_eq!(class, "symbol_replace");
        assert_eq!(repl.unwrap(), " Michal Lukaszewski");
    }

    #[test]
    fn test_greek_letter_in_math() {
        // "compute α × β" — should be symbol_replace
        let (class, repl) = classify_comment(" compute \u{03B1} \u{00D7} \u{03B2}");
        assert_eq!(class, "symbol_replace");
        assert_eq!(repl.unwrap(), " compute alpha x beta");
    }

    #[test]
    fn test_german_umlauts() {
        let (class, repl) = classify_comment(" \u{00FC}ber die Gr\u{00F6}\u{00DF}e");
        assert_eq!(class, "symbol_replace");
        assert_eq!(repl.unwrap(), " uber die Grosse");
    }
}
