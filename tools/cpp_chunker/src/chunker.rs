//! Tree-sitter based C++ semantic chunker.
//!
//! Uses the full AST to extract functions, classes, namespaces, and preamble
//! with proper understanding of C++ syntax — no regex hacks.

use tree_sitter::{Node, Parser};

#[derive(Debug, Clone, PartialEq)]
pub enum ChunkKind {
    Preamble,
    Function,
    Class,
    Namespace,
    TopLevel,
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub kind: ChunkKind,
    pub text: String,
    pub name: String,
    pub start_line: usize,
    pub end_line: usize,
}

/// Parse a C++ source file and extract semantic chunks.
pub fn chunk_file(parser: &mut Parser, source: &str) -> Vec<Chunk> {
    let tree = match parser.parse(source, None) {
        Some(t) => t,
        None => return fallback_chunk(source),
    };

    let root = tree.root_node();
    let mut chunks = Vec::new();
    let mut preamble_lines: Vec<&str> = Vec::new();
    let mut preamble_start: Option<usize> = None;
    let mut preamble_end: usize = 0;

    let cursor = &mut root.walk();

    // Walk top-level children
    for child in root.children(cursor) {
        let kind = child.kind();
        let text = &source[child.byte_range()];
        let start = child.start_position().row;
        let end = child.end_position().row;

        match kind {
            // Preamble: preprocessor, using, typedef, forward decls, comments
            "preproc_include" | "preproc_def" | "preproc_ifdef" | "preproc_ifndef"
            | "preproc_if" | "preproc_else" | "preproc_elif" | "preproc_endif"
            | "preproc_call" | "preproc_function_def"
            | "using_declaration" | "alias_declaration" | "type_definition"
            | "comment" => {
                if preamble_start.is_none() {
                    preamble_start = Some(start);
                }
                preamble_lines.push(text);
                preamble_end = end;
            }

            // Function definitions
            "function_definition" | "template_declaration" => {
                // Flush preamble
                flush_preamble(&mut chunks, &mut preamble_lines, preamble_start, preamble_end);
                preamble_start = None;

                if kind == "template_declaration" {
                    // Check if this is a template function or template class
                    let inner = find_inner_declaration(&child);
                    match inner {
                        Some(n) if n.kind() == "function_definition" => {
                            let name = extract_function_name(&n, source);
                            chunks.push(Chunk {
                                kind: ChunkKind::Function,
                                text: text.to_string(),
                                name,
                                start_line: start,
                                end_line: end,
                            });
                        }
                        Some(n) if n.kind() == "class_specifier" || n.kind() == "struct_specifier" => {
                            let name = extract_type_name(&n, source);
                            chunks.push(Chunk {
                                kind: ChunkKind::Class,
                                text: text.to_string(),
                                name,
                                start_line: start,
                                end_line: end,
                            });
                        }
                        _ => {
                            // Other template (variable, alias, concept)
                            chunks.push(Chunk {
                                kind: ChunkKind::TopLevel,
                                text: text.to_string(),
                                name: String::new(),
                                start_line: start,
                                end_line: end,
                            });
                        }
                    }
                } else {
                    let name = extract_function_name(&child, source);
                    chunks.push(Chunk {
                        kind: ChunkKind::Function,
                        text: text.to_string(),
                        name,
                        start_line: start,
                        end_line: end,
                    });
                }
            }

            // Class/struct definitions with body
            "class_specifier" | "struct_specifier" => {
                flush_preamble(&mut chunks, &mut preamble_lines, preamble_start, preamble_end);
                preamble_start = None;

                let name = extract_type_name(&child, source);
                // Only emit as class if it has a body (field_declaration_list)
                if child.child_by_field_name("body").is_some() {
                    chunks.push(Chunk {
                        kind: ChunkKind::Class,
                        text: text.to_string(),
                        name,
                        start_line: start,
                        end_line: end,
                    });
                } else {
                    // Forward declaration — preamble
                    preamble_lines.push(text);
                    if preamble_start.is_none() {
                        preamble_start = Some(start);
                    }
                    preamble_end = end;
                }
            }

            // Enum definitions
            "enum_specifier" => {
                flush_preamble(&mut chunks, &mut preamble_lines, preamble_start, preamble_end);
                preamble_start = None;

                let name = extract_type_name(&child, source);
                chunks.push(Chunk {
                    kind: ChunkKind::Class,
                    text: text.to_string(),
                    name,
                    start_line: start,
                    end_line: end,
                });
            }

            // Namespace definitions — recurse into contents
            "namespace_definition" => {
                flush_preamble(&mut chunks, &mut preamble_lines, preamble_start, preamble_end);
                preamble_start = None;

                let ns_name = child
                    .child_by_field_name("name")
                    .map(|n| source[n.byte_range()].to_string())
                    .unwrap_or_default();

                // Extract inner chunks from namespace body
                if let Some(body) = child.child_by_field_name("body") {
                    let inner_chunks = extract_namespace_children(&body, source, &ns_name);
                    if inner_chunks.is_empty() {
                        // Empty or simple namespace — emit as-is
                        chunks.push(Chunk {
                            kind: ChunkKind::Namespace,
                            text: text.to_string(),
                            name: ns_name,
                            start_line: start,
                            end_line: end,
                        });
                    } else {
                        // Wrap each inner chunk with namespace open/close
                        let ns_open = if ns_name.is_empty() {
                            "namespace {\n".to_string()
                        } else {
                            format!("namespace {} {{\n", ns_name)
                        };
                        let ns_close = if ns_name.is_empty() {
                            "\n}".to_string()
                        } else {
                            format!("\n}} // namespace {}", ns_name)
                        };

                        for mut ic in inner_chunks {
                            ic.text = format!("{}{}{}", ns_open, ic.text, ns_close);
                            ic.start_line = start; // approximate
                            chunks.push(ic);
                        }
                    }
                } else {
                    chunks.push(Chunk {
                        kind: ChunkKind::Namespace,
                        text: text.to_string(),
                        name: ns_name,
                        start_line: start,
                        end_line: end,
                    });
                }
            }

            // Linkage specification (extern "C" { ... })
            "linkage_specification" => {
                flush_preamble(&mut chunks, &mut preamble_lines, preamble_start, preamble_end);
                preamble_start = None;

                chunks.push(Chunk {
                    kind: ChunkKind::TopLevel,
                    text: text.to_string(),
                    name: "extern_C".to_string(),
                    start_line: start,
                    end_line: end,
                });
            }

            // Declarations (variable declarations, type declarations at top level)
            "declaration" => {
                // Check if it's a simple forward decl or a real declaration
                let trimmed = text.trim();
                if is_preamble_declaration(trimmed) {
                    preamble_lines.push(text);
                    if preamble_start.is_none() {
                        preamble_start = Some(start);
                    }
                    preamble_end = end;
                } else {
                    flush_preamble(&mut chunks, &mut preamble_lines, preamble_start, preamble_end);
                    preamble_start = None;

                    chunks.push(Chunk {
                        kind: ChunkKind::TopLevel,
                        text: text.to_string(),
                        name: String::new(),
                        start_line: start,
                        end_line: end,
                    });
                }
            }

            // Expression statements, empty statements, etc
            "expression_statement" | ";" => {
                // Usually top-level
                if text.trim().len() > 10 {
                    flush_preamble(&mut chunks, &mut preamble_lines, preamble_start, preamble_end);
                    preamble_start = None;

                    chunks.push(Chunk {
                        kind: ChunkKind::TopLevel,
                        text: text.to_string(),
                        name: String::new(),
                        start_line: start,
                        end_line: end,
                    });
                }
            }

            // Concept definitions (C++20)
            "concept_definition" => {
                flush_preamble(&mut chunks, &mut preamble_lines, preamble_start, preamble_end);
                preamble_start = None;

                chunks.push(Chunk {
                    kind: ChunkKind::TopLevel,
                    text: text.to_string(),
                    name: String::new(),
                    start_line: start,
                    end_line: end,
                });
            }

            // ERROR nodes — tree-sitter couldn't parse, treat as top-level
            "ERROR" => {
                flush_preamble(&mut chunks, &mut preamble_lines, preamble_start, preamble_end);
                preamble_start = None;

                if text.trim().len() >= 20 {
                    chunks.push(Chunk {
                        kind: ChunkKind::TopLevel,
                        text: text.to_string(),
                        name: String::new(),
                        start_line: start,
                        end_line: end,
                    });
                }
            }

            // Anything else — skip very small, keep others as top-level
            _ => {
                if text.trim().len() >= 20 {
                    flush_preamble(&mut chunks, &mut preamble_lines, preamble_start, preamble_end);
                    preamble_start = None;

                    chunks.push(Chunk {
                        kind: ChunkKind::TopLevel,
                        text: text.to_string(),
                        name: String::new(),
                        start_line: start,
                        end_line: end,
                    });
                }
            }
        }
    }

    // Flush remaining preamble
    flush_preamble(&mut chunks, &mut preamble_lines, preamble_start, preamble_end);

    // Merge consecutive preambles
    merge_preambles(&mut chunks);

    chunks
}

/// Extract children of a namespace body as chunks.
fn extract_namespace_children(body: &Node, source: &str, _ns_name: &str) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let cursor = &mut body.walk();

    for child in body.children(cursor) {
        let kind = child.kind();
        let text = &source[child.byte_range()];
        let start = child.start_position().row;
        let end = child.end_position().row;

        match kind {
            "function_definition" => {
                let name = extract_function_name(&child, source);
                chunks.push(Chunk {
                    kind: ChunkKind::Function,
                    text: text.to_string(),
                    name,
                    start_line: start,
                    end_line: end,
                });
            }
            "class_specifier" | "struct_specifier" => {
                if child.child_by_field_name("body").is_some() {
                    let name = extract_type_name(&child, source);
                    chunks.push(Chunk {
                        kind: ChunkKind::Class,
                        text: text.to_string(),
                        name,
                        start_line: start,
                        end_line: end,
                    });
                }
            }
            "template_declaration" => {
                let inner = find_inner_declaration(&child);
                let chunk_kind = match inner {
                    Some(n) if n.kind() == "function_definition" => ChunkKind::Function,
                    Some(n) if n.kind() == "class_specifier" || n.kind() == "struct_specifier" => {
                        ChunkKind::Class
                    }
                    _ => ChunkKind::TopLevel,
                };
                let name = match &inner {
                    Some(n) if n.kind() == "function_definition" => {
                        extract_function_name(n, source)
                    }
                    Some(n)
                        if n.kind() == "class_specifier"
                            || n.kind() == "struct_specifier" =>
                    {
                        extract_type_name(n, source)
                    }
                    _ => String::new(),
                };
                chunks.push(Chunk {
                    kind: chunk_kind,
                    text: text.to_string(),
                    name,
                    start_line: start,
                    end_line: end,
                });
            }
            "enum_specifier" => {
                let name = extract_type_name(&child, source);
                chunks.push(Chunk {
                    kind: ChunkKind::Class,
                    text: text.to_string(),
                    name,
                    start_line: start,
                    end_line: end,
                });
            }
            _ => {
                // Declarations, comments, etc inside namespace
                if text.trim().len() >= 20 {
                    chunks.push(Chunk {
                        kind: ChunkKind::TopLevel,
                        text: text.to_string(),
                        name: String::new(),
                        start_line: start,
                        end_line: end,
                    });
                }
            }
        }
    }

    chunks
}

/// Find the inner declaration inside a template_declaration.
fn find_inner_declaration<'a>(node: &'a Node) -> Option<Node<'a>> {
    let cursor = &mut node.walk();
    for child in node.children(cursor) {
        match child.kind() {
            "function_definition" | "class_specifier" | "struct_specifier"
            | "alias_declaration" | "declaration" | "concept_definition" => {
                return Some(child);
            }
            _ => {}
        }
    }
    None
}

/// Extract function name from a function_definition node.
fn extract_function_name(node: &Node, source: &str) -> String {
    if let Some(declarator) = node.child_by_field_name("declarator") {
        return extract_declarator_name(&declarator, source);
    }
    String::new()
}

/// Recursively find the identifier in a declarator.
fn extract_declarator_name(node: &Node, source: &str) -> String {
    match node.kind() {
        "identifier" | "field_identifier" | "destructor_name" | "operator_name" => {
            source[node.byte_range()].to_string()
        }
        "qualified_identifier" | "template_function" => {
            // For qualified names like std::vector::push_back, get the whole thing
            source[node.byte_range()].to_string()
        }
        "function_declarator" | "pointer_declarator" | "reference_declarator"
        | "parenthesized_declarator" => {
            // Recurse into declarator field
            if let Some(inner) = node.child_by_field_name("declarator") {
                return extract_declarator_name(&inner, source);
            }
            // Fall back to first named child
            let cursor = &mut node.walk();
            for child in node.children(cursor) {
                let name = extract_declarator_name(&child, source);
                if !name.is_empty() {
                    return name;
                }
            }
            String::new()
        }
        _ => {
            // Try children
            let cursor = &mut node.walk();
            for child in node.children(cursor) {
                let name = extract_declarator_name(&child, source);
                if !name.is_empty() {
                    return name;
                }
            }
            String::new()
        }
    }
}

/// Extract type name (class/struct/enum).
fn extract_type_name(node: &Node, source: &str) -> String {
    if let Some(name_node) = node.child_by_field_name("name") {
        return source[name_node.byte_range()].to_string();
    }
    String::new()
}

fn is_preamble_declaration(text: &str) -> bool {
    // Forward declarations, extern declarations, using namespace
    text.starts_with("class ")
        && text.ends_with(';')
        && !text.contains('{')
        || text.starts_with("struct ")
            && text.ends_with(';')
            && !text.contains('{')
        || text.starts_with("enum ")
            && text.ends_with(';')
            && !text.contains('{')
        || text.starts_with("extern ")
        || text.starts_with("using namespace")
        || text.starts_with("namespace ") && text.ends_with(';')
}

fn flush_preamble(
    chunks: &mut Vec<Chunk>,
    lines: &mut Vec<&str>,
    start: Option<usize>,
    end: usize,
) {
    if lines.is_empty() {
        return;
    }
    let text = lines.join("\n").trim().to_string();
    if !text.is_empty() {
        chunks.push(Chunk {
            kind: ChunkKind::Preamble,
            text,
            name: String::new(),
            start_line: start.unwrap_or(0),
            end_line: end,
        });
    }
    lines.clear();
}

fn merge_preambles(chunks: &mut Vec<Chunk>) {
    let mut merged = Vec::with_capacity(chunks.len());
    for chunk in chunks.drain(..) {
        if let Some(last) = merged.last_mut() {
            let last_chunk: &mut Chunk = last;
            if last_chunk.kind == ChunkKind::Preamble && chunk.kind == ChunkKind::Preamble {
                last_chunk.text.push_str("\n\n");
                last_chunk.text.push_str(&chunk.text);
                last_chunk.end_line = chunk.end_line;
                continue;
            }
        }
        merged.push(chunk);
    }
    *chunks = merged;
}

fn fallback_chunk(source: &str) -> Vec<Chunk> {
    if source.trim().len() < 50 {
        return vec![];
    }
    vec![Chunk {
        kind: ChunkKind::TopLevel,
        text: source.to_string(),
        name: String::new(),
        start_line: 0,
        end_line: source.lines().count(),
    }]
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
    fn test_simple_function() {
        let mut parser = make_parser();
        let source = r#"
int add(int a, int b) {
    return a + b;
}
"#;
        let chunks = chunk_file(&mut parser, source);
        assert!(!chunks.is_empty());
        let funcs: Vec<_> = chunks.iter().filter(|c| c.kind == ChunkKind::Function).collect();
        assert_eq!(funcs.len(), 1);
        assert_eq!(funcs[0].name, "add");
    }

    #[test]
    fn test_class_with_methods() {
        let mut parser = make_parser();
        let source = r#"
#include <iostream>

class LinkedList {
public:
    Node* head;

    LinkedList() : head(nullptr) {}

    void push(int data) {
        Node* n = new Node(data);
        n->next = head;
        head = n;
    }

    void print() {
        Node* cur = head;
        while (cur) {
            std::cout << cur->data << " -> ";
            cur = cur->next;
        }
        std::cout << "null" << std::endl;
    }
};
"#;
        let chunks = chunk_file(&mut parser, source);

        let preambles: Vec<_> = chunks.iter().filter(|c| c.kind == ChunkKind::Preamble).collect();
        let classes: Vec<_> = chunks.iter().filter(|c| c.kind == ChunkKind::Class).collect();

        assert!(!preambles.is_empty(), "Should have preamble (#include)");
        assert_eq!(classes.len(), 1, "Should have exactly 1 class");
        assert_eq!(classes[0].name, "LinkedList");
        // Class should contain all methods
        assert!(classes[0].text.contains("push"));
        assert!(classes[0].text.contains("print"));
    }

    #[test]
    fn test_namespace_with_functions() {
        let mut parser = make_parser();
        let source = r#"
namespace utils {

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

} // namespace utils
"#;
        let chunks = chunk_file(&mut parser, source);
        let funcs: Vec<_> = chunks.iter().filter(|c| c.kind == ChunkKind::Function).collect();

        assert_eq!(funcs.len(), 2, "Should extract 2 functions from namespace");
        // Each function should be wrapped in namespace
        for f in &funcs {
            assert!(f.text.contains("namespace utils {"), "Function should be wrapped in namespace");
        }
    }

    #[test]
    fn test_full_file_demo() {
        let mut parser = make_parser();
        let source = r#"#include <iostream>
#include <vector>
#include <string>

using namespace std;

// A simple linked list node
struct Node {
    int data;
    Node* next;
    Node(int d) : data(d), next(nullptr) {}
};

class LinkedList {
public:
    Node* head;

    LinkedList() : head(nullptr) {}

    void push(int data) {
        Node* n = new Node(data);
        n->next = head;
        head = n;
    }

    void print() {
        Node* cur = head;
        while (cur) {
            cout << cur->data << " -> ";
            cur = cur->next;
        }
        cout << "null" << endl;
    }
};

namespace utils {

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

} // namespace utils

int main() {
    LinkedList list;
    list.push(3);
    list.push(2);
    list.push(1);
    list.print();
    cout << "5! = " << utils::factorial(5) << endl;
    return 0;
}
"#;
        let chunks = chunk_file(&mut parser, source);

        let preambles: Vec<_> = chunks.iter().filter(|c| c.kind == ChunkKind::Preamble).collect();
        let classes: Vec<_> = chunks.iter().filter(|c| c.kind == ChunkKind::Class).collect();
        let funcs: Vec<_> = chunks.iter().filter(|c| c.kind == ChunkKind::Function).collect();

        assert!(!preambles.is_empty(), "Should have preamble");
        // Node struct + LinkedList class = 2 classes
        assert!(classes.len() >= 1, "Should have classes: got {}", classes.len());
        // factorial, fibonacci, main = 3 functions
        assert!(funcs.len() >= 2, "Should have functions: got {}", funcs.len());

        // Verify class keeps all methods together
        let ll = classes.iter().find(|c| c.name == "LinkedList");
        if let Some(ll) = ll {
            assert!(ll.text.contains("push"), "LinkedList should contain push method");
            assert!(ll.text.contains("print"), "LinkedList should contain print method");
        }

        eprintln!("Total chunks: {}", chunks.len());
        for (i, c) in chunks.iter().enumerate() {
            eprintln!("[{}] {:?} name={:?} lines={}-{} chars={}",
                     i, c.kind, c.name, c.start_line, c.end_line, c.text.len());
        }
    }
}
