use clap::Parser as ClapParser;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::{self, BufRead, BufWriter, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

mod chunker;
mod deps;
use chunker::{Chunk, ChunkKind};

#[derive(ClapParser)]
#[command(name = "cpp-chunker", about = "Syntax-aware C++ chunker using tree-sitter")]
struct Args {
    /// Input JSONL files
    #[arg(long, num_args = 1..)]
    inputs: Vec<String>,

    /// Output JSONL path
    #[arg(long)]
    output: String,

    /// Target max tokens per document
    #[arg(long, default_value = "1024")]
    max_tokens: usize,

    /// Max records per input file (0 = all)
    #[arg(long, default_value = "0")]
    max_records: usize,

    /// Number of threads (0 = auto)
    #[arg(long, default_value = "0")]
    threads: usize,

    /// Batch size for parallel processing
    #[arg(long, default_value = "1000")]
    batch_size: usize,

    /// Enable dependency-aware chunking (bottom-up ordering with call graph analysis)
    #[arg(long, default_value = "false")]
    dep_aware: bool,
}

#[derive(Deserialize)]
struct InputRecord {
    text: String,
}

#[derive(Serialize)]
struct OutputRecord<'a> {
    text: &'a str,
}

fn estimate_tokens(text: &str) -> usize {
    std::cmp::max(1, text.len() / 4)
}

/// Split oversized text into max_tokens-sized pieces at blank line boundaries.
fn split_oversized(text: &str, max_tokens: usize) -> Vec<String> {
    let total = estimate_tokens(text);
    if total <= max_tokens {
        return vec![text.to_string()];
    }

    let max_chars = max_tokens * 4;
    let mut pieces = Vec::new();
    let mut current = String::new();

    for line in text.lines() {
        if current.len() + line.len() + 1 > max_chars && !current.is_empty() {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                pieces.push(trimmed);
            }
            current.clear();
        }
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        pieces.push(trimmed);
    }

    pieces
}

/// Group chunks from a single file into training-ready documents.
fn group_chunks(chunks: &[Chunk], preamble: &str, max_tokens: usize) -> Vec<String> {
    if chunks.is_empty() && preamble.is_empty() {
        return vec![];
    }

    let _preamble_tokens = if preamble.is_empty() { 0 } else { estimate_tokens(preamble) };

    // Separate by kind
    let mut functions: Vec<&Chunk> = Vec::new();
    let mut classes: Vec<&Chunk> = Vec::new();
    let mut others: Vec<&Chunk> = Vec::new();

    for c in chunks {
        match c.kind {
            ChunkKind::Function => functions.push(c),
            ChunkKind::Class => classes.push(c),
            ChunkKind::Namespace => others.push(c), // namespace contents already extracted
            ChunkKind::TopLevel => others.push(c),
            ChunkKind::Preamble => {} // already separated
        }
    }

    // Check if full file fits
    let mut full_parts: Vec<&str> = Vec::new();
    if !preamble.is_empty() {
        full_parts.push(preamble);
    }
    for c in chunks {
        if c.kind != ChunkKind::Preamble {
            full_parts.push(&c.text);
        }
    }
    let full_text = full_parts.join("\n\n");
    if estimate_tokens(&full_text) <= max_tokens {
        if full_text.len() >= 50 {
            return vec![full_text];
        }
        return vec![];
    }

    let mut documents: Vec<String> = Vec::new();

    // Each class as its own document
    for cls in &classes {
        let doc = if !preamble.is_empty() {
            format!("{}\n\n{}", preamble, cls.text)
        } else {
            cls.text.clone()
        };
        if estimate_tokens(&doc) <= max_tokens * 2 {
            documents.push(doc);
        } else {
            documents.push(cls.text.clone());
        }
    }

    // Group functions into chains of 2-5
    if functions.len() >= 2 {
        let mut i = 0;
        // Simple deterministic chain sizing based on position
        while i < functions.len() {
            let remaining = functions.len() - i;
            let chain_size = std::cmp::min(3, remaining); // default chain of 3

            let mut parts: Vec<&str> = Vec::new();
            if !preamble.is_empty() {
                parts.push(preamble);
            }
            for f in &functions[i..i + chain_size] {
                parts.push(&f.text);
            }
            let chain_text = parts.join("\n\n");

            if estimate_tokens(&chain_text) <= max_tokens {
                documents.push(chain_text);
                i += chain_size;
            } else {
                // Chain too big — emit singles
                for f in &functions[i..i + chain_size] {
                    let doc = if !preamble.is_empty() {
                        format!("{}\n\n{}", preamble, f.text)
                    } else {
                        f.text.clone()
                    };
                    documents.push(doc);
                }
                i += chain_size;
            }
        }
    } else {
        for f in &functions {
            let doc = if !preamble.is_empty() {
                format!("{}\n\n{}", preamble, f.text)
            } else {
                f.text.clone()
            };
            documents.push(doc);
        }
    }

    // Other top-level chunks
    for other in &others {
        if estimate_tokens(&other.text) >= 20 {
            let doc = if !preamble.is_empty() {
                format!("{}\n\n{}", preamble, other.text)
            } else {
                other.text.clone()
            };
            if estimate_tokens(&doc) <= max_tokens * 2 {
                documents.push(doc);
            } else {
                documents.push(other.text.clone());
            }
        }
    }

    documents
}

/// Process a batch with dependency-aware chunking.
fn process_batch_dep_aware(texts: &[String], max_tokens: usize) -> Vec<String> {
    let mut parser = tree_sitter::Parser::new();
    let lang = tree_sitter_cpp::LANGUAGE;
    parser
        .set_language(&lang.into())
        .expect("Failed to set C++ language");

    let mut all_docs = Vec::new();

    for text in texts {
        if text.len() < 50 {
            continue;
        }

        let dep_info = deps::analyze_file(&mut parser, text);
        let docs = deps::build_dep_aware_documents(&dep_info, max_tokens);

        for doc in docs {
            if estimate_tokens(&doc) > max_tokens * 2 {
                all_docs.extend(split_oversized(&doc, max_tokens));
            } else {
                all_docs.push(doc);
            }
        }
    }

    all_docs
}

/// Process a batch of source texts, returning chunked documents.
fn process_batch(texts: &[String], max_tokens: usize) -> Vec<String> {
    let mut parser = tree_sitter::Parser::new();
    let lang = tree_sitter_cpp::LANGUAGE;
    parser
        .set_language(&lang.into())
        .expect("Failed to set C++ language");

    let mut all_docs = Vec::new();

    for text in texts {
        if text.len() < 50 {
            continue;
        }

        let chunks = chunker::chunk_file(&mut parser, text);

        // Separate preamble
        let preamble_parts: Vec<&str> = chunks
            .iter()
            .filter(|c| c.kind == ChunkKind::Preamble)
            .map(|c| c.text.as_str())
            .collect();
        let preamble = preamble_parts.join("\n\n");

        let content_chunks: Vec<&Chunk> = chunks
            .iter()
            .filter(|c| c.kind != ChunkKind::Preamble)
            .collect();

        if content_chunks.is_empty() && preamble.is_empty() {
            continue;
        }

        let docs = group_chunks(&chunks, &preamble, max_tokens);

        // Split any oversized documents
        for doc in docs {
            if estimate_tokens(&doc) > max_tokens * 2 {
                all_docs.extend(split_oversized(&doc, max_tokens));
            } else {
                all_docs.push(doc);
            }
        }

        // If no content chunks but we have a preamble, split the preamble into docs
        if content_chunks.is_empty() && !preamble.is_empty() && estimate_tokens(&preamble) >= 20 {
            all_docs.extend(split_oversized(&preamble, max_tokens));
        }
    }

    all_docs
}

fn main() {
    let args = Args::parse();

    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .expect("Failed to set thread count");
    }

    let num_threads = rayon::current_num_threads();
    eprintln!("cpp-chunker: {} threads, max_tokens={}, batch_size={}, dep_aware={}",
              num_threads, args.max_tokens, args.batch_size, args.dep_aware);

    let output_file = std::fs::File::create(&args.output).expect("Failed to create output file");
    let writer = Mutex::new(BufWriter::with_capacity(64 * 1024 * 1024, output_file));

    let total_files = AtomicU64::new(0);
    let total_docs = AtomicU64::new(0);
    let total_dedup = AtomicU64::new(0);
    let seen_hashes: Mutex<HashSet<[u8; 16]>> = Mutex::new(HashSet::new());

    let t0 = Instant::now();

    for input_path in &args.inputs {
        let file_size = std::fs::metadata(input_path)
            .map(|m| m.len() as f64 / 1e9)
            .unwrap_or(0.0);
        eprintln!("\nProcessing: {} ({:.1} GB)", input_path, file_size);

        let file = std::fs::File::open(input_path).expect("Failed to open input file");
        let reader = io::BufReader::with_capacity(16 * 1024 * 1024, file);

        // Read lines in batches, process each batch in parallel
        let mut batch: Vec<String> = Vec::with_capacity(args.batch_size);
        let mut line_count: u64 = 0;
        let input_t0 = Instant::now();

        for line_result in reader.lines() {
            let line = match line_result {
                Ok(l) => l,
                Err(_) => continue,
            };
            if line.is_empty() {
                continue;
            }

            let record: InputRecord = match serde_json::from_str(&line) {
                Ok(r) => r,
                Err(_) => continue,
            };

            batch.push(record.text);
            line_count += 1;

            if args.max_records > 0 && line_count >= args.max_records as u64 {
                break;
            }

            if batch.len() >= args.batch_size {
                // Process batch in parallel using rayon chunks
                let chunk_size = std::cmp::max(1, batch.len() / num_threads);
                let dep_aware = args.dep_aware;
                let batch_docs: Vec<Vec<String>> = batch
                    .par_chunks(chunk_size)
                    .map(|sub_batch| {
                        if dep_aware {
                            process_batch_dep_aware(sub_batch, args.max_tokens)
                        } else {
                            process_batch(sub_batch, args.max_tokens)
                        }
                    })
                    .collect();

                // Dedup and write
                let mut w = writer.lock().unwrap();
                let mut seen = seen_hashes.lock().unwrap();
                for docs in batch_docs {
                    for doc in docs {
                        let hash: [u8; 16] = md5::compute(doc.as_bytes()).into();
                        if seen.contains(&hash) {
                            total_dedup.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                        seen.insert(hash);

                        let out = OutputRecord { text: &doc };
                        serde_json::to_writer(&mut *w, &out).unwrap();
                        w.write_all(b"\n").unwrap();
                        total_docs.fetch_add(1, Ordering::Relaxed);
                    }
                }
                drop(w);
                drop(seen);

                total_files.fetch_add(batch.len() as u64, Ordering::Relaxed);
                batch.clear();

                if line_count % 100_000 < args.batch_size as u64 {
                    let elapsed = input_t0.elapsed().as_secs_f64();
                    let rate = line_count as f64 / elapsed;
                    eprintln!(
                        "  {}: {:>10} files → {:>12} docs ({:.0} files/sec)",
                        input_path,
                        format_num(line_count),
                        format_num(total_docs.load(Ordering::Relaxed)),
                        rate
                    );
                }
            }
        }

        // Process remaining batch
        if !batch.is_empty() {
            let chunk_size = std::cmp::max(1, batch.len() / num_threads);
            let dep_aware = args.dep_aware;
            let batch_docs: Vec<Vec<String>> = batch
                .par_chunks(chunk_size)
                .map(|sub_batch| {
                    if dep_aware {
                        process_batch_dep_aware(sub_batch, args.max_tokens)
                    } else {
                        process_batch(sub_batch, args.max_tokens)
                    }
                })
                .collect();

            let mut w = writer.lock().unwrap();
            let mut seen = seen_hashes.lock().unwrap();
            for docs in batch_docs {
                for doc in docs {
                    let hash: [u8; 16] = md5::compute(doc.as_bytes()).into();
                    if seen.contains(&hash) {
                        total_dedup.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    seen.insert(hash);

                    let out = OutputRecord { text: &doc };
                    serde_json::to_writer(&mut *w, &out).unwrap();
                    w.write_all(b"\n").unwrap();
                    total_docs.fetch_add(1, Ordering::Relaxed);
                }
            }

            total_files.fetch_add(batch.len() as u64, Ordering::Relaxed);
        }

        let elapsed = input_t0.elapsed().as_secs_f64();
        eprintln!(
            "  {}: {} files processed in {:.0}s",
            input_path,
            format_num(line_count),
            elapsed
        );
    }

    // Flush writer
    writer.lock().unwrap().flush().unwrap();

    let elapsed = t0.elapsed().as_secs_f64();
    let files = total_files.load(Ordering::Relaxed);
    let docs = total_docs.load(Ordering::Relaxed);
    let dedup = total_dedup.load(Ordering::Relaxed);
    let output_size = std::fs::metadata(&args.output)
        .map(|m| m.len() as f64 / 1e9)
        .unwrap_or(0.0);

    eprintln!("\n{}", "=".repeat(60));
    eprintln!("RESULTS:");
    eprintln!("  Files processed: {}", format_num(files));
    eprintln!("  Documents output: {}", format_num(docs));
    eprintln!("  Deduplicated: {}", format_num(dedup));
    eprintln!("  Output: {} ({:.2} GB)", args.output, output_size);
    eprintln!("  Time: {:.1}s ({:.0} files/sec)", elapsed, files as f64 / elapsed);
    eprintln!("{}", "=".repeat(60));
}

fn format_num(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}
