use clap::Parser as ClapParser;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::{self, BufRead, BufWriter, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

mod chunker;
mod compilable;
mod deps;
mod global_index;
mod project_graph;
use chunker::{Chunk, ChunkKind};
use global_index::GlobalIndex;

#[derive(ClapParser)]
#[command(name = "cpp-chunker", about = "Syntax-aware C++ chunker using tree-sitter")]
struct Args {
    /// Input JSONL files (for JSONL mode)
    #[arg(long, num_args = 1..)]
    inputs: Vec<String>,

    /// Output JSONL path
    #[arg(long)]
    output: String,

    /// Directory containing project subdirectories (for project-aware mode).
    /// Each subdirectory is treated as a separate project. Dependencies are
    /// auto-detected from #include directives and projects are processed in
    /// topological order (foundational libraries first).
    #[arg(long)]
    project_dirs: Option<String>,

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

    /// Enable cross-file dependency resolution (two-pass: index all files, then resolve).
    /// Implies --dep-aware. Pulls in function definitions from other files when they
    /// have a unique name in the corpus.
    #[arg(long, default_value = "false")]
    cross_file: bool,

    /// Max depth for cross-file dependency BFS (default: 3)
    #[arg(long, default_value = "3")]
    cross_depth: usize,

    /// Max input file size in bytes to process (skip larger files).
    /// Very large files (auto-generated code) can stall tree-sitter parsing.
    #[arg(long, default_value = "500000")]
    max_file_bytes: usize,

    /// Generate near-compilable C++ chunks with proper ordering:
    /// preamble → type definitions (topo order) → functions (bottom-up).
    /// Use with --project_dirs for best results.
    #[arg(long, default_value = "false")]
    compilable: bool,
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
fn process_batch_dep_aware(texts: &[String], max_tokens: usize, max_file_bytes: usize) -> Vec<String> {
    let mut parser = tree_sitter::Parser::new();
    let lang = tree_sitter_cpp::LANGUAGE;
    parser
        .set_language(&lang.into())
        .expect("Failed to set C++ language");

    let mut all_docs = Vec::new();

    for text in texts {
        if text.len() < 50 || text.len() > max_file_bytes {
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
fn process_batch(texts: &[String], max_tokens: usize, max_file_bytes: usize) -> Vec<String> {
    let mut parser = tree_sitter::Parser::new();
    let lang = tree_sitter_cpp::LANGUAGE;
    parser
        .set_language(&lang.into())
        .expect("Failed to set C++ language");

    let mut all_docs = Vec::new();

    for text in texts {
        if text.len() < 50 || text.len() > max_file_bytes {
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

    // Project-aware mode: reads raw project directories with dependency ordering
    if let Some(ref proj_dir) = args.project_dirs {
        eprintln!(
            "cpp-chunker: {} threads, max_tokens={}, project_mode=true, cross_depth={}",
            num_threads, args.max_tokens, args.cross_depth
        );
        run_project_mode(&args, proj_dir, num_threads);
        return;
    }

    if args.cross_file {
        eprintln!(
            "cpp-chunker: {} threads, max_tokens={}, batch_size={}, cross_file=true, cross_depth={}",
            num_threads, args.max_tokens, args.batch_size, args.cross_depth
        );
        run_cross_file_mode(&args, num_threads);
        return;
    }

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
                            process_batch_dep_aware(sub_batch, args.max_tokens, args.max_file_bytes)
                        } else {
                            process_batch(sub_batch, args.max_tokens, args.max_file_bytes)
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
                        process_batch_dep_aware(sub_batch, args.max_tokens, args.max_file_bytes)
                    } else {
                        process_batch(sub_batch, args.max_tokens, args.max_file_bytes)
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

/// Two-pass cross-file mode:
/// Phase 1: Build global function index from all input files.
/// Phase 2: Process all files again, resolving cross-file dependencies.
fn run_cross_file_mode(args: &Args, num_threads: usize) {
    // Phase 1: Build global index
    eprintln!("\n=== Phase 1: Building global function index ===");
    let t0 = Instant::now();
    let global_index = build_global_index(args, num_threads);
    let phase1_time = t0.elapsed().as_secs_f64();
    eprintln!(
        "Index built in {:.1}s: {} names, {} total defs, {} uniquely resolvable, ~{:.1} GB memory",
        phase1_time,
        format_num(global_index.name_count() as u64),
        format_num(global_index.total_defs() as u64),
        format_num(global_index.unique_count() as u64),
        global_index.memory_bytes() as f64 / 1e9
    );

    // Phase 2: Process with cross-file resolution
    eprintln!("\n=== Phase 2: Cross-file dependency document generation ===");
    let t1 = Instant::now();

    let output_file = std::fs::File::create(&args.output).expect("Failed to create output file");
    let writer = Mutex::new(BufWriter::with_capacity(64 * 1024 * 1024, output_file));
    let total_files = AtomicU64::new(0);
    let total_docs = AtomicU64::new(0);
    let total_dedup = AtomicU64::new(0);
    let seen_hashes: Mutex<HashSet<[u8; 16]>> = Mutex::new(HashSet::new());

    for input_path in &args.inputs {
        let file_size = std::fs::metadata(input_path)
            .map(|m| m.len() as f64 / 1e9)
            .unwrap_or(0.0);
        eprintln!("\nProcessing: {} ({:.1} GB)", input_path, file_size);

        let file = std::fs::File::open(input_path).expect("Failed to open input file");
        let reader = io::BufReader::with_capacity(16 * 1024 * 1024, file);
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
                let chunk_size = std::cmp::max(1, batch.len() / num_threads);
                let cross_depth = args.cross_depth;
                let max_tokens = args.max_tokens;
                let max_file_bytes = args.max_file_bytes;
                let batch_docs: Vec<Vec<String>> = batch
                    .par_chunks(chunk_size)
                    .map(|sub_batch| {
                        process_batch_cross_file(sub_batch, max_tokens, &global_index, cross_depth, max_file_bytes, args.compilable)
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
                drop(w);
                drop(seen);

                total_files.fetch_add(batch.len() as u64, Ordering::Relaxed);
                batch.clear();

                if line_count % 100_000 < args.batch_size as u64 {
                    let elapsed = input_t0.elapsed().as_secs_f64();
                    let rate = line_count as f64 / elapsed;
                    eprintln!(
                        "  {}: {:>10} files -> {:>12} docs ({:.0} files/sec)",
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
            let cross_depth = args.cross_depth;
            let max_tokens = args.max_tokens;
            let max_file_bytes = args.max_file_bytes;
            let batch_docs: Vec<Vec<String>> = batch
                .par_chunks(chunk_size)
                .map(|sub_batch| {
                    process_batch_cross_file(sub_batch, max_tokens, &global_index, cross_depth, max_file_bytes, args.compilable)
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

    writer.lock().unwrap().flush().unwrap();

    let phase2_time = t1.elapsed().as_secs_f64();
    let total_time = t0.elapsed().as_secs_f64();
    let files = total_files.load(Ordering::Relaxed);
    let docs = total_docs.load(Ordering::Relaxed);
    let dedup = total_dedup.load(Ordering::Relaxed);
    let output_size = std::fs::metadata(&args.output)
        .map(|m| m.len() as f64 / 1e9)
        .unwrap_or(0.0);

    eprintln!("\n{}", "=".repeat(60));
    eprintln!("RESULTS (cross-file mode):");
    eprintln!("  Phase 1 (indexing): {:.1}s", phase1_time);
    eprintln!("  Phase 2 (processing): {:.1}s", phase2_time);
    eprintln!("  Total time: {:.1}s", total_time);
    eprintln!("  Files processed: {}", format_num(files));
    eprintln!("  Documents output: {}", format_num(docs));
    eprintln!("  Deduplicated: {}", format_num(dedup));
    eprintln!("  Output: {} ({:.2} GB)", args.output, output_size);
    eprintln!("  Rate: {:.0} files/sec", files as f64 / total_time);
    eprintln!("{}", "=".repeat(60));
}

/// Project-aware mode: reads raw project directories, detects dependencies,
/// processes in topological order (foundational libraries first) with full
/// parallelism. No serial JSONL bottleneck.
fn run_project_mode(args: &Args, projects_dir: &str, num_threads: usize) {
    let t0 = Instant::now();

    // Phase 0: Discover projects and build dependency DAG
    eprintln!("\n=== Phase 0: Project discovery and dependency analysis ===");
    let projects = project_graph::plan_processing_order(
        std::path::Path::new(projects_dir),
        args.max_file_bytes,
    );

    let total_src: usize = projects.iter().map(|p| p.source_files.len()).sum();
    let total_hdr: usize = projects.iter().map(|p| p.header_files.len()).sum();
    eprintln!("Total: {} source files, {} headers across {} projects",
        format_num(total_src as u64), format_num(total_hdr as u64), projects.len());

    // Prepare output
    let output_file = std::fs::File::create(&args.output).expect("Failed to create output file");
    let writer = Mutex::new(BufWriter::with_capacity(64 * 1024 * 1024, output_file));
    let seen_hashes: Mutex<HashSet<[u8; 16]>> = Mutex::new(HashSet::new());
    let total_docs = AtomicU64::new(0);
    let total_dedup = AtomicU64::new(0);

    // Global index that grows as we process projects in dependency order
    let mut global_index = GlobalIndex::new();

    // Phase 1+2 combined: For each project in dependency order, index and generate docs
    eprintln!("\n=== Processing projects in dependency order ===");

    for (rank, proj) in projects.iter().enumerate() {
        let proj_t0 = Instant::now();
        eprintln!("\n[{}/{}] {} ({} src, {} hdr, {} deps)",
            rank + 1, projects.len(), proj.name,
            proj.source_files.len(), proj.header_files.len(),
            proj.depends_on.len());

        if proj.source_files.is_empty() {
            eprintln!("  Skipping: no source files");
            continue;
        }

        // Read all source files in parallel - no serial bottleneck!
        let max_file_bytes = args.max_file_bytes;
        let file_contents: Vec<(String, String)> = proj.source_files
            .par_iter()
            .filter_map(|path| {
                let content = std::fs::read_to_string(path).ok()?;
                if content.len() < 50 || content.len() > max_file_bytes {
                    return None;
                }
                let rel_path = path.strip_prefix(&proj.path)
                    .unwrap_or(path)
                    .to_string_lossy()
                    .to_string();
                Some((rel_path, content))
            })
            .collect();

        eprintln!("  Read {} files in {:.1}s",
            file_contents.len(), proj_t0.elapsed().as_secs_f64());

        // Index: parse all files and add to global index (parallel)
        let index_t0 = Instant::now();
        let local_indexes: Vec<GlobalIndex> = file_contents
            .par_chunks(std::cmp::max(1, file_contents.len() / num_threads))
            .map(|chunk| {
                let mut local = GlobalIndex::new();
                let mut parser = tree_sitter::Parser::new();
                let lang = tree_sitter_cpp::LANGUAGE;
                parser.set_language(&lang.into()).unwrap();

                for (_path, text) in chunk {
                    let dep_info = deps::analyze_file(&mut parser, text);
                    for func in &dep_info.functions {
                        local.add(&func.name, func.text.clone(), func.callees.clone());
                    }
                }
                local
            })
            .collect();

        for local in local_indexes {
            global_index.merge(local);
        }

        eprintln!("  Indexed in {:.1}s, global: {} names, {} unique",
            index_t0.elapsed().as_secs_f64(),
            format_num(global_index.name_count() as u64),
            format_num(global_index.unique_count() as u64));

        // Generate documents using the global index
        // (which now includes all previously processed dependency projects)
        let doc_t0 = Instant::now();
        let cross_depth = args.cross_depth;
        let max_tokens = args.max_tokens;
        let use_compilable = args.compilable;
        let batch_docs: Vec<Vec<String>> = file_contents
            .par_chunks(std::cmp::max(1, file_contents.len() / num_threads))
            .map(|chunk| {
                let mut parser = tree_sitter::Parser::new();
                let lang = tree_sitter_cpp::LANGUAGE;
                parser.set_language(&lang.into()).unwrap();

                let mut docs = Vec::new();
                for (_path, text) in chunk {
                    let dep_info = deps::analyze_file(&mut parser, text);
                    let file_docs = if use_compilable {
                        compilable::build_compilable_documents(
                            &dep_info, &global_index, max_tokens, cross_depth,
                        )
                    } else {
                        deps::build_cross_file_documents(
                            &dep_info, &global_index, max_tokens, cross_depth,
                        )
                    };
                    for doc in file_docs {
                        if estimate_tokens(&doc) > max_tokens * 2 {
                            docs.extend(split_oversized(&doc, max_tokens));
                        } else {
                            docs.push(doc);
                        }
                    }
                }
                docs
            })
            .collect();

        // Write documents with dedup
        let mut proj_docs = 0u64;
        {
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
                    proj_docs += 1;
                }
            }
        }

        let proj_time = proj_t0.elapsed().as_secs_f64();
        eprintln!("  Generated {} docs in {:.1}s ({:.0} files/sec)",
            format_num(proj_docs), proj_time,
            file_contents.len() as f64 / proj_time);
    }

    writer.lock().unwrap().flush().unwrap();

    let total_time = t0.elapsed().as_secs_f64();
    let docs = total_docs.load(Ordering::Relaxed);
    let dedup = total_dedup.load(Ordering::Relaxed);
    let output_size = std::fs::metadata(&args.output)
        .map(|m| m.len() as f64 / 1e9)
        .unwrap_or(0.0);

    eprintln!("\n{}", "=".repeat(60));
    eprintln!("RESULTS (project-aware mode):");
    eprintln!("  Projects processed: {}", projects.len());
    eprintln!("  Source files: {}", format_num(total_src as u64));
    eprintln!("  Documents output: {}", format_num(docs));
    eprintln!("  Deduplicated: {}", format_num(dedup));
    eprintln!("  Global index: {} names, {} unique, ~{:.1} GB",
        format_num(global_index.name_count() as u64),
        format_num(global_index.unique_count() as u64),
        global_index.memory_bytes() as f64 / 1e9);
    eprintln!("  Output: {} ({:.2} GB)", args.output, output_size);
    eprintln!("  Total time: {:.1}s ({:.0} files/sec)",
        total_time, total_src as f64 / total_time);
    eprintln!("{}", "=".repeat(60));
}

/// Phase 1: Build the global function index by scanning all input files.
fn build_global_index(args: &Args, num_threads: usize) -> GlobalIndex {
    let mut global = GlobalIndex::new();
    let total_indexed = AtomicU64::new(0);

    for input_path in &args.inputs {
        let file_size = std::fs::metadata(input_path)
            .map(|m| m.len() as f64 / 1e9)
            .unwrap_or(0.0);
        eprintln!("  Indexing: {} ({:.1} GB)", input_path, file_size);

        let file = std::fs::File::open(input_path).expect("Failed to open input file");
        let reader = io::BufReader::with_capacity(16 * 1024 * 1024, file);
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
                let chunk_size = std::cmp::max(1, batch.len() / num_threads);
                let mfb = args.max_file_bytes;
                let local_indexes: Vec<GlobalIndex> = batch
                    .par_chunks(chunk_size)
                    .map(|sub_batch| {
                        let mut local = GlobalIndex::new();
                        let mut parser = tree_sitter::Parser::new();
                        let lang = tree_sitter_cpp::LANGUAGE;
                        parser.set_language(&lang.into()).unwrap();

                        for text in sub_batch {
                            if text.len() < 50 || text.len() > mfb {
                                continue;
                            }
                            let dep_info = deps::analyze_file(&mut parser, text);
                            for func in &dep_info.functions {
                                local.add(&func.name, func.text.clone(), func.callees.clone());
                            }
                        }
                        local
                    })
                    .collect();

                for local in local_indexes {
                    global.merge(local);
                }
                total_indexed.fetch_add(batch.len() as u64, Ordering::Relaxed);
                batch.clear();

                if line_count % 500_000 < args.batch_size as u64 {
                    let elapsed = input_t0.elapsed().as_secs_f64();
                    eprintln!(
                        "    {} files indexed ({:.0}/sec), {} names, ~{:.1} GB",
                        format_num(line_count),
                        line_count as f64 / elapsed,
                        format_num(global.name_count() as u64),
                        global.memory_bytes() as f64 / 1e9
                    );
                }
            }
        }

        // Process remaining batch
        if !batch.is_empty() {
            let chunk_size = std::cmp::max(1, batch.len() / num_threads);
            let mfb = args.max_file_bytes;
            let local_indexes: Vec<GlobalIndex> = batch
                .par_chunks(chunk_size)
                .map(|sub_batch| {
                    let mut local = GlobalIndex::new();
                    let mut parser = tree_sitter::Parser::new();
                    let lang = tree_sitter_cpp::LANGUAGE;
                    parser.set_language(&lang.into()).unwrap();

                    for text in sub_batch {
                        if text.len() < 50 || text.len() > mfb {
                            continue;
                        }
                        let dep_info = deps::analyze_file(&mut parser, text);
                        for func in &dep_info.functions {
                            local.add(&func.name, func.text.clone(), func.callees.clone());
                        }
                    }
                    local
                })
                .collect();

            for local in local_indexes {
                global.merge(local);
            }
            total_indexed.fetch_add(batch.len() as u64, Ordering::Relaxed);
        }

        let elapsed = input_t0.elapsed().as_secs_f64();
        eprintln!(
            "    {} files indexed in {:.0}s",
            format_num(line_count),
            elapsed
        );
    }

    // Prune ambiguous names to free memory before Phase 2
    let before_names = global.name_count();
    let before_mem = global.memory_bytes();
    global.prune_ambiguous();
    eprintln!(
        "  Pruned ambiguous: {} -> {} names, {:.1} GB -> {:.1} GB",
        format_num(before_names as u64),
        format_num(global.name_count() as u64),
        before_mem as f64 / 1e9,
        global.memory_bytes() as f64 / 1e9
    );

    global
}

/// Process a batch with cross-file dependency resolution.
fn process_batch_cross_file(
    texts: &[String],
    max_tokens: usize,
    global: &GlobalIndex,
    cross_depth: usize,
    max_file_bytes: usize,
    use_compilable: bool,
) -> Vec<String> {
    let mut parser = tree_sitter::Parser::new();
    let lang = tree_sitter_cpp::LANGUAGE;
    parser
        .set_language(&lang.into())
        .expect("Failed to set C++ language");

    let mut all_docs = Vec::new();

    for text in texts {
        if text.len() < 50 || text.len() > max_file_bytes {
            continue;
        }

        let dep_info = deps::analyze_file(&mut parser, text);
        let docs = if use_compilable {
            compilable::build_compilable_documents(&dep_info, global, max_tokens, cross_depth)
        } else {
            deps::build_cross_file_documents(&dep_info, global, max_tokens, cross_depth)
        };

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
