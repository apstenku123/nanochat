/// comment-extractor: Extract, classify, and process non-ASCII C++ comments.
///
/// Subcommands:
///   extract    — Scan C++ files, extract non-ASCII comments to JSONL
///   verify     — Verify JSONL byte offsets match actual file contents
///   apply-ascii — Apply symbol_replace substitutions in-place

mod classify;
mod models;
mod scanner;

use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use clap::{Parser, Subcommand};
use rayon::prelude::*;
use walkdir::WalkDir;

use models::CommentRecord;

#[derive(Parser)]
#[command(name = "comment-extractor", about = "Extract and process non-ASCII C++ comments")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Scan C++ files and extract non-ASCII comments to JSONL
    Extract {
        /// Input directory of C++ files
        #[arg(long, required = true)]
        input: String,

        /// Output JSONL file
        #[arg(long, required = true)]
        output: String,

        /// Number of threads (0 = auto)
        #[arg(long, default_value = "0")]
        threads: usize,

        /// C++ file extensions
        #[arg(long, default_value = "cpp,cc,cxx,c,h,hpp,hxx,cu,cuh,inl,ipp")]
        extensions: String,

        /// Max files to process (for testing)
        #[arg(long)]
        max_files: Option<usize>,
    },

    /// Verify JSONL byte offsets match actual file contents
    Verify {
        /// Input JSONL file
        #[arg(long, required = true)]
        input: String,

        /// Number of threads (0 = auto)
        #[arg(long, default_value = "0")]
        threads: usize,
    },

    /// Apply symbol_replace substitutions in-place, output remaining translate records
    ApplyAscii {
        /// Input JSONL file (from extract)
        #[arg(long, required = true)]
        input: String,

        /// Output JSONL file (only translate records, with updated offsets)
        #[arg(long, required = true)]
        output: String,
    },
}

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

// ─── Extract ──────────────────────────────────────────────────────────────────

fn collect_cpp_files(dir: &str, extensions: &str, max_files: Option<usize>) -> Vec<PathBuf> {
    let exts: Vec<&str> = extensions.split(',').collect();
    let mut files = Vec::new();

    for entry in WalkDir::new(dir)
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

    files.sort();
    if let Some(max) = max_files {
        files.truncate(max);
    }
    files
}

fn run_extract(input: &str, output: &str, threads: usize, extensions: &str, max_files: Option<usize>) {
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok();
    }
    let num_threads = rayon::current_num_threads();

    eprintln!("Collecting C++ files from {}...", input);
    let t0 = Instant::now();
    let files = collect_cpp_files(input, extensions, max_files);
    eprintln!("Found {} files in {:.1}s", format_num(files.len() as u64), t0.elapsed().as_secs_f64());

    if files.is_empty() {
        eprintln!("No files found");
        std::process::exit(1);
    }

    let input_root = PathBuf::from(input).canonicalize().unwrap_or_else(|_| PathBuf::from(input));

    let t1 = Instant::now();
    let files_done = AtomicU64::new(0);
    let total_files = files.len() as u64;
    let total_comments = AtomicU64::new(0);
    let total_translate = AtomicU64::new(0);
    let total_symbol = AtomicU64::new(0);
    let files_with_comments = AtomicU64::new(0);

    // Collect all comments into a thread-safe vec
    let all_comments: Mutex<Vec<CommentRecord>> = Mutex::new(Vec::new());

    let batch_size = std::cmp::max(100, files.len() / (num_threads * 4));
    let batches: Vec<&[PathBuf]> = files.chunks(batch_size).collect();

    eprintln!("Scanning {} batches using {} threads...", batches.len(), num_threads);

    batches.par_iter().for_each(|batch| {
        let mut local_comments = Vec::new();

        for path in *batch {
            let source = match fs::read(path) {
                Ok(s) => s,
                Err(_) => continue,
            };

            // Compute relative path for ID
            let rel_path = path
                .strip_prefix(&input_root)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| path.to_string_lossy().to_string());

            let file_path_str = path.to_string_lossy().to_string();
            let comments = scanner::scan_file(&source, &file_path_str, &rel_path);

            if !comments.is_empty() {
                files_with_comments.fetch_add(1, Ordering::Relaxed);
                let n = comments.len() as u64;
                total_comments.fetch_add(n, Ordering::Relaxed);

                for c in &comments {
                    if c.classification == "translate" {
                        total_translate.fetch_add(1, Ordering::Relaxed);
                    } else {
                        total_symbol.fetch_add(1, Ordering::Relaxed);
                    }
                }
                local_comments.extend(comments);
            }

            let done = files_done.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 50_000 == 0 || done == total_files {
                let elapsed = t1.elapsed().as_secs_f64();
                eprintln!(
                    "  [{}/{}] {:.0} files/sec, {} comments found, {:.1}s",
                    format_num(done),
                    format_num(total_files),
                    done as f64 / elapsed,
                    format_num(total_comments.load(Ordering::Relaxed)),
                    elapsed,
                );
            }
        }

        // Merge local comments
        let mut global = all_comments.lock().unwrap();
        global.extend(local_comments);
    });

    let mut comments = all_comments.into_inner().unwrap();
    comments.sort_by(|a, b| a.file_path.cmp(&b.file_path).then(a.byte_start.cmp(&b.byte_start)));

    let elapsed = t1.elapsed().as_secs_f64();
    let n_comments = comments.len();
    let n_translate = total_translate.load(Ordering::Relaxed);
    let n_symbol = total_symbol.load(Ordering::Relaxed);
    let n_files = files_with_comments.load(Ordering::Relaxed);

    eprintln!("\nExtraction complete in {:.1}s:", elapsed);
    eprintln!("  Files with non-ASCII comments: {}", format_num(n_files));
    eprintln!("  Total non-ASCII comments:      {}", format_num(n_comments as u64));
    eprintln!("  Symbol replace (local):        {} ({:.1}%)", format_num(n_symbol), n_symbol as f64 / n_comments.max(1) as f64 * 100.0);
    eprintln!("  Needs translation (Gemini):    {} ({:.1}%)", format_num(n_translate), n_translate as f64 / n_comments.max(1) as f64 * 100.0);

    // Write JSONL
    let file = fs::File::create(output).expect("Failed to create output file");
    let mut writer = BufWriter::with_capacity(16 * 1024 * 1024, file);
    for comment in &comments {
        serde_json::to_writer(&mut writer, comment).unwrap();
        writer.write_all(b"\n").unwrap();
    }
    writer.flush().unwrap();
    eprintln!("Written {} records to {}", format_num(n_comments as u64), output);
}

// ─── Verify ───────────────────────────────────────────────────────────────────

fn run_verify(input: &str, threads: usize) {
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok();
    }

    eprintln!("Loading records from {}...", input);
    let file = fs::File::open(input).expect("Failed to open input file");
    let reader = io::BufReader::new(file);
    let mut records: Vec<CommentRecord> = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        if line.is_empty() { continue; }
        records.push(serde_json::from_str(&line).unwrap());
    }
    eprintln!("Loaded {} records", format_num(records.len() as u64));

    // Group by file
    let mut by_file: HashMap<String, Vec<&CommentRecord>> = HashMap::new();
    for rec in &records {
        by_file.entry(rec.file_path.clone()).or_default().push(rec);
    }

    let files: Vec<(String, Vec<&CommentRecord>)> = by_file.into_iter().collect();
    let errors = AtomicU64::new(0);
    let verified = AtomicU64::new(0);

    files.par_iter().for_each(|(file_path, recs)| {
        let source = match fs::read(file_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  ERROR reading {}: {}", file_path, e);
                errors.fetch_add(recs.len() as u64, Ordering::Relaxed);
                return;
            }
        };

        for rec in recs {
            if rec.byte_end > source.len() {
                eprintln!("  OFFSET ERROR: {} byte_end {} > file size {}", rec.id, rec.byte_end, source.len());
                errors.fetch_add(1, Ordering::Relaxed);
                continue;
            }

            let extracted = &source[rec.byte_start..rec.byte_end];
            if let Ok(extracted_str) = std::str::from_utf8(extracted) {
                if extracted_str != rec.original_text {
                    eprintln!("  MISMATCH: {} expected {:?} got {:?}",
                        rec.id,
                        &rec.original_text[..rec.original_text.len().min(50)],
                        &extracted_str[..extracted_str.len().min(50)]);
                    errors.fetch_add(1, Ordering::Relaxed);
                } else {
                    verified.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                eprintln!("  UTF8 ERROR at {}", rec.id);
                errors.fetch_add(1, Ordering::Relaxed);
            }
        }
    });

    let n_errors = errors.load(Ordering::Relaxed);
    let n_verified = verified.load(Ordering::Relaxed);

    if n_errors == 0 {
        eprintln!("\nSUCCESS: All {} records verified byte-exact!", format_num(n_verified));
    } else {
        eprintln!("\nFAILED: {} errors, {} verified", format_num(n_errors), format_num(n_verified));
        std::process::exit(1);
    }
}

// ─── Apply ASCII ──────────────────────────────────────────────────────────────

fn run_apply_ascii(input: &str, output: &str) {
    eprintln!("Loading records from {}...", input);
    let file = fs::File::open(input).expect("Failed to open input file");
    let reader = io::BufReader::new(file);
    let mut records: Vec<CommentRecord> = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        if line.is_empty() { continue; }
        records.push(serde_json::from_str(&line).unwrap());
    }
    eprintln!("Loaded {} records", format_num(records.len() as u64));

    // Separate translate vs symbol_replace
    let mut translate_records: Vec<CommentRecord> = Vec::new();
    let mut replace_by_file: HashMap<String, Vec<CommentRecord>> = HashMap::new();

    for rec in records {
        if rec.classification == "translate" {
            translate_records.push(rec);
        } else if let Some(_) = &rec.ascii_replacement {
            replace_by_file.entry(rec.file_path.clone()).or_default().push(rec);
        }
    }

    eprintln!("Symbol replacements: {} records across {} files",
        format_num(replace_by_file.values().map(|v| v.len() as u64).sum::<u64>()),
        format_num(replace_by_file.len() as u64));
    eprintln!("Translate records (passed through): {}", format_num(translate_records.len() as u64));

    // Apply symbol replacements file by file
    let mut applied = 0u64;
    let mut modified_files = 0u64;

    for (file_path, mut recs) in replace_by_file {
        let source = match fs::read(&file_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  ERROR reading {}: {}", file_path, e);
                continue;
            }
        };

        // Sort by byte_start descending to apply from end
        recs.sort_by(|a, b| b.byte_start.cmp(&a.byte_start));

        let mut result = source.clone();
        let mut file_changed = false;

        for rec in &recs {
            let replacement_content = match &rec.ascii_replacement {
                Some(r) => r,
                None => continue,
            };

            // Reconstruct full comment with markers
            let new_text = match rec.comment_type.as_str() {
                "line" => format!("//{}", replacement_content),
                "block" => format!("/*{}*/", replacement_content),
                _ => continue,
            };

            let new_bytes = new_text.as_bytes();

            // Validate all ASCII
            if !new_bytes.iter().all(|&b| b < 128) {
                eprintln!("  WARNING: non-ASCII in replacement for {}, skipping", rec.id);
                continue;
            }

            // Verify the original text matches before replacing
            if rec.byte_end > result.len() {
                eprintln!("  WARNING: offset out of bounds for {}", rec.id);
                continue;
            }
            if let Ok(current) = std::str::from_utf8(&result[rec.byte_start..rec.byte_end]) {
                if current != rec.original_text {
                    eprintln!("  WARNING: content mismatch for {} (file already modified?)", rec.id);
                    continue;
                }
            }

            result.splice(rec.byte_start..rec.byte_end, new_bytes.iter().copied());
            file_changed = true;
            applied += 1;
        }

        if file_changed {
            fs::write(&file_path, &result).expect("Failed to write file");
            modified_files += 1;
        }
    }

    eprintln!("\nApplied {} ASCII replacements across {} files", format_num(applied), format_num(modified_files));

    // Write translate records to output JSONL
    // Note: byte offsets for translate records in already-modified files may have shifted.
    // We need to re-scan those files to get correct offsets.
    // For files that were NOT modified (no symbol_replace records), offsets are still valid.

    let out_file = fs::File::create(output).expect("Failed to create output file");
    let mut writer = BufWriter::new(out_file);
    for rec in &translate_records {
        serde_json::to_writer(&mut writer, rec).unwrap();
        writer.write_all(b"\n").unwrap();
    }
    writer.flush().unwrap();
    eprintln!("Written {} translate records to {}", format_num(translate_records.len() as u64), output);

    eprintln!("\nIMPORTANT: If files had BOTH symbol_replace and translate records,");
    eprintln!("           the translate byte offsets may have shifted. Re-run extract");
    eprintln!("           on modified files to get updated offsets before Gemini translation.");
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Extract {
            input,
            output,
            threads,
            extensions,
            max_files,
        } => run_extract(&input, &output, threads, &extensions, max_files),

        Commands::Verify { input, threads } => run_verify(&input, threads),

        Commands::ApplyAscii { input, output } => run_apply_ascii(&input, &output),
    }
}
