//! Project dependency graph for smart processing order.
//!
//! Scans raw C++ project directories, detects inter-project dependencies
//! from #include directives, and produces a topological ordering so that
//! foundational libraries (boost, fmt, openssl) are indexed before
//! projects that depend on them.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use rayon::prelude::*;

/// A project discovered on disk.
#[derive(Debug)]
pub struct ProjectNode {
    pub name: String,
    pub path: PathBuf,
    pub source_files: Vec<PathBuf>,
    pub header_files: Vec<PathBuf>,
    pub depends_on: HashSet<String>,
}

/// C/C++ source extensions.
const SOURCE_EXTS: &[&str] = &["c", "cc", "cpp", "cxx", "c++"];
/// Header extensions.
const HEADER_EXTS: &[&str] = &["h", "hh", "hpp", "hxx", "h++", "inl"];
/// Directories to skip.
const SKIP_DIRS: &[&str] = &[
    ".git", "build", "cmake-build", "__pycache__", "node_modules",
    ".vs", ".vscode", "test", "tests", "testing", "benchmarks",
    "examples", "example", "samples", "sample", "docs", "doc",
];

impl ProjectNode {
    /// Discover source and header files in a project directory.
    pub fn discover(name: String, path: PathBuf, max_file_bytes: usize) -> Self {
        let mut source_files = Vec::new();
        let mut header_files = Vec::new();
        collect_files(&path, &path, &mut source_files, &mut header_files, max_file_bytes);
        ProjectNode {
            name,
            path,
            source_files,
            header_files,
            depends_on: HashSet::new(),
        }
    }

    pub fn total_files(&self) -> usize {
        self.source_files.len() + self.header_files.len()
    }
}

fn collect_files(
    root: &Path,
    dir: &Path,
    sources: &mut Vec<PathBuf>,
    headers: &mut Vec<PathBuf>,
    max_file_bytes: usize,
) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            let name_lower = name.to_lowercase();
            if SKIP_DIRS.iter().any(|s| name_lower == *s) {
                continue;
            }
            // Skip third_party/external only at top level to avoid bloat
            let depth = path.strip_prefix(root).map(|p| p.components().count()).unwrap_or(0);
            if depth <= 2 && (name_lower == "third_party" || name_lower == "external"
                || name_lower == "vendor" || name_lower == "deps") {
                continue;
            }
            collect_files(root, &path, sources, headers, max_file_bytes);
        } else if path.is_file() {
            // Check size
            if let Ok(meta) = path.metadata() {
                if meta.len() > max_file_bytes as u64 {
                    continue;
                }
            }
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                let ext_lower = ext.to_lowercase();
                if SOURCE_EXTS.iter().any(|e| *e == ext_lower) {
                    sources.push(path);
                } else if HEADER_EXTS.iter().any(|e| *e == ext_lower) {
                    headers.push(path);
                }
            }
        }
    }
}

/// Build a map from normalized header subpath to project name.
/// E.g., "boost/asio.hpp" -> "boost", "fmt/core.h" -> "fmt"
pub fn build_header_ownership(projects: &[ProjectNode]) -> HashMap<String, String> {
    let mut map = HashMap::new();

    for proj in projects {
        for header in &proj.header_files {
            // Try multiple prefix strategies for matching includes
            let rel = header.strip_prefix(&proj.path).unwrap_or(header);
            let rel_str = rel.to_string_lossy().replace('\\', "/");

            // Full relative path: "src/include/boost/asio.hpp"
            map.insert(rel_str.clone(), proj.name.clone());

            // Try stripping common prefixes: include/, src/, lib/
            for prefix in &["include/", "src/", "lib/", "source/"] {
                if let Some(stripped) = rel_str.strip_prefix(prefix) {
                    map.insert(stripped.to_string(), proj.name.clone());
                }
            }

            // Just the filename for unique headers
            if let Some(fname) = header.file_name() {
                let fname_str = fname.to_string_lossy().to_string();
                // Only add filename mapping if it's reasonably unique
                if !["config.h", "types.h", "common.h", "utils.h", "defs.h",
                     "version.h", "platform.h", "debug.h", "error.h", "log.h"]
                    .contains(&fname_str.as_str())
                {
                    map.entry(fname_str).or_insert_with(|| proj.name.clone());
                }
            }
        }
    }

    map
}

/// Extract #include directives from a single file using simple line scanning.
/// Returns (quoted_includes, angle_includes).
fn extract_includes(path: &Path) -> (Vec<String>, Vec<String>) {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return (vec![], vec![]),
    };

    let mut quoted = Vec::new();
    let mut angle = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("#include") && !trimmed.starts_with("# include") {
            continue;
        }
        // Find the include path
        if let Some(start) = trimmed.find('"') {
            if let Some(end) = trimmed[start + 1..].find('"') {
                quoted.push(trimmed[start + 1..start + 1 + end].to_string());
            }
        } else if let Some(start) = trimmed.find('<') {
            if let Some(end) = trimmed[start + 1..].find('>') {
                angle.push(trimmed[start + 1..start + 1 + end].to_string());
            }
        }
    }

    (quoted, angle)
}

/// Detect inter-project dependencies by scanning include directives.
/// Uses parallel scanning for speed.
pub fn detect_dependencies(projects: &mut Vec<ProjectNode>, header_map: &HashMap<String, String>) {
    // Pre-compute project name lookup: lowercase name -> original name (O(1) instead of O(N))
    let mut name_lookup: HashMap<String, String> = HashMap::new();
    for proj in projects.iter() {
        let lower = proj.name.to_lowercase();
        name_lookup.insert(lower.clone(), proj.name.clone());
        // Also add underscore variant: "gcc-mirror" -> "gcc_mirror"
        let underscore = lower.replace('-', "_");
        if underscore != lower {
            name_lookup.insert(underscore, proj.name.clone());
        }
    }

    // Collect all includes from each project in parallel
    let project_includes: Vec<HashSet<String>> = projects
        .par_iter()
        .map(|proj| {
            let all_files: Vec<&PathBuf> = proj.source_files.iter()
                .chain(proj.header_files.iter())
                .collect();

            let mut deps = HashSet::new();

            // Sample up to 500 files for dependency detection (speed)
            let sample_size = std::cmp::min(500, all_files.len());
            let step = if all_files.len() > sample_size { all_files.len() / sample_size } else { 1 };

            for (i, file) in all_files.iter().enumerate() {
                if i % step != 0 && i >= sample_size {
                    continue;
                }
                let (quoted, angle) = extract_includes(file);

                for inc in quoted.iter().chain(angle.iter()) {
                    // Try to match include path to a project
                    let inc_normalized = inc.replace('\\', "/");

                    // Try exact match
                    if let Some(dep_project) = header_map.get(&inc_normalized) {
                        deps.insert(dep_project.clone());
                        continue;
                    }

                    // Try just the filename part
                    if let Some(fname) = Path::new(&inc_normalized).file_name() {
                        let fname_str = fname.to_string_lossy().to_string();
                        if let Some(dep_project) = header_map.get(&fname_str) {
                            deps.insert(dep_project.clone());
                            continue;
                        }
                    }

                    // Try well-known prefix patterns with O(1) HashMap lookup
                    if let Some(slash_pos) = inc_normalized.find('/') {
                        let top_dir = inc_normalized[..slash_pos].to_lowercase();
                        if let Some(proj_name) = name_lookup.get(&top_dir) {
                            deps.insert(proj_name.clone());
                        }
                    }
                }
            }

            deps
        })
        .collect();

    // Apply detected dependencies (removing self-deps)
    for (i, deps) in project_includes.into_iter().enumerate() {
        let self_name = projects[i].name.clone();
        projects[i].depends_on = deps.into_iter()
            .filter(|d| *d != self_name)
            .collect();
    }
}

/// Topological sort of projects. Returns indices in processing order
/// (dependencies first). Projects with no dependencies come first.
pub fn topo_sort(projects: &[ProjectNode]) -> Vec<usize> {
    let name_to_idx: HashMap<&str, usize> = projects.iter()
        .enumerate()
        .map(|(i, p)| (p.name.as_str(), i))
        .collect();

    // Compute in-degrees
    let mut in_degree = vec![0usize; projects.len()];
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); projects.len()];

    for (i, proj) in projects.iter().enumerate() {
        for dep_name in &proj.depends_on {
            if let Some(&dep_idx) = name_to_idx.get(dep_name.as_str()) {
                in_degree[i] += 1;
                dependents[dep_idx].push(i);
            }
        }
    }

    // Kahn's algorithm
    let mut queue: VecDeque<usize> = VecDeque::new();
    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            queue.push_back(i);
        }
    }

    let mut order = Vec::with_capacity(projects.len());
    while let Some(idx) = queue.pop_front() {
        order.push(idx);
        for &dep_idx in &dependents[idx] {
            in_degree[dep_idx] -= 1;
            if in_degree[dep_idx] == 0 {
                queue.push_back(dep_idx);
            }
        }
    }

    // Handle cycles: add remaining nodes at the end
    if order.len() < projects.len() {
        for i in 0..projects.len() {
            if !order.contains(&i) {
                order.push(i);
            }
        }
    }

    order
}

/// Discover all projects in a directory, detect dependencies, and return
/// them in topological processing order.
pub fn plan_processing_order(
    projects_dir: &Path,
    max_file_bytes: usize,
) -> Vec<ProjectNode> {
    eprintln!("Discovering projects in {}...", projects_dir.display());

    // Collect project directories first, then discover in parallel
    let project_dirs: Vec<(String, PathBuf)> = std::fs::read_dir(projects_dir)
        .expect("Failed to read projects directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            let path = e.path();
            (name, path)
        })
        .collect();

    eprintln!("  Found {} directories, scanning in parallel...", project_dirs.len());

    // Discover files in parallel across all projects
    let mut projects: Vec<ProjectNode> = project_dirs
        .into_par_iter()
        .map(|(name, path)| {
            let proj = ProjectNode::discover(name.clone(), path, max_file_bytes);
            eprintln!("  Scanning: {} ({} files)", name, proj.total_files());
            proj
        })
        .filter(|p| p.total_files() > 0)
        .collect();

    eprintln!("Found {} projects with source files", projects.len());

    // Build header ownership map
    eprintln!("Building header ownership map...");
    let header_map = build_header_ownership(&projects);
    eprintln!("  {} header paths mapped", header_map.len());

    // Detect dependencies
    eprintln!("Detecting inter-project dependencies...");
    detect_dependencies(&mut projects, &header_map);

    for proj in &projects {
        if !proj.depends_on.is_empty() {
            eprintln!("  {} -> depends on: {:?}", proj.name,
                proj.depends_on.iter().take(10).collect::<Vec<_>>());
        }
    }

    // Topological sort
    let order = topo_sort(&projects);
    eprintln!("\nProcessing order ({} projects):", order.len());

    // Reorder projects
    let mut ordered: Vec<ProjectNode> = Vec::with_capacity(projects.len());
    // We need to move items out of projects by index in order
    let mut taken = vec![false; projects.len()];
    for &idx in &order {
        taken[idx] = true;
    }
    // Convert to indexed removal
    let mut projects_vec: Vec<Option<ProjectNode>> = projects.into_iter().map(Some).collect();
    for (rank, &idx) in order.iter().enumerate() {
        if let Some(proj) = projects_vec[idx].take() {
            eprintln!("  {:>3}. {} ({} src, {} hdr, deps: {})",
                rank + 1, proj.name, proj.source_files.len(),
                proj.header_files.len(), proj.depends_on.len());
            ordered.push(proj);
        }
    }

    ordered
}
