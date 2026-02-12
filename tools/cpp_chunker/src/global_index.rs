//! Global function index for cross-file dependency resolution.
//!
//! Phase 1 of the two-pass architecture: scan all input files, extract every
//! function definition, and build a name -> definition map. Functions with
//! exactly ONE definition are "uniquely resolvable" and can be pulled into
//! other files' training documents as cross-file dependencies.

use std::collections::HashMap;
use crate::deps::{normalize_name, is_system_call};

/// A function definition stored in the global index.
#[derive(Debug, Clone)]
pub struct IndexedFunction {
    pub name: String,
    pub text: String,
    pub callees: Vec<String>,
}

/// Global index mapping normalized function names to their definitions
/// across all input files. Used for cross-file dependency resolution.
pub struct GlobalIndex {
    /// normalized_name -> list of definitions (multiple = ambiguous)
    functions: HashMap<String, Vec<IndexedFunction>>,
}

impl GlobalIndex {
    pub fn new() -> Self {
        GlobalIndex {
            functions: HashMap::with_capacity(1_000_000),
        }
    }

    /// Add a function definition to the index.
    pub fn add(&mut self, name: &str, text: String, callees: Vec<String>) {
        let norm = normalize_name(name);
        if norm.is_empty() || norm.len() < 2 {
            return;
        }
        if is_system_call(&norm) {
            return;
        }
        self.functions
            .entry(norm)
            .or_default()
            .push(IndexedFunction {
                name: name.to_string(),
                text,
                callees,
            });
    }

    /// Resolve a function name. Returns Some only if exactly ONE definition
    /// exists globally (unambiguous resolution).
    pub fn resolve(&self, name: &str) -> Option<&IndexedFunction> {
        let norm = normalize_name(name);
        match self.functions.get(&norm) {
            Some(defs) if defs.len() == 1 => Some(&defs[0]),
            _ => None,
        }
    }

    /// Number of distinct normalized function names.
    pub fn name_count(&self) -> usize {
        self.functions.len()
    }

    /// Total number of function definitions (including duplicates).
    pub fn total_defs(&self) -> usize {
        self.functions.values().map(|v| v.len()).sum()
    }

    /// Number of uniquely resolvable functions (exactly 1 definition).
    pub fn unique_count(&self) -> usize {
        self.functions.values().filter(|v| v.len() == 1).count()
    }

    /// Estimated memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.functions
            .values()
            .flat_map(|defs| defs.iter())
            .map(|f| f.name.len() + f.text.len() + f.callees.iter().map(|c| c.len()).sum::<usize>() + 64)
            .sum()
    }

    /// Merge another index into this one (for combining thread-local indexes).
    pub fn merge(&mut self, other: GlobalIndex) {
        for (name, defs) in other.functions {
            self.functions.entry(name).or_default().extend(defs);
        }
    }

    /// Remove entries with multiple definitions (ambiguous, not useful for resolution).
    /// Call after merging all thread-local indexes to free memory before Phase 2.
    pub fn prune_ambiguous(&mut self) {
        self.functions.retain(|_, defs| defs.len() == 1);
        self.functions.shrink_to_fit();
    }
}
