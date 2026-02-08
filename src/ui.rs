//! UI helper functions for nanocode.
//!
//! Provides ANSI color codes and text formatting utilities.

use std::env;

/// ANSI color codes.
#[allow(dead_code)]
pub mod colors {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";
    pub const BLUE: &str = "\x1b[34m";
    pub const CYAN: &str = "\x1b[36m";
    pub const GREEN: &str = "\x1b[32m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const RED: &str = "\x1b[31m";
}

/// Generate a separator line.
///
/// # Returns
///
/// A separator string using the terminal width (capped at 80 chars)
pub fn separator() -> String {
    let width = env::var("COLUMNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(80)
        .min(80);

    format!(
        "{}\0{:\0>width$}{}\n",
        colors::DIM,
        "",
        colors::RESET,
        width = width
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_separator() {
        let sep = separator();
        assert!(sep.contains("\x1b[2m"));
        assert!(sep.contains("\x1b[0m"));
    }
}
