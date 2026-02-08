//! Tool implementations for nanocode.
//!
//! Provides six async tools: read, write, edit, glob, grep, bash.

use anyhow::{Context, Result};
use tokio::fs;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

/// Read file contents with line numbers.
///
/// # Arguments
///
/// * `path` - File path to read
/// * `offset` - Starting line number (0-based, default 0)
/// * `limit` - Maximum number of lines to read (default: all)
///
/// # Returns
///
/// File contents with line numbers in format "    1| line content"
pub async fn read_tool(path: &str, offset: Option<usize>, limit: Option<usize>) -> Result<String> {
    let content = fs::read_to_string(path)
        .await
        .with_context(|| format!("Failed to read file: {path}"))?;

    let lines: Vec<&str> = content.lines().collect();
    let offset = offset.unwrap_or(0);
    let limit = limit.unwrap_or(lines.len());

    let selected: Vec<&str> = lines.iter().skip(offset).take(limit).copied().collect();

    let result = selected
        .iter()
        .enumerate()
        .map(|(idx, line)| format!("{:4}| {}", offset + idx + 1, line))
        .collect::<Vec<_>>()
        .join("\n");

    Ok(result)
}

/// Write content to a file.
///
/// # Arguments
///
/// * `path` - File path to write
/// * `content` - Content to write
///
/// # Returns
///
/// "ok" on success
pub async fn write_tool(path: &str, content: &str) -> Result<String> {
    fs::write(path, content)
        .await
        .with_context(|| format!("Failed to write file: {path}"))?;

    Ok("ok".to_string())
}

/// Edit file by replacing old string with new string.
///
/// # Arguments
///
/// * `path` - File path to edit
/// * `old` - Old string to replace
/// * `new` - New string to replace with
/// * `all` - Replace all occurrences if true, else require unique match
///
/// # Returns
///
/// "ok" on success, error message on failure
pub async fn edit_tool(path: &str, old: &str, new: &str, all: Option<bool>) -> Result<String> {
    let content = fs::read_to_string(path)
        .await
        .with_context(|| format!("Failed to read file: {path}"))?;

    if !content.contains(old) {
        return Ok("error: old_string not found".to_string());
    }

    let count = content.matches(old).count();
    let replace_all = all.unwrap_or(false);

    if !replace_all && count > 1 {
        return Ok(format!(
            "error: old_string appears {count} times, must be unique (use all=true)"
        ));
    }

    let replacement = if replace_all {
        content.replacen(old, new, count)
    } else {
        content.replacen(old, new, 1)
    };

    fs::write(path, replacement)
        .await
        .with_context(|| format!("Failed to write file: {path}"))?;

    Ok("ok".to_string())
}

/// Find files matching a glob pattern, sorted by modification time.
///
/// # Arguments
///
/// * `pat` - Glob pattern (e.g., "**/*.rs")
/// * `path` - Base directory for search (default: ".")
///
/// # Returns
///
/// Newline-separated list of matching files, or "none" if no matches
pub fn glob_tool(pat: &str, path: Option<&str>) -> Result<String> {
    let base = path.unwrap_or(".");
    let pattern = format!("{}/{}", base.replace('\\', "/"), pat).replace("//", "/");

    let mut files = Vec::new();

    for entry in
        glob::glob(&pattern).with_context(|| format!("Failed to read glob pattern: {pattern}"))?
    {
        match entry {
            Ok(path) => {
                if path.is_file() {
                    let mtime =
                        path.metadata()
                            .ok()
                            .and_then(|m| m.modified().ok())
                            .map_or(0, |t| {
                                t.duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_secs()
                            });
                    files.push((path, mtime));
                }
            }
            Err(e) => {
                eprintln!("Glob error: {e}");
            }
        }
    }

    // Sort by modification time (newest first)
    files.sort_by(|a, b| b.1.cmp(&a.1));

    if files.is_empty() {
        Ok("none".to_string())
    } else {
        let paths: Vec<String> = files
            .into_iter()
            .map(|(p, _)| p.display().to_string().replace('\\', "/"))
            .collect();
        Ok(paths.join("\n"))
    }
}

/// Search files for regex pattern matches.
///
/// # Arguments
///
/// * `pat` - Regular expression pattern
/// * `path` - Base directory for search (default: ".")
///
/// # Returns
///
/// Newline-separated matches in format "path:line:content", up to 50 matches
pub async fn grep_tool(pat: &str, path: Option<&str>) -> Result<String> {
    let base = path.unwrap_or(".");
    let pattern =
        regex::Regex::new(pat).with_context(|| format!("Invalid regex pattern: {pat}"))?;

    let mut hits = Vec::new();

    for entry in walkdir::WalkDir::new(base)
        .follow_links(true)
        .max_depth(100)
        .into_iter()
        .filter_map(std::result::Result::ok)
    {
        let filepath = entry.path();
        if filepath.is_file()
            && let Ok(content) = fs::read_to_string(filepath).await
        {
            for (line_num, line) in content.lines().enumerate() {
                if pattern.is_match(line) {
                    hits.push(format!(
                        "{}:{}:{}",
                        filepath.display().to_string().replace('\\', "/"),
                        line_num + 1,
                        line.trim()
                    ));
                }
            }
        }
    }

    if hits.is_empty() {
        Ok("none".to_string())
    } else {
        Ok(hits.iter().take(50).cloned().collect::<Vec<_>>().join("\n"))
    }
}

/// Execute a shell command and return output.
///
/// # Arguments
///
/// * `cmd` - Shell command to execute
///
/// # Returns
///
/// Command output, or "(empty)" if no output
#[cfg(unix)]
pub async fn bash_tool(cmd: &str) -> Result<String> {
    use std::process::Stdio;

    let mut child = Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("Failed to execute command: {cmd}"))?;

    let stdout = child.stdout.take().expect("Failed to capture stdout");
    let stderr = child.stderr.take().expect("Failed to capture stderr");

    let stdout_reader = BufReader::new(stdout);
    let mut output_lines = Vec::new();

    let mut lines = stdout_reader.lines();
    while let Ok(Some(line)) = lines.next_line().await {
        println!("  │ {line}");
        output_lines.push(line);
    }

    // Also capture stderr
    let stderr_reader = BufReader::new(stderr);
    let mut stderr_lines = stderr_reader.lines();
    while let Ok(Some(line)) = stderr_lines.next_line().await {
        println!("  │ {line}");
        output_lines.push(line);
    }

    let status = child
        .wait()
        .await
        .with_context(|| format!("Failed to wait for command: {cmd}"))?;

    if !status.success() {
        let code = status.code().unwrap_or(-1);
        output_lines.push(format!("(exit code: {code})"));
    }

    let result = output_lines.join("\n");
    if result.is_empty() {
        Ok("(empty)".to_string())
    } else {
        Ok(result.trim().to_string())
    }
}

/// Execute a shell command and return output (Windows version).
///
/// # Arguments
///
/// * `cmd` - Shell command to execute
///
/// # Returns
///
/// Command output, or "(empty)" if no output
#[cfg(windows)]
pub async fn bash_tool(cmd: &str) -> Result<String> {
    use std::process::Stdio;

    let mut child = Command::new("cmd")
        .args(["/C", cmd])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("Failed to execute command: {cmd}"))?;

    let stdout = child.stdout.take().expect("Failed to capture stdout");
    let stderr = child.stderr.take().expect("Failed to capture stderr");

    let stdout_reader = BufReader::new(stdout);
    let mut output_lines = Vec::new();

    let mut lines = stdout_reader.lines();
    while let Ok(Some(line)) = lines.next_line().await {
        println!("  │ {line}");
        output_lines.push(line);
    }

    // Also capture stderr
    let stderr_reader = BufReader::new(stderr);
    let mut stderr_lines = stderr_reader.lines();
    while let Ok(Some(line)) = stderr_lines.next_line().await {
        println!("  │ {line}");
        output_lines.push(line);
    }

    let status = child
        .wait()
        .await
        .with_context(|| format!("Failed to wait for command: {cmd}"))?;

    if !status.success() {
        let code = status.code().unwrap_or(-1);
        output_lines.push(format!("(exit code: {code})"));
    }

    let result = output_lines.join("\n");
    if result.is_empty() {
        Ok("(empty)".to_string())
    } else {
        Ok(result.trim().to_string())
    }
}
