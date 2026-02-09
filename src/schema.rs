//! Tool schema generation for Anthropic API.
//!
//! Generates JSON Schema definitions for the tools.

use serde_json::{Value, json};

/// Tool definition with description and parameters.
pub struct ToolDef {
    pub name: &'static str,
    pub description: &'static str,
    pub params: ToolParams,
}

/// Tool parameter definitions.
pub struct ToolParams {
    pub params: &'static [(&'static str, &'static str, &'static str)], // (name, type, description)
}

/// Create the tool definition for the 'read' tool.
fn read_tool_def() -> ToolDef {
    ToolDef {
        name: "read",
        description: "Read the contents of a file and display with line numbers. Use this tool to examine source code, configuration files, documentation, and any text-based files. The path must be a valid absolute file path (not a directory). Optional offset and limit parameters allow reading specific portions of large files.",
        params: ToolParams {
            params: &[
                (
                    "path",
                    "string",
                    "Absolute path to the file to read. Must be a valid file path, not a directory. Example: '/home/user/project/src/main.rs' or 'C:\\Users\\user\\project\\src\\main.rs'",
                ),
                (
                    "offset",
                    "number?",
                    "Optional line number to start reading from (0-based). If not specified, starts from the beginning of the file. For example, offset=10 starts reading from line 11.",
                ),
                (
                    "limit",
                    "number?",
                    "Optional maximum number of lines to read. If not specified, reads the entire file from the offset. Useful for reading specific portions of large files.",
                ),
            ],
        },
    }
}

/// Create the tool definition for the 'write' tool.
fn write_tool_def() -> ToolDef {
    ToolDef {
        name: "write",
        description: "Write content to a file, creating it if it doesn't exist or overwriting if it does. Use this tool to create new files or completely replace existing file contents. The file will be created with appropriate permissions. For partial file modifications, use the edit tool instead.",
        params: ToolParams {
            params: &[
                (
                    "path",
                    "string",
                    "Absolute path to the file to write. The file will be created if it doesn't exist, or completely overwritten if it does. Example: '/home/user/project/src/new_file.rs'",
                ),
                (
                    "content",
                    "string",
                    "The complete content to write to the file. This will replace all existing content in the file. For partial modifications, use the edit tool instead.",
                ),
            ],
        },
    }
}

/// Create the tool definition for the 'edit' tool.
fn edit_tool_def() -> ToolDef {
    ToolDef {
        name: "edit",
        description: "Replace occurrences of a string with another string in a file. Use this tool to make targeted changes to existing files. By default, only the first occurrence is replaced. Set all=true to replace all occurrences (use carefully, as the old string must be unique unless all=true is specified).",
        params: ToolParams {
            params: &[
                (
                    "path",
                    "string",
                    "Absolute path to the file to edit. Must be an existing file. Example: '/home/user/project/src/main.rs'",
                ),
                (
                    "old",
                    "string",
                    "The exact string to search for and replace. Must be unique in the file unless all=true is specified. The replacement is case-sensitive and must match exactly.",
                ),
                (
                    "new",
                    "string",
                    "The replacement string that will replace all occurrences of the old string.",
                ),
                (
                    "all",
                    "boolean?",
                    "If true, replace all occurrences of the old string. If false or not specified, only the first occurrence is replaced. Use with caution - ensure the old string is unique when using this option.",
                ),
            ],
        },
    }
}

/// Create the tool definition for the 'glob' tool.
fn glob_tool_def() -> ToolDef {
    ToolDef {
        name: "glob",
        description: "Find files matching a glob pattern, sorted by modification time (newest first). Use this tool to discover files in a project, especially when you know part of the filename or want to find all files of a certain type. Supports standard glob patterns like *.rs, **/*.toml, etc.",
        params: ToolParams {
            params: &[
                (
                    "pat",
                    "string",
                    "Glob pattern to match files against. Supports wildcards: * matches any characters, ** matches directories recursively. Examples: '*.rs' (all .rs files in current directory), '**/*.toml' (all .toml files in all subdirectories), 'src/**/*.rs' (all .rs files in src directory tree)",
                ),
                (
                    "path",
                    "string?",
                    "Optional base directory to search in. If not specified, searches in the current working directory. Must be a valid directory path. Example: '/home/user/project'",
                ),
            ],
        },
    }
}

/// Create the tool definition for the 'grep' tool.
fn grep_tool_def() -> ToolDef {
    ToolDef {
        name: "grep",
        description: "Search for a regular expression pattern in files. Use this tool to find specific text, function definitions, or patterns across multiple files. Supports standard regex syntax. This is powerful for finding where functions are called, variables are used, or specific patterns appear in code.",
        params: ToolParams {
            params: &[
                (
                    "pat",
                    "string",
                    "Regular expression pattern to search for. Supports standard regex syntax. Examples: 'fn main' (find function definitions), 'TODO|FIXME' (find markers), 'struct \\w+' (find struct definitions). The search is case-sensitive by default.",
                ),
                (
                    "path",
                    "string?",
                    "Optional directory or file path to search in. If not specified, searches in the current working directory recursively. Can be a directory path (searches all files recursively) or a specific file path. Example: '/home/user/project/src'",
                ),
            ],
        },
    }
}

/// Create the tool definition for the 'bash' tool.
fn bash_tool_def() -> ToolDef {
    ToolDef {
        name: "bash",
        description: "Execute a shell command and return the output. Use this tool to run build commands, tests, Git operations, or any system commands. The command runs in the current working directory. Commands are executed asynchronously with a timeout.",
        params: ToolParams {
            params: &[(
                "cmd",
                "string",
                "The shell command to execute. Can be any valid shell command. Examples: 'cargo build', 'git status', 'ls -la', 'npm test'. The command will be executed in the current working directory.",
            )],
        },
    }
}

/// Generate tool schemas for Anthropic API.
///
/// # Returns
///
/// Vector of tool schema definitions compatible with Anthropic Messages API.
pub fn make_schema() -> Vec<Value> {
    let tools = vec![
        read_tool_def(),
        write_tool_def(),
        edit_tool_def(),
        glob_tool_def(),
        grep_tool_def(),
        bash_tool_def(),
    ];

    tools
        .into_iter()
        .map(|tool| {
            let mut properties = serde_json::Map::new();
            let mut required = Vec::new();

            for (param_name, param_type, param_description) in tool.params.params {
                let is_optional = param_type.ends_with('?');
                let base_type = param_type.trim_end_matches('?');

                let json_type = match base_type {
                    "number" => "number",
                    "integer" => "integer",
                    "boolean" => "boolean",
                    _ => "string",
                };

                properties.insert(
                    param_name.to_string(),
                    json!({
                        "type": json_type,
                        "description": param_description
                    }),
                );

                if !is_optional {
                    required.push(param_name.to_string());
                }
            }

            json!({
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            })
        })
        .collect()
}
