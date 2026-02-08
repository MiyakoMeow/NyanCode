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
    pub params: &'static [(&'static str, &'static str)],
}

/// Generate tool schemas for Anthropic API.
///
/// # Returns
///
/// Vector of tool schema definitions compatible with Anthropic Messages API.
pub fn make_schema() -> Vec<Value> {
    let tools = vec![
        ToolDef {
            name: "read",
            description: "Read file with line numbers (file path, not directory)",
            params: ToolParams {
                params: &[
                    ("path", "string"),
                    ("offset", "number?"),
                    ("limit", "number?"),
                ],
            },
        },
        ToolDef {
            name: "write",
            description: "Write content to file",
            params: ToolParams {
                params: &[("path", "string"), ("content", "string")],
            },
        },
        ToolDef {
            name: "edit",
            description: "Replace old with new in file (old must be unique unless all=true)",
            params: ToolParams {
                params: &[
                    ("path", "string"),
                    ("old", "string"),
                    ("new", "string"),
                    ("all", "boolean?"),
                ],
            },
        },
        ToolDef {
            name: "glob",
            description: "Find files by pattern, sorted by mtime",
            params: ToolParams {
                params: &[("pat", "string"), ("path", "string?")],
            },
        },
        ToolDef {
            name: "grep",
            description: "Search files for regex pattern",
            params: ToolParams {
                params: &[("pat", "string"), ("path", "string?")],
            },
        },
        ToolDef {
            name: "bash",
            description: "Run shell command",
            params: ToolParams {
                params: &[("cmd", "string")],
            },
        },
    ];

    tools
        .into_iter()
        .map(|tool| {
            let mut properties = serde_json::Map::new();
            let mut required = Vec::new();

            for (param_name, param_type) in tool.params.params {
                let is_optional = param_type.ends_with('?');
                let base_type = param_type.trim_end_matches('?');

                let json_type = match base_type {
                    "number" => "integer",
                    "boolean" => "boolean",
                    _ => "string",
                };

                properties.insert(param_name.to_string(), json!({ "type": json_type }));

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
