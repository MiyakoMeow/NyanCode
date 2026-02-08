//! Anthropic API client for nanocode.
//!
//! Handles API calls and the agentic loop for tool execution.

use crate::config::Config;
use crate::schema;
use crate::tools;
use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::{Value, json};

/// Anthropic API client.
pub struct AnthropicClient {
    client: Client,
    config: Config,
}

impl AnthropicClient {
    /// Create a new API client with the given configuration.
    pub fn new(config: Config) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    /// Call the Anthropic Messages API.
    ///
    /// # Arguments
    ///
    /// * `messages` - Conversation history
    /// * `system_prompt` - System prompt for the model
    ///
    /// # Returns
    ///
    /// API response as JSON
    async fn call_api(&self, messages: &[Value], system_prompt: &str) -> Result<Value> {
        let url = format!("{}/v1/messages", self.config.base_url);

        let request_body = json!({
            "model": self.config.model,
            "max_tokens": 8192,
            "system": system_prompt,
            "messages": messages,
            "tools": schema::make_schema(),
        });

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .header("x-api-key", &self.config.api_key)
            .json(&request_body)
            .send()
            .await
            .context("Failed to send API request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("API error: {status} - {error_text}");
        }

        let response_json: Value = response
            .json()
            .await
            .context("Failed to parse API response")?;

        Ok(response_json)
    }

    /// Run the agentic loop: keep calling API until no more tool calls.
    ///
    /// # Arguments
    ///
    /// * `messages` - Mutable reference to conversation history
    /// * `system_prompt` - System prompt for the model
    ///
    /// # Returns
    ///
    /// Ok(()) on success, Err on failure
    pub async fn run_agent_loop(
        &self,
        messages: &mut Vec<Value>,
        system_prompt: &str,
    ) -> Result<()> {
        loop {
            let response = self.call_api(messages, system_prompt).await?;
            let content_blocks = response
                .get("content")
                .and_then(|v| v.as_array())
                .ok_or_else(|| anyhow::anyhow!("Invalid response: missing content"))?;

            let mut tool_results = Vec::new();

            for block in content_blocks {
                let block_type = block
                    .get("type")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("Invalid block: missing type"))?;

                match block_type {
                    "text" => {
                        if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                            println!("\n⏺ {text}");
                        }
                    }
                    "tool_use" => {
                        let id = block
                            .get("id")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| anyhow::anyhow!("Invalid tool_use: missing id"))?;
                        let name = block
                            .get("name")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| anyhow::anyhow!("Invalid tool_use: missing name"))?;
                        let input = block
                            .get("input")
                            .and_then(|v| v.as_object())
                            .ok_or_else(|| anyhow::anyhow!("Invalid tool_use: missing input"))?;

                        // Show tool call
                        let arg_preview = input
                            .values()
                            .next()
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .chars()
                            .take(50)
                            .collect::<String>();
                        println!("\n⏺ {name}({arg_preview})");

                        // Execute tool
                        let result: String = self.run_tool(name, input).await;

                        // Show preview
                        let result_lines: Vec<&str> = result.lines().collect();
                        let preview = result_lines
                            .first()
                            .map(|s: &&str| s.chars().take(60).collect::<String>())
                            .unwrap_or_default();
                        if result_lines.len() > 1 || preview.len() < result.lines().count() {
                            println!("  ⎿  {preview} ... +{} lines", result_lines.len() - 1);
                        } else {
                            println!("  ⎿  {preview}");
                        }

                        tool_results.push(json!({
                            "type": "tool_result",
                            "tool_use_id": id,
                            "content": result,
                        }));
                    }
                    _ => {
                        anyhow::bail!("Unknown block type: {block_type}");
                    }
                }
            }

            // Add assistant response to messages
            messages.push(json!({
                "role": "assistant",
                "content": content_blocks,
            }));

            // If no tool results, we're done
            if tool_results.is_empty() {
                break;
            }

            // Add tool results as user message
            messages.push(json!({
                "role": "user",
                "content": tool_results,
            }));
        }

        Ok(())
    }

    /// Execute a single tool call.
    ///
    /// # Arguments
    ///
    /// * `name` - Tool name
    /// * `input` - Tool input parameters
    ///
    /// # Returns
    ///
    /// Tool result as string, or error message
    async fn run_tool(&self, name: &str, input: &serde_json::Map<String, Value>) -> String {
        let result = async {
            match name {
                "read" => {
                    let path = input
                        .get("path")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow::anyhow!("Missing path"))?;
                    let offset = input
                        .get("offset")
                        .and_then(serde_json::Value::as_i64)
                        .and_then(|v| usize::try_from(v).ok());
                    let limit = input
                        .get("limit")
                        .and_then(serde_json::Value::as_i64)
                        .and_then(|v| usize::try_from(v).ok());
                    tools::read_tool(path, offset, limit).await
                }
                "write" => {
                    let path = input
                        .get("path")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow::anyhow!("Missing path"))?;
                    let content = input
                        .get("content")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow::anyhow!("Missing content"))?;
                    tools::write_tool(path, content).await
                }
                "edit" => {
                    let path = input
                        .get("path")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow::anyhow!("Missing path"))?;
                    let old = input
                        .get("old")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow::anyhow!("Missing old"))?;
                    let new = input
                        .get("new")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow::anyhow!("Missing new"))?;
                    let all = input.get("all").and_then(serde_json::Value::as_bool);
                    tools::edit_tool(path, old, new, all).await
                }
                "glob" => {
                    let pat = input
                        .get("pat")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow::anyhow!("Missing pat"))?;
                    let path = input.get("path").and_then(|v| v.as_str());
                    Ok(tools::glob_tool(pat, path)?)
                }
                "grep" => {
                    let pat = input
                        .get("pat")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow::anyhow!("Missing pat"))?;
                    let path = input.get("path").and_then(|v| v.as_str());
                    tools::grep_tool(pat, path).await
                }
                "bash" => {
                    let cmd = input
                        .get("cmd")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow::anyhow!("Missing cmd"))?;
                    tools::bash_tool(cmd).await
                }
                _ => anyhow::bail!("Unknown tool: {name}"),
            }
        }
        .await;

        match result {
            Ok(s) => s,
            Err(e) => format!("error: {e}"),
        }
    }
}
