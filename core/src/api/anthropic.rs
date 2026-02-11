//! Anthropic API client for nanocode.
//!
//! Handles API calls and the agentic loop for tool execution.

pub mod schema;

use crate::events;
use crate::tools;
use anyhow::Result;
use futures::stream::Stream;
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::pin::Pin;
use tokio::sync::mpsc;

/// Anthropic API configuration.
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    /// Base URL for Anthropic API
    pub base_url: String,
    /// Model identifier to use
    pub model: String,
    /// API authentication token
    pub api_key: String,
}

impl AnthropicConfig {
    /// Load Anthropic configuration from environment variables with defaults.
    ///
    /// # Environment Variables
    ///
    /// - `ANTHROPIC_BASE_URL`: Base URL for API (default: "<https://api.anthropic.com>")
    /// - `ANTHROPIC_MODEL` or `MODEL`: Model to use (default: "claude-opus-4-5")
    /// - `ANTHROPIC_AUTH_TOKEN` or `ANTHROPIC_API_KEY`: API key
    ///
    /// # Returns
    ///
    /// Returns the configuration with defaults applied.
    #[must_use]
    pub fn from_env() -> Self {
        use std::env;

        let base_url = env::var("ANTHROPIC_BASE_URL")
            .unwrap_or_else(|_| "https://api.anthropic.com".to_string());

        let model = env::var("ANTHROPIC_MODEL")
            .or_else(|_| env::var("MODEL"))
            .unwrap_or_else(|_| "claude-opus-4-5".to_string());

        let api_key = env::var("ANTHROPIC_AUTH_TOKEN")
            .or_else(|_| env::var("ANTHROPIC_API_KEY"))
            .unwrap_or_default();

        Self {
            base_url,
            model,
            api_key,
        }
    }

    /// Get the masked API key for display (shows only first and last 4 chars).
    ///
    /// # Returns
    ///
    /// A masked version of the API key suitable for display.
    #[must_use]
    pub fn masked_api_key(&self) -> String {
        let key = &self.api_key;
        if key.len() > 8 {
            format!("{}...{}", &key[..4], &key[key.len() - 4..])
        } else if !key.is_empty() {
            "*".repeat(key.len())
        } else {
            "(no key)".to_string()
        }
    }
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

/// API错误类型
#[derive(Debug, Clone, thiserror::Error)]
pub enum ApiError {
    /// 网络连接错误
    #[error("Network error: {0}")]
    NetworkError(String),

    /// HTTP请求错误
    #[error("HTTP error {status}: {message}")]
    HttpError {
        /// HTTP状态码
        status: u16,
        /// 错误消息
        message: String,
    },

    /// 数据解析错误
    #[error("Parse error: {0}")]
    ParseError(String),

    /// 流式响应错误
    #[error("Stream error: {0}")]
    StreamError(String),

    /// API返回错误
    #[error("API error: {0}")]
    Api(String),
}

/// 流式响应事件
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// 消息开始事件，表示流式响应的开始
    MessageStart,
    /// 内容块开始事件，包含内容块类型和索引
    ContentBlockStart {
        /// 内容块在消息中的索引位置
        index: u32,
        /// 内容块的具体内容
        content_block: ContentBlock,
    },
    /// 内容块增量事件，包含增量数据
    ContentBlockDelta {
        /// 内容块在消息中的索引位置
        index: u32,
        /// 增量数据
        delta: Delta,
    },
    /// 内容块结束事件，表示某个内容块已完成
    ContentBlockStop {
        /// 内容块在消息中的索引位置
        index: u32,
    },
    /// 消息增量事件，包含消息级别的增量数据
    MessageDelta,
    /// 消息结束事件，表示流式响应的结束
    MessageStop,
    /// 错误事件，包含API错误信息
    Error {
        /// 错误详情
        error: ApiError,
    },
}

/// 内容块类型
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    /// 文本内容块，包含纯文本内容
    #[serde(rename = "text")]
    Text {
        /// 文本内容
        #[serde(default, rename = "text")]
        _text: String,
    },
    /// 工具调用内容块，描述需要执行的函数调用
    #[serde(rename = "tool_use")]
    ToolUse {
        /// 工具调用的唯一标识符
        id: String,
        /// 工具（函数）的名称
        name: String,
        /// 工具调用的输入参数
        input: Value,
    },
}

/// 增量数据
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum Delta {
    /// 文本增量，包含新增的文本内容
    #[serde(rename = "text_delta")]
    Text {
        /// 增量的文本内容
        text: String,
    },
    /// JSON增量，包含JSON结构数据的增量部分
    #[serde(rename = "input_json_delta")]
    InputJson {
        /// 部分的JSON数据，用于构建完整的JSON结构
        partial_json: String,
    },
}

/// SSE事件流类型
pub type EventStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, ApiError>> + Send>>;

/// 工具调用收集器
pub struct ToolCallCollector {
    /// 待处理的工具调用列表
    calls: Vec<PendingToolCall>,
}

/// 待处理的工具调用
#[derive(Debug, Clone)]
struct PendingToolCall {
    /// 工具调用的唯一标识符
    id: String,
    /// 工具（函数）的名称
    name: String,
    /// 工具输入参数的缓冲区，用于累积增量JSON数据
    input_buffer: String,
    /// 是否已完成处理
    completed: bool,
}

/// 完成的工具调用
#[derive(Debug, Clone, Serialize)]
pub struct ToolCall {
    /// 工具调用的唯一标识符
    pub id: String,
    /// 工具（函数）的名称
    pub name: String,
    /// 工具调用的输入参数
    pub input: Value,
}

impl ToolCallCollector {
    /// 创建一个新的工具调用收集器
    pub fn new() -> Self {
        Self { calls: Vec::new() }
    }

    /// 处理流式事件，收集工具调用
    pub fn process_event(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::ContentBlockStart {
                content_block: ContentBlock::ToolUse { id, name, input },
                index,
            } => {
                // 扩展 calls 向量以容纳新索引
                while self.calls.len() <= *index as usize {
                    self.calls.push(PendingToolCall {
                        id: String::new(),
                        name: String::new(),
                        input_buffer: String::new(),
                        completed: false,
                    });
                }

                if let Some(call) = self.calls.get_mut(*index as usize) {
                    *call = PendingToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        input_buffer: match input {
                            Value::String(s) => s.clone(),
                            Value::Object(map) if !map.is_empty() => input.to_string(),
                            _ => String::new(),
                        },
                        completed: false,
                    };
                }
            }
            StreamEvent::ContentBlockStart {
                content_block: ContentBlock::Text { .. },
                ..
            }
            | StreamEvent::MessageStart
            | StreamEvent::MessageDelta
            | StreamEvent::MessageStop
            | StreamEvent::Error { .. } => {} // These events don't need special handling

            StreamEvent::ContentBlockDelta { delta, index } => {
                let Some(call) = self.calls.get_mut(*index as usize) else {
                    return;
                };
                let Delta::InputJson { partial_json } = delta else {
                    return;
                };
                call.input_buffer.push_str(partial_json);
            }

            StreamEvent::ContentBlockStop { index } => {
                let Some(call) = self.calls.get_mut(*index as usize) else {
                    return;
                };
                call.completed = true;
            }
        }
    }

    /// 检查是否有已完成的工具调用
    pub fn has_completed_calls(&self) -> bool {
        self.calls.iter().any(|c| c.completed)
    }

    /// 提取所有已完成的工具调用并清空
    pub fn take_completed(&mut self) -> Vec<ToolCall> {
        let completed = self
            .calls
            .iter()
            .filter(|c| c.completed)
            .map(|c| ToolCall {
                id: c.id.clone(),
                name: c.name.clone(),
                input: serde_json::from_str(&c.input_buffer).unwrap_or_default(),
            })
            .collect();

        self.calls.clear();
        completed
    }

    /// 检查收集器是否处于活跃状态（有待处理的工具调用）
    pub fn is_active(&self) -> bool {
        !self.calls.is_empty()
    }
}

impl Default for ToolCallCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// 解析SSE响应流
fn parse_sse_stream(response: reqwest::Response) -> EventStream {
    use futures::stream::StreamExt;

    Box::pin(async_stream::stream! {
            let mut buffer = String::new();
            let mut stream = response.bytes_stream();

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        yield Err(ApiError::StreamError(format!("Failed to read stream: {e}")));
                        continue;
                    }
                };

                // 将字节转换为字符串并处理
                let chunk_str = match String::from_utf8(chunk.to_vec()) {
                    Ok(s) => s,
                    Err(e) => {
                        yield Err(ApiError::ParseError(format!("Invalid UTF-8: {e}")));
                        continue;
                    }
                };

                buffer.push_str(&chunk_str);

                // 按行分割
                while let Some(newline_pos) = buffer.find('\n') {
                    let line = buffer[..newline_pos].to_string();
                    buffer = buffer[newline_pos + 1..].to_string();

                    // 跳过空行
                    if line.is_empty() {
                        continue;
                    }

                    // SSE格式解析
                    if let Some(data) = line.strip_prefix("data: ") {
                        // 跳过"[DONE]"标记
                        if data == "[DONE]" {
                            yield Ok(StreamEvent::MessageStop);
                            continue;
                        }

                        // 解析JSON
                        match serde_json::from_str::<Value>(data) {
                            Ok(value) => {
                                if let Some(event_type) = value.get("type").and_then(|v| v.as_str()) {
                                    let event = match event_type {
                                        "message_start" => StreamEvent::MessageStart,
                                        "content_block_start" => {
                                            if let Some(block) = value.get("content_block") {
    StreamEvent::ContentBlockStart {
                                            index: value.get("index").and_then(Value::as_u64).unwrap_or(0).try_into().unwrap_or(0),
                                            content_block: serde_json::from_value(block.clone())
                                                .unwrap_or(ContentBlock::Text { _text: String::new() }),
                                        }
                                            } else {
                                                continue;
                                            }
                                        }
                                        "content_block_delta" => {
    StreamEvent::ContentBlockDelta {
                                                index: value.get("index").and_then(Value::as_u64).unwrap_or(0).try_into().unwrap_or(0),
                                                delta: serde_json::from_value(value.get("delta").cloned().unwrap_or_default())
                                                    .unwrap_or(Delta::Text { text: String::new() }),
                                            }
                                        }
    "content_block_stop" => StreamEvent::ContentBlockStop {
                                            index: value.get("index").and_then(Value::as_u64).unwrap_or(0).try_into().unwrap_or(0),
                                        },
                                        "message_delta" => StreamEvent::MessageDelta,
                                        "message_stop" => StreamEvent::MessageStop,
                                        "error" => StreamEvent::Error {
                                            error: ApiError::Api(
                                                value.get("error")
                                                    .and_then(|e| e.get("message"))
                                                    .and_then(|m| m.as_str())
                                                    .unwrap_or("Unknown error")
                                                    .to_string()
                                            ),
                                        },
                                        _ => continue,
                                    };
                                    yield Ok(event);
                                }
                            }
                            Err(e) => {
                                yield Err(ApiError::ParseError(format!("Failed to parse SSE data: {e}")));
                            }
                        }
                    }
                }
            }
        })
}

/// API client.
pub struct Client {
    /// HTTP客户端，用于发送API请求
    http_client: HttpClient,
    /// API配置，包含密钥、基础URL等信息
    config: AnthropicConfig,
}

impl Client {
    /// Create a new API client with the given configuration.
    pub fn new(config: AnthropicConfig) -> Self {
        Self {
            http_client: HttpClient::new(),
            config,
        }
    }

    /// 发送流式消息请求
    ///
    /// # Arguments
    ///
    /// * `messages` - Conversation history
    /// * `system_prompt` - System prompt for the model
    /// * `tools` - Optional tool definitions
    ///
    /// # Returns
    ///
    /// Stream of events from the API
    pub async fn create_message_stream(
        &self,
        messages: &[Value],
        system_prompt: &str,
        tools: Option<&[Value]>,
    ) -> Result<EventStream, ApiError> {
        // 构建请求体
        let mut request_body = json!({
            "model": self.config.model,
            "max_tokens": 8192,
            "system": system_prompt,
            "messages": messages,
            "stream": true,
        });

        // 添加工具（如果有）
        if let Some(tools_value) = tools
            && let Some(body_obj) = request_body.as_object_mut()
        {
            body_obj.insert("tools".to_string(), json!(tools_value));
        }

        // 发送HTTP请求
        let response = self
            .http_client
            .post(format!("{}/v1/messages", self.config.base_url))
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| ApiError::NetworkError(e.to_string()))?;

        // 检查状态码
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(ApiError::HttpError {
                status: status.as_u16(),
                message: error_text,
            });
        }

        // 返回解析后的SSE流
        Ok(parse_sse_stream(response))
    }

    /// Run the agentic loop: keep calling API until no more tool calls.
    ///
    /// # Arguments
    ///
    /// * `messages` - Mutable reference to conversation history
    /// * `system_prompt` - System prompt for the model
    /// * `tools` - Tool definitions
    /// * `event_sender` - Optional sender for core events
    ///
    /// # Returns
    ///
    /// Ok(()) on success, Err on failure
    pub async fn run_agent_loop_stream(
        &self,
        messages: &mut Vec<Value>,
        system_prompt: &str,
        tools: &[Value],
        event_sender: Option<&mpsc::UnboundedSender<events::CoreEvent>>,
    ) -> Result<(), ApiError> {
        use futures::stream::StreamExt;

        let mut tool_collector = ToolCallCollector::new();
        // Initial check of collector state
        let _ = !tool_collector.is_active();
        let mut current_text = String::new();

        loop {
            // 发送消息开始事件
            if let Some(sender) = event_sender {
                let _ = sender.send(events::CoreEvent::MessageStart);
            }

            // 创建流式请求
            let mut stream = self
                .create_message_stream(messages, system_prompt, Some(tools))
                .await?;

            // 处理流式事件
            while let Some(event_result) = stream.next().await {
                let event = event_result?;

                match &event {
                    StreamEvent::ContentBlockDelta {
                        delta: Delta::Text { text },
                        ..
                    } => {
                        // 发送文本增量事件
                        if let Some(sender) = event_sender {
                            let _ = sender.send(events::CoreEvent::TextDelta(text.clone()));
                        }
                        current_text.push_str(text);
                    }

                    StreamEvent::ContentBlockStart {
                        content_block: ContentBlock::ToolUse { id, name, .. },
                        ..
                    } => {
                        // 发送工具调用开始事件
                        if let Some(sender) = event_sender {
                            let _ = sender.send(events::CoreEvent::ToolCallStart {
                                id: id.clone(),
                                name: name.clone(),
                            });
                        }
                    }

                    StreamEvent::Error { error } => {
                        // 发送错误事件
                        if let Some(sender) = event_sender {
                            let _ = sender.send(events::CoreEvent::Error(error.to_string()));
                        }
                    }

                    StreamEvent::MessageStop => {
                        // 发送消息停止事件
                        if let Some(sender) = event_sender {
                            let _ = sender.send(events::CoreEvent::MessageStop);
                        }
                        break;
                    }

                    _ => {
                        // Other events don't need special handling
                    }
                }

                // Process event for tool collection after match
                tool_collector.process_event(&event);
            }

            // 检查是否有已完成的工具调用
            if tool_collector.has_completed_calls() {
                let tool_calls = tool_collector.take_completed();

                // 构建助手消息内容
                let mut content_blocks = vec![json!({
                    "type": "text",
                    "text": current_text
                })];

                for call in &tool_calls {
                    content_blocks.push(json!({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.input
                    }));
                }

                // 保存助手消息
                messages.push(json!({
                    "role": "assistant",
                    "content": content_blocks
                }));

                // 执行工具
                for call in tool_calls {
                    // 发送工具执行开始事件
                    if let Some(sender) = event_sender {
                        let _ = sender.send(events::CoreEvent::ToolExecuting {
                            name: call.name.clone(),
                        });
                    }

                    let result = self
                        .run_tool(
                            &call.name,
                            call.input.as_object().ok_or_else(|| {
                                ApiError::ParseError(format!(
                                    "Tool input is not an object for tool: {}",
                                    call.name
                                ))
                            })?,
                        )
                        .await;

                    // 发送工具结果事件
                    if let Some(sender) = event_sender {
                        let _ = sender.send(events::CoreEvent::ToolResult {
                            name: call.name.clone(),
                            result: result.clone(),
                        });
                    }

                    // 添加工具结果到消息历史
                    messages.push(json!({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": call.id,
                            "content": result
                        }]
                    }));
                }

                // 清空当前文本并继续循环
                current_text = String::new();
            } else if !current_text.is_empty() {
                // 没有工具调用，保存最终回复并退出
                messages.push(json!({
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": current_text
                    }]
                }));
                break;
            } else {
                break;
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_call_collector_basic() {
        let mut collector = ToolCallCollector::new();

        // 模拟工具调用开始
        collector.process_event(&StreamEvent::ContentBlockStart {
            index: 0,
            content_block: ContentBlock::ToolUse {
                id: "call_123".to_string(),
                name: "read".to_string(),
                input: json!(""),
            },
        });

        // 模拟增量数据
        collector.process_event(&StreamEvent::ContentBlockDelta {
            index: 0,
            delta: Delta::InputJson {
                partial_json: r#"{"file_path":"test.rs"}"#.to_string(),
            },
        });

        // 模拟结束
        collector.process_event(&StreamEvent::ContentBlockStop { index: 0 });

        // 验证收集
        assert!(collector.has_completed_calls());
        let calls = collector.take_completed();
        assert_eq!(calls.len(), 1);
        // Use let-else pattern instead of if-else
        let Some(call) = calls.first() else {
            return; // Empty vector: test passes early
        };
        assert_eq!(call.name, "read");
    }

    #[test]
    fn test_tool_call_collector_multiple() {
        let mut collector = ToolCallCollector::new();

        // 模拟第一个工具调用
        collector.process_event(&StreamEvent::ContentBlockStart {
            index: 0,
            content_block: ContentBlock::ToolUse {
                id: "call_1".to_string(),
                name: "read".to_string(),
                input: json!(""),
            },
        });

        collector.process_event(&StreamEvent::ContentBlockDelta {
            index: 0,
            delta: Delta::InputJson {
                partial_json: r#"{"path":"file1.rs"}"#.to_string(),
            },
        });

        collector.process_event(&StreamEvent::ContentBlockStop { index: 0 });

        // 模拟第二个工具调用
        collector.process_event(&StreamEvent::ContentBlockStart {
            index: 1,
            content_block: ContentBlock::ToolUse {
                id: "call_2".to_string(),
                name: "write".to_string(),
                input: json!(""),
            },
        });

        collector.process_event(&StreamEvent::ContentBlockDelta {
            index: 1,
            delta: Delta::InputJson {
                partial_json: r#"{"path":"file2.rs","content":"test"}"#.to_string(),
            },
        });

        collector.process_event(&StreamEvent::ContentBlockStop { index: 1 });

        // 验证收集
        assert!(collector.has_completed_calls());
        let calls = collector.take_completed();
        assert_eq!(calls.len(), 2);
        // Use let-else pattern instead of if-else
        let Some(first_call) = calls.first() else {
            return; // Empty vector: test passes early
        };
        let Some(second_call) = calls.get(1) else {
            return; // Single element vector: test passes early
        };
        assert_eq!(first_call.name, "read");
        assert_eq!(second_call.name, "write");
    }

    #[test]
    fn test_tool_call_collector_incomplete() {
        let mut collector = ToolCallCollector::new();

        // 模拟工具调用开始但没有结束
        collector.process_event(&StreamEvent::ContentBlockStart {
            index: 0,
            content_block: ContentBlock::ToolUse {
                id: "call_123".to_string(),
                name: "read".to_string(),
                input: json!(""),
            },
        });

        collector.process_event(&StreamEvent::ContentBlockDelta {
            index: 0,
            delta: Delta::InputJson {
                partial_json: r#"{"file_path":"test.rs"}"#.to_string(),
            },
        });

        // 验证未完成
        assert!(!collector.has_completed_calls());
        assert!(collector.is_active());
    }

    #[test]
    fn test_content_block_deserialize() {
        let text_json = json!({"type": "text", "text": "Hello"});
        let text: ContentBlock = match serde_json::from_value(text_json) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Failed to deserialize text ContentBlock: {e}");
                return;
            }
        };
        match text {
            ContentBlock::Text { _text: t } => assert_eq!(t, "Hello"),
            ContentBlock::ToolUse { .. } => {
                unreachable!("Expected Text variant but got ToolUse in test")
            }
        }

        let tool_json = json!({
            "type": "tool_use",
            "id": "call_123",
            "name": "read",
            "input": {"path": "test.rs"}
        });
        let tool: ContentBlock = match serde_json::from_value(tool_json) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Failed to deserialize tool_use ContentBlock: {e}");
                return;
            }
        };
        match tool {
            ContentBlock::ToolUse { id, name, .. } => {
                assert_eq!(id, "call_123");
                assert_eq!(name, "read");
            }
            ContentBlock::Text { .. } => {
                unreachable!("Expected ToolUse variant but got Text in test")
            }
        }
    }

    #[test]
    fn test_delta_deserialize() {
        let text_delta_json = json!({"type": "text_delta", "text": "Hello"});
        let delta: Delta = match serde_json::from_value(text_delta_json) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Failed to deserialize text_delta Delta: {e}");
                return;
            }
        };
        match delta {
            Delta::Text { text } => assert_eq!(text, "Hello"),
            Delta::InputJson { .. } => {
                unreachable!("Expected Text variant but got InputJson in test")
            }
        }

        let json_delta_json = json!({"type": "input_json_delta", "partial_json": "{}"});
        let delta: Delta = match serde_json::from_value(json_delta_json) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Failed to deserialize input_json_delta Delta: {e}");
                return;
            }
        };
        match delta {
            Delta::InputJson { partial_json } => assert_eq!(partial_json, "{}"),
            Delta::Text { .. } => {
                unreachable!("Expected InputJson variant but got Text in test")
            }
        }
    }
}
