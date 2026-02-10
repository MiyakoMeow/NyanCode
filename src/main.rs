//! nanocode - minimal Claude code alternative in Rust

/// API模块 - 处理与Claude AI的通信和交互
mod api;
/// 配置模块 - 管理应用配置和环境变量
mod config;
/// 模式定义模块 - 定义工具调用的JSON模式
mod schema;
/// 工具模块 - 提供AI助手可用的工具函数
mod tools;
/// 用户界面模块 - 处理终端显示和交互界面
mod ui;

use anyhow::Result;
use clap::Parser;
use serde_json::json;
use std::io::{self, Write};
use std::process::ExitCode;

use api::anthropic::Client;
use config::Config;
use ui::{colors, separator};

/// AI编程助手 - Claude Code Rust实现
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct CliArgs {
    /// 直接发送消息并执行（非交互模式）
    #[arg(short = 'm', long = "message")]
    message: Option<String>,
}

/// 运行交互式模式 - 进入REPL循环，与AI进行实时对话
async fn run_interactive_mode(
    client: &Client,
    system_prompt: &str,
    schema: &[serde_json::Value],
) -> Result<()> {
    let mut messages: Vec<serde_json::Value> = Vec::new();

    loop {
        print!("{}", separator());
        print!(
            "{}❯{} ",
            colors::BOLD.to_string() + colors::BLUE,
            colors::RESET
        );
        io::stdout().flush()?;

        let mut user_input = String::new();
        let bytes_read = io::stdin().read_line(&mut user_input)?;
        if bytes_read == 0 {
            break; // EOF，正常退出
        }
        let user_input = user_input.trim();

        print!("{}", separator());

        if user_input.is_empty() {
            continue;
        }

        // Handle commands
        match user_input {
            "/q" | "exit" => break,
            "/c" => {
                messages.clear();
                println!("{}⏺ Cleared conversation{}", colors::GREEN, colors::RESET);
                continue;
            }
            _ => {}
        }

        // Add user message
        messages.push(json!({
            "role": "user",
            "content": user_input,
        }));

        // Run agentic loop (streaming)
        if let Err(e) = client
            .run_agent_loop_stream(&mut messages, system_prompt, schema)
            .await
        {
            println!("{}⏺ Error: {}{}", colors::RED, e, colors::RESET);
        }

        println!();
    }

    Ok(())
}

/// 运行单条消息模式 - 发送单次消息并获取AI响应
async fn run_single_message_mode(
    message: String,
    client: &Client,
    system_prompt: &str,
    schema: &[serde_json::Value],
) -> Result<()> {
    let mut messages = Vec::new();

    // 添加用户消息
    messages.push(json!({
        "role": "user",
        "content": message
    }));

    // 调用流式响应（支持工具调用）
    client
        .run_agent_loop_stream(&mut messages, system_prompt, schema)
        .await?;

    Ok(())
}

fn main() -> ExitCode {
    // 解析命令行参数
    let args = CliArgs::parse();

    // 加载配置
    let config = Config::from_env();

    // 显示启动信息
    println!(
        "{}nanocode{} | {}{}{} | {}{}{} | {}{}{} | {}{}{}\n",
        colors::BOLD,
        colors::RESET,
        colors::DIM,
        config.model,
        colors::RESET,
        colors::YELLOW,
        config.masked_api_key(),
        colors::RESET,
        colors::DIM,
        config.base_url,
        colors::RESET,
        colors::DIM,
        config.cwd,
        colors::RESET,
    );

    // 创建运行时
    // SAFETY: Runtime creation failure is unrecoverable and should terminate the program.
    #[allow(clippy::expect_used)]
    let rt = tokio::runtime::Runtime::new().expect("Failed to create runtime");

    // 创建 API 客户端
    let client = Client::new(config.clone());

    // 准备系统提示和工具schema
    let system_prompt = format!("Concise coding assistant. cwd: {}", config.cwd);
    let schema = schema::make_schema();

    // 根据参数选择运行模式
    let result = rt.block_on(async {
        if let Some(message) = args.message {
            // 非交互模式：执行单次对话
            run_single_message_mode(message, &client, &system_prompt, &schema).await
        } else {
            // 交互模式：进入REPL
            run_interactive_mode(&client, &system_prompt, &schema).await
        }
    });

    // 处理结果并返回退出码
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("❌ Error: {e}");
            ExitCode::FAILURE
        }
    }
}
