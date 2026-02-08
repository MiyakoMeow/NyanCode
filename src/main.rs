//! nanocode - minimal Claude code alternative in Rust

mod api;
mod config;
mod schema;
mod tools;
mod ui;

use anyhow::Result;
use serde_json::json;
use std::io::{self, Write};

use api::AnthropicClient;
use config::Config;
use ui::{colors, separator};

#[tokio::main]
async fn main() -> Result<()> {
    // Load configuration
    let config = Config::from_env();

    // Print header
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

    // Initialize client and messages
    let client = AnthropicClient::new(config.clone());
    let mut messages: Vec<serde_json::Value> = Vec::new();
    let system_prompt = format!("Concise coding assistant. cwd: {}", config.cwd);

    // REPL loop
    loop {
        print!("{}", separator());
        print!(
            "{}❯{} ",
            colors::BOLD.to_string() + colors::BLUE,
            colors::RESET
        );
        io::stdout().flush()?;

        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)?;
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

        // Run agentic loop
        if let Err(e) = client.run_agent_loop(&mut messages, &system_prompt).await {
            println!("{}⏺ Error: {}{}", colors::RED, e, colors::RESET);
        }

        println!();
    }

    Ok(())
}
