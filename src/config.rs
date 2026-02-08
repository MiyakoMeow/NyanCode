//! Configuration management for nanocode
//!
//! Reads environment variables for Anthropic API configuration.

use std::env;

/// Application configuration read from environment variables.
#[derive(Debug, Clone)]
pub struct Config {
    /// Base URL for Anthropic API
    pub base_url: String,
    /// Model identifier to use
    pub model: String,
    /// API authentication token
    pub api_key: String,
    /// Current working directory
    pub cwd: String,
}

impl Config {
    /// Load configuration from environment variables with defaults.
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
        let base_url = env::var("ANTHROPIC_BASE_URL")
            .unwrap_or_else(|_| "https://api.anthropic.com".to_string());

        let model = env::var("ANTHROPIC_MODEL")
            .or_else(|_| env::var("MODEL"))
            .unwrap_or_else(|_| "claude-opus-4-5".to_string());

        let api_key = env::var("ANTHROPIC_AUTH_TOKEN")
            .or_else(|_| env::var("ANTHROPIC_API_KEY"))
            .unwrap_or_default();

        let cwd = env::current_dir().map_or_else(|_| ".".to_string(), |p| p.display().to_string());

        Self {
            base_url,
            model,
            api_key,
            cwd,
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

impl Default for Config {
    fn default() -> Self {
        Self::from_env()
    }
}
