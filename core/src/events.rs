//! Core event types for nanocode
//!
//! Defines the core events used throughout the application for communication
//! between different components of the system.

use serde::{Deserialize, Serialize};

/// Core event enumeration representing different types of events
/// that can occur during the execution of the system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoreEvent {
    /// Text delta event, representing incremental text output
    ///
    /// This event is used to stream text output as it's generated,
    /// allowing for real-time display of content.
    TextDelta(String),

    /// Tool call start event, indicating the beginning of a tool execution
    ///
    /// This event marks when a tool is about to be called, providing
    /// information about the tool's name and unique identifier.
    ToolCallStart {
        /// Unique identifier for the tool call
        id: String,
        /// Name of the tool being called
        name: String,
    },

    /// Tool executing event, indicating that a tool is currently being executed
    ///
    /// This event is emitted when a tool is actively processing and
    /// executing its operation.
    ToolExecuting {
        /// Name of the tool being executed
        name: String,
    },

    /// Tool result event, containing the result of a tool execution
    ///
    /// This event carries the output or result from a completed tool call,
    /// which can be used for further processing or display.
    ToolResult {
        /// Name of the tool that produced the result
        name: String,
        /// The result data from the tool execution
        result: String,
    },

    /// Error event, representing an error that occurred during execution
    ///
    /// This event carries error information when something goes wrong
    /// in the system, allowing for proper error handling and display.
    Error(String),

    /// Message start event, marking the beginning of a new message
    ///
    /// This event signals the start of a new conversation or message
    /// in the system, used to track message boundaries.
    MessageStart,

    /// Message stop event, marking the end of a message
    ///
    /// This event signals the completion of a message or conversation,
    /// used to indicate when processing should stop or continue.
    MessageStop,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_event_serialization() {
        let event = CoreEvent::TextDelta("Hello, world!".to_string());
        let serialized = serde_json::to_string(&event).unwrap();
        let deserialized: CoreEvent = serde_json::from_str(&serialized).unwrap();

        match deserialized {
            CoreEvent::TextDelta(text) => assert_eq!(text, "Hello, world!"),
            _ => panic!("Expected TextDelta variant"),
        }
    }

    #[test]
    fn test_core_event_tool_call_start() {
        let event = CoreEvent::ToolCallStart {
            id: "test-id".to_string(),
            name: "test-tool".to_string(),
        };

        match event {
            CoreEvent::ToolCallStart { id, name } => {
                assert_eq!(id, "test-id");
                assert_eq!(name, "test-tool");
            }
            _ => panic!("Expected ToolCallStart variant"),
        }
    }

    #[test]
    fn test_core_event_tool_result() {
        let event = CoreEvent::ToolResult {
            name: "read".to_string(),
            result: "File content".to_string(),
        };

        match event {
            CoreEvent::ToolResult { name, result } => {
                assert_eq!(name, "read");
                assert_eq!(result, "File content");
            }
            _ => panic!("Expected ToolResult variant"),
        }
    }

    #[test]
    fn test_core_event_clone() {
        let original = CoreEvent::TextDelta("test".to_string());
        let cloned = original.clone();

        match (original, cloned) {
            (CoreEvent::TextDelta(s1), CoreEvent::TextDelta(s2)) => {
                assert_eq!(s1, s2);
            }
            _ => panic!("Expected TextDelta variant"),
        }
    }
}
