use crate::error::ApiError;
use crate::types::{
    ContentBlockStartEvent, ContentBlockStopEvent, MessageStopEvent, OutputContentBlock, StreamEvent,
};

/// A tool call under construction. OpenAI-compatible streams deliver the name
/// and id once, then dribble the JSON arguments across many chunks, so a call
/// can only be emitted after the stream closes.
#[derive(Debug, Default, Clone)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Debug)]
pub struct SseParser {
    buffer: Vec<u8>,
    provider_is_openai: bool,
    tool_calls: Vec<PartialToolCall>,
}

impl SseParser {
    #[must_use]
    pub fn new(provider_is_openai: bool) -> Self {
        Self {
            buffer: Vec::new(),
            provider_is_openai,
            tool_calls: Vec::new(),
        }
    }

    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<StreamEvent>, ApiError> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();

        while let Some(frame) = self.next_frame() {
            events.extend(self.handle_frame(&frame)?);
        }

        Ok(events)
    }

    pub fn finish(&mut self) -> Result<Vec<StreamEvent>, ApiError> {
        if self.buffer.is_empty() {
            return Ok(Vec::new());
        }

        let trailing = std::mem::take(&mut self.buffer);
        self.handle_frame(&String::from_utf8_lossy(&trailing))
    }

    /// An Anthropic frame maps to at most one event, but an OpenAI-compatible
    /// one is stateful: tool call fragments accumulate until `[DONE]` closes the
    /// stream, at which point the completed calls plus the terminating
    /// `message_stop` are emitted together.
    fn handle_frame(&mut self, frame: &str) -> Result<Vec<StreamEvent>, ApiError> {
        if !self.provider_is_openai {
            return Ok(parse_frame(frame, false)?.into_iter().collect());
        }

        let Some(payload) = frame_payload(frame) else {
            return Ok(Vec::new());
        };

        // OpenAI-compatible streams carry no `message_stop` event — they end with
        // `[DONE]`, which arrives exactly once and always last.
        if payload == "[DONE]" {
            return Ok(self.flush_tool_calls());
        }

        let value = serde_json::from_str::<serde_json::Value>(&payload).map_err(ApiError::from)?;
        self.accumulate_tool_calls(&value);

        Ok(crate::client::translate_openai_chunk_to_event(value)
            .into_iter()
            .collect())
    }

    fn accumulate_tool_calls(&mut self, chunk: &serde_json::Value) {
        let Some(calls) = chunk
            .get("choices")
            .and_then(|choices| choices.as_array())
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("delta"))
            .and_then(|delta| delta.get("tool_calls"))
            .and_then(|calls| calls.as_array())
        else {
            return;
        };

        for call in calls {
            let index = usize::try_from(
                call.get("index")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
            )
            .unwrap_or(0);

            if self.tool_calls.len() <= index {
                self.tool_calls.resize(index + 1, PartialToolCall::default());
            }
            let slot = &mut self.tool_calls[index];

            if let Some(id) = call.get("id").and_then(serde_json::Value::as_str) {
                if !id.is_empty() {
                    slot.id = id.to_owned();
                }
            }
            if let Some(function) = call.get("function") {
                if let Some(name) = function.get("name").and_then(serde_json::Value::as_str) {
                    if !name.is_empty() {
                        slot.name = name.to_owned();
                    }
                }
                if let Some(args) = function
                    .get("arguments")
                    .and_then(serde_json::Value::as_str)
                {
                    slot.arguments.push_str(args);
                }
            }
        }
    }

    fn flush_tool_calls(&mut self) -> Vec<StreamEvent> {
        let mut events = Vec::new();

        for (position, call) in std::mem::take(&mut self.tool_calls)
            .into_iter()
            .filter(|call| !call.name.is_empty())
            .enumerate()
        {
            // A model that emits no arguments still produces a valid call with an
            // empty object, so an unparseable fragment must not sink the turn.
            let input = serde_json::from_str(&call.arguments)
                .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));

            let index = u32::try_from(position).unwrap_or(0);
            events.push(StreamEvent::ContentBlockStart(ContentBlockStartEvent {
                index,
                content_block: OutputContentBlock::ToolUse {
                    id: call.id,
                    name: call.name,
                    input,
                },
            }));
            events.push(StreamEvent::ContentBlockStop(ContentBlockStopEvent {
                index,
            }));
        }

        events.push(StreamEvent::MessageStop(MessageStopEvent {}));
        events
    }

    fn next_frame(&mut self) -> Option<String> {
        let separator = self
            .buffer
            .windows(2)
            .position(|window| window == b"\n\n")
            .map(|position| (position, 2))
            .or_else(|| {
                self.buffer
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .map(|position| (position, 4))
            })?;

        let (position, separator_len) = separator;
        let frame = self
            .buffer
            .drain(..position + separator_len)
            .collect::<Vec<_>>();
        let frame_len = frame.len().saturating_sub(separator_len);
        Some(String::from_utf8_lossy(&frame[..frame_len]).into_owned())
    }
}

/// Collapses an SSE frame down to its `data:` payload, discarding comments,
/// pings and heartbeat frames that carry no data.
fn frame_payload(frame: &str) -> Option<String> {
    let trimmed = frame.trim();
    if trimmed.is_empty() {
        return None;
    }

    let mut data_lines = Vec::new();
    let mut event_name: Option<&str> = None;

    for line in trimmed.lines() {
        if line.starts_with(':') {
            continue;
        }
        if let Some(name) = line.strip_prefix("event:") {
            event_name = Some(name.trim());
            continue;
        }
        if let Some(data) = line.strip_prefix("data:") {
            data_lines.push(data.trim_start());
        }
    }

    if matches!(event_name, Some("ping")) || data_lines.is_empty() {
        return None;
    }

    Some(data_lines.join("\n"))
}

pub fn parse_frame(frame: &str, is_openai: bool) -> Result<Option<StreamEvent>, ApiError> {
    let Some(payload) = frame_payload(frame) else {
        return Ok(None);
    };

    if payload == "[DONE]" {
        return Ok(None);
    }

    if is_openai {
        // We handle mapping inside the MessageStream because OpenAI chunks map to *multiple* events sometimes
        // For simplicity in the parser, if it's OpenAI, we just wrap the raw json string in a synthetic delta.
        // The actual client.rs will do the real translation.
        let val = serde_json::from_str::<serde_json::Value>(&payload).map_err(ApiError::from)?;
        return Ok(crate::client::translate_openai_chunk_to_event(val));
    }

    serde_json::from_str::<StreamEvent>(&payload)
        .map(Some)
        .map_err(ApiError::from)
}

#[cfg(test)]
mod tests {
    use super::{parse_frame, SseParser};
    use crate::types::{ContentBlockDelta, MessageDelta, OutputContentBlock, StreamEvent, Usage};

    #[test]
    fn parses_single_frame() {
        let frame = concat!(
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"Hi\"}}\n\n"
        );

        let event = parse_frame(frame, false).expect("frame should parse");
        assert_eq!(
            event,
            Some(StreamEvent::ContentBlockStart(
                crate::types::ContentBlockStartEvent {
                    index: 0,
                    content_block: OutputContentBlock::Text {
                        text: "Hi".to_string(),
                    },
                },
            ))
        );
    }

    #[test]
    fn parses_chunked_stream() {
        let mut parser = SseParser::new(false);
        let first = b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel";
        let second = b"lo\"}}\n\n";

        assert!(parser
            .push(first)
            .expect("first chunk should buffer")
            .is_empty());
        let events = parser.push(second).expect("second chunk should parse");

        assert_eq!(
            events,
            vec![StreamEvent::ContentBlockDelta(
                crate::types::ContentBlockDeltaEvent {
                    index: 0,
                    delta: ContentBlockDelta::TextDelta {
                        text: "Hello".to_string(),
                    },
                }
            )]
        );
    }

    #[test]
    fn ignores_ping_and_done() {
        let mut parser = SseParser::new(false);
        let payload = concat!(
            ": keepalive\n",
            "event: ping\n",
            "data: {\"type\":\"ping\"}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"input_tokens\":1,\"output_tokens\":2}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
            "data: [DONE]\n\n"
        );

        let events = parser
            .push(payload.as_bytes())
            .expect("parser should succeed");
        assert_eq!(
            events,
            vec![
                StreamEvent::MessageDelta(crate::types::MessageDeltaEvent {
                    delta: MessageDelta {
                        stop_reason: Some("tool_use".to_string()),
                        stop_sequence: None,
                    },
                    usage: Usage {
                        input_tokens: 1,
                        cache_creation_input_tokens: 0,
                        cache_read_input_tokens: 0,
                        output_tokens: 2,
                    },
                }),
                StreamEvent::MessageStop(crate::types::MessageStopEvent {}),
            ]
        );
    }

    #[test]
    fn ignores_data_less_event_frames() {
        let frame = "event: ping\n\n";
        let event = parse_frame(frame, false).expect("frame without data should be ignored");
        assert_eq!(event, None);
    }

    #[test]
    fn parses_split_json_across_data_lines() {
        let frame = concat!(
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\n",
            "data: \"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n"
        );

        let event = parse_frame(frame, false).expect("frame should parse");
        assert_eq!(
            event,
            Some(StreamEvent::ContentBlockDelta(
                crate::types::ContentBlockDeltaEvent {
                    index: 0,
                    delta: ContentBlockDelta::TextDelta {
                        text: "Hello".to_string(),
                    },
                }
            ))
        );
    }
}
