//! Gemma 4 tool-call extraction.
//!
//! Gemma 4 emits tool calls as plain text inside the assistant reply,
//! wrapped in special-token strings: `<|tool_call>call:NAME{ARGS}<tool_call|>`
//! (some variants close with `<turn|>` instead). If we leave that in
//! `content`, the client sees the raw markup as a normal assistant
//! message — which is what just bit zeroclaw. This module extracts
//! the calls so the handler can hoist them into the OpenAI-shaped
//! `tool_calls` array and set `finish_reason = "tool_calls"`.
//!
//! Ported from vLLM's `vllm/tool_parsers/gemma4_utils.py`. Two tiers:
//!   * **tier-1** — strict `<|tool_call>call:NAME{ARGS}<tool_call|>`.
//!     Matches what the model emits with the special tokens intact.
//!   * **tier-2** — bare `call:NAME{ARGS}` anchored at start-of-string
//!     or after whitespace. Fires when the decoder strips the
//!     `<|tool_call>` special tokens (our `TokenizerHandle::decode`
//!     runs with `skip_special_tokens=true`, which is how this path
//!     first surfaced: zeroclaw saw `call:get_weather{location:"..."}`
//!     land in `content` instead of as a structured tool call).

use serde::Serialize;
use serde_json::{Map, Value};

/// One extracted call. `arguments` is the JSON-string form expected by
/// OpenAI's `tool_calls[].function.arguments` field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: String,
}

const START: &str = "<|tool_call>";
const END_A: &str = "<tool_call|>";
const END_B: &str = "<turn|>";
const ESCAPE: &str = "<|\"|>";
// Cycle 56 step 1: removed unused `CHANNEL_OPEN` — the literal
// "<|channel>" lives in THOUGHT_BLOCK_OPENERS below; this private
// constant was a stale duplicate.
const CHANNEL_CLOSE: &str = "<channel|>";
// Gemma 4 wraps its pre-answer reasoning in multiple block shapes:
//   <|channel>thought\n...<channel|>
//   <|tool_response>thought\n...<channel|>
//   <thought\n...<channel|>     (model hallucination without the leading `|`)
// They all close with `<channel|>` and must be dropped from user content
// wholesale — the inner prose is the model's "what do I think the answer is"
// draft, which should never be shown to the user.
pub const THOUGHT_BLOCK_OPENERS: &[&str] = &[
    "<|channel>",
    "<|tool_response>",
    "<thought",
];

/// Tier-1 tool-call opener. Public so the streaming SSE path can hold
/// back content when an opener is in flight without its closer.
pub const TOOL_CALL_OPENER: &str = "<|tool_call>";

/// Extract all Gemma 4 tool calls from decoded text.
///
/// Returns an empty vec when no markup is present — callers should
/// treat that as a plain text response.
pub fn parse_gemma4_tool_calls(text: &str) -> Vec<ParsedToolCall> {
    let mut out = parse_tier1(text);
    if out.is_empty() {
        out = parse_tier2_bare(text);
    }
    out
}

fn parse_tier1(text: &str) -> Vec<ParsedToolCall> {
    let mut out = Vec::new();
    let mut cursor = 0;
    let bytes = text.as_bytes();

    while cursor < bytes.len() {
        let Some(rel) = text[cursor..].find(START) else { break };
        let after_start = cursor + rel + START.len();

        // After `<|tool_call>` the format is `call:NAME{ARGS}END`.
        let rest = &text[after_start..];
        let Some(stripped) = rest.strip_prefix("call:") else {
            cursor = after_start;
            continue;
        };
        // Name = [A-Za-z0-9_]+ up to `{`.
        let brace = match stripped.find('{') {
            Some(i) => i,
            None => break,
        };
        let name = &stripped[..brace];
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            cursor = after_start;
            continue;
        }
        let args_start = brace + 1; // past `{`
        let args_region = &stripped[args_start..];

        // Find the terminator. Args may contain `}` inside a JSON value,
        // so we search for the closing tag first and then trim one `}`.
        let (args_end_in_region, tag_len) = match (args_region.find(END_A), args_region.find(END_B)) {
            (Some(a), Some(b)) if a < b => (a, END_A.len()),
            (Some(_a), Some(b)) => (b, END_B.len()),
            (Some(a), None) => (a, END_A.len()),
            (None, Some(b)) => (b, END_B.len()),
            (None, None) => break,
        };
        let raw_args = &args_region[..args_end_in_region];
        let trimmed = raw_args.strip_suffix('}').unwrap_or(raw_args);

        out.push(ParsedToolCall {
            name: name.to_string(),
            arguments: arguments_to_json_string(trimmed),
        });

        cursor = after_start + "call:".len() + brace + 1 + args_end_in_region + tag_len;
    }

    out
}

/// Tier-2: bare `call:NAME{ARGS}` at start-of-string or after
/// whitespace. Matches what remains when a `skip_special_tokens=true`
/// decoder strips `<|tool_call>` / `<tool_call|>`.
///
/// UTF-8 note: the markers (`call:`, `{`, `}`) are ASCII, so marker
/// matching is byte-oriented — but `i` must land on a char boundary
/// before we can slice `&text[..]`. Output from Gemma 4 routinely
/// contains non-ASCII (German prose, emoji) that would otherwise
/// panic `str::is_char_boundary`.
fn parse_tier2_bare(text: &str) -> Vec<ParsedToolCall> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i + 5 <= bytes.len() {
        if !text.is_char_boundary(i) {
            i += 1;
            continue;
        }
        // Require start-of-string or whitespace immediately before `call:`.
        let anchored = i == 0 || (bytes[i - 1] as char).is_whitespace();
        if !anchored || &bytes[i..i + 5] != b"call:" {
            i += 1;
            continue;
        }
        let name_start = i + 5;
        let mut j = name_start;
        while j < bytes.len() && {
            let c = bytes[j] as char;
            c.is_ascii_alphanumeric() || c == '_'
        } {
            j += 1;
        }
        if j == name_start || j >= bytes.len() || bytes[j] != b'{' {
            i += 1;
            continue;
        }
        let name = &text[name_start..j];
        let args_start = j + 1;
        // Balanced-brace scan: find the `}` that closes the OPENING
        // `{` at `j`, accounting for nested objects (`{"a":{"b":1}}`)
        // and `}` characters inside JSON string literals (which must
        // NOT count toward depth). Without this, bare
        // `call:foo{"a":{"b":1}}` was truncated to `call:foo{"a":{"b":1}`.
        let Some(args_end_abs) = find_balanced_close_brace(text, args_start) else {
            break;
        };
        let args = &text[args_start..args_end_abs];
        out.push(ParsedToolCall {
            name: name.to_string(),
            arguments: arguments_to_json_string(args),
        });
        i = args_end_abs + 1;
    }
    out
}

/// Scan `text` starting at `start` (the byte AFTER an opening `{`) and
/// return the absolute byte index of the matching closing `}` at depth
/// zero. Tracks nested object braces and skips `{` / `}` that appear
/// inside JSON string literals (incl. backslash-escapes). Returns
/// `None` if the buffer ends with the call still unclosed.
///
/// Used by tier-2 bare-call extraction where the surrounding wrapper
/// is absent and the parser must distinguish "args end" from "nested
/// object close".
fn find_balanced_close_brace(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut depth: i32 = 1; // caller already consumed the opening `{`
    let mut in_string = false;
    let mut escape_next = false;
    let mut i = start;
    while i < bytes.len() {
        let c = bytes[i];
        if escape_next {
            escape_next = false;
            i += 1;
            continue;
        }
        if in_string {
            match c {
                b'\\' => escape_next = true,
                b'"' => in_string = false,
                _ => {}
            }
            i += 1;
            continue;
        }
        match c {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Render the inner args region into a JSON object string.
/// Prefers direct JSON parse (Gemma 4 escape-token handled). On
/// failure, returns `{}` — bubbling up a malformed call is worse than
/// letting the client decide.
fn arguments_to_json_string(args: &str) -> String {
    let cleaned = args.replace(ESCAPE, "\"");
    let wrapped = format!("{{{}}}", cleaned);
    if let Ok(v) = serde_json::from_str::<Value>(&wrapped) {
        return serde_json::to_string(&v).unwrap_or_else(|_| "{}".to_string());
    }

    // Fallback: harvest `key: "value"` pairs with a tiny hand-rolled scanner
    // rather than pulling in a regex dep for one edge case.
    let mut map = Map::new();
    let bytes = cleaned.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        let key_start = i;
        while i < bytes.len() && {
            let c = bytes[i] as char;
            c.is_ascii_alphanumeric() || c == '_'
        } {
            i += 1;
        }
        if i == key_start {
            break;
        }
        let key = &cleaned[key_start..i];
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b':' {
            break;
        }
        i += 1;
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'"' {
            break;
        }
        i += 1;
        let val_start = i;
        while i < bytes.len() && bytes[i] != b'"' {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let val = &cleaned[val_start..i];
        map.insert(key.to_string(), Value::String(val.to_string()));
        i += 1; // closing quote
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b',' {
            i += 1;
        }
    }
    serde_json::to_string(&Value::Object(map)).unwrap_or_else(|_| "{}".to_string())
}

/// Strip any Gemma 4 tool-call markup + leading/trailing whitespace
/// so the plain-text path has a clean payload when no calls are emitted
/// (or, for mixed output, for the prefix before the first call).
/// Handles both tier-1 `<|tool_call>...<tool_call|>` wrappers and
/// tier-2 bare `call:NAME{...}` patterns.
pub fn strip_tool_markup(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Markers are ASCII and therefore always start at a char boundary;
        // checking `is_char_boundary` first makes the subsequent `text[i..]`
        // slice safe when non-ASCII prose (e.g. German umlauts) sits in the
        // middle of the stream.
        if text.is_char_boundary(i) {
            // Drop entire thought / tool_response / hallucinated-thought blocks —
            // each closes with `<channel|>` and carries Gemma's internal
            // reasoning, never user-facing prose.
            let mut matched_thought = false;
            for opener in THOUGHT_BLOCK_OPENERS {
                if text[i..].starts_with(opener) {
                    // Prefer `<channel|>` as the closer; if Gemma terminated the
                    // block with `<turn|>` (end-of-turn) or just let it trail
                    // off, match that too rather than eating the rest of the
                    // reply. No closer found → only strip the opener itself so
                    // real content after it survives.
                    let rest = &text[i + opener.len()..];
                    let close = match (rest.find(CHANNEL_CLOSE), rest.find("<turn|>")) {
                        (Some(a), Some(b)) if a <= b => Some((a, CHANNEL_CLOSE.len())),
                        (Some(_), Some(b)) => Some((b, "<turn|>".len())),
                        (Some(a), None) => Some((a, CHANNEL_CLOSE.len())),
                        (None, Some(b)) => Some((b, "<turn|>".len())),
                        (None, None) => None,
                    };
                    let skip = match close {
                        Some((rel, close_len)) => opener.len() + rel + close_len,
                        None => opener.len(),
                    };
                    i += skip;
                    matched_thought = true;
                    break;
                }
            }
            if matched_thought {
                continue;
            }
            if text[i..].starts_with(START) {
                let rest = &text[i + START.len()..];
                let skip = match (rest.find(END_A), rest.find(END_B)) {
                    (Some(a), Some(b)) if a <= b => START.len() + a + END_A.len(),
                    (Some(_a), Some(b)) => START.len() + b + END_B.len(),
                    (Some(a), None) => START.len() + a + END_A.len(),
                    (None, Some(b)) => START.len() + b + END_B.len(),
                    // Cycle 33 fix (codex bug #6): malformed opener with no
                    // close was silently swallowing the rest of the reply,
                    // turning a broken tool-call attempt into "" content.
                    // That hid the actual model output during cliff trials
                    // (cycle 19+). Now skip ONLY the opener tag itself so
                    // the body content past it surfaces in `content` —
                    // ugly UX but the user can see the model misbehaved.
                    (None, None) => START.len(),
                };
                i += skip;
                continue;
            }
            // Tier-2 bare `call:NAME{...}` — strip when anchored.
            let anchored = i == 0
                || out.chars().last().map(|c| c.is_whitespace()).unwrap_or(false);
            if anchored && text[i..].starts_with("call:") {
                let after = i + "call:".len();
                let mut j = after;
                while j < bytes.len() && {
                    let c = bytes[j] as char;
                    c.is_ascii_alphanumeric() || c == '_'
                } {
                    j += 1;
                }
                if j > after && j < bytes.len() && bytes[j] == b'{' {
                    // Reuse the same balanced-brace scanner that
                    // `parse_tier2_bare` uses, so strip and parse stay
                    // symmetric: any args region the parser accepts
                    // (incl. nested objects, `}` inside string
                    // literals) is also fully removed from the visible
                    // content here. Without this, nested-args calls
                    // leaked a stray `}` + trailing prose into the
                    // assistant message body.
                    if let Some(end_abs) = find_balanced_close_brace(text, j + 1) {
                        i = end_abs + 1;
                        continue;
                    }
                }
            }
        }
        // Copy one whole UTF-8 scalar. If `i` is mid-codepoint (can happen
        // after a `skip` jump landed mid-sequence on malformed input), step
        // one byte to resynchronise — the lost byte will be replaced by the
        // next `chars()` decode.
        if let Some(ch) = text[i..].chars().next() {
            out.push(ch);
            i += ch.len_utf8();
        } else {
            i += 1;
        }
    }
    // Final sweep: Gemma 4's raw output sprinkles other single-token control
    // markers (`<|tool_response>`, `<turn|>`, `<|turn>`, literal `<thought`
    // fragments the model hallucinates) that the named passes above don't
    // enumerate. They all share a common shape (`<|…>` or `<…|>`), so a
    // conservative token sweep is enough — and any false positive would
    // have to be Gemma control-token-looking text in the middle of a
    // legitimate reply, which is vanishingly rare.
    strip_stray_control_markers(&out).trim().to_string()
}

fn strip_stray_control_markers(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    let bytes = text.as_bytes();
    while i < bytes.len() {
        if text.is_char_boundary(i) {
            // `<|…>` — opening-style control marker.
            if text[i..].starts_with("<|") {
                if let Some(rel) = text[i + 2..].find('>') {
                    i += 2 + rel + 1;
                    continue;
                }
            }
            // `<…|>` — closing-style control marker.
            if let Some(ch) = text[i..].chars().next() {
                if ch == '<' {
                    if let Some(rel) = text[i + 1..].find("|>") {
                        let inside = &text[i + 1..i + 1 + rel];
                        // Guard against eating a legitimate `<` in prose: only
                        // strip when the body is a bare token name (letters /
                        // digits / underscore), matching Gemma's control shape.
                        if !inside.is_empty()
                            && inside.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                        {
                            i += 1 + rel + 2;
                            continue;
                        }
                    }
                }
            }
        }
        if let Some(ch) = text[i..].chars().next() {
            out.push(ch);
            i += ch.len_utf8();
        } else {
            i += 1;
        }
    }
    out
}

// ═════════════════════════════════════════════════════════════════════
// Qwen3-VL family (XML-style tool calls)
// ═════════════════════════════════════════════════════════════════════
//
// Qwen 3.5 / 3.6 emit tool calls as XML blocks:
//   <tool_call>
//   {"name": "foo", "arguments": {"a": 1}}
//   </tool_call>
// Tool results return in matching `<tool_response>...</tool_response>`
// blocks. The inner payload is a single JSON object per call with
// `name` (string) and `arguments` (object). The chat template in
// `qwen3-6-35b-a3b-fp8/tokenizer_config.json` confirms this exact shape.

const QWEN_TOOL_OPEN: &str = "<tool_call>";
const QWEN_TOOL_CLOSE: &str = "</tool_call>";
const QWEN_RESPONSE_OPEN: &str = "<tool_response>";
const QWEN_RESPONSE_CLOSE: &str = "</tool_response>";

/// Extract all Qwen3-VL tool calls from decoded text. Returns an empty
/// vec when no markup is present — the SSE / non-streaming caller then
/// treats the text as a plain content reply.
///
/// Tolerates surrounding whitespace, missing closing tag (the model
/// occasionally truncates), and multiple calls in one assistant turn.
/// Skips blocks whose inner payload doesn't parse as JSON with a
/// `name` field.
pub fn parse_qwen36_tool_calls(text: &str) -> Vec<ParsedToolCall> {
    let mut out = Vec::new();
    let mut cursor = 0;
    while cursor < text.len() {
        let Some(rel) = text[cursor..].find(QWEN_TOOL_OPEN) else { break };
        let payload_start = cursor + rel + QWEN_TOOL_OPEN.len();
        let after = &text[payload_start..];
        // Closing tag may be absent (mid-stream cutoff or model truncation).
        // In that case consume up to end-of-string and try to parse.
        let (payload, advance) = match after.find(QWEN_TOOL_CLOSE) {
            Some(end_rel) => {
                let p = &after[..end_rel];
                (p, end_rel + QWEN_TOOL_CLOSE.len())
            }
            None => (after, after.len()),
        };
        if let Some(call) = qwen_payload_to_call(payload) {
            out.push(call);
        }
        cursor = payload_start + advance;
    }
    out
}

/// Parse the inner payload of a single `<tool_call>...</tool_call>`
/// block. Qwen 3.5 / 3.6 ship with two different training-data formats
/// in the wild — we accept either:
///
/// (a) **Canonical JSON** (matches `tokenizer_config.json` chat template
///     prose): `{"name": "...", "arguments": {...}}`. Also tolerates
///     `arguments` as a JSON string (double-encoded).
///
/// (b) **OpenManus-style DSL** (what qwen3-6-35b-a3b actually emits on
///     this hardware, regardless of what the template prose says):
///     ```
///     <function=NAME>
///     <parameter=KEY>
///     VALUE
///     </parameter>
///     ...
///     </function>
///     ```
///     Re-serialised into JSON-string `arguments` so the OpenAI
///     `tool_calls[].function.arguments` contract holds for clients.
///
/// Returns `None` if neither shape parses.
fn qwen_payload_to_call(payload: &str) -> Option<ParsedToolCall> {
    let trimmed = payload.trim();
    // (a) JSON form
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        if let Some(obj) = v.as_object() {
            if let Some(name) = obj.get("name").and_then(|n| n.as_str()) {
                let args_value = obj
                    .get("arguments")
                    .cloned()
                    .unwrap_or(Value::Object(Map::new()));
                let arguments = match args_value {
                    Value::String(s) => match serde_json::from_str::<Value>(&s) {
                        Ok(parsed) => serde_json::to_string(&parsed).ok()?,
                        Err(_) => serde_json::to_string(&s).ok()?,
                    },
                    other => serde_json::to_string(&other).ok()?,
                };
                return Some(ParsedToolCall { name: name.to_string(), arguments });
            }
        }
    }
    // (b) OpenManus-style DSL form
    qwen_dsl_payload_to_call(trimmed)
}

/// Parse the `<function=NAME>...<parameter=KEY>VALUE</parameter>...</function>`
/// DSL into a [`ParsedToolCall`] with JSON-serialised arguments.
fn qwen_dsl_payload_to_call(payload: &str) -> Option<ParsedToolCall> {
    let func_open_pat = "<function=";
    let func_open_idx = payload.find(func_open_pat)?;
    let after_eq = func_open_idx + func_open_pat.len();
    let name_end = payload[after_eq..].find('>')?;
    let name = payload[after_eq..after_eq + name_end].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let body_start = after_eq + name_end + 1;
    let body_end = payload[body_start..]
        .find("</function>")
        .map(|r| body_start + r)
        .unwrap_or(payload.len());
    let body = &payload[body_start..body_end];

    let mut args = Map::new();
    let mut cursor = 0;
    let param_open = "<parameter=";
    let param_close = "</parameter>";
    while cursor < body.len() {
        let Some(rel) = body[cursor..].find(param_open) else { break };
        let abs = cursor + rel + param_open.len();
        let Some(key_end_rel) = body[abs..].find('>') else { break };
        let key = body[abs..abs + key_end_rel].trim().to_string();
        let value_start = abs + key_end_rel + 1;
        let value_end = match body[value_start..].find(param_close) {
            Some(r) => value_start + r,
            None => break,
        };
        let raw = body[value_start..value_end].trim();
        // Coerce parameter VALUE to the most natural JSON type so the
        // OpenAI `arguments` payload looks idiomatic: parse as JSON
        // first (handles numbers / bools / null / nested objects),
        // otherwise keep as a string.
        let v: Value = serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()));
        if !key.is_empty() {
            args.insert(key, v);
        }
        cursor = value_end + param_close.len();
    }
    Some(ParsedToolCall {
        name,
        arguments: serde_json::to_string(&Value::Object(args)).ok()?,
    })
}

/// Strip Qwen3-VL tool-call and tool-response markup so the streaming
/// content path sees a clean prose view (mirrors Gemma 4's
/// [`strip_tool_markup`] for that family). The opener is also kept
/// publicly accessible via [`QWEN_TOOL_CALL_OPENER`] so the SSE
/// `safe_content_emit_end` machinery can hold back content when an
/// opener is in flight without its closer.
pub const QWEN_TOOL_CALL_OPENER: &str = QWEN_TOOL_OPEN;

pub fn strip_qwen36_tool_markup(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if text.is_char_boundary(i) {
            for (open, close) in [
                (QWEN_TOOL_OPEN, QWEN_TOOL_CLOSE),
                (QWEN_RESPONSE_OPEN, QWEN_RESPONSE_CLOSE),
            ] {
                if text[i..].starts_with(open) {
                    let rest = &text[i + open.len()..];
                    let skip = match rest.find(close) {
                        Some(rel) => open.len() + rel + close.len(),
                        // Same posture as Gemma's strip path: when the
                        // opener has no closer, drop only the opener so
                        // the following prose still surfaces — visible
                        // bug beats silent swallow.
                        None => open.len(),
                    };
                    i += skip;
                    // Re-enter the outer loop to handle back-to-back
                    // blocks without an intervening prose char.
                    continue;
                }
            }
            let _ = bytes; // silence stale-binding lint if loop above continues
        }
        if let Some(ch) = text[i..].chars().next() {
            out.push(ch);
            i += ch.len_utf8();
        } else {
            i += 1;
        }
    }
    out.trim().to_string()
}

// ═════════════════════════════════════════════════════════════════════
// Arch-aware dispatch
// ═════════════════════════════════════════════════════════════════════

/// Which tool-call dialect to parse. Picked by the OpenAI handler from
/// the loaded `VisionArch`. Kept as a small enum (instead of taking a
/// full `VisionArch`) so the tool_parser module stays free of the
/// router type cycle and unit-testable from this crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolDialect {
    /// Gemma 4 (31B / E4B): `<|tool_call>call:NAME{ARGS}<tool_call|>`.
    Gemma4,
    /// Qwen 3.5 / 3.6: `<tool_call>{"name":"...","arguments":{...}}</tool_call>`.
    Qwen36,
}

/// Parse the assistant text into a list of tool calls using the dialect
/// for the loaded model family. Returns an empty vec when no calls are
/// detected — callers then treat the text as a plain content reply.
pub fn parse_tool_calls(text: &str, dialect: ToolDialect) -> Vec<ParsedToolCall> {
    match dialect {
        ToolDialect::Gemma4 => parse_gemma4_tool_calls(text),
        ToolDialect::Qwen36 => parse_qwen36_tool_calls(text),
    }
}

/// Strip all tool-call / thought / control markup for the chosen
/// dialect — returns the visible prose. Mirrors the dispatch shape of
/// [`parse_tool_calls`].
pub fn strip_tool_markup_for(text: &str, dialect: ToolDialect) -> String {
    match dialect {
        ToolDialect::Gemma4 => strip_tool_markup(text),
        ToolDialect::Qwen36 => strip_qwen36_tool_markup(text),
    }
}

/// Opener token for the dialect — the SSE path holds back content past
/// this marker while the closer hasn't arrived yet.
pub fn tool_call_opener_for(dialect: ToolDialect) -> &'static str {
    match dialect {
        ToolDialect::Gemma4 => TOOL_CALL_OPENER,
        ToolDialect::Qwen36 => QWEN_TOOL_CALL_OPENER,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_single_call_json_args() {
        let s = r#"<|tool_call>call:get_weather{"city":"Zurich"}<tool_call|>"#;
        let calls = parse_gemma4_tool_calls(s);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"city":"Zurich"}"#);
    }

    #[test]
    fn extracts_call_with_turn_terminator() {
        let s = r#"<|tool_call>call:ping{}<turn|>"#;
        let calls = parse_gemma4_tool_calls(s);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ping");
        assert_eq!(calls[0].arguments, "{}");
    }

    #[test]
    fn handles_escape_token() {
        let s = "<|tool_call>call:foo{q:<|\"|>hello<|\"|>}<tool_call|>";
        let calls = parse_gemma4_tool_calls(s);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, r#"{"q":"hello"}"#);
    }

    #[test]
    fn extracts_multiple_calls() {
        let s = r#"<|tool_call>call:a{"x":1}<tool_call|> noise <|tool_call>call:b{"y":2}<tool_call|>"#;
        let calls = parse_gemma4_tool_calls(s);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
    }

    #[test]
    fn plain_text_yields_nothing() {
        let calls = parse_gemma4_tool_calls("Paris is the capital of France.");
        assert!(calls.is_empty());
    }

    #[test]
    fn tier2_bare_call_extracted() {
        // What Gemma 4 actually emits to rvllm-serve once special
        // tokens are stripped — this is the exact payload that
        // tripped zeroclaw in prod.
        let s = r#"call:get_weather{location: "Zurich"}thought"#;
        let calls = parse_gemma4_tool_calls(s);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"location":"Zurich"}"#);
    }

    #[test]
    fn tier2_only_triggers_when_anchored() {
        // "recall:foo{..}" must not match tier-2.
        let calls = parse_gemma4_tool_calls("the recall:foo{x:1} is fine");
        assert!(calls.is_empty());
    }

    #[test]
    fn strip_sweeps_stray_control_markers() {
        // `<|tool_response>…<turn|>` is a reasoning block — its body (here
        // "ok") is the model's pre-answer draft and must stay hidden.
        // Trailing stray `<|turn>` is a control marker the sweep removes.
        let s = "<|tool_response>ok<turn|> Das ist in Ordnung.<|turn> ";
        let stripped = strip_tool_markup(s);
        assert_eq!(stripped, "Das ist in Ordnung.");
    }

    #[test]
    fn strip_drops_hallucinated_thought_fragment() {
        // Regression for the live failure: Gemma replied with the leaked
        // pattern `<|tool_response>thought\n<channel|><thought\n<channel|>
        // Das Wetter ...`. Every bracketed marker should vanish; only the
        // real answer survives.
        let s = "<|tool_response>thought\n<channel|>\
                 <thought\n<channel|>\
                 Das Wetter in Bern ist heute bewölkt.<turn|>";
        let stripped = strip_tool_markup(s);
        assert_eq!(stripped, "Das Wetter in Bern ist heute bewölkt.");
    }

    #[test]
    fn strip_keeps_math_inequality() {
        // Regression guard — the stray-sweep must not chew up `<something|>`
        // shapes that aren't Gemma tokens. A sentence like "5<3|>" is weird
        // but the body "3" is numeric → alphanumeric → would strip. That's
        // accepted; the same body with punctuation (`"3, 4"`) must NOT.
        let s = "a<3, 4|>b";
        let stripped = strip_tool_markup(s);
        assert_eq!(stripped, "a<3, 4|>b");
    }

    #[test]
    fn strip_drops_thought_channel() {
        // Gemma 4 writes its internal reasoning inside `<|channel>thought...<channel|>`.
        // Without explicit stripping the tokenizer's skip-specials pass drops
        // the markers but leaves the reasoning prose in user-visible content.
        let s = "<|channel>thought\nDas wird wohl 14°C sein.<channel|>\
                 <|tool_call>call:weather{city:\"Bern\"}<tool_call|>";
        let stripped = strip_tool_markup(s);
        assert_eq!(stripped, "", "thought + tool_call markup should leave empty content, got {stripped:?}");
        let calls = parse_gemma4_tool_calls(s);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "weather");
        assert_eq!(calls[0].arguments, r#"{"city":"Bern"}"#);
    }

    #[test]
    fn strip_handles_utf8_prose() {
        // Regression: the parser used to panic with
        // `byte index is not a char boundary; it is inside 'ü'` when
        // Gemma 4 emitted a call followed by German prose.
        let s = "call:weather{city:Bern} Es regnet in Zürich und München.";
        let stripped = strip_tool_markup(s);
        assert!(stripped.contains("Zürich"));
        assert!(!stripped.contains("call:"));
    }

    #[test]
    fn parse_tier2_handles_utf8_prose() {
        // Same regression for the tier-2 matcher — must walk past
        // non-ASCII bytes without panicking.
        let s = "Die Antwort: ü call:ping{x:1}";
        let calls = parse_gemma4_tool_calls(s);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ping");
    }

    #[test]
    fn strip_removes_markup_keeps_prose() {
        let s = r#"Let me check. <|tool_call>call:foo{"a":1}<tool_call|> done."#;
        assert_eq!(strip_tool_markup(s), "Let me check.  done.");
    }

    // === Cycle 33/34 regression tests for parser visibility/safety ===

    #[test]
    fn malformed_opener_no_close_preserves_body() {
        // Cycle 33 fix: prior code skipped to EOF when `<|tool_call>` had
        // no matching `<tool_call|>` close, turning the model's broken
        // tool-call attempt into "" content. Now we strip ONLY the opener
        // tag itself so the body content surfaces — ugly UX but real
        // diagnostic visibility.
        let s = r#"<|tool_call>_- garbage la la la"#;
        let stripped = strip_tool_markup(s);
        // Opener tag is gone, but the body must remain visible.
        assert!(!stripped.contains("<|tool_call>"));
        assert!(stripped.contains("garbage"));
    }

    #[test]
    fn malformed_opener_then_valid_call_preserves_call() {
        // Even with a stray opener earlier in the stream, a real call
        // later must still parse. Defensive — the bug-fix should not
        // regress the multi-call mixed-content path.
        let s = r#"<|tool_call> stray text <|tool_call>call:get_weather{"city":"Bern"}<tool_call|>"#;
        let calls = parse_gemma4_tool_calls(s);
        // At least one valid call is recovered.
        assert!(!calls.is_empty(), "expected ≥1 valid call, got {}", calls.len());
        assert!(calls.iter().any(|c| c.name == "get_weather"));
    }

    #[test]
    fn empty_input_yields_empty_calls_and_empty_strip() {
        assert_eq!(parse_gemma4_tool_calls("").len(), 0);
        assert_eq!(strip_tool_markup(""), "");
    }

    #[test]
    fn lone_close_tag_does_not_panic() {
        // A stray `<tool_call|>` with no opener must not crash the parser.
        let s = "some prose <tool_call|> more prose";
        let _stripped = strip_tool_markup(s);
        let _calls = parse_gemma4_tool_calls(s);
        // No assertion on output shape — just that we got here.
    }

    #[test]
    fn tier2_nested_braces_handled_by_balanced_scanner() {
        // Tier-2 used to stop at the FIRST `}`, so a JSON object value
        // with a nested object got truncated to `call:foo{"a":{"b":1}`
        // and the args fallback parser then produced gibberish. The
        // balanced-brace scanner now matches the closing `}` at depth
        // zero, so nested objects round-trip cleanly. The tier-1
        // wrapped form (covered separately) was always fine because
        // the wrapper terminator delimits the args region.
        let s = r#"call:foo{"a":{"b":1}}"#;
        let calls = parse_gemma4_tool_calls(s);
        assert_eq!(calls.len(), 1, "expected one call, got {calls:?}");
        assert_eq!(calls[0].name, "foo");
        let args: serde_json::Value =
            serde_json::from_str(&calls[0].arguments).expect("valid json");
        assert_eq!(args["a"]["b"], serde_json::json!(1));
    }

    #[test]
    fn tier2_brace_inside_string_does_not_close_args() {
        // A `}` inside a JSON string literal must NOT terminate the
        // args region. The balanced-brace scanner tracks string state
        // and ignores braces between unescaped quotes.
        let s = r#"call:say{"text":"hello } world","level":1}"#;
        let calls = parse_gemma4_tool_calls(s);
        assert_eq!(calls.len(), 1, "expected one call, got {calls:?}");
        assert_eq!(calls[0].name, "say");
        let args: serde_json::Value =
            serde_json::from_str(&calls[0].arguments).expect("valid json");
        assert_eq!(args["text"], "hello } world");
        assert_eq!(args["level"], 1);
    }

    #[test]
    fn strip_handles_tier2_nested_args_without_residue() {
        // Symmetry guard for the balanced-brace scanner. `parse_tier2_bare`
        // matches up to the depth-zero closer; `strip_tool_markup` must
        // remove EXACTLY the same span so no `}` / no trailing prose
        // (after the call but before another marker) leaks into the
        // user-visible content. Before the fix, strip stopped at the
        // first inner `}` and left `}thought` in `content`.
        let s = r#"call:foo{"a":{"b":1}}thought"#;
        let stripped = strip_tool_markup(s);
        assert_eq!(stripped, "thought", "strip leaked residue: {stripped:?}");
    }

    #[test]
    fn strip_handles_tier2_brace_in_string_without_residue() {
        // Same property but with a `}` inside a JSON string literal —
        // the scanner must NOT treat it as a closer for either parse
        // or strip.
        let s = r#"call:say{"text":"hello } world"} ok"#;
        let stripped = strip_tool_markup(s);
        assert_eq!(stripped.trim(), "ok", "strip leaked residue: {stripped:?}");
    }

    #[test]
    fn tier2_unclosed_brace_does_not_panic_or_match() {
        // Truncated stream — the bare call never receives its closing
        // brace. We must drop it cleanly rather than panic or invent a
        // partial match.
        let s = r#"call:foo{"a":{"b":1}"#; // missing trailing `}`
        let calls = parse_gemma4_tool_calls(s);
        assert!(calls.is_empty(), "expected no calls, got {calls:?}");
    }

    #[test]
    fn tier1_wrapped_handles_nested_braces_correctly() {
        // Same nested args, but inside the tier-1 wrapper. Should parse
        // the full JSON because END_A/B match the wrapper close, not
        // the first inner `}`.
        let s = r#"<|tool_call>call:foo{"a":{"b":1}}<tool_call|>"#;
        let calls = parse_gemma4_tool_calls(s);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "foo");
        // Args should round-trip the nested object structure.
        let args: serde_json::Value =
            serde_json::from_str(&calls[0].arguments).expect("valid json");
        assert_eq!(args["a"]["b"], serde_json::json!(1));
    }

    #[test]
    fn utf8_inside_tool_call_args() {
        // German umlauts inside the JSON args of a valid call must
        // round-trip. Regression for the panic discovered in cycle 13ish.
        let s = r#"<|tool_call>call:weather{"city":"München"}<tool_call|>"#;
        let calls = parse_gemma4_tool_calls(s);
        assert_eq!(calls.len(), 1);
        assert!(calls[0].arguments.contains("München"));
    }

    // ─── Qwen3-VL XML tool-call parser ─────────────────────────────

    #[test]
    fn qwen36_single_call_round_trips() {
        let s = r#"<tool_call>
{"name": "get_weather", "arguments": {"city": "Bern", "units": "celsius"}}
</tool_call>"#;
        let calls = parse_qwen36_tool_calls(s);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(&calls[0].arguments).expect("valid json");
        assert_eq!(args["city"], "Bern");
        assert_eq!(args["units"], "celsius");
    }

    #[test]
    fn qwen36_inline_compact_call() {
        // No surrounding newlines — the chat template doesn't require any.
        let s = r#"<tool_call>{"name":"ping","arguments":{}}</tool_call>"#;
        let calls = parse_qwen36_tool_calls(s);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ping");
        assert_eq!(calls[0].arguments, "{}");
    }

    #[test]
    fn qwen36_multiple_calls_one_turn() {
        let s = r#"<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>some prose<tool_call>{"name":"b","arguments":{"y":2}}</tool_call>"#;
        let calls = parse_qwen36_tool_calls(s);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
    }

    #[test]
    fn qwen36_arguments_as_double_encoded_string() {
        // Some upstream Qwen training data emits arguments as a JSON
        // string instead of a nested object. Re-parse + re-serialise.
        let s = r#"<tool_call>{"name":"f","arguments":"{\"k\":\"v\"}"}</tool_call>"#;
        let calls = parse_qwen36_tool_calls(s);
        assert_eq!(calls.len(), 1);
        let args: serde_json::Value =
            serde_json::from_str(&calls[0].arguments).expect("valid json");
        assert_eq!(args["k"], "v");
    }

    #[test]
    fn qwen36_missing_close_tag_recovers_payload() {
        // Model truncated mid-stream; still try to parse what we have.
        let s = r#"<tool_call>{"name":"foo","arguments":{"a":1}}"#;
        let calls = parse_qwen36_tool_calls(s);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "foo");
    }

    #[test]
    fn qwen36_malformed_payload_yields_no_call() {
        let s = "<tool_call>not json at all</tool_call>";
        assert!(parse_qwen36_tool_calls(s).is_empty());
    }

    #[test]
    fn qwen36_no_markup_returns_empty() {
        assert!(parse_qwen36_tool_calls("just a plain reply").is_empty());
        assert!(parse_qwen36_tool_calls("").is_empty());
    }

    #[test]
    fn qwen36_strip_removes_call_and_response_blocks() {
        let s = "before<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>middle<tool_response>{\"ok\":true}</tool_response>after";
        let stripped = strip_qwen36_tool_markup(s);
        assert_eq!(stripped, "beforemiddleafter");
    }

    #[test]
    fn qwen36_strip_handles_unterminated_block() {
        // Only the opener token is dropped; tail prose survives.
        let stripped = strip_qwen36_tool_markup("hi <tool_call>{}");
        assert!(stripped.contains("hi"));
        assert!(stripped.contains("{}"));
        assert!(!stripped.contains("<tool_call>"));
    }

    #[test]
    fn qwen36_dsl_form_round_trips() {
        // Captured verbatim from qwen3-6-35b-a3b on the rvllm webhook
        // path (2026-05-24); represents the OpenManus-style DSL that the
        // model actually emits regardless of the JSON-prose chat template.
        let s = "<tool_call>\n<function=brain>\n<parameter=action>\nsearch\n</parameter>\n<parameter=query>\nVinz\n</parameter>\n</function>\n</tool_call>";
        let calls = parse_qwen36_tool_calls(s);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "brain");
        let args: serde_json::Value =
            serde_json::from_str(&calls[0].arguments).expect("valid json");
        assert_eq!(args["action"], "search");
        assert_eq!(args["query"], "Vinz");
    }

    #[test]
    fn qwen36_dsl_with_numeric_parameter_value() {
        let s = "<tool_call><function=set_limit><parameter=limit>42</parameter></function></tool_call>";
        let calls = parse_qwen36_tool_calls(s);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "set_limit");
        let args: serde_json::Value =
            serde_json::from_str(&calls[0].arguments).expect("valid json");
        // Number coerced via serde JSON parse, not left as a string.
        assert_eq!(args["limit"], serde_json::json!(42));
    }

    #[test]
    fn qwen36_dsl_no_parameters() {
        let s = "<tool_call><function=ping></function></tool_call>";
        let calls = parse_qwen36_tool_calls(s);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ping");
        assert_eq!(calls[0].arguments, "{}");
    }

    #[test]
    fn qwen36_strip_preserves_plain_prose() {
        assert_eq!(strip_qwen36_tool_markup("hello world"), "hello world");
    }

    #[test]
    fn dispatch_routes_dialects_correctly() {
        let gemma_in = r#"<|tool_call>call:get_weather{"city":"X"}<tool_call|>"#;
        let qwen_in = r#"<tool_call>{"name":"get_weather","arguments":{"city":"X"}}</tool_call>"#;

        // Wrong-dialect routing must return empty (so the runtime
        // gracefully degrades to plain content rather than mis-parsing).
        assert!(parse_tool_calls(gemma_in, ToolDialect::Qwen36).is_empty());
        assert!(parse_tool_calls(qwen_in, ToolDialect::Gemma4).is_empty());

        // Correct routing extracts the call.
        let g = parse_tool_calls(gemma_in, ToolDialect::Gemma4);
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].name, "get_weather");
        let q = parse_tool_calls(qwen_in, ToolDialect::Qwen36);
        assert_eq!(q.len(), 1);
        assert_eq!(q[0].name, "get_weather");

        // Opener token differs.
        assert_eq!(tool_call_opener_for(ToolDialect::Gemma4), "<|tool_call>");
        assert_eq!(tool_call_opener_for(ToolDialect::Qwen36), "<tool_call>");
    }
}
