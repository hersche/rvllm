use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde_json::json;

use crate::router::AppState;

pub async fn audio_transcriptions(
    State(state): State<AppState>,
    axum::Extension(_admission): axum::Extension<
        std::sync::Arc<tokio::sync::OwnedSemaphorePermit>,
    >,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut filename: Option<String> = None;
    let mut model: Option<String> = None;
    let mut language: Option<String> = None;
    let mut response_format: String = "json".to_string();
    let mut prompt_text: Option<String> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| err(StatusCode::BAD_REQUEST, "multipart_parse", &e.to_string()))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                filename = field.file_name().map(|s| s.to_string());
                let data = field
                    .bytes()
                    .await
                    .map_err(|e| err(StatusCode::BAD_REQUEST, "multipart_read_file", &e.to_string()))?;
                file_bytes = Some(data.to_vec());
            }
            "model" => {
                model = Some(
                    field
                        .text()
                        .await
                        .map_err(|e| err(StatusCode::BAD_REQUEST, "multipart_read_model", &e.to_string()))?,
                );
            }
            "language" => {
                language = field.text().await.ok();
            }
            "response_format" => {
                if let Ok(s) = field.text().await {
                    response_format = s;
                }
            }
            "prompt" => {
                prompt_text = field.text().await.ok();
            }
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    let file_bytes = file_bytes
        .ok_or_else(|| err(StatusCode::BAD_REQUEST, "missing_file", "field `file` is required"))?;
    let model = model.unwrap_or_else(|| state.config.model_id.clone());

    if !matches!(response_format.as_str(), "json" | "text") {
        return Err(err(
            StatusCode::NOT_IMPLEMENTED,
            "response_format_unsupported",
            &format!(
                "response_format='{response_format}' not implemented; only 'json' and 'text' are supported"
            ),
        ));
    }

    let mime = guess_mime(filename.as_deref());
    let data_uri = format!(
        "data:{};base64,{}",
        mime,
        base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &file_bytes)
    );

    let lang = language.as_deref().unwrap_or("");
    let instruction = if lang.is_empty() {
        prompt_text
            .clone()
            .unwrap_or_else(|| "Transcribe the audio verbatim. Output only the transcript.".to_string())
    } else {
        prompt_text.clone().unwrap_or_else(|| {
            format!(
                "Transcribe the following audio verbatim in {lang}. Output only the transcript."
            )
        })
    };

    let chat_payload = json!({
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": data_uri}},
                {"type": "text", "text": instruction}
            ]
        }],
        "max_tokens": 1024,
        "temperature": 0.0,
        "stream": false
    });

    let port = std::env::var("RVLLM_PORT").unwrap_or_else(|_| "8010".to_string());
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .json(&chat_payload)
        .send()
        .await
        .map_err(|e| err(StatusCode::BAD_GATEWAY, "internal_chat_send", &e.to_string()))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(err(status, "internal_chat_error", &body));
    }
    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| err(StatusCode::BAD_GATEWAY, "internal_chat_parse", &e.to_string()))?;
    let text = body
        .pointer("/choices/0/message/content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    if response_format == "text" {
        Ok(([(axum::http::header::CONTENT_TYPE, "text/plain; charset=utf-8")], text).into_response())
    } else {
        Ok(Json(json!({"text": text})).into_response())
    }
}

fn err(status: StatusCode, code: &str, message: &str) -> (StatusCode, Json<serde_json::Value>) {
    (
        status,
        Json(json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": code,
            }
        })),
    )
}

fn guess_mime(filename: Option<&str>) -> &'static str {
    let name = filename.unwrap_or("").to_ascii_lowercase();
    if name.ends_with(".wav") {
        "audio/wav"
    } else if name.ends_with(".mp3") {
        "audio/mpeg"
    } else if name.ends_with(".flac") {
        "audio/flac"
    } else if name.ends_with(".ogg") || name.ends_with(".opus") {
        "audio/ogg"
    } else if name.ends_with(".m4a") || name.ends_with(".mp4") {
        "audio/mp4"
    } else {
        "audio/wav"
    }
}
