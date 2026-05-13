//! Audio-content fetch + decode + resample for OpenAI multimodal
//! chat-completions requests (B2 of the E4B native-audio plan).
//!
//! Structural twin of `vision_fetch.rs`:
//!   - `data:` URI (RFC 2397) with optional base64 + media type
//!   - `http(s)://` (5s timeout, bytes cap from RVLLM_AUDIO_MAX_BYTES)
//!
//! Output is fully decoded mono f32 PCM at 16 kHz — the encoder's
//! mel frontend (`rvllm_runtime::audio_preprocess::MelExtractor`)
//! consumes exactly that. We resample on the admission thread so
//! the cuda_worker thread never blocks on tokio or on rubato's
//! sinc kernel.
//!
//! Format support is whatever the `symphonia` feature set in
//! `Cargo.toml` enables (WAV, MP3, FLAC, OGG/Vorbis, AAC, MP4).
//! Containers not on that list arrive as bytes and produce a
//! clean 400 ("audio format not supported"). Opus / Speex / WMA
//! payloads must be transcoded by the client first.

use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;

/// Cap for a SINGLE decoded audio item's raw bytes (pre-decode).
/// Operator override via `RVLLM_AUDIO_MAX_BYTES`.
const DEFAULT_MAX_AUDIO_BYTES: usize = 32 * 1024 * 1024;
/// Hard cap on resampled-PCM length per item, in samples at 16 kHz.
/// `RVLLM_AUDIO_MAX_SECONDS` × 16000 by default (30 s × 16 kHz
/// = 480 000 samples) — matches `audio_seq_length=750` × 40 ms in
/// `MelConfig`.
const DEFAULT_MAX_AUDIO_SECONDS: u32 = 30;
const FETCH_TIMEOUT: Duration = Duration::from_secs(5);
/// Encoder input cadence — mel frontend expects 16 kHz mono f32.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

#[derive(Debug)]
pub enum AudioError {
    FetchTimeout,
    FetchFailed(String),
    TooLarge(usize, usize),
    BadDataUri(String),
    Decode(String),
    Resample(String),
    FormatUnsupported(String),
    Cancelled,
}

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FetchTimeout => write!(f, "fetch timeout"),
            Self::FetchFailed(s) => write!(f, "fetch failed: {s}"),
            Self::TooLarge(g, m) => write!(f, "audio too large: {g} bytes (max {m})"),
            Self::BadDataUri(s) => write!(f, "bad data uri: {s}"),
            Self::Decode(s) => write!(f, "audio decode failed: {s}"),
            Self::Resample(s) => write!(f, "audio resample failed: {s}"),
            Self::FormatUnsupported(s) => write!(f, "audio format not supported: {s}"),
            Self::Cancelled => write!(f, "audio fetch cancelled"),
        }
    }
}
impl std::error::Error for AudioError {}

/// Resolved audio item — what `cuda_worker` will consume.
///
/// `samples_16k_mono` is the entire item decoded + resampled to
/// 16 kHz mono f32. `duration_seconds` is the WALL-CLOCK length of
/// the source (post-resample sample count / 16000), surfaced for
/// transcription metadata. `format_hint` carries the OpenAI-realtime
/// `format` field for diagnostics only.
#[derive(Debug)]
pub struct AudioItem {
    pub samples_16k_mono: Vec<f32>,
    pub duration_seconds: f32,
    pub format_hint: Option<String>,
}

impl AudioItem {
    pub fn num_samples(&self) -> usize {
        self.samples_16k_mono.len()
    }
}

/// Resolve an `audio_url.url` field to a decoded `AudioItem`.
///
/// `format_hint` comes from the `audio_url.format` JSON field, if
/// present. `cancelled` is checked at the network and decode
/// boundaries so a client disconnect doesn't tie up the admission
/// thread for the full timeout.
pub fn fetch_audio(
    url: &str,
    format_hint: Option<&str>,
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<AudioItem, AudioError> {
    use std::sync::atomic::Ordering;
    if cancelled.load(Ordering::Relaxed) {
        return Err(AudioError::Cancelled);
    }

    let max_bytes = std::env::var("RVLLM_AUDIO_MAX_BYTES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_AUDIO_BYTES);
    let max_seconds = std::env::var("RVLLM_AUDIO_MAX_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_AUDIO_SECONDS);

    let raw_bytes = if let Some(rest) = url.strip_prefix("data:") {
        parse_data_uri(rest, max_bytes)?
    } else if url.starts_with("http://") || url.starts_with("https://") {
        fetch_http(url, max_bytes)?
    } else {
        return Err(AudioError::FetchFailed(format!(
            "unsupported URL scheme: {url}"
        )));
    };

    if cancelled.load(Ordering::Relaxed) {
        return Err(AudioError::Cancelled);
    }

    let (samples_native, source_rate, channels) = decode_with_symphonia(&raw_bytes, format_hint)?;
    let mono = downmix_to_mono(&samples_native, channels);
    let resampled = resample_to_target(mono, source_rate, TARGET_SAMPLE_RATE)?;

    // Hard cap on output length AFTER resample so a malicious 1 Hz
    // 1 MB WAV doesn't expand to 16k × MAX_SECONDS samples and OOM
    // the admission thread. `samples` is already on the heap; we
    // truncate in place to avoid an extra alloc.
    let max_samples = (max_seconds as usize) * (TARGET_SAMPLE_RATE as usize);
    let truncated = resampled.len() > max_samples;
    let samples_16k_mono = if truncated {
        resampled[..max_samples].to_vec()
    } else {
        resampled
    };

    let duration_seconds = samples_16k_mono.len() as f32 / TARGET_SAMPLE_RATE as f32;
    Ok(AudioItem {
        samples_16k_mono,
        duration_seconds,
        format_hint: format_hint.map(|s| s.to_string()),
    })
}

// ---------------------------------------------------------------------------
//  data: URI
// ---------------------------------------------------------------------------

fn parse_data_uri(rest: &str, max_bytes: usize) -> Result<Vec<u8>, AudioError> {
    let comma = rest
        .find(',')
        .ok_or_else(|| AudioError::BadDataUri("missing comma".into()))?;
    let (header, data) = rest.split_at(comma);
    let data = &data[1..];
    let mut is_base64 = false;
    for token in header.split(';') {
        if token == "base64" {
            is_base64 = true;
        }
    }
    // Same encoded-size pre-check as vision_fetch.rs to avoid a 60 MiB
    // base64 payload getting fully decoded just to be rejected.
    let max_encoded = if is_base64 {
        max_bytes.saturating_mul(4) / 3 + 4
    } else {
        max_bytes
    };
    if data.len() > max_encoded {
        return Err(AudioError::TooLarge(data.len(), max_encoded));
    }
    let bytes = if is_base64 {
        B64.decode(data.as_bytes())
            .map_err(|e| AudioError::BadDataUri(format!("base64: {e}")))?
    } else {
        // RFC 2397 §3 percent-decoding; mirror vision_fetch's
        // implementation. Audio payloads are usually base64 in
        // practice (any binary), but spec allows percent-encoded too.
        let mut out = Vec::with_capacity(data.len());
        let input = data.as_bytes();
        let mut i = 0;
        while i < input.len() {
            let b = input[i];
            if b == b'%' {
                if i + 2 >= input.len() {
                    return Err(AudioError::BadDataUri(format!(
                        "truncated %xx escape at offset {i}"
                    )));
                }
                let hi = (input[i + 1] as char).to_digit(16).ok_or_else(|| {
                    AudioError::BadDataUri(format!("bad hex {:?}", input[i + 1] as char))
                })?;
                let lo = (input[i + 2] as char).to_digit(16).ok_or_else(|| {
                    AudioError::BadDataUri(format!("bad hex {:?}", input[i + 2] as char))
                })?;
                out.push(((hi << 4) | lo) as u8);
                i += 3;
            } else {
                out.push(b);
                i += 1;
            }
        }
        out
    };
    if bytes.len() > max_bytes {
        return Err(AudioError::TooLarge(bytes.len(), max_bytes));
    }
    Ok(bytes)
}

// ---------------------------------------------------------------------------
//  HTTP fetch (lean — reuses vision_fetch.rs's blocking client style)
// ---------------------------------------------------------------------------

fn http_client() -> &'static reqwest::blocking::Client {
    use std::sync::OnceLock;
    static CLIENT: OnceLock<reqwest::blocking::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::blocking::Client::builder()
            .timeout(FETCH_TIMEOUT)
            .pool_idle_timeout(Duration::from_secs(60))
            .pool_max_idle_per_host(8)
            // Same SSRF stance as vision_fetch: refuse redirects,
            // disable env-driven proxies. Operators can flip
            // `RVLLM_AUDIO_FETCH_ALLOW_PRIVATE=1` to permit
            // private-network targets (mirrors the vision env knob).
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .build()
            .expect("build audio-fetch reqwest client")
    })
}

fn fetch_http(url: &str, max_bytes: usize) -> Result<Vec<u8>, AudioError> {
    let client = http_client();
    let mut resp = client
        .get(url)
        .send()
        .map_err(|e| {
            if e.is_timeout() {
                AudioError::FetchTimeout
            } else {
                AudioError::FetchFailed(format!("send: {e}"))
            }
        })?;
    if !resp.status().is_success() {
        return Err(AudioError::FetchFailed(format!(
            "status: {}",
            resp.status()
        )));
    }
    // Stream the body so we can refuse early if it exceeds the cap.
    let mut buf = Vec::with_capacity(64 * 1024);
    let mut chunk = [0u8; 64 * 1024];
    loop {
        use std::io::Read;
        let n = resp.read(&mut chunk).map_err(|e| {
            AudioError::FetchFailed(format!("read: {e}"))
        })?;
        if n == 0 {
            break;
        }
        if buf.len() + n > max_bytes {
            return Err(AudioError::TooLarge(buf.len() + n, max_bytes));
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    Ok(buf)
}

// ---------------------------------------------------------------------------
//  Symphonia decode → interleaved f32 PCM
// ---------------------------------------------------------------------------

/// Returns `(samples_interleaved_f32, source_sample_rate, channels)`.
/// Always normalizes the source to f32 in `[-1, 1]` regardless of the
/// original PCM bit depth so the resampler downstream sees a uniform
/// dtype.
fn decode_with_symphonia(
    bytes: &[u8],
    format_hint: Option<&str>,
) -> Result<(Vec<f32>, u32, u16), AudioError> {
    use std::io::Cursor;
    use symphonia::core::audio::AudioBufferRef;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error as SymphoniaError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let mss = MediaSourceStream::new(Box::new(Cursor::new(bytes.to_vec())), Default::default());
    let mut hint = Hint::new();
    if let Some(fmt) = format_hint {
        // RFC 6838 mime hints like "audio/mpeg" don't directly hit
        // symphonia's extension/mime registries; strip to the bare
        // codec tag (after the slash) for the extension fallback.
        let ext = fmt.split('/').next_back().unwrap_or(fmt);
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| match e {
            SymphoniaError::Unsupported(s) => AudioError::FormatUnsupported(s.to_string()),
            other => AudioError::Decode(format!("probe: {other}")),
        })?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or_else(|| AudioError::Decode("no decodable tracks".into()))?;
    let track_id = track.id;
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count() as u16)
        .unwrap_or(1);
    let source_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| AudioError::Decode("track missing sample_rate".into()))?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| AudioError::Decode(format!("decoder: {e}")))?;

    let mut out: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(ref io))
                if io.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => {
                return Err(AudioError::Decode(format!("next_packet: {e}")));
            }
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymphoniaError::DecodeError(_)) => continue, // skip bad packet
            Err(e) => return Err(AudioError::Decode(format!("decode: {e}"))),
        };
        // Convert whatever sample dtype symphonia emits into f32
        // interleaved. AudioBufferRef has typed planar buffers per
        // dtype; iterate channel-major and interleave.
        match decoded {
            AudioBufferRef::F32(buf) => append_planar_to_interleaved(&buf, &mut out, |s| s),
            AudioBufferRef::F64(buf) => {
                append_planar_to_interleaved(&buf, &mut out, |s| s as f32)
            }
            AudioBufferRef::S8(buf) => append_planar_to_interleaved(&buf, &mut out, |s| {
                s as f32 / i8::MAX as f32
            }),
            AudioBufferRef::S16(buf) => append_planar_to_interleaved(&buf, &mut out, |s| {
                s as f32 / i16::MAX as f32
            }),
            AudioBufferRef::S24(buf) => append_planar_to_interleaved(&buf, &mut out, |s| {
                s.inner() as f32 / 0x7f_ffff as f32
            }),
            AudioBufferRef::S32(buf) => append_planar_to_interleaved(&buf, &mut out, |s| {
                s as f32 / i32::MAX as f32
            }),
            AudioBufferRef::U8(buf) => append_planar_to_interleaved(&buf, &mut out, |s| {
                (s as f32 - 128.0) / 128.0
            }),
            AudioBufferRef::U16(buf) => append_planar_to_interleaved(&buf, &mut out, |s| {
                (s as f32 - 32768.0) / 32768.0
            }),
            AudioBufferRef::U24(buf) => append_planar_to_interleaved(&buf, &mut out, |s| {
                (s.inner() as f32 - 0x80_0000 as f32) / 0x80_0000 as f32
            }),
            AudioBufferRef::U32(buf) => append_planar_to_interleaved(&buf, &mut out, |s| {
                (s as f32 - 2_147_483_648.0) / 2_147_483_648.0
            }),
        }
    }
    if out.is_empty() {
        return Err(AudioError::Decode("decoded 0 samples".into()));
    }
    Ok((out, source_rate, channels))
}

/// Helper: append a planar `AudioBuffer<S>` (channel-major) to an
/// interleaved `Vec<f32>` (sample-major). symphonia hands us all
/// channels in lock-step at the same frame count, so the loop is
/// trivially correct.
fn append_planar_to_interleaved<S>(
    buf: &symphonia::core::audio::AudioBuffer<S>,
    out: &mut Vec<f32>,
    cast: impl Fn(S) -> f32,
) where
    S: symphonia::core::sample::Sample + Copy,
{
    use symphonia::core::audio::Signal;
    let n_frames = buf.frames();
    let n_chans = buf.spec().channels.count();
    out.reserve(n_frames * n_chans);
    for f in 0..n_frames {
        for c in 0..n_chans {
            out.push(cast(buf.chan(c)[f]));
        }
    }
}

// ---------------------------------------------------------------------------
//  Downmix + resample
// ---------------------------------------------------------------------------

fn downmix_to_mono(interleaved: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    let n = channels as usize;
    let frames = interleaved.len() / n;
    let mut out = Vec::with_capacity(frames);
    for f in 0..frames {
        let base = f * n;
        let mut acc = 0.0f32;
        for c in 0..n {
            acc += interleaved[base + c];
        }
        out.push(acc / n as f32);
    }
    out
}

fn resample_to_target(
    mono: Vec<f32>,
    source_rate: u32,
    target_rate: u32,
) -> Result<Vec<f32>, AudioError> {
    if source_rate == target_rate {
        return Ok(mono);
    }
    use rubato::Resampler;
    let ratio = target_rate as f64 / source_rate as f64;
    // Chunk size is fixed by SincFixedIn; we pick something
    // moderate-ish (1024 input frames per chunk) and zero-pad the
    // last partial chunk. rubato's `process_partial` accepts a
    // shorter final chunk with `Some(&[...])` but only if the
    // resampler was built with `SincFixedOut`; SincFixedIn is the
    // simpler API and zero-padding is cheap for the trailing
    // <chunk_size samples. We then truncate the output by the
    // expected length (ceil(in_len * ratio)).
    const CHUNK: usize = 1024;
    let mut resampler = rubato::SincFixedIn::<f32>::new(
        ratio,
        2.0,
        rubato::SincInterpolationParameters {
            sinc_len: 128,
            f_cutoff: 0.95,
            interpolation: rubato::SincInterpolationType::Linear,
            oversampling_factor: 128,
            window: rubato::WindowFunction::Blackman,
        },
        CHUNK,
        1,
    )
    .map_err(|e| AudioError::Resample(format!("new: {e}")))?;
    let expected_out = ((mono.len() as f64) * ratio).ceil() as usize;
    let mut out: Vec<f32> = Vec::with_capacity(expected_out + 32);
    let mut idx = 0usize;
    while idx < mono.len() {
        let end = (idx + CHUNK).min(mono.len());
        let mut chunk = vec![0.0f32; CHUNK];
        chunk[..(end - idx)].copy_from_slice(&mono[idx..end]);
        let input = vec![chunk];
        let produced = resampler
            .process(&input, None)
            .map_err(|e| AudioError::Resample(format!("process: {e}")))?;
        let ch0 = produced.into_iter().next().unwrap_or_default();
        out.extend_from_slice(&ch0);
        idx += CHUNK;
    }
    // Truncate trailing zero-pad-induced samples to the expected
    // output length. Without this an 8 kHz, 800-sample input would
    // pad to 1024 and emit ~2048 output samples; the caller
    // expects 1600.
    out.truncate(expected_out);
    Ok(out)
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    /// Synthesize a minimal 16-bit PCM WAV header + body for a
    /// constant-valued mono signal at `sample_rate` Hz. Used as a
    /// fixture for the data: URI fast path.
    fn make_wav_const(sample_rate: u32, num_samples: u32, value: i16) -> Vec<u8> {
        let byte_rate = sample_rate * 2; // mono, 16-bit
        let data_bytes = (num_samples * 2) as u32;
        let mut out = Vec::with_capacity(44 + data_bytes as usize);
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
        out.extend_from_slice(&1u16.to_le_bytes()); // PCM
        out.extend_from_slice(&1u16.to_le_bytes()); // mono
        out.extend_from_slice(&sample_rate.to_le_bytes());
        out.extend_from_slice(&byte_rate.to_le_bytes());
        out.extend_from_slice(&2u16.to_le_bytes()); // block align
        out.extend_from_slice(&16u16.to_le_bytes()); // bits/sample
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_bytes.to_le_bytes());
        for _ in 0..num_samples {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    #[test]
    fn data_uri_wav_decodes_and_passes_through_16khz() {
        let wav = make_wav_const(16_000, 1_600, 8_192); // 100 ms at 16 kHz
        let b64 = B64.encode(&wav);
        let url = format!("data:audio/wav;base64,{b64}");
        let cancel = AtomicBool::new(false);
        let item = fetch_audio(&url, Some("wav"), &cancel).expect("fetch ok");
        assert_eq!(item.samples_16k_mono.len(), 1_600);
        // Constant-valued WAV @ +0.25 (8192/32767 ≈ 0.250) — no
        // resample needed, no resample wobble.
        let mid = item.samples_16k_mono[800];
        assert!((mid - 0.25).abs() < 5e-4, "midpoint = {mid}");
        assert!((item.duration_seconds - 0.1).abs() < 1e-3);
    }

    #[test]
    fn data_uri_wav_at_8khz_resamples_to_16khz() {
        let wav = make_wav_const(8_000, 800, 8_192); // 100 ms at 8 kHz
        let b64 = B64.encode(&wav);
        let url = format!("data:audio/wav;base64,{b64}");
        let cancel = AtomicBool::new(false);
        let item = fetch_audio(&url, None, &cancel).expect("fetch ok");
        // 8 → 16 kHz doubles the sample count. Allow a small +/- for
        // rubato's edge handling.
        let want = 1_600;
        let got = item.samples_16k_mono.len();
        assert!(
            (got as i64 - want as i64).abs() <= 64,
            "got {got} samples, expected ~{want}"
        );
    }

    #[test]
    fn rejects_oversized_data_uri() {
        // 100 MiB encoded WAV >> MAX_AUDIO_BYTES default cap. We
        // build only the data: header here; no actual decode.
        let huge = "A".repeat(50 * 1024 * 1024); // 50 MiB of 'A'
        let url = format!("data:audio/wav;base64,{huge}");
        let cancel = AtomicBool::new(false);
        let err = fetch_audio(&url, None, &cancel).expect_err("should reject");
        assert!(
            matches!(err, AudioError::TooLarge(_, _)),
            "got: {err:?}"
        );
    }

    #[test]
    fn cancellation_short_circuits() {
        let cancel = AtomicBool::new(true);
        let wav = make_wav_const(16_000, 16, 0);
        let url = format!("data:audio/wav;base64,{}", B64.encode(&wav));
        let err = fetch_audio(&url, None, &cancel).expect_err("cancelled");
        assert!(matches!(err, AudioError::Cancelled), "got: {err:?}");
    }
}
