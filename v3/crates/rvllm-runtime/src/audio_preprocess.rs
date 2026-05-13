//! Pure-Rust mel-spectrogram extractor for the Gemma 4 audio
//! frontend (B1 of the E4B native-audio plan).
//!
//! ## Layout
//!
//! Gemma 4 E4B's `processor_config.json` ships an explicit
//! `Gemma4AudioFeatureExtractor` config with these knobs (read at
//! runtime — none are hardcoded):
//!
//! ```text
//! sampling_rate       16000  Hz
//! fft_length          512    bins (n_fft)
//! frame_length        320    samples Hann window (zero-padded → 512)
//! hop_length          160    samples (10 ms)
//! feature_size        128    mel filters
//! min_frequency         0.0  Hz
//! max_frequency      8000.0  Hz
//! mel_floor             0.001    clamp before log
//! preemphasis           0.0      no pre-emphasis filter
//! input_scale_factor    1.0
//! ```
//!
//! Pipeline (`compute_mel`):
//!   `[samples_f32]`
//!     → frame into 320-sample windows, hop=160, zero-pad to 512
//!     → apply Hann window (320 active, 192 zero tail)
//!     → real FFT (size 512, output 257 complex bins)
//!     → power spectrum
//!     → mel-filter-bank multiply (slaney-style triangular, librosa
//!       compatible)
//!     → max(mel_floor, .) + log
//!     → return `[T, n_mels]` f32 row-major
//!
//! Scope of this commit (B1): host CPU only. Promotion to a GPU
//! kernel is deferred until end-to-end profiling shows the encoder
//! forward is no longer the dominant cost per audio request.

use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub struct MelConfig {
    pub sampling_rate: u32,
    pub n_fft: usize,
    pub frame_length: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub min_frequency: f32,
    pub max_frequency: f32,
    pub mel_floor: f32,
    pub preemphasis: f32,
    pub input_scale_factor: f32,
}

impl MelConfig {
    /// Gemma 4 E4B defaults — verified against
    /// `/home/r00t/gemma4-e4b/processor_config.json` 2026-05-13.
    /// Used as the test fixture and as a fallback when the file is
    /// unreadable. Production callers should load
    /// `from_processor_config()`.
    pub const fn gemma4_e4b() -> Self {
        Self {
            sampling_rate: 16_000,
            n_fft: 512,
            frame_length: 320,
            hop_length: 160,
            n_mels: 128,
            min_frequency: 0.0,
            max_frequency: 8_000.0,
            mel_floor: 0.001,
            preemphasis: 0.0,
            input_scale_factor: 1.0,
        }
    }

    /// Parse from a Gemma 4 `processor_config.json`. Returns `None`
    /// on any structural failure so the caller can fall back to a
    /// known-good default. Uses `serde_json::Value` to avoid adding
    /// `serde` as a direct dep of `rvllm-runtime`.
    pub fn from_processor_config(json_bytes: &[u8]) -> Option<Self> {
        let v: serde_json::Value = serde_json::from_slice(json_bytes).ok()?;
        let fe = v.get("feature_extractor")?;
        let u = |k: &str| fe.get(k)?.as_u64();
        let f = |k: &str| fe.get(k)?.as_f64();
        Some(Self {
            sampling_rate: u("sampling_rate")? as u32,
            n_fft: u("fft_length")? as usize,
            frame_length: u("frame_length")? as usize,
            hop_length: u("hop_length")? as usize,
            n_mels: u("feature_size")? as usize,
            min_frequency: f("min_frequency")? as f32,
            max_frequency: f("max_frequency")? as f32,
            mel_floor: f("mel_floor")? as f32,
            preemphasis: f("preemphasis")? as f32,
            input_scale_factor: fe
                .get("input_scale_factor")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0) as f32,
        })
    }
}

/// Precomputed window + filter-bank + FFT planner. Cheap to
/// construct; re-use across requests when the model lives.
pub struct MelExtractor {
    cfg: MelConfig,
    window: Vec<f32>,      // length frame_length, Hann
    filter_bank: Vec<f32>, // [n_mels, n_fft/2 + 1] row-major
    fft: Arc<dyn rustfft::Fft<f32>>,
}

impl MelExtractor {
    pub fn new(cfg: MelConfig) -> Self {
        let window = hann_window(cfg.frame_length);
        let n_bins = cfg.n_fft / 2 + 1;
        let filter_bank = mel_filter_bank(
            cfg.n_mels,
            n_bins,
            cfg.sampling_rate as f32,
            cfg.min_frequency,
            cfg.max_frequency,
        );
        let mut planner = rustfft::FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(cfg.n_fft);
        Self { cfg, window, filter_bank, fft }
    }

    pub fn n_mels(&self) -> usize { self.cfg.n_mels }

    /// Number of frames produced for `n_samples` of f32 PCM with
    /// the configured hop.
    pub fn num_frames(&self, n_samples: usize) -> usize {
        if n_samples < self.cfg.frame_length {
            0
        } else {
            (n_samples - self.cfg.frame_length) / self.cfg.hop_length + 1
        }
    }

    /// Compute `[T, n_mels]` log-mel spectrogram as f32 row-major.
    /// `samples` is mono f32 PCM at `cfg.sampling_rate` Hz.
    pub fn compute_mel(&self, samples: &[f32]) -> Vec<f32> {
        let cfg = &self.cfg;
        let n_bins = cfg.n_fft / 2 + 1;
        let t = self.num_frames(samples.len());
        let mut out = vec![0.0f32; t * cfg.n_mels];
        if t == 0 {
            return out;
        }
        let mut scratch = vec![rustfft::num_complex::Complex::<f32>::new(0.0, 0.0); cfg.n_fft];
        let mut power = vec![0.0f32; n_bins];

        for frame in 0..t {
            let start = frame * cfg.hop_length;
            // Window + zero-pad up to n_fft.
            for i in 0..cfg.n_fft {
                let re = if i < cfg.frame_length {
                    samples[start + i] * self.window[i] * cfg.input_scale_factor
                } else {
                    0.0
                };
                scratch[i].re = re;
                scratch[i].im = 0.0;
            }
            self.fft.process(&mut scratch);
            for b in 0..n_bins {
                let c = scratch[b];
                power[b] = c.re * c.re + c.im * c.im;
            }
            // Mel filter bank: out[frame, m] = log(max(floor, sum_b fb[m,b] * power[b]))
            let frame_off = frame * cfg.n_mels;
            for m in 0..cfg.n_mels {
                let fb_row = &self.filter_bank[m * n_bins..(m + 1) * n_bins];
                let mut acc = 0.0f32;
                for b in 0..n_bins {
                    acc += fb_row[b] * power[b];
                }
                let clamped = acc.max(cfg.mel_floor);
                out[frame_off + m] = clamped.ln();
            }
        }
        out
    }
}

fn hann_window(n: usize) -> Vec<f32> {
    // Periodic Hann window (HF feature extractors use the periodic
    // variant by default — divisor `n`, not `n-1`).
    let mut w = Vec::with_capacity(n);
    let denom = n as f32;
    for i in 0..n {
        let phase = 2.0 * std::f32::consts::PI * (i as f32) / denom;
        w.push(0.5 - 0.5 * phase.cos());
    }
    w
}

fn hz_to_mel(hz: f32) -> f32 {
    // HTK formula (`preemphasis_htk_flavor=True` in processor_config).
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

/// Slaney-style triangular mel filter bank, librosa-compatible.
/// Returns `[n_mels, n_bins]` row-major f32.
fn mel_filter_bank(
    n_mels: usize,
    n_bins: usize,
    sr: f32,
    f_min: f32,
    f_max: f32,
) -> Vec<f32> {
    let mel_lo = hz_to_mel(f_min);
    let mel_hi = hz_to_mel(f_max);
    // n_mels + 2 edges = n_mels triangles + 2 outer anchors.
    let mut mel_pts = Vec::with_capacity(n_mels + 2);
    for i in 0..(n_mels + 2) {
        let t = i as f32 / (n_mels + 1) as f32;
        mel_pts.push(mel_lo + t * (mel_hi - mel_lo));
    }
    // Map mel anchors → linear Hz → fractional FFT-bin index.
    let n_fft = (n_bins - 1) * 2;
    let mut bin_pts = Vec::with_capacity(mel_pts.len());
    for &m in &mel_pts {
        let hz = mel_to_hz(m);
        bin_pts.push(hz * n_fft as f32 / sr);
    }
    let mut fb = vec![0.0f32; n_mels * n_bins];
    for m in 0..n_mels {
        let lo = bin_pts[m];
        let ctr = bin_pts[m + 1];
        let hi = bin_pts[m + 2];
        let inv_l = if ctr > lo { 1.0 / (ctr - lo) } else { 0.0 };
        let inv_r = if hi > ctr { 1.0 / (hi - ctr) } else { 0.0 };
        // Slaney normalisation: triangle peak height = 2/(hi-lo) so
        // equal-Hz-width filters integrate to the same area.
        let slaney = 2.0 / (hi - lo).max(1e-12);
        for b in 0..n_bins {
            let bf = b as f32;
            let l = (bf - lo) * inv_l;
            let r = (hi - bf) * inv_r;
            let tri = l.min(r).max(0.0);
            fb[m * n_bins + b] = tri * slaney;
        }
    }
    fb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_from_real_processor_config() {
        // The actual file on this machine; if it's missing, the
        // checkout is mis-provisioned and other E4B tests would
        // fail too — skip cleanly.
        let path = "/home/r00t/gemma4-e4b/processor_config.json";
        let Ok(bytes) = std::fs::read(path) else {
            eprintln!("skipping: {path} not present");
            return;
        };
        let cfg = MelConfig::from_processor_config(&bytes)
            .expect("parse E4B processor_config");
        assert_eq!(cfg.sampling_rate, 16_000);
        assert_eq!(cfg.n_fft, 512);
        assert_eq!(cfg.frame_length, 320);
        assert_eq!(cfg.hop_length, 160);
        assert_eq!(cfg.n_mels, 128);
        assert_eq!(cfg.min_frequency, 0.0);
        assert_eq!(cfg.max_frequency, 8_000.0);
        assert!((cfg.mel_floor - 0.001).abs() < 1e-9);
    }

    #[test]
    fn one_khz_sine_peaks_in_expected_mel_bin() {
        let cfg = MelConfig::gemma4_e4b();
        let extr = MelExtractor::new(cfg);
        // 1 second of 1 kHz at 16 kHz; small amplitude to stay well
        // below clipping.
        let sr = cfg.sampling_rate as f32;
        let n_samples = sr as usize;
        let f = 1000.0f32;
        let samples: Vec<f32> = (0..n_samples)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * f * (i as f32) / sr).sin())
            .collect();
        let mel = extr.compute_mel(&samples);
        let t = extr.num_frames(samples.len());
        assert!(t > 50, "expected many frames, got {t}");
        // Take a middle frame to skip leading-edge effects.
        let frame = &mel[t / 2 * cfg.n_mels..(t / 2 + 1) * cfg.n_mels];
        let (peak_bin, _peak_val) = frame
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        // Expected mel bin: hz_to_mel(1000) / (hz_to_mel(8000) /
        // (n_mels + 1)) — but use the actual filter bank to compute
        // the true argmax bin (filter-bank quantisation can shift
        // the peak by ±1 relative to the analytical center).
        let fb = mel_filter_bank(cfg.n_mels, cfg.n_fft / 2 + 1, sr, 0.0, 8000.0);
        // Sample bin closest to 1 kHz:
        let bin_1k = (1000.0 * cfg.n_fft as f32 / sr).round() as usize;
        let (expected_bin, _) = (0..cfg.n_mels)
            .map(|m| (m, fb[m * (cfg.n_fft / 2 + 1) + bin_1k]))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        // Allow ±1 bin slack to absorb the triangular filter overlap
        // — a 1 kHz tone often lands on the boundary between two
        // adjacent mel filters, and either one peaking is valid.
        assert!(
            (peak_bin as i32 - expected_bin as i32).abs() <= 1,
            "peak at {peak_bin}, expected near {expected_bin}"
        );
    }

    #[test]
    fn num_frames_matches_hop() {
        let extr = MelExtractor::new(MelConfig::gemma4_e4b());
        // 1 s of audio at 16 kHz, hop=160 → 16000-320)/160 + 1 = 99
        assert_eq!(extr.num_frames(16_000), 99);
        // Below frame_length → no frames.
        assert_eq!(extr.num_frames(100), 0);
        assert_eq!(extr.num_frames(320), 1);
    }

    #[test]
    fn hann_window_is_zero_at_endpoints() {
        let w = hann_window(320);
        assert!(w[0].abs() < 1e-6);
        // Periodic Hann is NOT zero at the last index; just check
        // it's non-negative and < 1.
        let last = w[319];
        assert!(last >= 0.0 && last < 1.0);
    }
}
