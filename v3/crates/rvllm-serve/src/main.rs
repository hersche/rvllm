//! `rvllm-server` scaffold entry point in the upstream `rvllm` repo.
//!
//! The functional OpenAI-compatible inference server lives in the
//! `rvllm-serve` fork (branch `rusty_sm121_inference_server`). This
//! binary is intentionally only a scaffold: it prints a clear notice
//! and exits with code 2 (LSB convention for misuse / unimplemented),
//! NOT 1, so an init system or wrapper script can distinguish a
//! "scaffold not yet wired" condition from a real runtime failure of
//! the production server.

use std::process::ExitCode;

const SCAFFOLD_MESSAGE: &str = "\
rvllm-server: this binary is the upstream `rvllm` workspace scaffold and \
does not start an HTTP server.\n\
\n\
The functional OpenAI-compatible server is in the rvllm-serve fork \
(branch `rusty_sm121_inference_server`). Build and run from there:\n\
\n\
  cargo run --release --bin rvllm-server --features cuda,gb10\n\
\n\
This scaffold exits with status 2 (\"unimplemented\") rather than 1 \
so callers can detect it.\n";

fn main() -> ExitCode {
    eprint!("{SCAFFOLD_MESSAGE}");
    ExitCode::from(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: the message mentions that this is a scaffold AND points
    /// at the functional server. Reviewers see drift loudly.
    #[test]
    fn scaffold_message_explains_redirection() {
        assert!(SCAFFOLD_MESSAGE.contains("scaffold"));
        assert!(SCAFFOLD_MESSAGE.contains("rvllm-serve"));
        assert!(SCAFFOLD_MESSAGE.contains("rusty_sm121_inference_server"));
    }
}
