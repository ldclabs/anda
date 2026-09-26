//! Bound the SDK's byte input without implementing JSON-RPC framing or decoding.

use std::{
    io,
    pin::Pin,
    process::Stdio,
    task::{Context, Poll},
};
use tokio::{
    io::{AsyncRead, ReadBuf},
    process::{Child, ChildStdin, ChildStdout, Command},
};

pub(super) struct LineLimitedReader<R> {
    inner: R,
    length: usize,
    limit: usize,
    failed: bool,
}
impl<R> LineLimitedReader<R> {
    pub fn new(inner: R, limit: usize) -> Self {
        Self {
            inner,
            length: 0,
            limit,
            failed: false,
        }
    }
}
impl<R: AsyncRead + Unpin> AsyncRead for LineLimitedReader<R> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        output: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        if self.failed {
            return Poll::Ready(Err(io::Error::other("MCP stdio message limit exceeded")));
        }
        if output.remaining() == 0 {
            return Poll::Ready(Ok(()));
        }
        let mut bytes = [0u8; 8192];
        let n = output.remaining().min(bytes.len());
        let mut input = ReadBuf::new(&mut bytes[..n]);
        match Pin::new(&mut self.inner).poll_read(cx, &mut input) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(Err(err)) => return Poll::Ready(Err(err)),
            Poll::Ready(Ok(())) => {}
        }
        for byte in input.filled() {
            if *byte == b'\n' {
                self.length = 0;
            } else {
                self.length += 1;
                if self.length > self.limit {
                    self.failed = true;
                    return Poll::Ready(Err(io::Error::other("MCP stdio message limit exceeded")));
                }
            }
        }
        output.put_slice(input.filled());
        Poll::Ready(Ok(()))
    }
}

pub(super) struct StdioProcess {
    child: Child,
}
impl Drop for StdioProcess {
    fn drop(&mut self) {
        // The child remains owned until this guard drops, including failed/cancelled handshakes.
        #[cfg(unix)]
        if let Some(pid) = self.child.id() {
            // SAFETY: only the process group created for this owned child is targeted.
            unsafe {
                libc::kill(-(pid as i32), libc::SIGKILL);
            }
        }
        let _ = self.child.start_kill();
    }
}

type StdioIo = (LineLimitedReader<ChildStdout>, ChildStdin);
pub(super) fn spawn(mut command: Command, limit: usize) -> io::Result<(StdioProcess, StdioIo)> {
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true);
    #[cfg(unix)]
    command.process_group(0);
    let mut process = StdioProcess {
        child: command.spawn()?,
    };
    let stdout = process
        .child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("MCP stdout missing"))?;
    let stdin = process
        .child
        .stdin
        .take()
        .ok_or_else(|| io::Error::other("MCP stdin missing"))?;
    Ok((process, (LineLimitedReader::new(stdout, limit), stdin)))
}
