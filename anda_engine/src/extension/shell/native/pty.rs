//! Unix PTY adapter; the host explicitly opts into terminal behavior.

use anda_core::BoxError;
use std::{
    fs::File,
    io::{Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::process::CommandExt,
    },
    pin::Pin,
    process::Stdio,
    sync::Arc,
    task::{Context, Poll, ready},
};
use tokio::{
    io::{AsyncRead, AsyncWrite, ReadBuf, unix::AsyncFd},
    process::{Child, Command},
};

struct Terminal(Arc<AsyncFd<File>>);

impl AsyncRead for Terminal {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        loop {
            let mut ready = ready!(self.0.poll_read_ready(cx))?;
            match ready.try_io(|fd| fd.get_ref().read(buf.initialize_unfilled())) {
                Ok(Ok(count)) => {
                    buf.advance(count);
                    return Poll::Ready(Ok(()));
                }
                Ok(Err(err)) if err.raw_os_error() == Some(libc::EIO) => {
                    return Poll::Ready(Ok(()));
                }
                Ok(Err(err)) => return Poll::Ready(Err(err)),
                Err(_) => continue,
            }
        }
    }
}

impl AsyncWrite for Terminal {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bytes: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        loop {
            let mut ready = ready!(self.0.poll_write_ready(cx))?;
            match ready.try_io(|fd| fd.get_ref().write(bytes)) {
                Ok(result) => return Poll::Ready(result),
                Err(_) => continue,
            }
        }
    }
    fn poll_flush(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }
    fn poll_shutdown(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

type Reader = Box<dyn AsyncRead + Send + Unpin>;
type Input = Box<dyn AsyncWrite + Send + Unpin>;

pub(super) fn spawn(
    mut command: std::process::Command,
) -> Result<(Child, Reader, Reader, Option<Input>), BoxError> {
    let mut master = -1;
    let mut slave = -1;
    let mut size = libc::winsize {
        ws_row: 24,
        ws_col: 120,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    // SAFETY: descriptors are written into valid storage and all optional pointers are null.
    if unsafe {
        libc::openpty(
            &mut master,
            &mut slave,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut size,
        )
    } != 0
    {
        return Err(std::io::Error::last_os_error().into());
    }
    // SAFETY: successful openpty transfers ownership of both descriptors.
    let master = unsafe { File::from_raw_fd(master) };
    let slave = unsafe { File::from_raw_fd(slave) };
    for file in [&master, &slave] {
        // SAFETY: each descriptor is live. No descriptor is inherited by unrelated children.
        if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) } < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    // SAFETY: only the master is made nonblocking; child terminal semantics stay unchanged.
    if unsafe { libc::fcntl(master.as_raw_fd(), libc::F_SETFL, libc::O_NONBLOCK) } < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    command
        .stdout(Stdio::from(slave.try_clone()?))
        .stderr(Stdio::from(slave.try_clone()?))
        .stdin(Stdio::from(slave));
    // SAFETY: only async-signal-safe libc operations run between fork and exec.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() < 0 || libc::ioctl(libc::STDIN_FILENO, libc::TIOCSCTTY as _, 0) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut command = Command::from(command);
    command.kill_on_drop(true);
    let child = command.spawn()?;
    drop(command);
    let fd = Arc::new(AsyncFd::new(master)?);
    Ok((
        child,
        Box::new(Terminal(fd.clone())),
        Box::new(tokio::io::empty()),
        Some(Box::new(Terminal(fd))),
    ))
}
