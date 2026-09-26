//! Optional OS process isolation. Missing backends are errors, never a fallback
//! to unrestricted execution. Authorization and approval UI remain host concerns.

use anda_core::BoxError;
use std::{
    path::{Path, PathBuf},
    process::Command,
};

/// Network access granted by the host to sandboxed commands.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SandboxNetwork {
    /// Block networking (the default).
    #[default]
    Deny,
    /// Allow networking without destination filtering.
    Allow,
}

/// Immutable host policy shared by all calls on a configured runtime.
#[derive(Debug, Clone)]
pub struct SandboxPolicy {
    readable: Vec<PathBuf>,
    writable: Vec<PathBuf>,
    network: SandboxNetwork,
}

impl SandboxPolicy {
    /// Grants writes only within this workspace, `/dev/null`, and (on Linux) a
    /// private `/tmp`; macOS needs an explicit scratch grant for `TMPDIR`. Common
    /// OS runtime files are readable; other paths require grants.
    pub fn workspace(workspace: impl AsRef<Path>) -> Result<Self, BoxError> {
        Ok(Self {
            readable: Vec::new(),
            writable: vec![root(workspace.as_ref(), true)?],
            network: SandboxNetwork::Deny,
        })
    }
    /// Grants read access to an existing host path.
    pub fn allow_read(mut self, path: impl AsRef<Path>) -> Result<Self, BoxError> {
        self.readable.push(root(path.as_ref(), false)?);
        Ok(self)
    }
    /// Grants writes to an existing host directory. The filesystem root is refused.
    pub fn allow_write(mut self, path: impl AsRef<Path>) -> Result<Self, BoxError> {
        self.writable.push(root(path.as_ref(), true)?);
        Ok(self)
    }
    /// Selects network access. The model cannot widen this policy through arguments.
    pub fn network(mut self, network: SandboxNetwork) -> Self {
        self.network = network;
        self
    }

    pub(super) fn validate_platform(&self) -> Result<(), BoxError> {
        #[cfg(target_os = "macos")]
        let backend = Path::new("/usr/bin/sandbox-exec");
        #[cfg(target_os = "linux")]
        let backend = Path::new("/usr/bin/bwrap");
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        return Err(
            "No built-in process sandbox for this platform; supply an isolated Executor".into(),
        );
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        if !backend.is_file() {
            return Err(format!(
                "Required sandbox backend is unavailable: {}",
                backend.display()
            )
            .into());
        }
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        Ok(())
    }

    pub(super) fn wrap(
        &self,
        command: Command,
        cwd: &Path,
        tty: bool,
    ) -> Result<Command, BoxError> {
        self.validate_platform()?;
        if !self
            .writable
            .iter()
            .chain(&self.readable)
            .any(|root| cwd.starts_with(root))
        {
            return Err("Shell cwd is not granted by the sandbox policy".into());
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            let _ = (command, tty, self.network);
            Err("Process sandbox unsupported".into())
        }
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        {
            #[cfg(target_os = "macos")]
            let mut wrapped = {
                let mut wrapper = Command::new("/usr/bin/sandbox-exec");
                wrapper.arg("-p").arg(self.seatbelt(tty)?).arg("--");
                wrapper
            };
            #[cfg(target_os = "linux")]
            let mut wrapped = {
                let mut wrapper = Command::new("/usr/bin/bwrap");
                wrapper.args(["--die-with-parent", "--unshare-all"]);
                // A PTY launch already runs in its own session, and that private
                // terminal must stay the controlling terminal of the command.
                if !tty {
                    wrapper.arg("--new-session");
                }
                if self.network == SandboxNetwork::Allow {
                    wrapper.arg("--share-net");
                }
                // Bind /tmp grants after the private mount so it cannot hide them.
                wrapper.args(["--tmpfs", "/tmp"]);
                for path in system_roots()
                    .into_iter()
                    .chain(self.readable.iter().cloned())
                {
                    wrapper.arg("--ro-bind").arg(&path).arg(&path);
                }
                for path in &self.writable {
                    wrapper.arg("--bind").arg(path).arg(path);
                }
                wrapper
                    .args(["--proc", "/proc", "--dev", "/dev", "--chdir"])
                    .arg(cwd)
                    .arg("--");
                wrapper
            };
            wrapped.arg(command.get_program()).args(command.get_args());
            for (key, value) in command.get_envs() {
                if let Some(value) = value {
                    wrapped.env(key, value);
                } else {
                    wrapped.env_remove(key);
                }
            }
            Ok(wrapped)
        }
    }

    #[cfg(any(target_os = "macos", test))]
    fn seatbelt(&self, tty: bool) -> Result<String, BoxError> {
        let mut profile = String::from(
            "(version 1)\n(deny default)\n(allow process-exec process-fork)\n(allow signal (target same-sandbox))\n(allow process-info* (target same-sandbox))\n(allow sysctl-read)\n(allow file-read-metadata)\n",
        );
        // dyld must map the system runtime already readable below. This does
        // not grant Mach services, additional reads, writes, or networking.
        // A handle to the root directory is needed for loader path traversal.
        // `literal` grants only that directory, never its descendants.
        profile.push_str("(allow file-read-data (literal \"/\"))\n");
        profile.push_str(
            "(allow file-map-executable (subpath \"/usr/lib\") (subpath \"/System/Library\") (subpath \"/System/Volumes/Preboot/Cryptexes/OS/System/Library/dyld\"))\n",
        );
        for path in system_roots()
            .into_iter()
            .chain(self.readable.iter().cloned())
            .chain(self.writable.iter().cloned())
        {
            profile.push_str(&format!(
                "(allow file-read* (subpath {}))\n",
                literal(&path)?
            ));
        }
        for path in &self.writable {
            profile.push_str(&format!(
                "(allow file-write* (subpath {}))\n",
                literal(path)?
            ));
        }
        profile.push_str("(allow file-read* file-write* (literal \"/dev/null\") (literal \"/dev/zero\"))\n(allow file-read* (literal \"/dev/random\") (literal \"/dev/urandom\"))\n");
        if tty {
            profile.push_str("(allow file-read* file-write* (literal \"/dev/tty\"))\n(allow file-ioctl (regex #\"^/dev/ttys[0-9]+$\"))\n");
        }
        if self.network == SandboxNetwork::Allow {
            profile.push_str("(allow network*)\n");
        }
        Ok(profile)
    }
}

fn root(path: &Path, writable: bool) -> Result<PathBuf, BoxError> {
    let path = std::fs::canonicalize(path)?;
    if writable && (!path.is_dir() || path.parent().is_none()) {
        return Err("Writable sandbox roots must be existing directories other than /".into());
    }
    if path
        .to_str()
        .is_none_or(|value| value.chars().any(char::is_control))
    {
        return Err("Sandbox roots must be UTF-8 without control characters".into());
    }
    Ok(path)
}

#[cfg(any(target_os = "macos", test))]
fn literal(path: &Path) -> Result<String, BoxError> {
    Ok(serde_json::to_string(
        path.to_str().ok_or("Non-UTF8 sandbox path")?,
    )?)
}

#[cfg(any(target_os = "macos", target_os = "linux", test))]
fn system_roots() -> Vec<PathBuf> {
    #[cfg(target_os = "macos")]
    let roots = [
        "/System",
        "/usr",
        "/bin",
        "/sbin",
        "/Library",
        "/private/etc/localtime",
        "/private/etc/hosts",
        "/private/etc/resolv.conf",
        "/private/var/db/timezone",
    ];
    #[cfg(not(target_os = "macos"))]
    let roots = [
        "/usr",
        "/bin",
        "/sbin",
        "/lib",
        "/lib64",
        "/etc/ld.so.cache",
        "/etc/ld.so.conf",
        "/etc/alternatives",
        "/etc/localtime",
        "/etc/hosts",
        "/etc/resolv.conf",
    ];
    roots
        .iter()
        .filter(|path| Path::new(path).exists())
        .map(|path| PathBuf::from(*path))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn policy_escapes_paths_and_has_no_implicit_network_grant() {
        let policy = SandboxPolicy {
            readable: vec![],
            writable: vec![PathBuf::from("/tmp/a\"b")],
            network: SandboxNetwork::Deny,
        };
        let profile = policy.seatbelt(false).unwrap();
        assert!(profile.contains("(subpath \"/tmp/a\\\"b\")"));
        assert!(!profile.contains("(allow network*)"));
        assert!(SandboxPolicy::workspace("/").is_err());
    }
}
