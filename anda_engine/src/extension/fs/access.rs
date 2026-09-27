//! Handle-based regular-file access and bounded, atomic replacement.

use super::{MAX_FILE_SIZE_BYTES, ensure_file_size_within_limit, ensure_regular_file};
use anda_core::BoxError;
use std::{
    fs::{File, Metadata, Permissions},
    io::Write,
    path::Path,
};

pub(super) async fn open_read(path: &Path) -> Result<(tokio::fs::File, Metadata), BoxError> {
    let path = path.to_owned();
    tokio::task::spawn_blocking(move || {
        let file = platform::open_read(&path)?;
        let metadata = file.metadata()?;
        ensure_regular_file(
            &metadata,
            &path,
            "Access to multiply-linked files is not allowed",
        )?;
        ensure_file_size_within_limit(&metadata, &path, MAX_FILE_SIZE_BYTES)?;
        Ok((tokio::fs::File::from_std(file), metadata))
    })
    .await?
}

#[derive(Clone, Copy)]
enum WriteMode {
    Replace,
    Create,
}

pub(super) async fn replace(
    path: &Path,
    bytes: &[u8],
    permissions: Option<&Permissions>,
) -> Result<(), BoxError> {
    write(path, bytes, permissions, WriteMode::Replace).await
}

pub(super) async fn create(
    path: &Path,
    bytes: &[u8],
    permissions: Option<&Permissions>,
) -> Result<(), BoxError> {
    write(path, bytes, permissions, WriteMode::Create).await
}

async fn write(
    path: &Path,
    bytes: &[u8],
    permissions: Option<&Permissions>,
    mode: WriteMode,
) -> Result<(), BoxError> {
    if bytes.len() as u64 > MAX_FILE_SIZE_BYTES {
        return Err("Write result exceeds the maximum file size of 10 MiB".into());
    }
    let path = path.to_owned();
    let bytes = bytes.to_vec();
    let permissions = permissions.cloned();
    tokio::task::spawn_blocking(move || platform::replace(&path, &bytes, permissions, mode)).await?
}

pub(super) async fn remove(path: &Path) -> Result<(), BoxError> {
    let path = path.to_owned();
    tokio::task::spawn_blocking(move || platform::remove(&path)).await?
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use tokio::io::AsyncReadExt;

    #[tokio::test]
    async fn concurrent_creation_never_overwrites_the_winner() {
        let root =
            std::env::temp_dir().join(format!("anda-create-{:032x}", rand::random::<u128>()));
        std::fs::create_dir(&root).unwrap();
        let root = root.canonicalize().unwrap();
        let path = root.join("new");
        let (a, b) = tokio::join!(
            create(&path, b"first", None),
            create(&path, b"second", None)
        );
        assert_ne!(a.is_ok(), b.is_ok());
        assert_eq!(
            std::fs::read(&path).unwrap(),
            if a.is_ok() {
                b"first".as_slice()
            } else {
                b"second".as_slice()
            }
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn opened_read_survives_path_replacement_without_following_new_link() {
        let root =
            std::env::temp_dir().join(format!("anda-handle-{:032x}", rand::random::<u128>()));
        std::fs::create_dir(&root).unwrap();
        let root = root.canonicalize().unwrap();
        let path = root.join("file");
        std::fs::write(&path, "original").unwrap();
        std::fs::write(root.join("replacement"), "different").unwrap();
        let (mut file, _) = open_read(&path).await.unwrap();
        std::fs::rename(&path, root.join("moved")).unwrap();
        std::os::unix::fs::symlink(root.join("replacement"), &path).unwrap();
        let mut content = String::new();
        file.read_to_string(&mut content).await.unwrap();
        assert_eq!(content, "original");
        assert!(open_read(&path).await.is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn search_only_ancestors_stay_accessible() {
        use std::os::unix::fs::PermissionsExt;
        let root =
            std::env::temp_dir().join(format!("anda-search-{:032x}", rand::random::<u128>()));
        let locked = root.join("locked");
        std::fs::create_dir_all(locked.join("nested")).unwrap();
        let root = root.canonicalize().unwrap();
        let locked = root.join("locked");
        std::fs::write(locked.join("nested/file"), "content").unwrap();
        // Writable and searchable, but not listable.
        std::fs::set_permissions(&locked, Permissions::from_mode(0o300)).unwrap();
        let result = async {
            let (mut file, _) = open_read(&locked.join("nested/file")).await?;
            let mut content = String::new();
            file.read_to_string(&mut content).await?;
            replace(&locked.join("created"), b"new", None).await?;
            create(&locked.join("new/deep"), b"deep", None).await?;
            remove(&locked.join("created")).await?;
            Ok::<_, BoxError>(content)
        }
        .await;
        std::fs::set_permissions(&locked, Permissions::from_mode(0o755)).unwrap();
        assert_eq!(result.unwrap(), "content");
        assert_eq!(std::fs::read(locked.join("new/deep")).unwrap(), b"deep");
        assert!(!locked.join("created").exists());
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(unix)]
mod platform {
    use super::*;
    use std::{
        ffi::{CString, OsStr},
        os::{
            fd::{AsRawFd, FromRawFd},
            unix::ffi::OsStrExt,
        },
        path::Component,
    };

    // Ancestors need only search permission, as with path-based access: a workspace below a
    // traversable but unlistable directory (Android's /data, 0711 homes) must stay reachable.
    // O_PATH and O_SEARCH descriptors still anchor the *at() calls below.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    const DIRECTORY: i32 = libc::O_PATH | libc::O_DIRECTORY;
    #[cfg(target_vendor = "apple")]
    const DIRECTORY: i32 = libc::O_SEARCH;
    #[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
    const DIRECTORY: i32 = libc::O_RDONLY | libc::O_DIRECTORY;

    fn name(value: &OsStr) -> std::io::Result<CString> {
        CString::new(value.as_bytes())
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "NUL in path"))
    }

    fn open_root() -> std::io::Result<File> {
        // SAFETY: the literal is NUL-terminated; a successful descriptor transfers to File.
        let fd = unsafe { libc::open(c"/".as_ptr(), DIRECTORY | libc::O_CLOEXEC) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(unsafe { File::from_raw_fd(fd) })
    }

    fn open_at(parent: &File, name: &CString, flags: i32) -> std::io::Result<File> {
        // SAFETY: parent and name remain live; successful descriptors transfer to File.
        let fd = unsafe {
            libc::openat(
                parent.as_raw_fd(),
                name.as_ptr(),
                flags | libc::O_CLOEXEC | libc::O_NOFOLLOW,
                0o666,
            )
        };
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(unsafe { File::from_raw_fd(fd) })
    }

    fn parent(path: &Path, create: bool) -> Result<(File, CString), BoxError> {
        if !path.is_absolute() {
            return Err("Expected an absolute resolved path".into());
        }
        let mut parts = Vec::new();
        for part in path.components() {
            match part {
                Component::RootDir => (),
                Component::Normal(value) => parts.push(name(value)?),
                _ => return Err("Expected a normalized resolved path".into()),
            }
        }
        let leaf = parts.pop().ok_or("Path must name a file")?;
        let mut directory = open_root()?;
        for component in parts {
            directory = match open_at(&directory, &component, DIRECTORY) {
                Ok(file) => file,
                Err(err) if create && err.kind() == std::io::ErrorKind::NotFound => {
                    // SAFETY: the parent descriptor and terminated component are valid.
                    let result =
                        unsafe { libc::mkdirat(directory.as_raw_fd(), component.as_ptr(), 0o777) };
                    if result < 0 {
                        let err = std::io::Error::last_os_error();
                        if err.kind() != std::io::ErrorKind::AlreadyExists {
                            return Err(err.into());
                        }
                    }
                    open_at(&directory, &component, DIRECTORY)?
                }
                Err(err) => return Err(err.into()),
            };
        }
        Ok((directory, leaf))
    }

    fn check(file: &File, path: &Path) -> Result<(), BoxError> {
        ensure_regular_file(
            &file.metadata()?,
            path,
            "Access to multiply-linked files is not allowed",
        )
    }

    pub(super) fn open_read(path: &Path) -> Result<File, BoxError> {
        let (parent, leaf) = parent(path, false)?;
        let file = open_at(&parent, &leaf, libc::O_RDONLY | libc::O_NONBLOCK)?;
        check(&file, path)?;
        Ok(file)
    }

    struct Temporary<'a> {
        parent: &'a File,
        name: CString,
    }
    impl Drop for Temporary<'_> {
        fn drop(&mut self) {
            // SAFETY: both descriptor and name outlive this cleanup.
            unsafe {
                libc::unlinkat(self.parent.as_raw_fd(), self.name.as_ptr(), 0);
            }
        }
    }

    pub(super) fn remove(path: &Path) -> Result<(), BoxError> {
        let (parent, leaf) = parent(path, false)?;
        let file = open_at(&parent, &leaf, libc::O_RDONLY | libc::O_NONBLOCK)?;
        check(&file, path)?;
        // SAFETY: remove only this leaf under the pinned parent; never recurse.
        if unsafe { libc::unlinkat(parent.as_raw_fd(), leaf.as_ptr(), 0) } < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(())
    }

    pub(super) fn replace(
        path: &Path,
        bytes: &[u8],
        permissions: Option<Permissions>,
        mode: WriteMode,
    ) -> Result<(), BoxError> {
        let (parent, leaf) = parent(path, true)?;
        let validate = || -> Result<(), BoxError> {
            match open_at(&parent, &leaf, libc::O_RDONLY | libc::O_NONBLOCK) {
                Ok(file) => check(&file, path),
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(err) => Err(err.into()),
            }
        };
        validate()?;
        let temporary = Temporary {
            parent: &parent,
            name: CString::new(format!(".anda-tmp-{:032x}", rand::random::<u128>()))?,
        };
        let mut file = open_at(
            &parent,
            &temporary.name,
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL,
        )?;
        if let Some(permissions) = permissions {
            file.set_permissions(permissions)?;
        }
        file.write_all(bytes)?;
        file.sync_all()?;
        validate()?;
        // SAFETY: both names are relative to the same held directory descriptor.
        // linkat publishes a complete new file and atomically refuses any existing
        // destination. Temporary cleanup removes the second link immediately.
        let result = unsafe {
            match mode {
                WriteMode::Replace => libc::renameat(
                    parent.as_raw_fd(),
                    temporary.name.as_ptr(),
                    parent.as_raw_fd(),
                    leaf.as_ptr(),
                ),
                WriteMode::Create => libc::linkat(
                    parent.as_raw_fd(),
                    temporary.name.as_ptr(),
                    parent.as_raw_fd(),
                    leaf.as_ptr(),
                    0,
                ),
            }
        };
        if result < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(())
    }
}

#[cfg(not(unix))]
mod platform {
    use super::*;

    // Windows directory handles without FILE_SHARE_DELETE pin the ancestry until
    // the file is opened or the atomic replacement has committed.
    #[cfg(windows)]
    fn pin_parents(path: &Path, create: bool) -> Result<Vec<File>, BoxError> {
        use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
        use windows_sys::Win32::Storage::FileSystem::{
            FILE_ATTRIBUTE_REPARSE_POINT, FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT,
            FILE_SHARE_READ, FILE_SHARE_WRITE,
        };
        let mut handles = Vec::new();
        let mut current = std::path::PathBuf::new();
        for component in path
            .parent()
            .ok_or("Missing parent directory")?
            .components()
        {
            current.push(component);
            if matches!(
                component,
                std::path::Component::Prefix(_) | std::path::Component::RootDir
            ) {
                continue;
            }
            if create && !current.exists() {
                match std::fs::create_dir(&current) {
                    Ok(()) => (),
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => (),
                    Err(error) => return Err(error.into()),
                }
            }
            let handle = std::fs::OpenOptions::new()
                .read(true)
                .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
                .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS)
                .open(&current)?;
            let metadata = handle.metadata()?;
            if !metadata.is_dir() || metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
            {
                return Err("Unsafe directory component".into());
            }
            handles.push(handle);
        }
        Ok(handles)
    }

    #[cfg(not(windows))]
    fn pin_parents(_path: &Path, _create: bool) -> Result<Vec<File>, BoxError> {
        Ok(Vec::new())
    }

    pub(super) fn remove(path: &Path) -> Result<(), BoxError> {
        let _parents = pin_parents(path, false)?;
        open_read(path)?;
        std::fs::remove_file(path)?;
        Ok(())
    }

    pub(super) fn open_read(path: &Path) -> Result<File, BoxError> {
        let _parents = pin_parents(path, false)?;
        let mut options = std::fs::OpenOptions::new();
        options.read(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            options.custom_flags(
                windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT,
            );
        }
        let file = options.open(path)?;
        validate(&file, path)?;
        Ok(file)
    }

    fn validate(file: &File, path: &Path) -> Result<(), BoxError> {
        let metadata = file.metadata()?;
        ensure_regular_file(
            &metadata,
            path,
            "Access to multiply-linked files is not allowed",
        )?;
        #[cfg(windows)]
        {
            use std::os::windows::{fs::MetadataExt, io::AsRawHandle};
            use windows_sys::Win32::Storage::FileSystem::{
                BY_HANDLE_FILE_INFORMATION, FILE_ATTRIBUTE_REPARSE_POINT,
                GetFileInformationByHandle,
            };
            if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
                return Err("Reparse-point files are not allowed".into());
            }
            let mut info = std::mem::MaybeUninit::<BY_HANDLE_FILE_INFORMATION>::zeroed();
            // SAFETY: the handle is live and info points to writable storage.
            if unsafe { GetFileInformationByHandle(file.as_raw_handle() as _, info.as_mut_ptr()) }
                == 0
            {
                return Err(std::io::Error::last_os_error().into());
            }
            if unsafe { info.assume_init() }.nNumberOfLinks > 1 {
                return Err("Access to multiply-linked files is not allowed".into());
            }
        }
        Ok(())
    }

    pub(super) fn replace(
        path: &Path,
        bytes: &[u8],
        permissions: Option<Permissions>,
        mode: WriteMode,
    ) -> Result<(), BoxError> {
        let _parents = pin_parents(path, true)?;
        if path.exists() {
            open_read(path)?;
        }
        let parent = path.parent().ok_or("Missing parent directory")?;
        std::fs::create_dir_all(parent)?;
        let temporary = parent.join(format!(".anda-tmp-{:032x}", rand::random::<u128>()));
        let result = (|| -> Result<(), BoxError> {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            file.write_all(bytes)?;
            if let Some(permissions) = permissions {
                file.set_permissions(permissions)?;
            }
            file.sync_all()?;
            drop(file);
            if path.exists() {
                open_read(path)?;
            }
            match mode {
                WriteMode::Replace => std::fs::rename(&temporary, path)?,
                WriteMode::Create => {
                    #[cfg(windows)]
                    {
                        use std::os::windows::ffi::OsStrExt;
                        let from: Vec<u16> =
                            temporary.as_os_str().encode_wide().chain(Some(0)).collect();
                        let to: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
                        // SAFETY: terminated strings live through the call. Flags 0 forbid replacement.
                        if unsafe {
                            windows_sys::Win32::Storage::FileSystem::MoveFileExW(
                                from.as_ptr(),
                                to.as_ptr(),
                                0,
                            )
                        } == 0
                        {
                            return Err(std::io::Error::last_os_error().into());
                        }
                    }
                    #[cfg(not(windows))]
                    std::fs::hard_link(&temporary, path)?;
                }
            }
            Ok(())
        })();
        let _ = std::fs::remove_file(&temporary);
        result
    }
}
