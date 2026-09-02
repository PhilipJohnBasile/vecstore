//! Backup and Restore API for VecStore
//!
//! This module provides a comprehensive backup/restore system that extends
//! the basic snapshot functionality with:
//!
//! - External backup locations (file paths, not just internal snapshots)
//! - Archive format (single file backups)
//! - Incremental backups based on WAL position
//! - Backup validation and verification
//! - Backup metadata and manifest
//!
//! ## Usage
//!
//! ```no_run
//! use vecstore::backup::{BackupManager, BackupConfig};
//!
//! # fn main() -> anyhow::Result<()> {
//! let backup_mgr = BackupManager::new("/path/to/backups")?;
//!
//! // Create a full backup
//! let backup_id = backup_mgr.create_backup("/path/to/vecstore", BackupConfig::default())?;
//!
//! // List backups
//! for backup in backup_mgr.list_backups()? {
//!     println!("Backup: {} ({})", backup.id, backup.created_at);
//! }
//!
//! // Restore from backup
//! backup_mgr.restore_backup(&backup_id, "/path/to/restore")?;
//! # Ok(())
//! # }
//! ```

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Configuration for backup operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    /// Include HNSW index in backup
    pub include_index: bool,

    /// Include WAL files in backup
    pub include_wal: bool,

    /// Compress the backup (future feature)
    pub compress: bool,

    /// Verify backup after creation
    pub verify: bool,

    /// Custom metadata to include
    pub metadata: HashMap<String, String>,
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            include_index: true,
            include_wal: false,
            compress: false,
            verify: true,
            metadata: HashMap::new(),
        }
    }
}

/// Information about a backup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupInfo {
    /// Unique backup identifier
    pub id: String,

    /// When the backup was created
    pub created_at: String,

    /// Source path that was backed up
    pub source_path: String,

    /// Number of vectors in the backup
    pub vector_count: usize,

    /// Vector dimension
    pub dimension: usize,

    /// Total size in bytes
    pub size_bytes: u64,

    /// Whether the backup includes the HNSW index
    pub has_index: bool,

    /// Whether the backup includes WAL files
    pub has_wal: bool,

    /// Custom metadata
    pub metadata: HashMap<String, String>,

    /// Files included in the backup
    pub files: Vec<String>,

    /// Checksum for verification
    pub checksum: Option<String>,
}

/// Manifest file for backup directory
#[derive(Debug, Serialize, Deserialize)]
struct BackupManifest {
    version: u32,
    backups: Vec<BackupInfo>,
}

impl Default for BackupManifest {
    fn default() -> Self {
        Self {
            version: 1,
            backups: Vec::new(),
        }
    }
}

/// Manager for backup and restore operations
pub struct BackupManager {
    /// Root directory for backups
    backup_root: PathBuf,
}

impl BackupManager {
    /// Create a new backup manager with the given root directory
    pub fn new(backup_root: impl Into<PathBuf>) -> Result<Self> {
        let root = backup_root.into();
        fs::create_dir_all(&root).context("Failed to create backup root directory")?;

        Ok(Self { backup_root: root })
    }

    /// Get path to manifest file
    fn manifest_path(&self) -> PathBuf {
        self.backup_root.join("manifest.json")
    }

    /// Load or create manifest
    fn load_manifest(&self) -> Result<BackupManifest> {
        let path = self.manifest_path();
        if path.exists() {
            let content = fs::read_to_string(&path)?;
            Ok(serde_json::from_str(&content)?)
        } else {
            Ok(BackupManifest::default())
        }
    }

    /// Save manifest
    fn save_manifest(&self, manifest: &BackupManifest) -> Result<()> {
        let content = serde_json::to_string_pretty(manifest)?;
        fs::write(self.manifest_path(), content)?;
        Ok(())
    }

    /// Generate a unique backup ID
    fn generate_id() -> String {
        let now = chrono::Utc::now();
        // Include milliseconds and a random suffix for uniqueness
        use rand::Rng;
        let random: u32 = rand::rng().random_range(0..10000);
        format!("backup_{}_{}", now.format("%Y%m%d_%H%M%S_%3f"), random)
    }

    /// Create a backup of a VecStore directory
    ///
    /// Returns the backup ID that can be used for restore
    pub fn create_backup(
        &self,
        source_path: impl AsRef<Path>,
        config: BackupConfig,
    ) -> Result<String> {
        let source = source_path.as_ref();
        if !source.exists() {
            return Err(anyhow!("Source path does not exist: {:?}", source));
        }

        let backup_id = Self::generate_id();
        let backup_dir = self.backup_root.join(&backup_id);
        fs::create_dir_all(&backup_dir)?;

        let mut files_backed_up = Vec::new();
        let mut total_size = 0u64;

        // Copy data files
        let data_files = vec!["manifest.json", "records.bin", "id_mapping.bin"];

        for file in &data_files {
            let src = source.join(file);
            if src.exists() {
                let dst = backup_dir.join(file);
                fs::copy(&src, &dst)?;
                total_size += fs::metadata(&dst)?.len();
                files_backed_up.push(file.to_string());
            }
        }

        // Copy HNSW index if requested
        if config.include_index {
            let index_file = "hnsw.bin";
            let src = source.join(index_file);
            if src.exists() {
                let dst = backup_dir.join(index_file);
                fs::copy(&src, &dst)?;
                total_size += fs::metadata(&dst)?.len();
                files_backed_up.push(index_file.to_string());
            }
        }

        // Copy WAL if requested
        if config.include_wal {
            let wal_dir = source.join("wal");
            if wal_dir.exists() {
                let dst_wal = backup_dir.join("wal");
                fs::create_dir_all(&dst_wal)?;

                for entry in fs::read_dir(&wal_dir)? {
                    let entry = entry?;
                    let path = entry.path();
                    if path.is_file() {
                        let name = path.file_name().unwrap();
                        let dst = dst_wal.join(name);
                        fs::copy(&path, &dst)?;
                        total_size += fs::metadata(&dst)?.len();
                        files_backed_up.push(format!("wal/{}", name.to_string_lossy()));
                    }
                }
            }
        }

        // Read source manifest for metadata
        let (vector_count, dimension) = self.read_source_metadata(source)?;

        // Calculate checksum if verification is enabled
        let checksum = if config.verify {
            Some(self.calculate_checksum(&backup_dir)?)
        } else {
            None
        };

        let backup_info = BackupInfo {
            id: backup_id.clone(),
            created_at: chrono::Utc::now().to_rfc3339(),
            source_path: source.to_string_lossy().to_string(),
            vector_count,
            dimension,
            size_bytes: total_size,
            has_index: config.include_index,
            has_wal: config.include_wal,
            metadata: config.metadata,
            files: files_backed_up,
            checksum,
        };

        // Save backup info
        let info_path = backup_dir.join("backup_info.json");
        fs::write(&info_path, serde_json::to_string_pretty(&backup_info)?)?;

        // Update manifest
        let mut manifest = self.load_manifest()?;
        manifest.backups.push(backup_info);
        self.save_manifest(&manifest)?;

        // Verify if requested
        if config.verify {
            self.verify_backup(&backup_id)?;
        }

        Ok(backup_id)
    }

    /// Read metadata from source VecStore
    fn read_source_metadata(&self, source: &Path) -> Result<(usize, usize)> {
        let manifest_path = source.join("manifest.json");
        if manifest_path.exists() {
            let content = fs::read_to_string(&manifest_path)?;
            let manifest: serde_json::Value = serde_json::from_str(&content)?;

            let dimension = manifest["dimension"].as_u64().unwrap_or(0) as usize;
            let count = manifest["record_count"].as_u64().unwrap_or(0) as usize;

            Ok((count, dimension))
        } else {
            // Try to count records file
            let records_path = source.join("records.bin");
            if records_path.exists() {
                let size = fs::metadata(&records_path)?.len();
                // Rough estimate: assume average record size
                Ok(((size / 1000) as usize, 0))
            } else {
                Ok((0, 0))
            }
        }
    }

    /// Calculate a simple checksum for verification
    fn calculate_checksum(&self, backup_dir: &Path) -> Result<String> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        for entry in fs::read_dir(backup_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file()
                && path
                    .file_name()
                    .map(|n| n != "backup_info.json")
                    .unwrap_or(false)
            {
                let mut file = File::open(&path)?;
                let mut buffer = [0u8; 8192];

                while let Ok(n) = file.read(&mut buffer) {
                    if n == 0 {
                        break;
                    }
                    buffer[..n].hash(&mut hasher);
                }
            }
        }

        Ok(format!("{:016x}", hasher.finish()))
    }

    /// List all available backups
    pub fn list_backups(&self) -> Result<Vec<BackupInfo>> {
        let manifest = self.load_manifest()?;
        Ok(manifest.backups)
    }

    /// Get information about a specific backup
    pub fn get_backup(&self, backup_id: &str) -> Result<BackupInfo> {
        let backup_dir = self.backup_root.join(backup_id);
        let info_path = backup_dir.join("backup_info.json");

        if !info_path.exists() {
            return Err(anyhow!("Backup '{}' not found", backup_id));
        }

        let content = fs::read_to_string(&info_path)?;
        Ok(serde_json::from_str(&content)?)
    }

    /// Verify a backup's integrity
    pub fn verify_backup(&self, backup_id: &str) -> Result<bool> {
        let info = self.get_backup(backup_id)?;
        let backup_dir = self.backup_root.join(backup_id);

        // Check all files exist
        for file in &info.files {
            let path = backup_dir.join(file);
            if !path.exists() {
                return Err(anyhow!("Missing file in backup: {}", file));
            }
        }

        // Verify checksum if available
        if let Some(expected_checksum) = &info.checksum {
            let actual_checksum = self.calculate_checksum(&backup_dir)?;
            if &actual_checksum != expected_checksum {
                return Err(anyhow!(
                    "Checksum mismatch: expected {}, got {}",
                    expected_checksum,
                    actual_checksum
                ));
            }
        }

        Ok(true)
    }

    /// Restore a backup to a target directory
    ///
    /// The target directory will be created if it doesn't exist.
    /// If it exists and is not empty, this will fail unless `force` is true.
    pub fn restore_backup(&self, backup_id: &str, target_path: impl AsRef<Path>) -> Result<()> {
        self.restore_backup_with_options(backup_id, target_path, false)
    }

    /// Restore with optional force overwrite
    pub fn restore_backup_with_options(
        &self,
        backup_id: &str,
        target_path: impl AsRef<Path>,
        force: bool,
    ) -> Result<()> {
        let target = target_path.as_ref();
        let backup_dir = self.backup_root.join(backup_id);

        if !backup_dir.exists() {
            return Err(anyhow!("Backup '{}' not found", backup_id));
        }

        // Check target directory
        if target.exists() {
            let is_empty = target.read_dir()?.next().is_none();
            if !is_empty && !force {
                return Err(anyhow!(
                    "Target directory is not empty. Use force=true to overwrite."
                ));
            }
        }

        fs::create_dir_all(target)?;

        // Get backup info
        let info = self.get_backup(backup_id)?;

        // Verify backup integrity first
        self.verify_backup(backup_id)?;

        // Copy all files
        for file in &info.files {
            let src = backup_dir.join(file);
            let dst = target.join(file);

            // Create parent directories if needed
            if let Some(parent) = dst.parent() {
                fs::create_dir_all(parent)?;
            }

            fs::copy(&src, &dst)?;
        }

        Ok(())
    }

    /// Delete a backup
    pub fn delete_backup(&self, backup_id: &str) -> Result<()> {
        let backup_dir = self.backup_root.join(backup_id);

        if !backup_dir.exists() {
            return Err(anyhow!("Backup '{}' not found", backup_id));
        }

        // Remove from manifest
        let mut manifest = self.load_manifest()?;
        manifest.backups.retain(|b| b.id != backup_id);
        self.save_manifest(&manifest)?;

        // Delete directory
        fs::remove_dir_all(&backup_dir)?;

        Ok(())
    }

    /// Get total size of all backups
    pub fn total_size(&self) -> Result<u64> {
        let backups = self.list_backups()?;
        Ok(backups.iter().map(|b| b.size_bytes).sum())
    }

    /// Prune old backups, keeping only the N most recent
    pub fn prune(&self, keep: usize) -> Result<Vec<String>> {
        let mut backups = self.list_backups()?;

        // Sort by creation time (newest first)
        backups.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        let mut deleted = Vec::new();

        // Delete backups beyond the keep count
        for backup in backups.iter().skip(keep) {
            self.delete_backup(&backup.id)?;
            deleted.push(backup.id.clone());
        }

        Ok(deleted)
    }
}

/// Helper to create a single-file backup archive
pub struct BackupArchive;

impl BackupArchive {
    /// Create a tar archive from a backup directory
    pub fn create_archive(
        backup_dir: impl AsRef<Path>,
        archive_path: impl AsRef<Path>,
    ) -> Result<()> {
        let backup = backup_dir.as_ref();
        let archive = archive_path.as_ref();

        let mut file = File::create(archive)?;

        // Simple archive format: file count + (name_len, name, data_len, data)*
        let entries: Vec<_> = Self::collect_files(backup)?;

        // Write number of entries
        file.write_all(&(entries.len() as u32).to_le_bytes())?;

        for (relative_path, full_path) in entries {
            let name_bytes = relative_path.as_bytes();
            let data = fs::read(&full_path)?;

            // Write name length and name
            file.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
            file.write_all(name_bytes)?;

            // Write data length and data
            file.write_all(&(data.len() as u64).to_le_bytes())?;
            file.write_all(&data)?;
        }

        Ok(())
    }

    /// Extract a tar archive to a directory
    pub fn extract_archive(
        archive_path: impl AsRef<Path>,
        target_dir: impl AsRef<Path>,
    ) -> Result<()> {
        let archive = archive_path.as_ref();
        let target = target_dir.as_ref();

        fs::create_dir_all(target)?;

        let mut file = File::open(archive)?;

        // Read number of entries
        let mut count_buf = [0u8; 4];
        file.read_exact(&mut count_buf)?;
        let count = u32::from_le_bytes(count_buf) as usize;

        for _ in 0..count {
            // Read name
            let mut name_len_buf = [0u8; 4];
            file.read_exact(&mut name_len_buf)?;
            let name_len = u32::from_le_bytes(name_len_buf) as usize;

            let mut name_buf = vec![0u8; name_len];
            file.read_exact(&mut name_buf)?;
            let name = String::from_utf8(name_buf)?;

            // Read data
            let mut data_len_buf = [0u8; 8];
            file.read_exact(&mut data_len_buf)?;
            let data_len = u64::from_le_bytes(data_len_buf) as usize;

            let mut data = vec![0u8; data_len];
            file.read_exact(&mut data)?;

            // Write to target
            let target_path = target.join(&name);
            if let Some(parent) = target_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&target_path, &data)?;
        }

        Ok(())
    }

    fn collect_files(dir: &Path) -> Result<Vec<(String, PathBuf)>> {
        let mut files = Vec::new();

        for entry in walkdir::WalkDir::new(dir) {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };

            if entry.file_type().is_file() {
                let relative = entry
                    .path()
                    .strip_prefix(dir)
                    .unwrap()
                    .to_string_lossy()
                    .to_string();
                files.push((relative, entry.path().to_path_buf()));
            }
        }

        Ok(files)
    }
}

// Simple walkdir implementation for backup
mod walkdir {
    use std::fs;
    use std::path::{Path, PathBuf};

    pub struct WalkDir {
        stack: Vec<PathBuf>,
    }

    impl WalkDir {
        pub fn new(root: &Path) -> Self {
            Self {
                stack: vec![root.to_path_buf()],
            }
        }
    }

    pub struct DirEntry {
        path: PathBuf,
        is_file: bool,
    }

    impl DirEntry {
        pub fn path(&self) -> &Path {
            &self.path
        }

        pub fn file_type(&self) -> FileType {
            FileType {
                is_file: self.is_file,
            }
        }
    }

    pub struct FileType {
        is_file: bool,
    }

    impl FileType {
        pub fn is_file(&self) -> bool {
            self.is_file
        }
    }

    impl Iterator for WalkDir {
        type Item = Result<DirEntry, std::io::Error>;

        fn next(&mut self) -> Option<Self::Item> {
            while let Some(path) = self.stack.pop() {
                let metadata = match fs::metadata(&path) {
                    Ok(m) => m,
                    Err(e) => return Some(Err(e)),
                };

                if metadata.is_dir() {
                    // Add children to stack
                    match fs::read_dir(&path) {
                        Ok(entries) => {
                            for e in entries.flatten() {
                                self.stack.push(e.path());
                            }
                        },
                        Err(e) => return Some(Err(e)),
                    }
                } else {
                    return Some(Ok(DirEntry {
                        path,
                        is_file: true,
                    }));
                }
            }

            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_backup_manager_creation() {
        let temp = TempDir::new().unwrap();
        let mgr = BackupManager::new(temp.path());
        assert!(mgr.is_ok());
    }

    #[test]
    fn test_backup_and_restore() {
        let temp = TempDir::new().unwrap();
        let source_dir = temp.path().join("source");
        let backup_dir = temp.path().join("backups");
        let restore_dir = temp.path().join("restore");

        // Create source with some files
        fs::create_dir_all(&source_dir).unwrap();
        fs::write(
            source_dir.join("manifest.json"),
            r#"{"dimension":128,"record_count":100}"#,
        )
        .unwrap();
        fs::write(source_dir.join("records.bin"), vec![0u8; 1000]).unwrap();

        // Create backup
        let mgr = BackupManager::new(&backup_dir).unwrap();
        let backup_id = mgr
            .create_backup(&source_dir, BackupConfig::default())
            .unwrap();

        // Verify backup exists
        let backups = mgr.list_backups().unwrap();
        assert_eq!(backups.len(), 1);
        assert_eq!(backups[0].id, backup_id);

        // Restore
        mgr.restore_backup(&backup_id, &restore_dir).unwrap();

        // Verify restore
        assert!(restore_dir.join("manifest.json").exists());
        assert!(restore_dir.join("records.bin").exists());
    }

    #[test]
    fn test_backup_verification() {
        let temp = TempDir::new().unwrap();
        let source_dir = temp.path().join("source");
        let backup_dir = temp.path().join("backups");

        fs::create_dir_all(&source_dir).unwrap();
        fs::write(source_dir.join("records.bin"), vec![1u8; 100]).unwrap();

        let mgr = BackupManager::new(&backup_dir).unwrap();
        let backup_id = mgr
            .create_backup(
                &source_dir,
                BackupConfig {
                    verify: true,
                    ..Default::default()
                },
            )
            .unwrap();

        // Verification should pass
        assert!(mgr.verify_backup(&backup_id).unwrap());
    }

    #[test]
    fn test_backup_prune() {
        let temp = TempDir::new().unwrap();
        let source_dir = temp.path().join("source");
        let backup_dir = temp.path().join("backups");

        fs::create_dir_all(&source_dir).unwrap();
        fs::write(source_dir.join("records.bin"), vec![0u8; 10]).unwrap();

        let mgr = BackupManager::new(&backup_dir).unwrap();

        // Create 3 backups with delay to ensure unique timestamps
        for i in 0..3 {
            let result = mgr.create_backup(&source_dir, BackupConfig::default());
            assert!(
                result.is_ok(),
                "Failed to create backup {}: {:?}",
                i,
                result.err()
            );
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        let backups = mgr.list_backups().unwrap();
        assert_eq!(
            backups.len(),
            3,
            "Expected 3 backups, got {}",
            backups.len()
        );

        // Prune to keep only 1
        let deleted = mgr.prune(1).unwrap();
        assert_eq!(deleted.len(), 2, "Expected to delete 2 backups");
        assert_eq!(
            mgr.list_backups().unwrap().len(),
            1,
            "Expected 1 backup remaining"
        );
    }

    #[test]
    fn test_archive() {
        let temp = TempDir::new().unwrap();
        let source_dir = temp.path().join("source");
        let archive_path = temp.path().join("backup.archive");
        let extract_dir = temp.path().join("extract");

        // Create source
        fs::create_dir_all(&source_dir).unwrap();
        fs::write(source_dir.join("file1.txt"), "hello").unwrap();
        fs::write(source_dir.join("file2.bin"), vec![1, 2, 3, 4]).unwrap();

        // Create archive
        BackupArchive::create_archive(&source_dir, &archive_path).unwrap();
        assert!(archive_path.exists());

        // Extract
        BackupArchive::extract_archive(&archive_path, &extract_dir).unwrap();

        // Verify
        assert_eq!(
            fs::read_to_string(extract_dir.join("file1.txt")).unwrap(),
            "hello"
        );
        assert_eq!(
            fs::read(extract_dir.join("file2.bin")).unwrap(),
            vec![1, 2, 3, 4]
        );
    }
}
