//! Write-Ahead Logging (WAL) for crash recovery
//!
//! This module provides a write-ahead log implementation that ensures
//! durability and enables crash recovery. All writes are first logged
//! to a sequential append-only file before being applied to the main store.
//!
//! ## Features
//!
//! - Append-only log for fast writes
//! - Crash recovery via log replay
//! - Checkpointing for log compaction
//! - Concurrent readers during write operations
//!
//! ## Usage
//!
//! ```no_run
//! use vecstore::wal::{WriteAheadLog, LogEntry, Operation};
//!
//! # fn main() -> anyhow::Result<()> {
//! let mut wal = WriteAheadLog::open("store.wal")?;
//!
//! // Log an operation
//! let entry = LogEntry::Insert {
//!     id: "doc1".to_string(),
//!     vector: vec![0.1, 0.2, 0.3],
//! };
//! wal.append(entry)?;
//!
//! // Recover from crash
//! let entries = wal.replay()?;
//! for entry in entries {
//!     // Apply entry to main store
//! }
//!
//! // Checkpoint and truncate
//! wal.checkpoint()?;
//! # Ok(())
//! # }
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// A single entry in the write-ahead log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogEntry {
    /// Insert a new vector
    Insert { id: String, vector: Vec<f32> },

    /// Update an existing vector
    Update { id: String, vector: Vec<f32> },

    /// Delete a vector
    Delete { id: String },

    /// Begin a transaction
    BeginTx { tx_id: u64 },

    /// Commit a transaction
    CommitTx { tx_id: u64 },

    /// Abort a transaction
    AbortTx { tx_id: u64 },

    /// Checkpoint marker
    Checkpoint { sequence: u64 },
}

/// Write-Ahead Log implementation
pub struct WriteAheadLog {
    file: File,
    writer: BufWriter<File>,
    next_sequence: u64,
    last_checkpoint: u64,
    entry_count: u64,
}

impl WriteAheadLog {
    /// Open or create a WAL file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .context("Failed to open WAL file")?;

        let writer_file = file.try_clone()?;
        let writer = BufWriter::new(writer_file);

        // Read existing log to get next sequence number and entry count
        let (entry_count, last_seq) = Self::scan_log(&file)?;
        let next_sequence = if entry_count > 0 { last_seq + 1 } else { 0 };

        Ok(Self {
            file,
            writer,
            next_sequence,
            last_checkpoint: 0,
            entry_count,
        })
    }

    /// Append an entry to the log
    pub fn append(&mut self, entry: LogEntry) -> Result<u64> {
        let sequence = self.next_sequence;
        self.next_sequence += 1;
        self.entry_count += 1;

        // Serialize the entry with its sequence number
        let record = LogRecord { sequence, entry };
        let serialized = bincode::serialize(&record).context("Failed to serialize log entry")?;

        // Write length prefix (for easy recovery)
        let len = serialized.len() as u32;
        self.writer.write_all(&len.to_le_bytes())?;

        // Write the serialized entry
        self.writer.write_all(&serialized)?;

        // Flush to ensure durability
        self.writer.flush()?;

        Ok(sequence)
    }

    /// Replay all log entries since the last checkpoint
    pub fn replay(&mut self) -> Result<Vec<LogEntry>> {
        let mut reader = BufReader::new(self.file.try_clone()?);
        reader.seek(SeekFrom::Start(0))?;

        let mut entries = Vec::new();
        let mut last_checkpoint_seq = 0;

        loop {
            // Read length prefix
            let mut len_bytes = [0u8; 4];
            match reader.read_exact(&mut len_bytes) {
                Ok(_) => {}
                Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }

            let len = u32::from_le_bytes(len_bytes) as usize;

            // Read the entry
            let mut buffer = vec![0u8; len];
            reader.read_exact(&mut buffer)?;

            let record: LogRecord =
                bincode::deserialize(&buffer).context("Failed to deserialize log entry")?;

            // Track checkpoints
            if let LogEntry::Checkpoint { sequence } = record.entry {
                last_checkpoint_seq = sequence;
                entries.clear(); // Discard entries before checkpoint
            } else {
                entries.push(record.entry);
            }
        }

        self.last_checkpoint = last_checkpoint_seq;

        Ok(entries)
    }

    /// Write a checkpoint marker and truncate log
    pub fn checkpoint(&mut self) -> Result<()> {
        if self.entry_count == 0 {
            // No entries to checkpoint
            return Ok(());
        }

        let checkpoint_seq = self.next_sequence - 1; // Last written sequence

        // Manually write checkpoint (not through append to avoid incrementing sequence)
        let record = LogRecord {
            sequence: checkpoint_seq,
            entry: LogEntry::Checkpoint {
                sequence: checkpoint_seq,
            },
        };

        let serialized = bincode::serialize(&record).context("Failed to serialize checkpoint")?;

        let len = serialized.len() as u32;
        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(&serialized)?;
        self.entry_count += 1;

        // Flush everything
        self.writer.flush()?;
        self.file.sync_all()?;

        self.last_checkpoint = checkpoint_seq;

        Ok(())
    }

    /// Truncate the log (call after checkpoint and successful apply)
    pub fn truncate(&mut self) -> Result<()> {
        // Flush and sync before truncating
        self.writer.flush()?;
        self.file.sync_all()?;

        // Truncate to zero (start fresh)
        self.file.set_len(0)?;
        self.file.seek(SeekFrom::Start(0))?;

        self.next_sequence = 0;
        self.last_checkpoint = 0;
        self.entry_count = 0;

        Ok(())
    }

    /// Get the number of entries in the log
    pub fn len(&self) -> u64 {
        self.entry_count
    }

    /// Check if the log is empty
    pub fn is_empty(&self) -> bool {
        self.entry_count == 0
    }

    /// Scan the log and return (entry_count, last_sequence)
    fn scan_log(file: &File) -> Result<(u64, u64)> {
        let mut reader = BufReader::new(file.try_clone()?);
        reader.seek(SeekFrom::Start(0))?;

        let mut entry_count = 0u64;
        let mut last_seq = 0u64;

        loop {
            let mut len_bytes = [0u8; 4];
            match reader.read_exact(&mut len_bytes) {
                Ok(_) => {}
                Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }

            let len = u32::from_le_bytes(len_bytes) as usize;
            let mut buffer = vec![0u8; len];
            reader.read_exact(&mut buffer)?;

            if let Ok(record) = bincode::deserialize::<LogRecord>(&buffer) {
                last_seq = record.sequence;
                entry_count += 1;
            }
        }

        Ok((entry_count, last_seq))
    }
}

/// Internal log record with sequence number
#[derive(Debug, Serialize, Deserialize)]
struct LogRecord {
    sequence: u64,
    entry: LogEntry,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_create_wal() {
        let temp_file = NamedTempFile::new().unwrap();
        let wal = WriteAheadLog::open(temp_file.path()).unwrap();

        assert_eq!(wal.len(), 0);
        assert!(wal.is_empty());
    }

    #[test]
    fn test_append_entry() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut wal = WriteAheadLog::open(temp_file.path()).unwrap();

        let entry = LogEntry::Insert {
            id: "doc1".to_string(),
            vector: vec![1.0, 2.0, 3.0],
        };

        let seq = wal.append(entry).unwrap();
        assert_eq!(seq, 0);
        assert_eq!(wal.len(), 1);
    }

    #[test]
    fn test_replay_empty() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut wal = WriteAheadLog::open(temp_file.path()).unwrap();

        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 0);
    }

    #[test]
    fn test_replay_entries() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut wal = WriteAheadLog::open(temp_file.path()).unwrap();

        // Append some entries
        wal.append(LogEntry::Insert {
            id: "doc1".to_string(),
            vector: vec![1.0],
        })
        .unwrap();

        wal.append(LogEntry::Update {
            id: "doc1".to_string(),
            vector: vec![2.0],
        })
        .unwrap();

        wal.append(LogEntry::Delete {
            id: "doc1".to_string(),
        })
        .unwrap();

        // Replay
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 3);

        match &entries[0] {
            LogEntry::Insert { id, .. } => assert_eq!(id, "doc1"),
            _ => panic!("Expected Insert"),
        }

        match &entries[1] {
            LogEntry::Update { id, .. } => assert_eq!(id, "doc1"),
            _ => panic!("Expected Update"),
        }

        match &entries[2] {
            LogEntry::Delete { id } => assert_eq!(id, "doc1"),
            _ => panic!("Expected Delete"),
        }
    }

    #[test]
    fn test_checkpoint() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut wal = WriteAheadLog::open(temp_file.path()).unwrap();

        // Append entries before checkpoint
        wal.append(LogEntry::Insert {
            id: "doc1".to_string(),
            vector: vec![1.0],
        })
        .unwrap();

        wal.append(LogEntry::Insert {
            id: "doc2".to_string(),
            vector: vec![2.0],
        })
        .unwrap();

        // Checkpoint
        wal.checkpoint().unwrap();

        // Append more entries after checkpoint
        wal.append(LogEntry::Insert {
            id: "doc3".to_string(),
            vector: vec![3.0],
        })
        .unwrap();

        // Replay should only return entries after checkpoint
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);

        match &entries[0] {
            LogEntry::Insert { id, .. } => assert_eq!(id, "doc3"),
            _ => panic!("Expected Insert"),
        }
    }

    #[test]
    fn test_truncate() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut wal = WriteAheadLog::open(temp_file.path()).unwrap();

        // Append entries
        for i in 0..10 {
            wal.append(LogEntry::Insert {
                id: format!("doc{}", i),
                vector: vec![i as f32],
            })
            .unwrap();
        }

        assert_eq!(wal.len(), 10);

        // Truncate
        wal.truncate().unwrap();

        assert_eq!(wal.len(), 0);
        assert!(wal.is_empty());

        // Replay should return nothing
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 0);
    }

    #[test]
    fn test_crash_recovery() {
        let temp_file = NamedTempFile::new().unwrap();

        // Write some entries and "crash"
        {
            let mut wal = WriteAheadLog::open(temp_file.path()).unwrap();
            for i in 0..5 {
                wal.append(LogEntry::Insert {
                    id: format!("doc{}", i),
                    vector: vec![i as f32],
                })
                .unwrap();
            }
            // WAL is dropped here (simulating crash)
        }

        // Reopen and replay
        {
            let mut wal = WriteAheadLog::open(temp_file.path()).unwrap();
            let entries = wal.replay().unwrap();

            assert_eq!(entries.len(), 5);
            for (i, entry) in entries.iter().enumerate() {
                match entry {
                    LogEntry::Insert { id, .. } => assert_eq!(id, &format!("doc{}", i)),
                    _ => panic!("Expected Insert"),
                }
            }
        }
    }
}
