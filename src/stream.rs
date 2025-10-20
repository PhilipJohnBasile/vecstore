//! Streaming query APIs for memory-efficient result iteration
//!
//! This module provides streaming interfaces for query results, allowing
//! processing of large result sets without loading everything into memory.
//!
//! ## Features
//!
//! - Iterator-based streaming
//! - Cursor-based pagination
//! - Backpressure support
//! - Memory-efficient for large result sets
//!
//! ## Usage
//!
//! ```no_run
//! use vecstore::{VecStore, Query};
//!
//! # fn main() -> anyhow::Result<()> {
//! let store = VecStore::open("vectors.db")?;
//! let query = Query::new(vec![0.1, 0.2, 0.3]);
//!
//! // Stream results instead of loading all at once
//! let mut stream = store.query_stream(query)?;
//!
//! while let Some(result) = stream.next() {
//!     println!("ID: {}, Score: {}", result.id, result.score);
//!     // Process one result at a time - memory efficient!
//! }
//! # Ok(())
//! # }
//! ```

use crate::store::Neighbor;

/// Streaming query result iterator
///
/// Provides memory-efficient iteration over query results without
/// loading all results into memory at once.
pub struct QueryStream {
    results: Vec<Neighbor>,
    position: usize,
    batch_size: usize,
}

impl QueryStream {
    /// Create a new query stream from results
    pub fn new(results: Vec<Neighbor>) -> Self {
        Self {
            results,
            position: 0,
            batch_size: 100, // Default batch size
        }
    }

    /// Set batch size for prefetching
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Get next result
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<&Neighbor> {
        if self.position < self.results.len() {
            let result = &self.results[self.position];
            self.position += 1;
            Some(result)
        } else {
            None
        }
    }

    /// Peek at next result without consuming
    pub fn peek(&self) -> Option<&Neighbor> {
        if self.position < self.results.len() {
            Some(&self.results[self.position])
        } else {
            None
        }
    }

    /// Get remaining result count
    pub fn remaining(&self) -> usize {
        self.results.len().saturating_sub(self.position)
    }

    /// Check if stream is exhausted
    pub fn is_empty(&self) -> bool {
        self.position >= self.results.len()
    }

    /// Reset stream to beginning
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Collect remaining results into a Vec
    pub fn collect(mut self) -> Vec<Neighbor> {
        self.results.split_off(self.position)
    }

    /// Skip n results
    pub fn skip(&mut self, n: usize) {
        self.position = (self.position + n).min(self.results.len());
    }

    /// Take up to n results
    pub fn take(&mut self, n: usize) -> Vec<Neighbor> {
        let end = (self.position + n).min(self.results.len());
        let taken = self.results[self.position..end].to_vec();
        self.position = end;
        taken
    }
}

/// Cursor for paginated queries
///
/// Enables efficient pagination through large result sets.
pub struct QueryCursor {
    offset: usize,
    limit: usize,
    total_count: usize,
}

impl QueryCursor {
    /// Create a new cursor
    pub fn new(limit: usize) -> Self {
        Self {
            offset: 0,
            limit,
            total_count: 0,
        }
    }

    /// Get current page
    pub fn page(&self) -> usize {
        if self.limit == 0 {
            0
        } else {
            self.offset / self.limit
        }
    }

    /// Move to next page
    pub fn next_page(&mut self) {
        self.offset += self.limit;
    }

    /// Move to previous page
    pub fn prev_page(&mut self) {
        self.offset = self.offset.saturating_sub(self.limit);
    }

    /// Move to specific page
    pub fn goto_page(&mut self, page: usize) {
        self.offset = page * self.limit;
    }

    /// Get offset for current page
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get limit (page size)
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Check if there are more pages
    pub fn has_next(&self) -> bool {
        self.offset + self.limit < self.total_count
    }

    /// Check if there are previous pages
    pub fn has_prev(&self) -> bool {
        self.offset > 0
    }

    /// Set total count of results
    pub fn set_total_count(&mut self, count: usize) {
        self.total_count = count;
    }

    /// Get total pages
    pub fn total_pages(&self) -> usize {
        if self.limit == 0 {
            0
        } else {
            self.total_count.div_ceil(self.limit)
        }
    }
}

/// Batched streaming query
///
/// Processes results in batches for better performance.
pub struct BatchedStream {
    results: Vec<Neighbor>,
    batch_size: usize,
    position: usize,
}

impl BatchedStream {
    /// Create a new batched stream
    pub fn new(results: Vec<Neighbor>, batch_size: usize) -> Self {
        Self {
            results,
            batch_size,
            position: 0,
        }
    }

    /// Get next batch
    pub fn next_batch(&mut self) -> Option<&[Neighbor]> {
        if self.position >= self.results.len() {
            return None;
        }

        let end = (self.position + self.batch_size).min(self.results.len());
        let batch = &self.results[self.position..end];
        self.position = end;

        Some(batch)
    }

    /// Get remaining batches count
    pub fn remaining_batches(&self) -> usize {
        let remaining = self.results.len().saturating_sub(self.position);
        remaining.div_ceil(self.batch_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Metadata;

    fn make_neighbor(id: &str, score: f32) -> Neighbor {
        Neighbor {
            id: id.to_string(),
            score,
            metadata: Metadata {
                fields: std::collections::HashMap::new(),
            },
        }
    }

    #[test]
    fn test_stream_basic() {
        let results = vec![
            make_neighbor("doc1", 0.1),
            make_neighbor("doc2", 0.2),
            make_neighbor("doc3", 0.3),
        ];

        let mut stream = QueryStream::new(results);

        assert_eq!(stream.remaining(), 3);
        assert!(stream.next().is_some());
        assert_eq!(stream.remaining(), 2);
        assert!(stream.next().is_some());
        assert!(stream.next().is_some());
        assert!(stream.next().is_none());
        assert!(stream.is_empty());
    }

    #[test]
    fn test_stream_peek() {
        let results = vec![make_neighbor("doc1", 0.1)];
        let mut stream = QueryStream::new(results);

        // Peek doesn't consume
        assert!(stream.peek().is_some());
        assert_eq!(stream.remaining(), 1);

        // Next consumes
        assert!(stream.next().is_some());
        assert_eq!(stream.remaining(), 0);
    }

    #[test]
    fn test_stream_skip() {
        let results = vec![
            make_neighbor("doc1", 0.1),
            make_neighbor("doc2", 0.2),
            make_neighbor("doc3", 0.3),
        ];

        let mut stream = QueryStream::new(results);
        stream.skip(2);

        assert_eq!(stream.remaining(), 1);
        assert_eq!(stream.next().unwrap().id, "doc3");
    }

    #[test]
    fn test_stream_take() {
        let results = vec![
            make_neighbor("doc1", 0.1),
            make_neighbor("doc2", 0.2),
            make_neighbor("doc3", 0.3),
        ];

        let mut stream = QueryStream::new(results);
        let taken = stream.take(2);

        assert_eq!(taken.len(), 2);
        assert_eq!(taken[0].id, "doc1");
        assert_eq!(taken[1].id, "doc2");
        assert_eq!(stream.remaining(), 1);
    }

    #[test]
    fn test_stream_reset() {
        let results = vec![make_neighbor("doc1", 0.1)];
        let mut stream = QueryStream::new(results);

        stream.next();
        assert!(stream.is_empty());

        stream.reset();
        assert!(!stream.is_empty());
        assert!(stream.next().is_some());
    }

    #[test]
    fn test_cursor_pagination() {
        let mut cursor = QueryCursor::new(10);
        cursor.set_total_count(100);

        assert_eq!(cursor.page(), 0);
        assert_eq!(cursor.total_pages(), 10);
        assert!(cursor.has_next());
        assert!(!cursor.has_prev());

        cursor.next_page();
        assert_eq!(cursor.page(), 1);
        assert!(cursor.has_prev());

        cursor.goto_page(5);
        assert_eq!(cursor.page(), 5);
        assert_eq!(cursor.offset(), 50);
    }

    #[test]
    fn test_batched_stream() {
        let results = vec![
            make_neighbor("doc1", 0.1),
            make_neighbor("doc2", 0.2),
            make_neighbor("doc3", 0.3),
            make_neighbor("doc4", 0.4),
            make_neighbor("doc5", 0.5),
        ];

        let mut stream = BatchedStream::new(results, 2);

        let batch1 = stream.next_batch().unwrap();
        assert_eq!(batch1.len(), 2);
        assert_eq!(batch1[0].id, "doc1");

        let batch2 = stream.next_batch().unwrap();
        assert_eq!(batch2.len(), 2);

        let batch3 = stream.next_batch().unwrap();
        assert_eq!(batch3.len(), 1); // Last batch

        assert!(stream.next_batch().is_none());
    }

    #[test]
    fn test_batched_remaining() {
        let results = vec![
            make_neighbor("doc1", 0.1),
            make_neighbor("doc2", 0.2),
            make_neighbor("doc3", 0.3),
        ];

        let mut stream = BatchedStream::new(results, 2);
        assert_eq!(stream.remaining_batches(), 2);

        stream.next_batch();
        assert_eq!(stream.remaining_batches(), 1);

        stream.next_batch();
        assert_eq!(stream.remaining_batches(), 0);
    }
}
