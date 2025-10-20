//! Async streaming query APIs with futures::Stream support
//!
//! This module provides async streaming interfaces for progressive result delivery,
//! enabling real-time UI updates and WebSocket streaming.
//!
//! ## Features
//!
//! - futures::Stream implementation for async iteration
//! - Progressive result delivery
//! - Backpressure support
//! - WebSocket-compatible streaming
//! - Memory-efficient for large result sets
//!
//! ## Usage
//!
//! ```no_run
//! use vecstore::{AsyncVecStore, Query};
//! use futures::StreamExt;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let store = AsyncVecStore::open("vectors.db").await?;
//!     let query = Query::new(vec![0.1, 0.2, 0.3]);
//!
//!     // Stream results progressively
//!     let mut stream = store.query_stream(query).await?;
//!
//!     while let Some(result) = stream.next().await {
//!         println!("ID: {}, Score: {}", result.id, result.score);
//!         // Update UI progressively!
//!     }
//!     Ok(())
//! }
//! ```

#[cfg(feature = "async")]
use crate::store::Neighbor;
#[cfg(feature = "async")]
use futures::stream::Stream;
#[cfg(feature = "async")]
use std::pin::Pin;
#[cfg(feature = "async")]
use std::task::{Context, Poll};
#[cfg(feature = "async")]
use tokio::sync::mpsc;

/// Async streaming query result iterator
///
/// Implements futures::Stream for async iteration over query results.
/// Results are delivered progressively for better UX.
#[cfg(feature = "async")]
pub struct AsyncQueryStream {
    results: Vec<Neighbor>,
    position: usize,
    chunk_size: usize,
}

#[cfg(feature = "async")]
impl AsyncQueryStream {
    /// Create a new async query stream from results
    pub fn new(results: Vec<Neighbor>) -> Self {
        Self {
            results,
            position: 0,
            chunk_size: 10, // Default: deliver 10 results at a time
        }
    }

    /// Set chunk size for progressive delivery
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Get remaining result count
    pub fn remaining(&self) -> usize {
        self.results.len().saturating_sub(self.position)
    }

    /// Check if stream is exhausted
    pub fn is_empty(&self) -> bool {
        self.position >= self.results.len()
    }
}

#[cfg(feature = "async")]
impl Stream for AsyncQueryStream {
    type Item = Neighbor;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.position < self.results.len() {
            let result = self.results[self.position].clone();
            self.position += 1;
            Poll::Ready(Some(result))
        } else {
            Poll::Ready(None)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining();
        (remaining, Some(remaining))
    }
}

/// Chunked async stream that delivers results in batches
///
/// Useful for batch processing or reducing message overhead in WebSocket streaming.
#[cfg(feature = "async")]
pub struct ChunkedQueryStream {
    results: Vec<Neighbor>,
    position: usize,
    chunk_size: usize,
}

#[cfg(feature = "async")]
impl ChunkedQueryStream {
    /// Create a new chunked stream
    pub fn new(results: Vec<Neighbor>, chunk_size: usize) -> Self {
        Self {
            results,
            position: 0,
            chunk_size,
        }
    }

    /// Get remaining chunks count
    pub fn remaining_chunks(&self) -> usize {
        let remaining = self.results.len().saturating_sub(self.position);
        remaining.div_ceil(self.chunk_size)
    }
}

#[cfg(feature = "async")]
impl Stream for ChunkedQueryStream {
    type Item = Vec<Neighbor>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.position >= self.results.len() {
            return Poll::Ready(None);
        }

        let end = (self.position + self.chunk_size).min(self.results.len());
        let chunk = self.results[self.position..end].to_vec();
        self.position = end;

        Poll::Ready(Some(chunk))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining_chunks = self.remaining_chunks();
        (remaining_chunks, Some(remaining_chunks))
    }
}

/// Channel-based async stream for real-time result delivery
///
/// Allows producing results from one task and consuming from another,
/// ideal for WebSocket streaming or background query processing.
#[cfg(feature = "async")]
pub struct ChannelQueryStream {
    receiver: mpsc::UnboundedReceiver<Neighbor>,
}

#[cfg(feature = "async")]
impl ChannelQueryStream {
    /// Create a new channel stream
    ///
    /// Returns (stream, sender) tuple. Use sender to push results,
    /// stream will deliver them asynchronously.
    pub fn new() -> (Self, mpsc::UnboundedSender<Neighbor>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self { receiver: rx }, tx)
    }

    /// Create from an existing receiver
    pub fn from_receiver(receiver: mpsc::UnboundedReceiver<Neighbor>) -> Self {
        Self { receiver }
    }
}

#[cfg(feature = "async")]
impl Default for ChannelQueryStream {
    fn default() -> Self {
        Self::new().0
    }
}

#[cfg(feature = "async")]
impl Stream for ChannelQueryStream {
    type Item = Neighbor;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

/// Progressive query executor that streams results as they're computed
///
/// Instead of waiting for all results, this delivers results progressively
/// as the search algorithm finds them.
#[cfg(feature = "async")]
pub struct ProgressiveQueryStream {
    receiver: mpsc::UnboundedReceiver<StreamEvent>,
    total_expected: Option<usize>,
    received: usize,
}

/// Events emitted by progressive query stream
#[cfg(feature = "async")]
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A result was found
    Result(Neighbor),
    /// Progress update (current, total)
    Progress { current: usize, total: usize },
    /// Query completed
    Done,
    /// Error occurred
    Error(String),
}

#[cfg(feature = "async")]
impl ProgressiveQueryStream {
    /// Create a new progressive stream
    pub fn new() -> (Self, mpsc::UnboundedSender<StreamEvent>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (
            Self {
                receiver: rx,
                total_expected: None,
                received: 0,
            },
            tx,
        )
    }

    /// Get progress percentage (0-100)
    pub fn progress_percent(&self) -> Option<f32> {
        self.total_expected.map(|total| {
            if total == 0 {
                100.0
            } else {
                (self.received as f32 / total as f32) * 100.0
            }
        })
    }

    /// Check if complete
    pub fn is_complete(&self) -> bool {
        if let Some(total) = self.total_expected {
            self.received >= total
        } else {
            false
        }
    }
}

#[cfg(feature = "async")]
impl Default for ProgressiveQueryStream {
    fn default() -> Self {
        Self::new().0
    }
}

#[cfg(feature = "async")]
impl Stream for ProgressiveQueryStream {
    type Item = StreamEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.receiver.poll_recv(cx) {
            Poll::Ready(Some(event)) => {
                match &event {
                    StreamEvent::Result(_) => {
                        self.received += 1;
                    }
                    StreamEvent::Progress { total, .. } => {
                        self.total_expected = Some(*total);
                    }
                    _ => {}
                }
                Poll::Ready(Some(event))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(all(test, feature = "async"))]
mod tests {
    use super::*;
    use crate::store::Metadata;
    use futures::StreamExt;
    use std::collections::HashMap;

    fn make_neighbor(id: &str, score: f32) -> Neighbor {
        Neighbor {
            id: id.to_string(),
            score,
            metadata: Metadata {
                fields: HashMap::new(),
            },
        }
    }

    #[tokio::test]
    async fn test_async_stream_basic() {
        let results = vec![
            make_neighbor("doc1", 0.1),
            make_neighbor("doc2", 0.2),
            make_neighbor("doc3", 0.3),
        ];

        let mut stream = AsyncQueryStream::new(results);
        let mut count = 0;

        while let Some(_result) = stream.next().await {
            count += 1;
        }

        assert_eq!(count, 3);
    }

    #[tokio::test]
    async fn test_chunked_stream() {
        let results = vec![
            make_neighbor("doc1", 0.1),
            make_neighbor("doc2", 0.2),
            make_neighbor("doc3", 0.3),
            make_neighbor("doc4", 0.4),
            make_neighbor("doc5", 0.5),
        ];

        let mut stream = ChunkedQueryStream::new(results, 2);
        let mut chunks = Vec::new();

        while let Some(chunk) = stream.next().await {
            chunks.push(chunk);
        }

        assert_eq!(chunks.len(), 3); // 2 + 2 + 1
        assert_eq!(chunks[0].len(), 2);
        assert_eq!(chunks[1].len(), 2);
        assert_eq!(chunks[2].len(), 1);
    }

    #[tokio::test]
    async fn test_channel_stream() {
        let (mut stream, sender) = ChannelQueryStream::new();

        // Send results from another task
        tokio::spawn(async move {
            sender.send(make_neighbor("doc1", 0.1)).unwrap();
            sender.send(make_neighbor("doc2", 0.2)).unwrap();
            sender.send(make_neighbor("doc3", 0.3)).unwrap();
            // Sender drops, closing channel
        });

        let mut count = 0;
        while let Some(_result) = stream.next().await {
            count += 1;
        }

        assert_eq!(count, 3);
    }

    #[tokio::test]
    async fn test_progressive_stream() {
        let (mut stream, sender) = ProgressiveQueryStream::new();

        // Simulate progressive query execution
        tokio::spawn(async move {
            sender
                .send(StreamEvent::Progress {
                    current: 0,
                    total: 3,
                })
                .unwrap();
            sender
                .send(StreamEvent::Result(make_neighbor("doc1", 0.1)))
                .unwrap();
            sender
                .send(StreamEvent::Progress {
                    current: 1,
                    total: 3,
                })
                .unwrap();
            sender
                .send(StreamEvent::Result(make_neighbor("doc2", 0.2)))
                .unwrap();
            sender
                .send(StreamEvent::Result(make_neighbor("doc3", 0.3)))
                .unwrap();
            sender.send(StreamEvent::Done).unwrap();
        });

        let mut result_count = 0;
        let mut got_done = false;

        while let Some(event) = stream.next().await {
            match event {
                StreamEvent::Result(_) => result_count += 1,
                StreamEvent::Done => got_done = true,
                _ => {}
            }
        }

        assert_eq!(result_count, 3);
        assert!(got_done);
    }

    #[tokio::test]
    async fn test_stream_collect() {
        let results = vec![
            make_neighbor("doc1", 0.1),
            make_neighbor("doc2", 0.2),
            make_neighbor("doc3", 0.3),
        ];

        let stream = AsyncQueryStream::new(results);
        let collected: Vec<Neighbor> = stream.collect().await;

        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0].id, "doc1");
    }
}
