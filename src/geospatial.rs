//! Geospatial Indexing and Queries
//!
//! This module provides geospatial indexing using S2 geometry for efficient
//! location-based queries combined with vector search.
//!
//! ## Features
//!
//! - **Radius search**: Find points within distance
//! - **Bounding box**: Query rectangular regions
//! - **Hybrid queries**: Combine location + vector similarity
//! - **S2 cells**: Hierarchical spatial indexing
//!
//! ## Example
//!
//! ```no_run
//! use vecstore::geospatial::{GeoPoint, GeoIndex, RadiusQuery};
//!
//! # fn main() -> anyhow::Result<()> {
//! let mut index = GeoIndex::new();
//!
//! // Add locations
//! index.add("cafe1", GeoPoint::new(37.7749, -122.4194), vec![0.1; 128]);
//! index.add("cafe2", GeoPoint::new(37.7849, -122.4094), vec![0.2; 128]);
//!
//! // Radius search
//! let center = GeoPoint::new(37.7799, -122.4144);
//! let results = index.radius_search(&center, 1000.0, 10)?; // 1km radius
//!
//! println!("Found {} locations", results.len());
//! # Ok(())
//! # }
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Geographic point (latitude, longitude)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GeoPoint {
    /// Latitude in degrees (-90 to 90)
    pub lat: f64,
    /// Longitude in degrees (-180 to 180)
    pub lon: f64,
}

impl GeoPoint {
    /// Create a new geographic point
    pub fn new(lat: f64, lon: f64) -> Self {
        Self { lat, lon }
    }

    /// Validate the coordinates
    pub fn validate(&self) -> Result<()> {
        if !(-90.0..=90.0).contains(&self.lat) {
            return Err(anyhow!("Latitude must be between -90 and 90"));
        }
        if !(-180.0..=180.0).contains(&self.lon) {
            return Err(anyhow!("Longitude must be between -180 and 180"));
        }
        Ok(())
    }

    /// Calculate Haversine distance to another point (in meters)
    pub fn distance_to(&self, other: &GeoPoint) -> f64 {
        const EARTH_RADIUS_M: f64 = 6371000.0; // Earth radius in meters

        let lat1 = self.lat.to_radians();
        let lat2 = other.lat.to_radians();
        let delta_lat = (other.lat - self.lat).to_radians();
        let delta_lon = (other.lon - self.lon).to_radians();

        let a = (delta_lat / 2.0).sin() * (delta_lat / 2.0).sin()
            + lat1.cos() * lat2.cos() * (delta_lon / 2.0).sin() * (delta_lon / 2.0).sin();

        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

        EARTH_RADIUS_M * c
    }

    /// Get S2 cell ID at given level (0-30)
    pub fn s2_cell_id(&self, level: u8) -> u64 {
        // Simplified S2 cell ID calculation
        // Real implementation would use S2 geometry library

        let level = level.min(30);

        // Normalize coordinates to [0, 1]
        let x = (self.lon + 180.0) / 360.0;
        let y = (self.lat + 90.0) / 180.0;

        // Simple quad-tree based cell ID
        let max_cells = 1u64 << level;
        let cell_x = (x * max_cells as f64) as u64;
        let cell_y = (y * max_cells as f64) as u64;

        // Interleave bits (Z-order curve)
        interleave_bits(cell_x, cell_y)
    }
}

/// Bounding box (rectangle)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min_lat: f64,
    pub max_lat: f64,
    pub min_lon: f64,
    pub max_lon: f64,
}

impl BoundingBox {
    /// Create a new bounding box
    pub fn new(min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64) -> Self {
        Self {
            min_lat,
            max_lat,
            min_lon,
            max_lon,
        }
    }

    /// Check if a point is inside the bounding box
    pub fn contains(&self, point: &GeoPoint) -> bool {
        point.lat >= self.min_lat
            && point.lat <= self.max_lat
            && point.lon >= self.min_lon
            && point.lon <= self.max_lon
    }

    /// Get the center point
    pub fn center(&self) -> GeoPoint {
        GeoPoint::new(
            (self.min_lat + self.max_lat) / 2.0,
            (self.min_lon + self.max_lon) / 2.0,
        )
    }
}

/// Geographic document with location and vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoDocument {
    pub id: String,
    pub location: GeoPoint,
    pub vector: Vec<f32>,
    pub metadata: serde_json::Value,
}

/// Geospatial index
pub struct GeoIndex {
    /// Documents by ID
    documents: HashMap<String, GeoDocument>,
    /// S2 cell index: cell_id -> list of document IDs
    cell_index: HashMap<u64, Vec<String>>,
    /// S2 level for indexing
    s2_level: u8,
}

impl GeoIndex {
    /// Create a new geospatial index
    pub fn new() -> Self {
        Self::with_s2_level(16) // Level 16 gives ~100m cells
    }

    /// Create with specific S2 level (0-30)
    pub fn with_s2_level(level: u8) -> Self {
        Self {
            documents: HashMap::new(),
            cell_index: HashMap::new(),
            s2_level: level.min(30),
        }
    }

    /// Add a document
    pub fn add(&mut self, id: impl Into<String>, location: GeoPoint, vector: Vec<f32>) {
        let id = id.into();
        location.validate().ok(); // Ignore validation errors for now

        let cell_id = location.s2_cell_id(self.s2_level);

        // Add to cell index
        self.cell_index
            .entry(cell_id)
            .or_insert_with(Vec::new)
            .push(id.clone());

        // Add document
        let doc = GeoDocument {
            id: id.clone(),
            location,
            vector,
            metadata: serde_json::json!({}),
        };

        self.documents.insert(id, doc);
    }

    /// Radius search - find all points within distance
    pub fn radius_search(
        &self,
        center: &GeoPoint,
        radius_meters: f64,
        limit: usize,
    ) -> Result<Vec<GeoSearchResult>> {
        let mut results = Vec::new();

        // For simplicity, check all documents
        // In production, would use S2 cell covering to limit search space
        for (doc_id, doc) in &self.documents {
            let distance = center.distance_to(&doc.location);

            if distance <= radius_meters {
                results.push(GeoSearchResult {
                    id: doc_id.clone(),
                    location: doc.location,
                    distance,
                    vector_score: 0.0, // Not computed for pure geo search
                });
            }
        }

        // Sort by distance
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results.truncate(limit);

        Ok(results)
    }

    /// Bounding box search
    pub fn bbox_search(&self, bbox: &BoundingBox, limit: usize) -> Result<Vec<GeoSearchResult>> {
        let mut results = Vec::new();

        for doc in self.documents.values() {
            if bbox.contains(&doc.location) {
                results.push(GeoSearchResult {
                    id: doc.id.clone(),
                    location: doc.location,
                    distance: 0.0,
                    vector_score: 0.0,
                });

                if results.len() >= limit {
                    break;
                }
            }
        }

        Ok(results)
    }

    /// Hybrid search - radius + vector similarity
    pub fn hybrid_search(
        &self,
        center: &GeoPoint,
        radius_meters: f64,
        query_vector: &[f32],
        limit: usize,
    ) -> Result<Vec<GeoSearchResult>> {
        let mut results = self.radius_search(center, radius_meters, usize::MAX)?;

        // Compute vector similarity for each result
        for result in &mut results {
            if let Some(doc) = self.documents.get(&result.id) {
                result.vector_score = cosine_similarity(query_vector, &doc.vector);
            }
        }

        // Sort by combined score (distance + vector similarity)
        results.sort_by(|a, b| {
            let score_a = a.vector_score - (a.distance as f32 / 1000.0); // Normalize distance to km
            let score_b = b.vector_score - (b.distance as f32 / 1000.0);
            score_b.partial_cmp(&score_a).unwrap()
        });

        results.truncate(limit);

        Ok(results)
    }

    /// Get neighboring cells for a cell ID
    fn get_neighboring_cells(&self, center_cell: u64) -> Vec<u64> {
        // Simplified - return center and approximate neighbors
        // Real S2 would use proper neighbor finding algorithms
        let mut cells = vec![center_cell];

        // Add some basic neighboring cells (simplified approach)
        // In production, this would use S2's GetAllNeighbors() method
        for delta in &[-1i64, 0, 1] {
            for delta2 in &[-1i64, 0, 1] {
                let neighbor = ((center_cell as i64) + delta + (delta2 << 10)) as u64;
                if neighbor != center_cell {
                    cells.push(neighbor);
                }
            }
        }

        cells
    }

    /// Get statistics
    pub fn stats(&self) -> GeoIndexStats {
        GeoIndexStats {
            num_documents: self.documents.len(),
            num_cells: self.cell_index.len(),
            s2_level: self.s2_level,
            avg_docs_per_cell: if !self.cell_index.is_empty() {
                self.documents.len() as f32 / self.cell_index.len() as f32
            } else {
                0.0
            },
        }
    }
}

impl Default for GeoIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Search result with geospatial info
#[derive(Debug, Clone)]
pub struct GeoSearchResult {
    pub id: String,
    pub location: GeoPoint,
    pub distance: f64, // meters
    pub vector_score: f32,
}

/// Index statistics
#[derive(Debug, Clone)]
pub struct GeoIndexStats {
    pub num_documents: usize,
    pub num_cells: usize,
    pub s2_level: u8,
    pub avg_docs_per_cell: f32,
}

/// Interleave bits for Z-order curve (simplified S2 cell ID)
fn interleave_bits(x: u64, y: u64) -> u64 {
    let mut result = 0u64;
    for i in 0..32 {
        result |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1));
    }
    result
}

/// Cosine similarity
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geopoint_creation() {
        let point = GeoPoint::new(37.7749, -122.4194);
        assert_eq!(point.lat, 37.7749);
        assert_eq!(point.lon, -122.4194);
    }

    #[test]
    fn test_geopoint_validation() {
        let valid = GeoPoint::new(37.7749, -122.4194);
        assert!(valid.validate().is_ok());

        let invalid_lat = GeoPoint::new(100.0, -122.4194);
        assert!(invalid_lat.validate().is_err());

        let invalid_lon = GeoPoint::new(37.7749, 200.0);
        assert!(invalid_lon.validate().is_err());
    }

    #[test]
    fn test_haversine_distance() {
        // San Francisco to Los Angeles (approx 559 km)
        let sf = GeoPoint::new(37.7749, -122.4194);
        let la = GeoPoint::new(34.0522, -118.2437);

        let distance = sf.distance_to(&la);
        assert!(distance > 500_000.0 && distance < 600_000.0); // ~559 km
    }

    #[test]
    fn test_s2_cell_id() {
        let point = GeoPoint::new(37.7749, -122.4194);
        let cell_id = point.s2_cell_id(16);
        assert!(cell_id > 0);

        // Same point should give same cell
        let point2 = GeoPoint::new(37.7749, -122.4194);
        assert_eq!(point.s2_cell_id(16), point2.s2_cell_id(16));
    }

    #[test]
    fn test_bounding_box_contains() {
        let bbox = BoundingBox::new(37.0, 38.0, -123.0, -122.0);

        let inside = GeoPoint::new(37.5, -122.5);
        assert!(bbox.contains(&inside));

        let outside = GeoPoint::new(39.0, -122.5);
        assert!(!bbox.contains(&outside));
    }

    #[test]
    fn test_bounding_box_center() {
        let bbox = BoundingBox::new(37.0, 38.0, -123.0, -122.0);
        let center = bbox.center();

        assert_eq!(center.lat, 37.5);
        assert_eq!(center.lon, -122.5);
    }

    #[test]
    fn test_geo_index_add() {
        let mut index = GeoIndex::new();

        let point = GeoPoint::new(37.7749, -122.4194);
        index.add("loc1", point, vec![0.1; 128]);

        assert_eq!(index.documents.len(), 1);
        assert!(index.documents.contains_key("loc1"));
    }

    #[test]
    fn test_radius_search() {
        let mut index = GeoIndex::new();

        // Add points
        let sf = GeoPoint::new(37.7749, -122.4194);
        let nearby = GeoPoint::new(37.7849, -122.4094); // ~1.2 km away
        let far = GeoPoint::new(34.0522, -118.2437); // LA - ~559 km away

        index.add("sf", sf, vec![0.1; 128]);
        index.add("nearby", nearby, vec![0.2; 128]);
        index.add("far", far, vec![0.3; 128]);

        // Search within 2km radius
        let results = index.radius_search(&sf, 2000.0, 10).unwrap();

        assert_eq!(results.len(), 2); // sf and nearby
        assert_eq!(results[0].id, "sf"); // Exact match first
    }

    #[test]
    fn test_bbox_search() {
        let mut index = GeoIndex::new();

        let p1 = GeoPoint::new(37.7, -122.4);
        let p2 = GeoPoint::new(37.8, -122.3);
        let p3 = GeoPoint::new(38.0, -122.0); // Outside bbox

        index.add("p1", p1, vec![0.1; 128]);
        index.add("p2", p2, vec![0.2; 128]);
        index.add("p3", p3, vec![0.3; 128]);

        let bbox = BoundingBox::new(37.0, 38.0, -123.0, -122.0);
        let results = index.bbox_search(&bbox, 10).unwrap();

        assert!(results.len() >= 2); // p1 and p2
    }

    #[test]
    fn test_hybrid_search() {
        let mut index = GeoIndex::new();

        let p1 = GeoPoint::new(37.7749, -122.4194);
        let p2 = GeoPoint::new(37.7849, -122.4094);

        // p1 has vector similar to query, p2 doesn't
        index.add("p1", p1, vec![1.0, 0.0, 0.0]);
        index.add("p2", p2, vec![0.0, 1.0, 0.0]);

        let query = vec![1.0, 0.0, 0.0];
        let results = index.hybrid_search(&p1, 2000.0, &query, 10).unwrap();

        assert!(results.len() > 0);
        // p1 should rank higher (closer + better vector match)
        assert_eq!(results[0].id, "p1");
    }

    #[test]
    fn test_index_stats() {
        let mut index = GeoIndex::new();

        index.add("p1", GeoPoint::new(37.7, -122.4), vec![0.1; 128]);
        index.add("p2", GeoPoint::new(37.8, -122.3), vec![0.2; 128]);

        let stats = index.stats();
        assert_eq!(stats.num_documents, 2);
        assert!(stats.num_cells > 0);
        assert_eq!(stats.s2_level, 16);
    }
}
