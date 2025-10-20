//! Advanced filtering capabilities
//!
//! This module extends the basic filtering system with:
//! - Geospatial queries (radius, bounding box)
//! - Date range filtering
//! - Numeric range queries
//! - Array/list operations

use super::types::*;
use serde_json::Value as JsonValue;

/// Geospatial point (latitude, longitude)
#[derive(Debug, Clone, Copy)]
pub struct GeoPoint {
    pub lat: f64,
    pub lon: f64,
}

impl GeoPoint {
    pub fn new(lat: f64, lon: f64) -> Self {
        Self { lat, lon }
    }

    /// Calculate distance to another point using Haversine formula (in kilometers)
    pub fn distance_to(&self, other: &GeoPoint) -> f64 {
        let r = 6371.0; // Earth radius in km
        let lat1 = self.lat.to_radians();
        let lat2 = other.lat.to_radians();
        let delta_lat = (other.lat - self.lat).to_radians();
        let delta_lon = (other.lon - self.lon).to_radians();

        let a = (delta_lat / 2.0).sin().powi(2)
            + lat1.cos() * lat2.cos() * (delta_lon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

        r * c
    }
}

/// Geospatial filters
pub enum GeoFilter {
    /// Within radius (center, radius_km)
    Radius { center: GeoPoint, radius_km: f64 },

    /// Within bounding box (southwest corner, northeast corner)
    BoundingBox { sw: GeoPoint, ne: GeoPoint },
}

impl GeoFilter {
    /// Check if a point matches this geo filter
    pub fn matches(&self, point: &GeoPoint) -> bool {
        match self {
            GeoFilter::Radius { center, radius_km } => center.distance_to(point) <= *radius_km,
            GeoFilter::BoundingBox { sw, ne } => {
                point.lat >= sw.lat
                    && point.lat <= ne.lat
                    && point.lon >= sw.lon
                    && point.lon <= ne.lon
            }
        }
    }

    /// Extract GeoPoint from metadata
    ///
    /// Expects metadata with "lat" and "lon" fields
    pub fn extract_point(metadata: &Metadata) -> Option<GeoPoint> {
        let lat = metadata.fields.get("lat")?.as_f64()?;
        let lon = metadata.fields.get("lon")?.as_f64()?;
        Some(GeoPoint::new(lat, lon))
    }
}

/// Date range filter
pub struct DateRangeFilter {
    pub start: Option<i64>, // Unix timestamp (seconds)
    pub end: Option<i64>,
}

impl DateRangeFilter {
    /// Create a filter for dates after a timestamp
    pub fn after(timestamp: i64) -> Self {
        Self {
            start: Some(timestamp),
            end: None,
        }
    }

    /// Create a filter for dates before a timestamp
    pub fn before(timestamp: i64) -> Self {
        Self {
            start: None,
            end: Some(timestamp),
        }
    }

    /// Create a filter for dates between two timestamps
    pub fn between(start: i64, end: i64) -> Self {
        Self {
            start: Some(start),
            end: Some(end),
        }
    }

    /// Check if a timestamp matches this filter
    pub fn matches(&self, timestamp: i64) -> bool {
        if let Some(start) = self.start {
            if timestamp < start {
                return false;
            }
        }
        if let Some(end) = self.end {
            if timestamp > end {
                return false;
            }
        }
        true
    }

    /// Extract timestamp from metadata field
    pub fn extract_timestamp(metadata: &Metadata, field: &str) -> Option<i64> {
        metadata.fields.get(field)?.as_i64()
    }
}

/// Numeric range filter
pub struct NumericRangeFilter {
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl NumericRangeFilter {
    pub fn new(min: Option<f64>, max: Option<f64>) -> Self {
        Self { min, max }
    }

    pub fn matches(&self, value: f64) -> bool {
        if let Some(min) = self.min {
            if value < min {
                return false;
            }
        }
        if let Some(max) = self.max {
            if value > max {
                return false;
            }
        }
        true
    }
}

/// Array/list operations
pub enum ArrayFilter {
    /// Check if array contains a value
    Contains(JsonValue),

    /// Check if array contains all values
    ContainsAll(Vec<JsonValue>),

    /// Check if array contains any of the values
    ContainsAny(Vec<JsonValue>),

    /// Check array length
    Length {
        min: Option<usize>,
        max: Option<usize>,
    },
}

impl ArrayFilter {
    pub fn matches(&self, array: &JsonValue) -> bool {
        let arr = match array.as_array() {
            Some(a) => a,
            None => return false,
        };

        match self {
            ArrayFilter::Contains(val) => arr.contains(val),

            ArrayFilter::ContainsAll(values) => values.iter().all(|v| arr.contains(v)),

            ArrayFilter::ContainsAny(values) => values.iter().any(|v| arr.contains(v)),

            ArrayFilter::Length { min, max } => {
                let len = arr.len();
                if let Some(min_len) = min {
                    if len < *min_len {
                        return false;
                    }
                }
                if let Some(max_len) = max {
                    if len > *max_len {
                        return false;
                    }
                }
                true
            }
        }
    }
}

/// Combined advanced filter
pub enum AdvancedFilter {
    Geo(GeoFilter),
    DateRange {
        field: String,
        filter: DateRangeFilter,
    },
    NumericRange {
        field: String,
        filter: NumericRangeFilter,
    },
    Array {
        field: String,
        filter: ArrayFilter,
    },
}

impl AdvancedFilter {
    /// Check if metadata matches this advanced filter
    pub fn matches(&self, metadata: &Metadata) -> bool {
        match self {
            AdvancedFilter::Geo(geo_filter) => {
                if let Some(point) = GeoFilter::extract_point(metadata) {
                    geo_filter.matches(&point)
                } else {
                    false
                }
            }

            AdvancedFilter::DateRange { field, filter } => {
                if let Some(timestamp) = DateRangeFilter::extract_timestamp(metadata, field) {
                    filter.matches(timestamp)
                } else {
                    false
                }
            }

            AdvancedFilter::NumericRange { field, filter } => {
                if let Some(value) = metadata.fields.get(field).and_then(|v| v.as_f64()) {
                    filter.matches(value)
                } else {
                    false
                }
            }

            AdvancedFilter::Array { field, filter } => {
                if let Some(array) = metadata.fields.get(field) {
                    filter.matches(array)
                } else {
                    false
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    fn make_metadata(fields: HashMap<String, serde_json::Value>) -> Metadata {
        Metadata { fields }
    }

    #[test]
    fn test_geo_radius() {
        let center = GeoPoint::new(40.7128, -74.0060); // NYC
        let filter = GeoFilter::Radius {
            center,
            radius_km: 100.0,
        };

        let nearby = GeoPoint::new(40.7589, -73.9851); // Times Square
        assert!(filter.matches(&nearby));

        let far = GeoPoint::new(34.0522, -118.2437); // LA
        assert!(!filter.matches(&far));
    }

    #[test]
    fn test_geo_bbox() {
        let sw = GeoPoint::new(40.0, -75.0);
        let ne = GeoPoint::new(41.0, -73.0);
        let filter = GeoFilter::BoundingBox { sw, ne };

        let inside = GeoPoint::new(40.5, -74.0);
        assert!(filter.matches(&inside));

        let outside = GeoPoint::new(42.0, -74.0);
        assert!(!filter.matches(&outside));
    }

    #[test]
    fn test_date_range() {
        let filter = DateRangeFilter::between(100, 200);

        assert!(!filter.matches(50));
        assert!(filter.matches(150));
        assert!(!filter.matches(250));
    }

    #[test]
    fn test_numeric_range() {
        let filter = NumericRangeFilter::new(Some(10.0), Some(20.0));

        assert!(!filter.matches(5.0));
        assert!(filter.matches(15.0));
        assert!(!filter.matches(25.0));
    }

    #[test]
    fn test_array_contains() {
        let filter = ArrayFilter::Contains(json!("rust"));
        let array = json!(["rust", "python", "go"]);

        assert!(filter.matches(&array));

        let array2 = json!(["java", "cpp"]);
        assert!(!filter.matches(&array2));
    }

    #[test]
    fn test_array_contains_all() {
        let filter = ArrayFilter::ContainsAll(vec![json!("rust"), json!("python")]);
        let array = json!(["rust", "python", "go"]);

        assert!(filter.matches(&array));

        let array2 = json!(["rust", "go"]);
        assert!(!filter.matches(&array2));
    }

    #[test]
    fn test_advanced_filter_geo() {
        let mut fields = HashMap::new();
        fields.insert("lat".to_string(), json!(40.7128));
        fields.insert("lon".to_string(), json!(-74.0060));
        let metadata = make_metadata(fields);

        let center = GeoPoint::new(40.7128, -74.0060);
        let filter = AdvancedFilter::Geo(GeoFilter::Radius {
            center,
            radius_km: 1.0,
        });

        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_advanced_filter_date_range() {
        let mut fields = HashMap::new();
        fields.insert("created_at".to_string(), json!(150));
        let metadata = make_metadata(fields);

        let filter = AdvancedFilter::DateRange {
            field: "created_at".to_string(),
            filter: DateRangeFilter::between(100, 200),
        };

        assert!(filter.matches(&metadata));
    }
}
