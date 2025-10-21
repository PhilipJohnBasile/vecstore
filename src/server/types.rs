//! Type conversions between protobuf and vecstore types

use crate::namespace::{Namespace, NamespaceQuotas, NamespaceStatus};
use crate::store::{Metadata, Neighbor, Query};
use anyhow::Result;
use std::collections::HashMap;

// Re-export generated protobuf types
pub mod pb {
    include!("../generated/vecstore.rs");
}

/// Convert protobuf Value to serde_json::Value
pub fn pb_value_to_json(value: &pb::Value) -> Result<serde_json::Value> {
    use pb::value::Kind;

    match &value.kind {
        Some(Kind::StringValue(s)) => Ok(serde_json::Value::String(s.clone())),
        Some(Kind::NumberValue(n)) => Ok(serde_json::json!(n)),
        Some(Kind::BoolValue(b)) => Ok(serde_json::Value::Bool(*b)),
        Some(Kind::ArrayValue(arr)) => {
            let values: Result<Vec<_>> = arr.values.iter().map(pb_value_to_json).collect();
            Ok(serde_json::Value::Array(values?))
        }
        Some(Kind::ObjectValue(obj)) => {
            let mut map = serde_json::Map::new();
            for (k, v) in &obj.fields {
                map.insert(k.clone(), pb_value_to_json(v)?);
            }
            Ok(serde_json::Value::Object(map))
        }
        Some(Kind::NullValue(_)) => Ok(serde_json::Value::Null),
        None => Ok(serde_json::Value::Null),
    }
}

/// Convert serde_json::Value to protobuf Value
pub fn json_to_pb_value(value: &serde_json::Value) -> pb::Value {
    use pb::value::Kind;

    let kind = match value {
        serde_json::Value::String(s) => Some(Kind::StringValue(s.clone())),
        serde_json::Value::Number(n) => Some(Kind::NumberValue(n.as_f64().unwrap_or(0.0))),
        serde_json::Value::Bool(b) => Some(Kind::BoolValue(*b)),
        serde_json::Value::Array(arr) => {
            let values = arr.iter().map(json_to_pb_value).collect();
            Some(Kind::ArrayValue(pb::ArrayValue { values }))
        }
        serde_json::Value::Object(obj) => {
            let fields = obj
                .iter()
                .map(|(k, v)| (k.clone(), json_to_pb_value(v)))
                .collect();
            Some(Kind::ObjectValue(pb::ObjectValue { fields }))
        }
        serde_json::Value::Null => Some(Kind::NullValue(pb::NullValue::NullValue as i32)),
    };

    pb::Value { kind }
}

/// Convert protobuf metadata map to Metadata
pub fn pb_metadata_to_metadata(pb_meta: &HashMap<String, pb::Value>) -> Result<Metadata> {
    let mut fields = HashMap::new();

    for (key, pb_value) in pb_meta {
        fields.insert(key.clone(), pb_value_to_json(pb_value)?);
    }

    Ok(Metadata { fields })
}

/// Convert Metadata to protobuf metadata map
pub fn metadata_to_pb_metadata(metadata: &Metadata) -> HashMap<String, pb::Value> {
    metadata
        .fields
        .iter()
        .map(|(k, v)| (k.clone(), json_to_pb_value(v)))
        .collect()
}

/// Convert Neighbor to protobuf QueryResult
pub fn neighbor_to_query_result(neighbor: &Neighbor) -> pb::QueryResult {
    pb::QueryResult {
        id: neighbor.id.clone(),
        score: neighbor.score,
        metadata: metadata_to_pb_metadata(&neighbor.metadata),
    }
}

/// Convert protobuf QueryRequest to Query
pub fn pb_query_to_query(req: &pb::QueryRequest) -> Result<Query> {
    let filter = if let Some(ref filter_str) = req.filter {
        Some(crate::store::parse_filter(filter_str)?)
    } else {
        None
    };

    Ok(Query {
        vector: req.vector.clone(),
        k: req.limit as usize,
        filter,
    })
}

// ==================== Namespace Type Conversions ====================

/// Convert Namespace to protobuf NamespaceInfo
pub fn namespace_info_to_proto(ns: &Namespace) -> pb::NamespaceInfo {
    pb::NamespaceInfo {
        id: ns.id.clone(),
        name: ns.name.clone(),
        description: ns.description.clone(),
        quotas: Some(namespace_quotas_to_proto(&ns.quotas)),
        status: namespace_status_to_proto(ns.status) as i32,
        created_at: ns.created_at as i64,
        updated_at: ns.updated_at as i64,
        metadata: ns.metadata.clone(),
    }
}

/// Convert NamespaceQuotas to protobuf
pub fn namespace_quotas_to_proto(quotas: &NamespaceQuotas) -> pb::NamespaceQuotas {
    pb::NamespaceQuotas {
        max_vectors: quotas.max_vectors.map(|v| v as i64),
        max_storage_bytes: quotas.max_storage_bytes.map(|v| v as i64),
        max_requests_per_second: quotas.max_requests_per_second,
        max_concurrent_queries: quotas.max_concurrent_queries.map(|v| v as i32),
        max_dimension: quotas.max_dimension.map(|v| v as i32),
        max_results_per_query: quotas.max_results_per_query.map(|v| v as i32),
        max_batch_size: quotas.max_batch_size.map(|v| v as i32),
    }
}

/// Convert protobuf NamespaceQuotas to Rust type
pub fn namespace_quotas_from_proto(pb: pb::NamespaceQuotas) -> Result<NamespaceQuotas> {
    Ok(NamespaceQuotas {
        max_vectors: pb.max_vectors.map(|v| v as usize),
        max_storage_bytes: pb.max_storage_bytes.map(|v| v as u64),
        max_requests_per_second: pb.max_requests_per_second,
        max_concurrent_queries: pb.max_concurrent_queries.map(|v| v as usize),
        max_dimension: pb.max_dimension.map(|v| v as usize),
        max_results_per_query: pb.max_results_per_query.map(|v| v as usize),
        max_batch_size: pb.max_batch_size.map(|v| v as usize),
    })
}

/// Convert NamespaceStatus to protobuf enum
pub fn namespace_status_to_proto(status: NamespaceStatus) -> pb::NamespaceStatus {
    match status {
        NamespaceStatus::Active => pb::NamespaceStatus::NamespaceActive,
        NamespaceStatus::Suspended => pb::NamespaceStatus::NamespaceSuspended,
        NamespaceStatus::ReadOnly => pb::NamespaceStatus::NamespaceReadOnly,
        NamespaceStatus::PendingDeletion => pb::NamespaceStatus::NamespacePendingDeletion,
    }
}

/// Convert protobuf NamespaceStatus to Rust enum
pub fn namespace_status_from_proto(status: i32) -> Option<NamespaceStatus> {
    match pb::NamespaceStatus::try_from(status).ok()? {
        pb::NamespaceStatus::NamespaceActive => Some(NamespaceStatus::Active),
        pb::NamespaceStatus::NamespaceSuspended => Some(NamespaceStatus::Suspended),
        pb::NamespaceStatus::NamespaceReadOnly => Some(NamespaceStatus::ReadOnly),
        pb::NamespaceStatus::NamespacePendingDeletion => Some(NamespaceStatus::PendingDeletion),
    }
}
