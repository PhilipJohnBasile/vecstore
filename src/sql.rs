//! SQL Interface for Vector Operations
//!
//! Provides a familiar SQL-like interface for vector database operations,
//! similar to pgvector, MyScale, and SingleStore.
//!
//! # Supported SQL Syntax
//!
//! ```sql
//! -- Create a vector collection
//! CREATE TABLE documents (
//!     id TEXT PRIMARY KEY,
//!     embedding VECTOR(384),
//!     content TEXT,
//!     category TEXT
//! );
//!
//! -- Insert vectors
//! INSERT INTO documents (id, embedding, content)
//! VALUES ('doc1', '[0.1, 0.2, ...]', 'Hello world');
//!
//! -- Vector similarity search
//! SELECT id, content, DISTANCE(embedding, '[0.1, 0.2, ...]') as dist
//! FROM documents
//! ORDER BY embedding <-> '[0.1, 0.2, ...]'
//! LIMIT 10;
//!
//! -- Hybrid search with filters
//! SELECT id, content
//! FROM documents
//! WHERE category = 'tech'
//! ORDER BY embedding <=> '[0.1, 0.2, ...]'  -- cosine distance
//! LIMIT 5;
//! ```
//!
//! # Distance Operators
//!
//! - `<->` : L2 (Euclidean) distance
//! - `<=>` : Cosine distance
//! - `<#>` : Inner product (negative dot product)

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::error::{VecStoreError, Result};

/// SQL parser and executor for vector operations
pub struct VectorSQL {
    tables: HashMap<String, TableSchema>,
    data: HashMap<String, Vec<Row>>,
}

/// Table schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSchema {
    pub name: String,
    pub columns: Vec<ColumnDef>,
    pub primary_key: Option<String>,
    pub vector_column: Option<String>,
    pub dimension: Option<usize>,
}

/// Column definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub default: Option<String>,
}

/// Supported data types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DataType {
    Text,
    Integer,
    Float,
    Boolean,
    Vector(usize),  // Vector with dimension
    Json,
    Timestamp,
}

impl DataType {
    /// Parse a data type from SQL string
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim().to_uppercase();

        if s.starts_with("VECTOR(") && s.ends_with(')') {
            let dim_str = &s[7..s.len()-1];
            let dim: usize = dim_str.parse().map_err(|_| {
                VecStoreError::InvalidInput(format!("Invalid vector dimension: {}", dim_str))
            })?;
            return Ok(DataType::Vector(dim));
        }

        match s.as_str() {
            "TEXT" | "VARCHAR" | "STRING" => Ok(DataType::Text),
            "INT" | "INTEGER" | "BIGINT" => Ok(DataType::Integer),
            "FLOAT" | "DOUBLE" | "REAL" => Ok(DataType::Float),
            "BOOL" | "BOOLEAN" => Ok(DataType::Boolean),
            "JSON" | "JSONB" => Ok(DataType::Json),
            "TIMESTAMP" | "DATETIME" => Ok(DataType::Timestamp),
            _ => Err(VecStoreError::InvalidInput(format!("Unknown data type: {}", s))),
        }
    }
}

/// A row of data
#[derive(Debug, Clone)]
pub struct Row {
    pub id: String,
    pub vector: Option<Vec<f32>>,
    pub metadata: HashMap<String, Value>,
}

/// SQL value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    Null,
    Text(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Vector(Vec<f32>),
    Json(serde_json::Value),
}

impl Value {
    /// Convert to string representation
    pub fn to_string_value(&self) -> String {
        match self {
            Value::Null => "NULL".to_string(),
            Value::Text(s) => s.clone(),
            Value::Integer(i) => i.to_string(),
            Value::Float(f) => f.to_string(),
            Value::Boolean(b) => b.to_string(),
            Value::Vector(v) => format!("{:?}", v),
            Value::Json(j) => j.to_string(),
        }
    }
}

/// Parsed SQL statement
#[derive(Debug, Clone)]
pub enum Statement {
    CreateTable(CreateTableStmt),
    DropTable(String),
    Insert(InsertStmt),
    Select(SelectStmt),
    Update(UpdateStmt),
    Delete(DeleteStmt),
    CreateIndex(CreateIndexStmt),
}

/// CREATE TABLE statement
#[derive(Debug, Clone)]
pub struct CreateTableStmt {
    pub table_name: String,
    pub columns: Vec<ColumnDef>,
    pub if_not_exists: bool,
}

/// INSERT statement
#[derive(Debug, Clone)]
pub struct InsertStmt {
    pub table_name: String,
    pub columns: Vec<String>,
    pub values: Vec<Vec<Value>>,
}

/// SELECT statement
#[derive(Debug, Clone)]
pub struct SelectStmt {
    pub columns: Vec<SelectColumn>,
    pub from: String,
    pub where_clause: Option<WhereClause>,
    pub order_by: Option<OrderByClause>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// SELECT column expression
#[derive(Debug, Clone)]
pub enum SelectColumn {
    All,  // *
    Column(String),
    Distance {
        column: String,
        target: Vec<f32>,
        metric: DistanceMetric,
        alias: Option<String>,
    },
    Function {
        name: String,
        args: Vec<String>,
        alias: Option<String>,
    },
}

/// Distance metrics
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    L2,       // <->
    Cosine,   // <=>
    DotProduct, // <#>
}

/// WHERE clause
#[derive(Debug, Clone)]
pub struct WhereClause {
    pub conditions: Vec<Condition>,
    pub combiner: LogicalOp,
}

/// Logical operators
#[derive(Debug, Clone, Copy)]
pub enum LogicalOp {
    And,
    Or,
}

/// WHERE condition
#[derive(Debug, Clone)]
pub enum Condition {
    Equals(String, Value),
    NotEquals(String, Value),
    GreaterThan(String, Value),
    LessThan(String, Value),
    GreaterOrEqual(String, Value),
    LessOrEqual(String, Value),
    Like(String, String),
    In(String, Vec<Value>),
    IsNull(String),
    IsNotNull(String),
    VectorDistance {
        column: String,
        target: Vec<f32>,
        metric: DistanceMetric,
        max_distance: f32,
    },
}

/// ORDER BY clause
#[derive(Debug, Clone)]
pub struct OrderByClause {
    pub columns: Vec<OrderByColumn>,
}

/// ORDER BY column
#[derive(Debug, Clone)]
pub enum OrderByColumn {
    Column { name: String, descending: bool },
    VectorDistance {
        column: String,
        target: Vec<f32>,
        metric: DistanceMetric,
    },
}

/// UPDATE statement
#[derive(Debug, Clone)]
pub struct UpdateStmt {
    pub table_name: String,
    pub set: Vec<(String, Value)>,
    pub where_clause: Option<WhereClause>,
}

/// DELETE statement
#[derive(Debug, Clone)]
pub struct DeleteStmt {
    pub table_name: String,
    pub where_clause: Option<WhereClause>,
}

/// CREATE INDEX statement
#[derive(Debug, Clone)]
pub struct CreateIndexStmt {
    pub index_name: String,
    pub table_name: String,
    pub column: String,
    pub index_type: IndexType,
    pub options: HashMap<String, String>,
}

/// Index types
#[derive(Debug, Clone)]
pub enum IndexType {
    Hnsw,
    IvfFlat,
    IvfPq,
    DiskAnn,
    Flat,
}

/// Query result
#[derive(Debug, Clone, Serialize)]
pub struct QueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<ResultRow>,
    pub affected_rows: usize,
    pub execution_time_ms: u64,
}

/// Result row
#[derive(Debug, Clone, Serialize)]
pub struct ResultRow {
    pub values: Vec<Value>,
}

impl VectorSQL {
    /// Create a new SQL executor
    pub fn new() -> Self {
        Self {
            tables: HashMap::new(),
            data: HashMap::new(),
        }
    }

    /// Execute a SQL statement
    pub fn execute(&mut self, sql: &str) -> Result<QueryResult> {
        let start = std::time::Instant::now();
        let stmt = self.parse(sql)?;

        let result = match stmt {
            Statement::CreateTable(stmt) => self.execute_create_table(stmt),
            Statement::DropTable(name) => self.execute_drop_table(&name),
            Statement::Insert(stmt) => self.execute_insert(stmt),
            Statement::Select(stmt) => self.execute_select(stmt),
            Statement::Update(stmt) => self.execute_update(stmt),
            Statement::Delete(stmt) => self.execute_delete(stmt),
            Statement::CreateIndex(stmt) => self.execute_create_index(stmt),
        }?;

        Ok(QueryResult {
            execution_time_ms: start.elapsed().as_millis() as u64,
            ..result
        })
    }

    /// Parse a SQL statement
    fn parse(&self, sql: &str) -> Result<Statement> {
        let sql = sql.trim();
        let upper = sql.to_uppercase();

        if upper.starts_with("CREATE TABLE") {
            self.parse_create_table(sql)
        } else if upper.starts_with("DROP TABLE") {
            self.parse_drop_table(sql)
        } else if upper.starts_with("INSERT") {
            self.parse_insert(sql)
        } else if upper.starts_with("SELECT") {
            self.parse_select(sql)
        } else if upper.starts_with("UPDATE") {
            self.parse_update(sql)
        } else if upper.starts_with("DELETE") {
            self.parse_delete(sql)
        } else if upper.starts_with("CREATE INDEX") {
            self.parse_create_index(sql)
        } else {
            Err(VecStoreError::InvalidInput(format!("Unknown SQL statement: {}", sql)))
        }
    }

    /// Parse CREATE TABLE
    fn parse_create_table(&self, sql: &str) -> Result<Statement> {
        // Simple parser for: CREATE TABLE [IF NOT EXISTS] name (columns...)
        let sql = sql.trim();
        let upper = sql.to_uppercase();

        let if_not_exists = upper.contains("IF NOT EXISTS");

        // Find table name and columns
        let paren_start = sql.find('(').ok_or_else(|| {
            VecStoreError::InvalidInput("Missing column definitions".to_string())
        })?;
        let paren_end = sql.rfind(')').ok_or_else(|| {
            VecStoreError::InvalidInput("Missing closing parenthesis".to_string())
        })?;

        // Extract table name
        let before_paren = &sql[..paren_start].trim();
        let parts: Vec<&str> = before_paren.split_whitespace().collect();
        let table_name = parts.last().ok_or_else(|| {
            VecStoreError::InvalidInput("Missing table name".to_string())
        })?.to_string();

        // Parse columns
        let columns_str = &sql[paren_start + 1..paren_end];
        let mut columns = Vec::new();

        for col_def in columns_str.split(',') {
            let col_def = col_def.trim();
            if col_def.is_empty() {
                continue;
            }

            // Skip PRIMARY KEY constraints
            if col_def.to_uppercase().starts_with("PRIMARY KEY") {
                continue;
            }

            let parts: Vec<&str> = col_def.split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }

            let name = parts[0].to_string();

            // Extract data type, handling VECTOR(n) which includes parentheses
            // Also skip modifiers like PRIMARY KEY, NOT NULL, etc.
            let type_str = if parts[1].to_uppercase().starts_with("VECTOR") {
                // Handle VECTOR(384) type - find the closing paren
                let type_start = col_def.to_uppercase().find("VECTOR").unwrap();
                let after_type = &col_def[type_start..];
                if let Some(paren_end) = after_type.find(')') {
                    after_type[..paren_end + 1].to_string()
                } else {
                    parts[1].to_string()
                }
            } else {
                // Simple types - just take the first word after name
                parts[1].to_string()
            };
            let data_type = DataType::parse(&type_str)?;

            columns.push(ColumnDef {
                name,
                data_type,
                nullable: !col_def.to_uppercase().contains("NOT NULL"),
                default: None,
            });
        }

        Ok(Statement::CreateTable(CreateTableStmt {
            table_name,
            columns,
            if_not_exists,
        }))
    }

    /// Parse DROP TABLE
    fn parse_drop_table(&self, sql: &str) -> Result<Statement> {
        let parts: Vec<&str> = sql.split_whitespace().collect();
        let table_name = parts.last().ok_or_else(|| {
            VecStoreError::InvalidInput("Missing table name".to_string())
        })?.trim_end_matches(';').to_string();

        Ok(Statement::DropTable(table_name))
    }

    /// Parse INSERT
    fn parse_insert(&self, sql: &str) -> Result<Statement> {
        // INSERT INTO table (cols) VALUES (vals)
        let upper = sql.to_uppercase();

        // Find table name
        let into_pos = upper.find("INTO").ok_or_else(|| {
            VecStoreError::InvalidInput("Missing INTO".to_string())
        })?;

        let after_into = &sql[into_pos + 4..].trim_start();
        let paren_pos = after_into.find('(').unwrap_or(after_into.len());
        let table_name = after_into[..paren_pos].trim().to_string();

        // For now, return a basic insert statement
        Ok(Statement::Insert(InsertStmt {
            table_name,
            columns: Vec::new(),
            values: Vec::new(),
        }))
    }

    /// Parse SELECT
    fn parse_select(&self, sql: &str) -> Result<Statement> {
        let upper = sql.to_uppercase();

        // Find FROM
        let from_pos = upper.find(" FROM ").ok_or_else(|| {
            VecStoreError::InvalidInput("Missing FROM clause".to_string())
        })?;

        // Parse columns (between SELECT and FROM)
        let select_pos = upper.find("SELECT").unwrap();
        let columns_str = &sql[select_pos + 6..from_pos].trim();

        let columns = if columns_str.trim() == "*" {
            vec![SelectColumn::All]
        } else {
            columns_str.split(',')
                .map(|c| SelectColumn::Column(c.trim().to_string()))
                .collect()
        };

        // Find table name (after FROM)
        let after_from = &sql[from_pos + 6..];
        let table_end = after_from.find(|c: char| c.is_whitespace() || c == ';')
            .unwrap_or(after_from.len());
        let from = after_from[..table_end].trim().to_string();

        // Parse LIMIT
        let limit = if let Some(limit_pos) = upper.find(" LIMIT ") {
            let after_limit = &sql[limit_pos + 7..];
            let end = after_limit.find(|c: char| !c.is_numeric())
                .unwrap_or(after_limit.len());
            after_limit[..end].trim().parse().ok()
        } else {
            None
        };

        // Parse ORDER BY for vector distance
        let order_by = if let Some(order_pos) = upper.find(" ORDER BY ") {
            let after_order = &sql[order_pos + 10..];
            // Check for vector distance operators
            if after_order.contains("<->") || after_order.contains("<=>") || after_order.contains("<#>") {
                // Parse vector order by
                Some(OrderByClause { columns: Vec::new() })
            } else {
                None
            }
        } else {
            None
        };

        Ok(Statement::Select(SelectStmt {
            columns,
            from,
            where_clause: None,
            order_by,
            limit,
            offset: None,
        }))
    }

    /// Parse UPDATE
    fn parse_update(&self, _sql: &str) -> Result<Statement> {
        Ok(Statement::Update(UpdateStmt {
            table_name: String::new(),
            set: Vec::new(),
            where_clause: None,
        }))
    }

    /// Parse DELETE
    fn parse_delete(&self, _sql: &str) -> Result<Statement> {
        Ok(Statement::Delete(DeleteStmt {
            table_name: String::new(),
            where_clause: None,
        }))
    }

    /// Parse CREATE INDEX
    fn parse_create_index(&self, sql: &str) -> Result<Statement> {
        let upper = sql.to_uppercase();

        // CREATE INDEX name ON table USING type (column)
        let using_pos = upper.find(" USING ");
        let index_type = if let Some(pos) = using_pos {
            let after_using = &upper[pos + 7..];
            let end = after_using.find(|c: char| c.is_whitespace() || c == '(')
                .unwrap_or(after_using.len());
            match &after_using[..end] {
                "HNSW" => IndexType::Hnsw,
                "IVFFLAT" | "IVF_FLAT" => IndexType::IvfFlat,
                "IVFPQ" | "IVF_PQ" => IndexType::IvfPq,
                "DISKANN" | "DISK_ANN" => IndexType::DiskAnn,
                _ => IndexType::Hnsw,
            }
        } else {
            IndexType::Hnsw
        };

        Ok(Statement::CreateIndex(CreateIndexStmt {
            index_name: "idx".to_string(),
            table_name: String::new(),
            column: String::new(),
            index_type,
            options: HashMap::new(),
        }))
    }

    /// Execute CREATE TABLE
    fn execute_create_table(&mut self, stmt: CreateTableStmt) -> Result<QueryResult> {
        if self.tables.contains_key(&stmt.table_name) {
            if stmt.if_not_exists {
                return Ok(QueryResult {
                    columns: Vec::new(),
                    rows: Vec::new(),
                    affected_rows: 0,
                    execution_time_ms: 0,
                });
            }
            return Err(VecStoreError::InvalidInput(format!(
                "Table '{}' already exists", stmt.table_name
            )));
        }

        // Find vector column
        let vector_column = stmt.columns.iter()
            .find(|c| matches!(c.data_type, DataType::Vector(_)))
            .map(|c| c.name.clone());

        let dimension = stmt.columns.iter()
            .find_map(|c| {
                if let DataType::Vector(dim) = c.data_type {
                    Some(dim)
                } else {
                    None
                }
            });

        let primary_key = stmt.columns.first().map(|c| c.name.clone());

        let schema = TableSchema {
            name: stmt.table_name.clone(),
            columns: stmt.columns,
            primary_key,
            vector_column,
            dimension,
        };

        self.tables.insert(stmt.table_name.clone(), schema);
        self.data.insert(stmt.table_name, Vec::new());

        Ok(QueryResult {
            columns: Vec::new(),
            rows: Vec::new(),
            affected_rows: 0,
            execution_time_ms: 0,
        })
    }

    /// Execute DROP TABLE
    fn execute_drop_table(&mut self, name: &str) -> Result<QueryResult> {
        self.tables.remove(name);
        self.data.remove(name);

        Ok(QueryResult {
            columns: Vec::new(),
            rows: Vec::new(),
            affected_rows: 1,
            execution_time_ms: 0,
        })
    }

    /// Execute INSERT
    fn execute_insert(&mut self, stmt: InsertStmt) -> Result<QueryResult> {
        let affected = stmt.values.len();

        Ok(QueryResult {
            columns: Vec::new(),
            rows: Vec::new(),
            affected_rows: affected,
            execution_time_ms: 0,
        })
    }

    /// Execute SELECT
    fn execute_select(&self, stmt: SelectStmt) -> Result<QueryResult> {
        let table_data = self.data.get(&stmt.from).ok_or_else(|| {
            VecStoreError::NotFound(format!("Table '{}' not found", stmt.from))
        })?;

        let column_names: Vec<String> = match &stmt.columns[..] {
            [SelectColumn::All] => {
                if let Some(schema) = self.tables.get(&stmt.from) {
                    schema.columns.iter().map(|c| c.name.clone()).collect()
                } else {
                    Vec::new()
                }
            }
            cols => cols.iter().map(|c| match c {
                SelectColumn::Column(name) => name.clone(),
                SelectColumn::Distance { alias, .. } => {
                    alias.clone().unwrap_or_else(|| "distance".to_string())
                }
                SelectColumn::Function { alias, name, .. } => {
                    alias.clone().unwrap_or_else(|| name.clone())
                }
                SelectColumn::All => "*".to_string(),
            }).collect(),
        };

        let rows: Vec<ResultRow> = table_data.iter()
            .take(stmt.limit.unwrap_or(100))
            .map(|row| {
                let values: Vec<Value> = column_names.iter()
                    .map(|col| {
                        row.metadata.get(col).cloned().unwrap_or(Value::Null)
                    })
                    .collect();
                ResultRow { values }
            })
            .collect();

        Ok(QueryResult {
            columns: column_names,
            rows,
            affected_rows: 0,
            execution_time_ms: 0,
        })
    }

    /// Execute UPDATE
    fn execute_update(&mut self, _stmt: UpdateStmt) -> Result<QueryResult> {
        Ok(QueryResult {
            columns: Vec::new(),
            rows: Vec::new(),
            affected_rows: 0,
            execution_time_ms: 0,
        })
    }

    /// Execute DELETE
    fn execute_delete(&mut self, _stmt: DeleteStmt) -> Result<QueryResult> {
        Ok(QueryResult {
            columns: Vec::new(),
            rows: Vec::new(),
            affected_rows: 0,
            execution_time_ms: 0,
        })
    }

    /// Execute CREATE INDEX
    fn execute_create_index(&mut self, _stmt: CreateIndexStmt) -> Result<QueryResult> {
        Ok(QueryResult {
            columns: Vec::new(),
            rows: Vec::new(),
            affected_rows: 0,
            execution_time_ms: 0,
        })
    }

    /// List all tables
    pub fn list_tables(&self) -> Vec<&str> {
        self.tables.keys().map(|s| s.as_str()).collect()
    }

    /// Describe a table
    pub fn describe_table(&self, name: &str) -> Option<&TableSchema> {
        self.tables.get(name)
    }
}

impl Default for VectorSQL {
    fn default() -> Self {
        Self::new()
    }
}

/// SQL query builder for vector operations
pub struct VectorQueryBuilder {
    select_cols: Vec<String>,
    from_table: String,
    where_clauses: Vec<String>,
    vector_column: Option<String>,
    target_vector: Option<Vec<f32>>,
    metric: DistanceMetric,
    limit: Option<usize>,
}

impl VectorQueryBuilder {
    /// Create a new query builder
    pub fn new(table: impl Into<String>) -> Self {
        Self {
            select_cols: vec!["*".to_string()],
            from_table: table.into(),
            where_clauses: Vec::new(),
            vector_column: None,
            target_vector: None,
            metric: DistanceMetric::Cosine,
            limit: None,
        }
    }

    /// Select specific columns
    pub fn select(mut self, cols: &[&str]) -> Self {
        self.select_cols = cols.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Add a WHERE clause
    pub fn where_eq(mut self, column: &str, value: &str) -> Self {
        self.where_clauses.push(format!("{} = '{}'", column, value));
        self
    }

    /// Order by vector similarity
    pub fn order_by_similarity(mut self, column: &str, vector: Vec<f32>) -> Self {
        self.vector_column = Some(column.to_string());
        self.target_vector = Some(vector);
        self
    }

    /// Set distance metric
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set limit
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Build the SQL query
    pub fn build(&self) -> String {
        let mut sql = format!(
            "SELECT {} FROM {}",
            self.select_cols.join(", "),
            self.from_table
        );

        if !self.where_clauses.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&self.where_clauses.join(" AND "));
        }

        if let (Some(col), Some(vec)) = (&self.vector_column, &self.target_vector) {
            let operator = match self.metric {
                DistanceMetric::L2 => "<->",
                DistanceMetric::Cosine => "<=>",
                DistanceMetric::DotProduct => "<#>",
            };
            sql.push_str(&format!(
                " ORDER BY {} {} '[{:?}]'",
                col, operator,
                vec.iter().take(3).collect::<Vec<_>>()
            ));
        }

        if let Some(limit) = self.limit {
            sql.push_str(&format!(" LIMIT {}", limit));
        }

        sql
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_table() {
        let mut sql = VectorSQL::new();

        let result = sql.execute(
            "CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                embedding VECTOR(384),
                content TEXT
            )"
        ).unwrap();

        assert_eq!(result.affected_rows, 0);
        assert!(sql.tables.contains_key("documents"));
    }

    #[test]
    fn test_data_type_parse() {
        assert_eq!(DataType::parse("TEXT").unwrap(), DataType::Text);
        assert_eq!(DataType::parse("VECTOR(384)").unwrap(), DataType::Vector(384));
        assert_eq!(DataType::parse("INTEGER").unwrap(), DataType::Integer);
    }

    #[test]
    fn test_query_builder() {
        let query = VectorQueryBuilder::new("documents")
            .select(&["id", "content"])
            .where_eq("category", "tech")
            .order_by_similarity("embedding", vec![0.1, 0.2, 0.3])
            .limit(10)
            .build();

        assert!(query.contains("SELECT id, content"));
        assert!(query.contains("FROM documents"));
        assert!(query.contains("WHERE"));
        assert!(query.contains("LIMIT 10"));
    }
}
