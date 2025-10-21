// Filter Parser - SQL-like syntax for metadata filters
//
// Supported syntax:
//   field = 'value'
//   field != 'value'
//   field > 10
//   field >= 10
//   field < 10
//   field <= 10
//   field CONTAINS 'substring'
//   condition AND condition
//   condition OR condition
//   NOT condition
//   (condition)
//
// Examples:
//   "age > 18 AND role = 'admin'"
//   "score >= 50 AND (category = 'A' OR category = 'B')"
//   "NOT archived AND created_at > 1234567890"

use crate::store::types::{FilterExpr, FilterOp};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("Unexpected end of input")]
    UnexpectedEof,

    #[error("Unexpected token: {0}")]
    UnexpectedToken(String),

    #[error("Expected {expected}, got {got}")]
    Expected { expected: String, got: String },

    #[error("Invalid number: {0}")]
    InvalidNumber(String),

    #[error("Unclosed string literal")]
    UnclosedString,

    #[error("Empty filter expression")]
    EmptyExpression,
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    String(String),
    Number(f64),
    // Operators
    Eq,  // =
    Neq, // !=
    Gt,  // >
    Gte, // >=
    Lt,  // <
    Lte, // <=
    // Keywords
    And,
    Or,
    Not,
    Contains,
    In,         // IN operator (Major Issue #9 fix)
    NotIn,      // NOT IN operator (Major Issue #9 fix)
    StartsWith, // STARTSWITH operator (Major Issue #13 fix)
    // Delimiters
    LParen,
    RParen,
    LBracket, // [ for array literals
    RBracket, // ] for array literals
    Comma,    // , for array elements
    Eof,
}

struct Lexer {
    input: Vec<char>,
    pos: usize,
}

impl Lexer {
    fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.pos += 1;
        Some(ch)
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn read_string(&mut self, quote: char) -> Result<String, ParseError> {
        let mut s = String::new();

        while let Some(ch) = self.advance() {
            if ch == quote {
                return Ok(s);
            } else if ch == '\\' {
                // Handle escape sequences
                if let Some(escaped) = self.advance() {
                    match escaped {
                        'n' => s.push('\n'),
                        't' => s.push('\t'),
                        'r' => s.push('\r'),
                        '\\' => s.push('\\'),
                        '\'' => s.push('\''),
                        '"' => s.push('"'),
                        _ => {
                            s.push('\\');
                            s.push(escaped);
                        }
                    }
                }
            } else {
                s.push(ch);
            }
        }

        Err(ParseError::UnclosedString)
    }

    fn read_ident_or_keyword(&mut self) -> String {
        let mut s = String::new();

        while let Some(ch) = self.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                s.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        s
    }

    fn read_number(&mut self) -> Result<f64, ParseError> {
        let mut s = String::new();

        // Handle negative numbers
        if self.peek() == Some('-') {
            s.push('-');
            self.advance();
        }

        while let Some(ch) = self.peek() {
            if ch.is_numeric() || ch == '.' {
                s.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        s.parse().map_err(|_| ParseError::InvalidNumber(s))
    }

    fn next_token(&mut self) -> Result<Token, ParseError> {
        self.skip_whitespace();

        match self.peek() {
            None => Ok(Token::Eof),
            Some('(') => {
                self.advance();
                Ok(Token::LParen)
            }
            Some(')') => {
                self.advance();
                Ok(Token::RParen)
            }
            Some('[') => {
                self.advance();
                Ok(Token::LBracket)
            }
            Some(']') => {
                self.advance();
                Ok(Token::RBracket)
            }
            Some(',') => {
                self.advance();
                Ok(Token::Comma)
            }
            Some('=') => {
                self.advance();
                Ok(Token::Eq)
            }
            Some('!') => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    Ok(Token::Neq)
                } else {
                    Err(ParseError::UnexpectedToken("!".to_string()))
                }
            }
            Some('>') => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    Ok(Token::Gte)
                } else {
                    Ok(Token::Gt)
                }
            }
            Some('<') => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    Ok(Token::Lte)
                } else {
                    Ok(Token::Lt)
                }
            }
            Some('\'') | Some('"') => {
                let quote = self.advance().unwrap();
                let s = self.read_string(quote)?;
                Ok(Token::String(s))
            }
            Some(ch) if ch.is_numeric() || ch == '-' => {
                let n = self.read_number()?;
                Ok(Token::Number(n))
            }
            Some(ch) if ch.is_alphabetic() || ch == '_' => {
                let ident = self.read_ident_or_keyword();
                let upper = ident.to_uppercase();

                // Check for "NOT IN" two-word operator (Major Issue #9 fix)
                if upper == "NOT" {
                    // Save position in case we need to backtrack
                    let saved_pos = self.pos;
                    self.skip_whitespace();

                    // Try to read next word
                    if self.peek().map_or(false, |c| c.is_alphabetic()) {
                        let next_ident = self.read_ident_or_keyword();
                        if next_ident.to_uppercase() == "IN" {
                            return Ok(Token::NotIn);
                        }
                    }

                    // Not "NOT IN", restore position and return NOT
                    self.pos = saved_pos;
                    return Ok(Token::Not);
                }

                match upper.as_str() {
                    "AND" => Ok(Token::And),
                    "OR" => Ok(Token::Or),
                    "CONTAINS" => Ok(Token::Contains),
                    "IN" => Ok(Token::In), // Major Issue #9 fix
                    "STARTSWITH" => Ok(Token::StartsWith), // Major Issue #13 fix
                    _ => Ok(Token::Ident(ident)),
                }
            }
            Some(ch) => Err(ParseError::UnexpectedToken(ch.to_string())),
        }
    }
}

struct Parser {
    lexer: Lexer,
    current: Token,
}

impl Parser {
    fn new(input: &str) -> Result<Self, ParseError> {
        let mut lexer = Lexer::new(input);
        let current = lexer.next_token()?;
        Ok(Self { lexer, current })
    }

    fn advance(&mut self) -> Result<(), ParseError> {
        self.current = self.lexer.next_token()?;
        Ok(())
    }

    fn expect(&mut self, expected: Token) -> Result<(), ParseError> {
        if std::mem::discriminant(&self.current) == std::mem::discriminant(&expected) {
            self.advance()?;
            Ok(())
        } else {
            Err(ParseError::Expected {
                expected: format!("{:?}", expected),
                got: format!("{:?}", self.current),
            })
        }
    }

    // Grammar:
    // expr     := or_expr
    // or_expr  := and_expr ( OR and_expr )*
    // and_expr := not_expr ( AND not_expr )*
    // not_expr := NOT not_expr | primary
    // primary  := ( expr ) | comparison
    // comparison := field op value

    pub fn parse(&mut self) -> Result<FilterExpr, ParseError> {
        if self.current == Token::Eof {
            return Err(ParseError::EmptyExpression);
        }
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<FilterExpr, ParseError> {
        let mut left = self.parse_and()?;

        while self.current == Token::Or {
            self.advance()?;
            let right = self.parse_and()?;
            left = FilterExpr::Or(vec![left, right]);
        }

        Ok(left)
    }

    fn parse_and(&mut self) -> Result<FilterExpr, ParseError> {
        let mut left = self.parse_not()?;

        while self.current == Token::And {
            self.advance()?;
            let right = self.parse_not()?;
            left = FilterExpr::And(vec![left, right]);
        }

        Ok(left)
    }

    fn parse_not(&mut self) -> Result<FilterExpr, ParseError> {
        if self.current == Token::Not {
            self.advance()?;
            let expr = self.parse_not()?;
            Ok(FilterExpr::Not(Box::new(expr)))
        } else {
            self.parse_primary()
        }
    }

    fn parse_primary(&mut self) -> Result<FilterExpr, ParseError> {
        if self.current == Token::LParen {
            self.advance()?;
            let expr = self.parse_or()?;
            self.expect(Token::RParen)?;
            Ok(expr)
        } else {
            self.parse_comparison()
        }
    }

    fn parse_comparison(&mut self) -> Result<FilterExpr, ParseError> {
        let field = match &self.current {
            Token::Ident(name) => {
                let name = name.clone();
                self.advance()?;
                name
            }
            _ => {
                return Err(ParseError::Expected {
                    expected: "field name".to_string(),
                    got: format!("{:?}", self.current),
                })
            }
        };

        let op = match &self.current {
            Token::Eq => FilterOp::Eq,
            Token::Neq => FilterOp::Neq,
            Token::Gt => FilterOp::Gt,
            Token::Gte => FilterOp::Gte,
            Token::Lt => FilterOp::Lt,
            Token::Lte => FilterOp::Lte,
            Token::Contains => FilterOp::Contains,
            Token::In => FilterOp::In,       // Major Issue #9 fix
            Token::NotIn => FilterOp::NotIn, // Major Issue #9 fix
            Token::StartsWith => FilterOp::StartsWith, // Major Issue #13 fix
            _ => {
                return Err(ParseError::Expected {
                    expected: "operator (=, !=, >, >=, <, <=, CONTAINS, IN, STARTSWITH)"
                        .to_string(),
                    got: format!("{:?}", self.current),
                })
            }
        };
        self.advance()?;

        // Parse value - for IN/NOT IN, expect array literal
        let value = if matches!(op, FilterOp::In | FilterOp::NotIn) {
            // Expect array literal: ['value1', 'value2']
            if !matches!(self.current, Token::LBracket) {
                return Err(ParseError::Expected {
                    expected: "array literal [...]".to_string(),
                    got: format!("{:?}", self.current),
                });
            }
            self.advance()?; // consume [

            let mut elements = Vec::new();
            loop {
                // Check for empty array or end of array
                if matches!(self.current, Token::RBracket) {
                    self.advance()?;
                    break;
                }

                // Parse array element
                let elem = match &self.current {
                    Token::String(s) => {
                        let s = s.clone();
                        self.advance()?;
                        serde_json::json!(s)
                    }
                    Token::Number(n) => {
                        let n = *n;
                        self.advance()?;
                        if n.fract() == 0.0 && n.abs() < (i64::MAX as f64) {
                            serde_json::json!(n as i64)
                        } else {
                            serde_json::json!(n)
                        }
                    }
                    Token::Ident(s) => {
                        let s = s.clone();
                        self.advance()?;
                        match s.to_lowercase().as_str() {
                            "true" => serde_json::json!(true),
                            "false" => serde_json::json!(false),
                            "null" => serde_json::json!(null),
                            _ => serde_json::json!(s),
                        }
                    }
                    _ => {
                        return Err(ParseError::Expected {
                            expected: "array element (string, number, or identifier)".to_string(),
                            got: format!("{:?}", self.current),
                        })
                    }
                };
                elements.push(elem);

                // Check for comma or end
                match &self.current {
                    Token::Comma => {
                        self.advance()?;
                        // Continue to next element
                    }
                    Token::RBracket => {
                        self.advance()?;
                        break;
                    }
                    _ => {
                        return Err(ParseError::Expected {
                            expected: ", or ]".to_string(),
                            got: format!("{:?}", self.current),
                        })
                    }
                }
            }

            serde_json::json!(elements)
        } else {
            // Regular value parsing for other operators
            match &self.current {
                Token::String(s) => {
                    let s = s.clone();
                    self.advance()?;
                    serde_json::json!(s)
                }
                Token::Number(n) => {
                    let n = *n;
                    self.advance()?;
                    // Use integer if whole number
                    if n.fract() == 0.0 && n.abs() < (i64::MAX as f64) {
                        serde_json::json!(n as i64)
                    } else {
                        serde_json::json!(n)
                    }
                }
                Token::Ident(s) => {
                    let s = s.clone();
                    self.advance()?;
                    // Handle boolean literals
                    match s.to_lowercase().as_str() {
                        "true" => serde_json::json!(true),
                        "false" => serde_json::json!(false),
                        "null" => serde_json::json!(null),
                        _ => serde_json::json!(s),
                    }
                }
                _ => {
                    return Err(ParseError::Expected {
                        expected: "value (string, number, or identifier)".to_string(),
                        got: format!("{:?}", self.current),
                    })
                }
            }
        };

        Ok(FilterExpr::Cmp { field, op, value })
    }
}

/// Parse a filter expression from a SQL-like string
///
/// # Examples
///
/// ```
/// use vecstore::parse_filter;
///
/// let filter = parse_filter("age > 18 AND role = 'admin'").unwrap();
/// ```
pub fn parse_filter(input: &str) -> Result<FilterExpr, ParseError> {
    let mut parser = Parser::new(input)?;
    parser.parse()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_eq() {
        let filter = parse_filter("name = 'Alice'").unwrap();
        match filter {
            FilterExpr::Cmp { field, op, value } => {
                assert_eq!(field, "name");
                assert_eq!(op, FilterOp::Eq);
                assert_eq!(value, serde_json::json!("Alice"));
            }
            _ => panic!("Expected Cmp"),
        }
    }

    #[test]
    fn test_number_comparison() {
        let filter = parse_filter("age > 18").unwrap();
        match filter {
            FilterExpr::Cmp { field, op, value } => {
                assert_eq!(field, "age");
                assert_eq!(op, FilterOp::Gt);
                assert_eq!(value, serde_json::json!(18));
            }
            _ => panic!("Expected Cmp"),
        }
    }

    #[test]
    fn test_and() {
        let filter = parse_filter("age > 18 AND role = 'admin'").unwrap();
        match filter {
            FilterExpr::And(exprs) => {
                assert_eq!(exprs.len(), 2);
            }
            _ => panic!("Expected And"),
        }
    }

    #[test]
    fn test_or() {
        let filter = parse_filter("category = 'A' OR category = 'B'").unwrap();
        match filter {
            FilterExpr::Or(exprs) => {
                assert_eq!(exprs.len(), 2);
            }
            _ => panic!("Expected Or"),
        }
    }

    #[test]
    fn test_not() {
        let filter = parse_filter("NOT archived = true").unwrap();
        match filter {
            FilterExpr::Not(_) => {}
            _ => panic!("Expected Not"),
        }
    }

    #[test]
    fn test_parentheses() {
        let filter = parse_filter("(age > 18 AND role = 'admin') OR vip = true").unwrap();
        match filter {
            FilterExpr::Or(_) => {}
            _ => panic!("Expected Or"),
        }
    }

    #[test]
    fn test_contains() {
        let filter = parse_filter("description CONTAINS 'rust'").unwrap();
        match filter {
            FilterExpr::Cmp { op, .. } => {
                assert_eq!(op, FilterOp::Contains);
            }
            _ => panic!("Expected Cmp with Contains"),
        }
    }

    #[test]
    fn test_complex() {
        let filter = parse_filter(
            "score >= 50 AND (category = 'A' OR category = 'B') AND NOT archived = true",
        )
        .unwrap();

        // Should parse successfully
        match filter {
            FilterExpr::And(_) => {}
            _ => panic!("Expected And at top level"),
        }
    }

    #[test]
    fn test_boolean_literals() {
        let filter = parse_filter("active = true").unwrap();
        match filter {
            FilterExpr::Cmp { value, .. } => {
                assert_eq!(value, serde_json::json!(true));
            }
            _ => panic!("Expected Cmp"),
        }
    }

    #[test]
    fn test_escaped_strings() {
        let filter = parse_filter(r#"name = 'O\'Reilly'"#).unwrap();
        match filter {
            FilterExpr::Cmp { value, .. } => {
                assert_eq!(value, serde_json::json!("O'Reilly"));
            }
            _ => panic!("Expected Cmp"),
        }
    }

    #[test]
    fn test_negative_numbers() {
        let filter = parse_filter("temperature < -10").unwrap();
        match filter {
            FilterExpr::Cmp { value, .. } => {
                assert_eq!(value, serde_json::json!(-10));
            }
            _ => panic!("Expected Cmp"),
        }
    }

    #[test]
    fn test_empty_fails() {
        assert!(parse_filter("").is_err());
    }

    #[test]
    fn test_unclosed_string_fails() {
        assert!(parse_filter("name = 'Alice").is_err());
    }
}
