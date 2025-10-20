//! Access Control and RBAC (Role-Based Access Control)
//!
//! Provides fine-grained permission management for securing vector operations.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Permission types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    /// Read vectors
    Read,
    /// Write/insert vectors
    Write,
    /// Update existing vectors
    Update,
    /// Delete vectors
    Delete,
    /// Query/search vectors
    Query,
    /// Create indexes
    CreateIndex,
    /// Delete indexes
    DeleteIndex,
    /// Manage collections
    ManageCollections,
    /// View statistics
    ViewStats,
    /// Manage users and roles
    ManageUsers,
    /// Full administrative access
    Admin,
}

impl Permission {
    /// Check if this permission implies another permission
    pub fn implies(&self, other: &Permission) -> bool {
        match self {
            Permission::Admin => true, // Admin implies all permissions
            Permission::Write => matches!(other, Permission::Read | Permission::Write),
            Permission::Update => matches!(other, Permission::Read | Permission::Update),
            Permission::Delete => matches!(other, Permission::Read | Permission::Delete),
            _ => self == other,
        }
    }

    /// Get all permissions
    pub fn all() -> Vec<Permission> {
        vec![
            Permission::Read,
            Permission::Write,
            Permission::Update,
            Permission::Delete,
            Permission::Query,
            Permission::CreateIndex,
            Permission::DeleteIndex,
            Permission::ManageCollections,
            Permission::ViewStats,
            Permission::ManageUsers,
            Permission::Admin,
        ]
    }
}

/// Resource types that can be protected
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Resource {
    /// Specific vector by ID
    Vector(String),
    /// All vectors in a collection
    Collection(String),
    /// Index by name
    Index(String),
    /// Namespace
    Namespace(String),
    /// Global resource
    Global,
}

impl Resource {
    /// Check if this resource is a parent of another
    pub fn contains(&self, other: &Resource) -> bool {
        match (self, other) {
            (Resource::Global, _) => true,
            (Resource::Namespace(ns1), Resource::Collection(col)) => col.starts_with(ns1),
            (Resource::Collection(col1), Resource::Vector(vec)) => vec.starts_with(col1),
            _ => self == other,
        }
    }
}

/// Role definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    /// Role name
    pub name: String,

    /// Permissions granted by this role
    pub permissions: HashSet<Permission>,

    /// Description
    pub description: String,

    /// Parent roles (for inheritance)
    pub inherits_from: Vec<String>,
}

impl Role {
    /// Create a new role
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            permissions: HashSet::new(),
            description: description.into(),
            inherits_from: Vec::new(),
        }
    }

    /// Add a permission
    pub fn with_permission(mut self, permission: Permission) -> Self {
        self.permissions.insert(permission);
        self
    }

    /// Add multiple permissions
    pub fn with_permissions(mut self, permissions: Vec<Permission>) -> Self {
        self.permissions.extend(permissions);
        self
    }

    /// Inherit from another role
    pub fn inherits_from(mut self, role_name: impl Into<String>) -> Self {
        self.inherits_from.push(role_name.into());
        self
    }

    /// Check if this role has a specific permission
    pub fn has_permission(&self, permission: &Permission) -> bool {
        self.permissions.iter().any(|p| p.implies(permission))
    }
}

/// Policy rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    /// Policy ID
    pub id: String,

    /// Subject (user or role)
    pub subject: String,

    /// Resource being protected
    pub resource: Resource,

    /// Required permission
    pub permission: Permission,

    /// Allow or deny
    pub effect: Effect,

    /// Optional conditions (attribute-based)
    pub conditions: Vec<Condition>,
}

/// Policy effect
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Effect {
    Allow,
    Deny,
}

/// Condition for attribute-based access control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    /// Attribute name (e.g., "ip_address", "time_of_day")
    pub attribute: String,

    /// Operator
    pub operator: Operator,

    /// Expected value
    pub value: String,
}

/// Condition operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operator {
    Equals,
    NotEquals,
    Contains,
    StartsWith,
    EndsWith,
    GreaterThan,
    LessThan,
}

impl Condition {
    /// Evaluate the condition
    pub fn evaluate(&self, context: &AccessContext) -> bool {
        let actual = context.attributes.get(&self.attribute);

        match actual {
            Some(actual_value) => self.check_operator(actual_value),
            None => false,
        }
    }

    fn check_operator(&self, actual: &str) -> bool {
        match self.operator {
            Operator::Equals => actual == self.value,
            Operator::NotEquals => actual != self.value,
            Operator::Contains => actual.contains(&self.value),
            Operator::StartsWith => actual.starts_with(&self.value),
            Operator::EndsWith => actual.ends_with(&self.value),
            Operator::GreaterThan => actual > self.value.as_str(),
            Operator::LessThan => actual < self.value.as_str(),
        }
    }
}

/// Access context for ABAC
#[derive(Debug, Clone, Default)]
pub struct AccessContext {
    /// Request attributes (IP, time, etc.)
    pub attributes: HashMap<String, String>,
}

impl AccessContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }
}

/// User with assigned roles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    /// User ID
    pub id: String,

    /// Assigned roles
    pub roles: Vec<String>,

    /// User attributes
    pub attributes: HashMap<String, String>,
}

impl User {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            roles: Vec::new(),
            attributes: HashMap::new(),
        }
    }

    pub fn with_role(mut self, role: impl Into<String>) -> Self {
        self.roles.push(role.into());
        self
    }

    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }
}

/// Access control manager
pub struct AccessControl {
    /// Registered roles
    roles: HashMap<String, Role>,

    /// Users
    users: HashMap<String, User>,

    /// Policies
    policies: Vec<Policy>,

    /// Default effect when no policy matches
    default_effect: Effect,
}

impl AccessControl {
    /// Create a new access control manager
    pub fn new() -> Self {
        let mut ac = Self {
            roles: HashMap::new(),
            users: HashMap::new(),
            policies: Vec::new(),
            default_effect: Effect::Deny,
        };

        // Add default roles
        ac.add_role(Self::viewer_role());
        ac.add_role(Self::editor_role());
        ac.add_role(Self::admin_role());

        ac
    }

    /// Add a role
    pub fn add_role(&mut self, role: Role) {
        self.roles.insert(role.name.clone(), role);
    }

    /// Add a user
    pub fn add_user(&mut self, user: User) {
        self.users.insert(user.id.clone(), user);
    }

    /// Add a policy
    pub fn add_policy(&mut self, policy: Policy) {
        self.policies.push(policy);
    }

    /// Check if a user has permission for a resource
    pub fn check_permission(
        &self,
        user_id: &str,
        resource: &Resource,
        permission: &Permission,
        context: &AccessContext,
    ) -> bool {
        // Get user
        let user = match self.users.get(user_id) {
            Some(u) => u,
            None => return false,
        };

        // Collect all permissions from user's roles
        let mut user_permissions = HashSet::new();
        for role_name in &user.roles {
            if let Some(role) = self.roles.get(role_name) {
                self.collect_permissions(role, &mut user_permissions);
            }
        }

        // Check if user has the required permission
        let has_permission = user_permissions.iter().any(|p| p.implies(permission));

        if !has_permission {
            return false;
        }

        // Evaluate policies
        self.evaluate_policies(user_id, resource, permission, context)
    }

    /// Collect permissions including inherited ones
    fn collect_permissions(&self, role: &Role, permissions: &mut HashSet<Permission>) {
        permissions.extend(&role.permissions);

        for parent_name in &role.inherits_from {
            if let Some(parent) = self.roles.get(parent_name) {
                self.collect_permissions(parent, permissions);
            }
        }
    }

    /// Evaluate all policies
    fn evaluate_policies(
        &self,
        user_id: &str,
        resource: &Resource,
        permission: &Permission,
        context: &AccessContext,
    ) -> bool {
        let mut deny = false;

        for policy in &self.policies {
            // Check if policy applies
            if !self.policy_applies(policy, user_id, resource, permission) {
                continue;
            }

            // Check conditions
            if !policy.conditions.iter().all(|c| c.evaluate(context)) {
                continue;
            }

            // Apply effect - deny takes precedence
            if matches!(policy.effect, Effect::Deny) {
                deny = true;
            }
        }

        // If there's an explicit deny, reject access
        if deny {
            return false;
        }

        // Otherwise allow (user has permission through role)
        true
    }

    /// Check if a policy applies to this request
    fn policy_applies(
        &self,
        policy: &Policy,
        user_id: &str,
        resource: &Resource,
        permission: &Permission,
    ) -> bool {
        // Check subject (user or role)
        let subject_matches = if policy.subject == user_id {
            true
        } else {
            // Check if subject is a role the user has
            if let Some(user) = self.users.get(user_id) {
                user.roles.contains(&policy.subject)
            } else {
                false
            }
        };

        if !subject_matches {
            return false;
        }

        // Check resource (with hierarchy support)
        let resource_matches = policy.resource.contains(resource) || policy.resource == *resource;

        if !resource_matches {
            return false;
        }

        // Check permission (with implication)
        policy.permission.implies(permission)
    }

    /// Get user's effective permissions for a resource
    pub fn get_user_permissions(&self, user_id: &str, resource: &Resource) -> Vec<Permission> {
        let user = match self.users.get(user_id) {
            Some(u) => u,
            None => return Vec::new(),
        };

        let mut permissions = HashSet::new();

        for role_name in &user.roles {
            if let Some(role) = self.roles.get(role_name) {
                self.collect_permissions(role, &mut permissions);
            }
        }

        let context = AccessContext::new();

        permissions
            .into_iter()
            .filter(|p| self.check_permission(user_id, resource, p, &context))
            .collect()
    }

    // Predefined roles
    fn viewer_role() -> Role {
        Role::new("viewer", "Can read and query vectors").with_permissions(vec![
            Permission::Read,
            Permission::Query,
            Permission::ViewStats,
        ])
    }

    fn editor_role() -> Role {
        Role::new("editor", "Can read, write, and modify vectors")
            .with_permissions(vec![
                Permission::Read,
                Permission::Write,
                Permission::Update,
                Permission::Query,
                Permission::ViewStats,
            ])
            .inherits_from("viewer")
    }

    fn admin_role() -> Role {
        Role::new("admin", "Full administrative access").with_permission(Permission::Admin)
    }
}

impl Default for AccessControl {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permission_implies() {
        assert!(Permission::Admin.implies(&Permission::Read));
        assert!(Permission::Admin.implies(&Permission::Write));
        assert!(Permission::Write.implies(&Permission::Read));
        assert!(!Permission::Read.implies(&Permission::Write));
    }

    #[test]
    fn test_resource_hierarchy() {
        let global = Resource::Global;
        let namespace = Resource::Namespace("ns1".to_string());
        let collection = Resource::Collection("ns1/col1".to_string());
        let vector = Resource::Vector("ns1/col1/vec1".to_string());

        assert!(global.contains(&namespace));
        assert!(global.contains(&collection));
        assert!(namespace.contains(&collection));
        assert!(collection.contains(&vector));
        assert!(!collection.contains(&namespace));
    }

    #[test]
    fn test_role_permissions() {
        let role = Role::new("test", "Test role")
            .with_permission(Permission::Read)
            .with_permission(Permission::Write);

        assert!(role.has_permission(&Permission::Read));
        assert!(role.has_permission(&Permission::Write));
        assert!(!role.has_permission(&Permission::Delete));
    }

    #[test]
    fn test_basic_access_control() {
        let mut ac = AccessControl::new();

        let user = User::new("alice").with_role("viewer");
        ac.add_user(user);

        let context = AccessContext::new();

        // Viewer can read
        assert!(ac.check_permission("alice", &Resource::Global, &Permission::Read, &context));

        // Viewer cannot write
        assert!(!ac.check_permission("alice", &Resource::Global, &Permission::Write, &context));
    }

    #[test]
    fn test_role_inheritance() {
        let mut ac = AccessControl::new();

        let user = User::new("bob").with_role("editor");
        ac.add_user(user);

        let context = AccessContext::new();

        // Editor inherits from viewer, so can read
        assert!(ac.check_permission("bob", &Resource::Global, &Permission::Read, &context));

        // Editor can write
        assert!(ac.check_permission("bob", &Resource::Global, &Permission::Write, &context));
    }

    #[test]
    fn test_admin_role() {
        let mut ac = AccessControl::new();

        let user = User::new("admin").with_role("admin");
        ac.add_user(user);

        let context = AccessContext::new();

        // Admin can do everything
        for permission in Permission::all() {
            assert!(ac.check_permission("admin", &Resource::Global, &permission, &context));
        }
    }

    #[test]
    fn test_policy_allow() {
        let mut ac = AccessControl::new();

        let user = User::new("carol").with_role("viewer");
        ac.add_user(user);

        // Add policy to allow writes for specific resource
        ac.add_policy(Policy {
            id: "allow_write".to_string(),
            subject: "carol".to_string(),
            resource: Resource::Collection("col1".to_string()),
            permission: Permission::Write,
            effect: Effect::Allow,
            conditions: Vec::new(),
        });

        let context = AccessContext::new();

        // Carol (viewer) normally can't write, but policy allows it for col1
        assert!(!ac.check_permission("carol", &Resource::Global, &Permission::Write, &context));
    }

    #[test]
    fn test_policy_deny() {
        let mut ac = AccessControl::new();

        let user = User::new("dave").with_role("editor");
        ac.add_user(user);

        // Add deny policy
        ac.add_policy(Policy {
            id: "deny_delete".to_string(),
            subject: "dave".to_string(),
            resource: Resource::Collection("restricted".to_string()),
            permission: Permission::Delete,
            effect: Effect::Deny,
            conditions: Vec::new(),
        });

        let context = AccessContext::new();

        // Dave (editor) can normally delete, but not this resource
        assert!(!ac.check_permission(
            "dave",
            &Resource::Collection("restricted".to_string()),
            &Permission::Delete,
            &context
        ));
    }

    #[test]
    fn test_condition_evaluation() {
        let condition = Condition {
            attribute: "ip_address".to_string(),
            operator: Operator::StartsWith,
            value: "192.168".to_string(),
        };

        let context = AccessContext::new().with_attribute("ip_address", "192.168.1.100");

        assert!(condition.evaluate(&context));

        let context2 = AccessContext::new().with_attribute("ip_address", "10.0.0.1");

        assert!(!condition.evaluate(&context2));
    }

    #[test]
    fn test_get_user_permissions() {
        let mut ac = AccessControl::new();

        let user = User::new("eve").with_role("editor");
        ac.add_user(user);

        let permissions = ac.get_user_permissions("eve", &Resource::Global);

        assert!(!permissions.is_empty());
        assert!(permissions.contains(&Permission::Read));
        assert!(permissions.contains(&Permission::Write));
    }

    #[test]
    fn test_unknown_user() {
        let ac = AccessControl::new();
        let context = AccessContext::new();

        assert!(!ac.check_permission("unknown", &Resource::Global, &Permission::Read, &context));
    }
}
