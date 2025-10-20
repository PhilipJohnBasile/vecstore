//! Access Control and RBAC Demo
//!
//! Demonstrates role-based and attribute-based access control.

use vecstore::*;

fn main() -> anyhow::Result<()> {
    println!("\n🔒 Access Control and RBAC Demo\n");
    println!("{}", "=".repeat(70));

    // Create access control manager
    let mut ac = AccessControl::new();
    let context = AccessContext::new();

    // Test 1: Default Roles
    println!("\n[1/7] Default Roles");
    println!("{}", "-".repeat(70));

    println!("Built-in roles:");
    println!("  • viewer:  Read and query access");
    println!("  • editor:  Read, write, update, query");
    println!("  • admin:   Full administrative access");

    // Test 2: User with Viewer Role
    println!("\n[2/7] Viewer Role Permissions");
    println!("{}", "-".repeat(70));

    let alice = User::new("alice").with_role("viewer");
    ac.add_user(alice);

    println!("User: alice (viewer)");

    let tests = vec![
        (Permission::Read, "Read"),
        (Permission::Write, "Write"),
        (Permission::Delete, "Delete"),
        (Permission::Query, "Query"),
    ];

    for (perm, name) in &tests {
        let allowed = ac.check_permission("alice", &Resource::Global, perm, &context);
        let status = if allowed { "✓ ALLOWED" } else { "✗ DENIED" };
        println!("  {:<15} {}", name, status);
    }

    // Test 3: User with Editor Role
    println!("\n[3/7] Editor Role Permissions");
    println!("{}", "-".repeat(70));

    let bob = User::new("bob").with_role("editor");
    ac.add_user(bob);

    println!("User: bob (editor)");

    for (perm, name) in &tests {
        let allowed = ac.check_permission("bob", &Resource::Global, perm, &context);
        let status = if allowed { "✓ ALLOWED" } else { "✗ DENIED" };
        println!("  {:<15} {}", name, status);
    }

    // Test 4: User with Admin Role
    println!("\n[4/7] Admin Role Permissions");
    println!("{}", "-".repeat(70));

    let admin = User::new("admin_user").with_role("admin");
    ac.add_user(admin);

    println!("User: admin_user (admin)");

    for (perm, name) in &tests {
        let allowed = ac.check_permission("admin_user", &Resource::Global, perm, &context);
        let status = if allowed { "✓ ALLOWED" } else { "✗ DENIED" };
        println!("  {:<15} {}", name, status);
    }

    println!("\n✨ Admin has all permissions!");

    // Test 5: Custom Roles
    println!("\n[5/7] Custom Role Creation");
    println!("{}", "-".repeat(70));

    let analyst =
        Role::new("analyst", "Data analyst with query-only access").with_permissions(vec![
            Permission::Read,
            Permission::Query,
            Permission::ViewStats,
        ]);

    ac.add_role(analyst);

    let charlie = User::new("charlie").with_role("analyst");
    ac.add_user(charlie);

    println!("Custom role: analyst");
    println!("  Description: Data analyst with query-only access");
    println!("  Permissions: Read, Query, ViewStats");

    println!("\nUser: charlie (analyst)");

    for (perm, name) in &tests {
        let allowed = ac.check_permission("charlie", &Resource::Global, perm, &context);
        let status = if allowed { "✓ ALLOWED" } else { "✗ DENIED" };
        println!("  {:<15} {}", name, status);
    }

    // Test 6: Policy-Based Access Control
    println!("\n[6/7] Policy-Based Access Control");
    println!("{}", "-".repeat(70));

    // Add a deny policy for a specific resource
    ac.add_policy(Policy {
        id: "deny_delete_prod".to_string(),
        subject: "bob".to_string(),
        resource: Resource::Collection("production".to_string()),
        permission: Permission::Delete,
        effect: Effect::Deny,
        conditions: Vec::new(),
    });

    println!("Policy added:");
    println!("  Subject:    bob");
    println!("  Resource:   Collection 'production'");
    println!("  Permission: Delete");
    println!("  Effect:     DENY");

    println!("\nBob's access to production collection:");

    let prod_resource = Resource::Collection("production".to_string());

    let can_read = ac.check_permission("bob", &prod_resource, &Permission::Read, &context);
    let can_write = ac.check_permission("bob", &prod_resource, &Permission::Write, &context);
    let can_delete = ac.check_permission("bob", &prod_resource, &Permission::Delete, &context);

    println!(
        "  Read:       {}",
        if can_read {
            "✓ ALLOWED"
        } else {
            "✗ DENIED"
        }
    );
    println!(
        "  Write:      {}",
        if can_write {
            "✓ ALLOWED"
        } else {
            "✗ DENIED"
        }
    );
    println!(
        "  Delete:     {}",
        if can_delete {
            "✓ ALLOWED"
        } else {
            "✗ DENIED"
        }
    );

    // Test 7: Attribute-Based Access Control (ABAC)
    println!("\n[7/7] Attribute-Based Access Control (ABAC)");
    println!("{}", "-".repeat(70));

    // Add policy with IP-based condition
    ac.add_policy(Policy {
        id: "ip_restricted".to_string(),
        subject: "dave".to_string(),
        resource: Resource::Collection("secure".to_string()),
        permission: Permission::Read,
        effect: Effect::Deny,
        conditions: vec![Condition {
            attribute: "ip_address".to_string(),
            operator: Operator::NotEquals,
            value: "192.168.1.100".to_string(),
        }],
    });

    let dave = User::new("dave").with_role("viewer");
    ac.add_user(dave);

    println!("Policy added:");
    println!("  Subject:    dave");
    println!("  Resource:   Collection 'secure'");
    println!("  Permission: Read");
    println!("  Effect:     DENY");
    println!("  Condition:  IP != 192.168.1.100");

    let secure_resource = Resource::Collection("secure".to_string());

    // Test from allowed IP
    let context_allowed = AccessContext::new().with_attribute("ip_address", "192.168.1.100");

    let allowed = ac.check_permission(
        "dave",
        &secure_resource,
        &Permission::Read,
        &context_allowed,
    );
    println!("\nDave accessing from 192.168.1.100:");
    println!(
        "  Read:       {}",
        if allowed { "✓ ALLOWED" } else { "✗ DENIED" }
    );

    // Test from denied IP
    let context_denied = AccessContext::new().with_attribute("ip_address", "10.0.0.50");

    let denied = ac.check_permission("dave", &secure_resource, &Permission::Read, &context_denied);
    println!("\nDave accessing from 10.0.0.50:");
    println!(
        "  Read:       {}",
        if denied { "✓ ALLOWED" } else { "✗ DENIED" }
    );

    // Test 8: Resource Hierarchy
    println!("\n[8/8] Resource Hierarchy");
    println!("{}", "-".repeat(70));

    println!("Resource hierarchy:");
    println!("  Global");
    println!("    └─ Namespace: 'ns1'");
    println!("         └─ Collection: 'ns1/col1'");
    println!("              └─ Vector: 'ns1/col1/vec123'");

    let global = Resource::Global;
    let namespace = Resource::Namespace("ns1".to_string());
    let collection = Resource::Collection("ns1/col1".to_string());
    let vector = Resource::Vector("ns1/col1/vec123".to_string());

    println!("\nHierarchy relationships:");
    println!(
        "  Global contains Namespace:   {}",
        global.contains(&namespace)
    );
    println!(
        "  Global contains Collection:  {}",
        global.contains(&collection)
    );
    println!(
        "  Namespace contains Collection: {}",
        namespace.contains(&collection)
    );
    println!(
        "  Collection contains Vector:  {}",
        collection.contains(&vector)
    );
    println!(
        "  Collection contains Namespace: {}",
        collection.contains(&namespace)
    );

    // Summary
    println!("\n{}", "=".repeat(70));
    println!("✅ Demo Complete!");
    println!("{}", "=".repeat(70));

    println!("\n✨ Key Features Demonstrated:");
    println!("  ✓ Role-Based Access Control (RBAC)");
    println!("  ✓ Predefined roles (viewer, editor, admin)");
    println!("  ✓ Custom role creation");
    println!("  ✓ Permission inheritance");
    println!("  ✓ Policy-based access control");
    println!("  ✓ Allow and deny policies");
    println!("  ✓ Attribute-Based Access Control (ABAC)");
    println!("  ✓ Conditional access (IP-based, time-based, etc.)");
    println!("  ✓ Resource hierarchy");
    println!("  ✓ Fine-grained permissions");

    println!("\n📋 Permission Types:");
    println!("  • Read:              Read vector data");
    println!("  • Write:             Insert new vectors");
    println!("  • Update:            Modify existing vectors");
    println!("  • Delete:            Remove vectors");
    println!("  • Query:             Search operations");
    println!("  • CreateIndex:       Create new indexes");
    println!("  • DeleteIndex:       Remove indexes");
    println!("  • ManageCollections: Collection operations");
    println!("  • ViewStats:         View statistics");
    println!("  • ManageUsers:       User management");
    println!("  • Admin:             Full access");

    println!("\n🎯 Use Cases:");
    println!("  • Multi-tenant SaaS applications");
    println!("  • Enterprise data governance");
    println!("  • Compliance and regulatory requirements");
    println!("  • Secure data sharing");
    println!("  • Fine-grained access control");
    println!("  • IP-based restrictions");
    println!("  • Time-based access windows");

    println!();

    Ok(())
}
