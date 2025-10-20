"""
Snapshots and Backup Example

This example demonstrates:
- Creating named snapshots
- Restoring from snapshots
- Using snapshots for version control
- Backup and recovery workflows
"""

import random
from vecstore import VecStore


def mock_embed(text: str) -> list[float]:
    """Mock embedding function"""
    random.seed(hash(text) % (2**32))
    return [random.random() for _ in range(384)]


def print_store_state(store: VecStore, label: str):
    """Helper to print current store state"""
    print(f"\n{label}:")
    print(f"  Total vectors: {len(store)}")
    if len(store) > 0:
        # Query to see what's in the store
        sample_vec = mock_embed("test")
        results = store.query(sample_vec, k=len(store))
        print(f"  Vectors in store:")
        for r in results:
            print(f"    - {r.id}: {r.metadata.get('text', 'N/A')[:40]}...")


def main():
    print("=" * 60)
    print("VecStore - Snapshots & Backup Example")
    print("=" * 60)

    # Create store
    print("\n1. Creating vector store...")
    store = VecStore("./snapshots_demo_db")
    print(f"   ✓ Store created: {store}")

    # Add initial data
    print("\n2. Adding initial data (Version 1)...")
    documents_v1 = [
        {"id": "doc1", "text": "Rust programming language"},
        {"id": "doc2", "text": "Python data science"},
        {"id": "doc3", "text": "JavaScript web development"},
    ]

    for doc in documents_v1:
        vector = mock_embed(doc["text"])
        store.upsert(doc["id"], vector, {"text": doc["text"], "version": 1})
        print(f"   ✓ Added {doc['id']}")

    print_store_state(store, "   State after Version 1")

    # Create snapshot v1
    print("\n3. Creating snapshot 'version_1'...")
    store.create_snapshot("version_1")
    print("   ✓ Snapshot 'version_1' created")

    # Add more data
    print("\n4. Adding more data (Version 2)...")
    documents_v2 = [
        {"id": "doc4", "text": "Go systems programming"},
        {"id": "doc5", "text": "TypeScript static typing"},
    ]

    for doc in documents_v2:
        vector = mock_embed(doc["text"])
        store.upsert(doc["id"], vector, {"text": doc["text"], "version": 2})
        print(f"   ✓ Added {doc['id']}")

    print_store_state(store, "   State after Version 2")

    # Create snapshot v2
    print("\n5. Creating snapshot 'version_2'...")
    store.create_snapshot("version_2")
    print("   ✓ Snapshot 'version_2' created")

    # Modify data (Version 3)
    print("\n6. Modifying data (Version 3)...")
    store.remove("doc2")  # Remove Python doc
    print("   ✓ Removed doc2")

    # Update doc1
    vector = mock_embed("Rust - safe systems programming")
    store.upsert("doc1", vector, {"text": "Rust - safe systems programming", "version": 3})
    print("   ✓ Updated doc1")

    print_store_state(store, "   State after Version 3")

    # List all snapshots
    print("\n7. Listing all snapshots...")
    snapshots = store.list_snapshots()
    print(f"   Available snapshots: {snapshots}")

    # Restore to version 1
    print("\n8. Restoring to 'version_1'...")
    store.restore_snapshot("version_1")
    print("   ✓ Restored to version_1")
    print_store_state(store, "   State after restore to version_1")

    # Restore to version 2
    print("\n9. Restoring to 'version_2'...")
    store.restore_snapshot("version_2")
    print("   ✓ Restored to version_2")
    print_store_state(store, "   State after restore to version_2")

    # Demonstrate backup workflow
    print("\n10. Simulating backup workflow...")
    print("   Creating daily backup...")
    import datetime
    backup_name = f"daily_backup_{datetime.datetime.now().strftime('%Y%m%d')}"
    store.create_snapshot(backup_name)
    print(f"   ✓ Created backup: {backup_name}")

    # Show all snapshots
    snapshots = store.list_snapshots()
    print(f"\n   All snapshots:")
    for snapshot in snapshots:
        print(f"     - {snapshot}")

    # Best practices
    print("\n" + "=" * 60)
    print("Snapshot Best Practices:")
    print("=" * 60)
    print("""
1. Regular Backups:
   - Create daily/weekly snapshots for disaster recovery
   - Use timestamps in snapshot names

2. Version Control:
   - Create snapshots before major changes
   - Keep snapshots of stable versions

3. Testing:
   - Create snapshot before testing
   - Restore if tests reveal issues

4. Deployment:
   - Snapshot before deploying new code
   - Quick rollback if deployment fails

5. Cleanup:
   - Regularly review and remove old snapshots
   - Keep only critical version snapshots long-term
    """)

    print("\n" + "=" * 60)
    print("✓ Snapshots demo complete!")
    print("=" * 60)

    # Cleanup
    import shutil
    shutil.rmtree("./snapshots_demo_db", ignore_errors=True)
    print("\n✓ Cleaned up demo database")


if __name__ == "__main__":
    main()
