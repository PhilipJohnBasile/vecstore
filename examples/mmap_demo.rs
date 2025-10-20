use std::time::Instant;
use vecstore::mmap::{MmapConfig, MmapVectorStore};

fn main() -> anyhow::Result<()> {
    println!("=== Memory-Mapped Vector Storage Demo ===\n");

    let path = "/tmp/vecstore_mmap_demo.bin";

    // Create a memory-mapped vector store
    println!("Creating memory-mapped store at: {}", path);
    let config = MmapConfig {
        vector_dim: 128,
        initial_capacity: 1000,
        use_huge_pages: false,
        populate: false,
    };

    let mut store = MmapVectorStore::create(path, config)?;
    println!("Initial capacity: {} vectors\n", store.capacity());

    // Insert some vectors
    println!("Inserting 100 vectors...");
    let start = Instant::now();

    for i in 0..100 {
        let vector: Vec<f32> = (0..128).map(|j| (i * 128 + j) as f32 / 1000.0).collect();
        store.insert(i, &vector)?;
    }

    let insert_time = start.elapsed();
    println!("Inserted 100 vectors in {:?}", insert_time);
    println!("Current count: {}", store.len());

    // Read vectors back
    println!("\nReading vectors back...");
    let start = Instant::now();

    for i in 0..100 {
        let _ = store.get(i)?;
    }

    let read_time = start.elapsed();
    println!("Read 100 vectors in {:?}", read_time);

    // Flush to disk
    println!("\nFlushing to disk...");
    store.flush()?;
    println!("Flushed successfully");

    // Drop the store to close it
    drop(store);

    // Reopen and verify
    println!("\nReopening store from disk...");
    let config = MmapConfig {
        vector_dim: 128,
        initial_capacity: 1000,
        use_huge_pages: false,
        populate: false,
    };

    let store = MmapVectorStore::open(path, config)?;
    println!("Reopened with {} vectors", store.len());

    // Verify a few vectors
    println!("\nVerifying vectors...");
    for i in [0, 50, 99] {
        let vector = store.get(i)?;
        println!(
            "Vector {}: first 5 elements = [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
            i, vector[0], vector[1], vector[2], vector[3], vector[4]
        );
    }

    println!("\n=== Demo Complete ===");
    println!("\nMemory-mapped storage benefits:");
    println!("  • Vectors stored on disk, accessed on-demand");
    println!("  • Only accessed pages loaded into RAM");
    println!("  • Can handle datasets larger than available RAM");
    println!("  • Fast random access (OS page cache optimization)");
    println!("  • Automatic persistence");

    Ok(())
}
