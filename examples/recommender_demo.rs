//! Recommender system demonstration
//!
//! Shows how to build recommendation systems:
//! - Content-based: Recommend based on item features
//! - Collaborative filtering: Recommend based on user behavior
//! - Hybrid: Combine both approaches
//! - Similar items: Find related content

use anyhow::Result;
use std::collections::HashMap;
use vecstore::Metadata;
use vecstore::recommender::{
    CollaborativeRecommender, ContentBasedRecommender, HybridRecommender, UserPreference,
};

fn create_metadata(title: &str, genre: &str, year: i32) -> Metadata {
    let mut fields = HashMap::new();
    fields.insert("title".to_string(), serde_json::json!(title));
    fields.insert("genre".to_string(), serde_json::json!(genre));
    fields.insert("year".to_string(), serde_json::json!(year));
    Metadata { fields }
}

fn main() -> Result<()> {
    println!("🎬 VecStore Recommender System Demo\n");
    println!("{}", "=".repeat(80));

    // Setup: Create movie catalog with feature vectors
    // Features: [action, comedy, drama, sci-fi, romance]
    println!("\n[1/4] Setting up movie catalog...");

    let movies = vec![
        (
            "movie1",
            "The Matrix",
            "Sci-Fi Action",
            vec![0.9, 0.1, 0.0, 1.0, 0.0],
            1999,
        ),
        (
            "movie2",
            "Inception",
            "Sci-Fi Thriller",
            vec![0.8, 0.0, 0.2, 0.9, 0.0],
            2010,
        ),
        (
            "movie3",
            "Superbad",
            "Comedy",
            vec![0.0, 1.0, 0.0, 0.0, 0.2],
            2007,
        ),
        (
            "movie4",
            "The Hangover",
            "Comedy",
            vec![0.1, 0.9, 0.0, 0.0, 0.1],
            2009,
        ),
        (
            "movie5",
            "The Notebook",
            "Romance Drama",
            vec![0.0, 0.0, 0.8, 0.0, 1.0],
            2004,
        ),
        (
            "movie6",
            "Titanic",
            "Romance Drama",
            vec![0.2, 0.0, 0.7, 0.0, 0.9],
            1997,
        ),
        (
            "movie7",
            "Die Hard",
            "Action",
            vec![1.0, 0.2, 0.0, 0.0, 0.0],
            1988,
        ),
        (
            "movie8",
            "Mad Max",
            "Action Sci-Fi",
            vec![0.9, 0.1, 0.0, 0.7, 0.0],
            2015,
        ),
    ];

    println!("   Movie catalog:");
    for (id, title, genre, _, year) in &movies {
        println!("   • {} ({}): {} - {}", id, year, title, genre);
    }

    // Content-Based Recommendations
    println!("\n[2/4] Content-Based Recommendations...");
    let mut content_recommender = ContentBasedRecommender::new();

    for (id, title, genre, vector, year) in &movies {
        content_recommender.add_item(*id, vector.clone(), create_metadata(title, genre, *year))?;
    }

    // User likes "The Matrix"
    println!("\n   Scenario: User watched and loved 'The Matrix' (Sci-Fi Action)");
    let preferences = vec![UserPreference::new("movie1", 5.0)];

    let content_recs = content_recommender.recommend(&preferences, 3)?;
    println!("   Content-based recommendations:");
    for (i, rec) in content_recs.iter().enumerate() {
        let title = rec
            .metadata
            .as_ref()
            .and_then(|m| m.fields.get("title"))
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");
        println!(
            "   {}. {} - Score: {:.3} ({})",
            i + 1,
            title,
            rec.score,
            rec.reason
        );
    }

    // Similar Items
    println!("\n   Finding movies similar to 'The Matrix'...");
    let similar = content_recommender.similar_items("movie1", 3)?;
    println!("   Similar movies:");
    for (i, rec) in similar.iter().enumerate() {
        let title = rec
            .metadata
            .as_ref()
            .and_then(|m| m.fields.get("title"))
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");
        println!("   {}. {} - Similarity: {:.3}", i + 1, title, rec.score);
    }

    // Collaborative Filtering
    println!("\n[3/4] Collaborative Filtering...");
    let mut collab_recommender = CollaborativeRecommender::new();

    // Simulate user ratings
    println!("   User ratings:");

    // User Alice (likes sci-fi action)
    collab_recommender.add_rating("alice", "movie1", 5.0); // The Matrix
    collab_recommender.add_rating("alice", "movie2", 5.0); // Inception
    collab_recommender.add_rating("alice", "movie7", 4.0); // Die Hard
    collab_recommender.add_rating("alice", "movie5", 2.0); // The Notebook
    println!("   • Alice: Loves Matrix (5★), Inception (5★), Die Hard (4★)");

    // User Bob (similar to Alice)
    collab_recommender.add_rating("bob", "movie1", 5.0);
    collab_recommender.add_rating("bob", "movie2", 4.0);
    collab_recommender.add_rating("bob", "movie8", 5.0); // Mad Max (Alice hasn't seen)
    println!("   • Bob: Loves Matrix (5★), Inception (4★), Mad Max (5★)");

    // User Carol (likes comedy)
    collab_recommender.add_rating("carol", "movie3", 5.0);
    collab_recommender.add_rating("carol", "movie4", 5.0);
    collab_recommender.add_rating("carol", "movie1", 2.0);
    println!("   • Carol: Loves Superbad (5★), Hangover (5★)");

    // User Dave (likes romance)
    collab_recommender.add_rating("dave", "movie5", 5.0);
    collab_recommender.add_rating("dave", "movie6", 5.0);
    println!("   • Dave: Loves Notebook (5★), Titanic (5★)");

    println!("\n   Recommendations for Alice (based on similar users):");
    let collab_recs = collab_recommender.recommend("alice", 3)?;
    for (i, rec) in collab_recs.iter().enumerate() {
        // Find movie title
        let movie = movies
            .iter()
            .find(|(id, _, _, _, _)| id == &rec.item_id.as_str());
        let title = movie.map(|(_, t, _, _, _)| *t).unwrap_or("Unknown");
        println!(
            "   {}. {} ({}) - Score: {:.3}",
            i + 1,
            title,
            rec.item_id,
            rec.score
        );
    }

    // Hybrid Recommendations
    println!("\n[4/4] Hybrid Recommendations...");
    let mut hybrid_recommender = HybridRecommender::new(0.6); // 60% content, 40% collaborative

    // Add movies
    for (id, title, genre, vector, year) in &movies {
        hybrid_recommender.add_item(*id, vector.clone(), create_metadata(title, genre, *year))?;
    }

    // Add ratings
    hybrid_recommender.add_rating("alice", "movie1", 5.0);
    hybrid_recommender.add_rating("alice", "movie2", 5.0);
    hybrid_recommender.add_rating("alice", "movie7", 4.0);

    hybrid_recommender.add_rating("bob", "movie1", 5.0);
    hybrid_recommender.add_rating("bob", "movie2", 4.0);
    hybrid_recommender.add_rating("bob", "movie8", 5.0);

    println!("   Hybrid recommendations for Alice:");
    println!("   (60% content-based + 40% collaborative)");

    let alice_prefs = vec![
        UserPreference::new("movie1", 5.0),
        UserPreference::new("movie2", 5.0),
    ];

    let hybrid_recs = hybrid_recommender.recommend("alice", &alice_prefs, 3)?;
    for (i, rec) in hybrid_recs.iter().enumerate() {
        let movie = movies
            .iter()
            .find(|(id, _, _, _, _)| id == &rec.item_id.as_str());
        let title = movie.map(|(_, t, _, _, _)| *t).unwrap_or("Unknown");
        println!(
            "   {}. {} - Score: {:.3} ({})",
            i + 1,
            title,
            rec.score,
            rec.reason
        );
    }

    // Summary
    println!("\n{}", "=".repeat(80));
    println!("📊 Summary");
    println!("{}", "=".repeat(80));

    println!("\n✅ Recommender systems working!");

    println!("\n🔍 Algorithm Comparison:");
    println!("\n   Content-Based:");
    println!("   • Based on: Item features/attributes");
    println!("   • Pros: No cold-start, explainable, works for new users");
    println!("   • Cons: Limited diversity, requires good features");
    println!("   • Best for: Items with rich metadata, new users");

    println!("\n   Collaborative Filtering:");
    println!("   • Based on: User behavior and ratings");
    println!("   • Pros: Discovers hidden patterns, no item features needed");
    println!("   • Cons: Cold-start problem, sparse data issues");
    println!("   • Best for: Established users, rich interaction data");

    println!("\n   Hybrid:");
    println!("   • Based on: Combination of both methods");
    println!("   • Pros: Best of both worlds, more robust");
    println!("   • Cons: More complex, requires tuning");
    println!("   • Best for: Production systems, diverse user base");

    println!("\n💡 Use Cases:");
    println!("   • E-commerce: Product recommendations");
    println!("   • Streaming: Movie/music suggestions");
    println!("   • Social media: Content discovery");
    println!("   • News: Article recommendations");
    println!("   • B2B: Partner/vendor matching");

    println!("\n⚙️  Configuration:");
    println!("   Content-Based:");
    println!("   • Feature engineering: Key to success");
    println!("   • Vector quality: Use embeddings for text/images");
    println!("   • Normalization: Ensure fair comparisons");

    println!("\n   Collaborative:");
    println!("   • Min interactions: 5-10 ratings per user");
    println!("   • Similarity metric: Pearson for sparse data");
    println!("   • Neighborhood size: 20-50 similar users");

    println!("\n   Hybrid:");
    println!("   • Content weight: 0.6-0.8 for cold-start scenarios");
    println!("   • Content weight: 0.3-0.5 for mature systems");
    println!("   • A/B test: Find optimal balance");

    println!("\n🎯 Best Practices:");
    println!("   1. Handle cold-start with content-based");
    println!("   2. Use implicit feedback (views, clicks)");
    println!("   3. Apply time decay for freshness");
    println!("   4. Filter inappropriate content");
    println!("   5. Diversify recommendations");
    println!("   6. Explain why items are recommended");
    println!("   7. A/B test different approaches");

    println!("\n📈 Evaluation Metrics:");
    println!("   • Precision@K: Relevant items in top K");
    println!("   • Recall@K: Coverage of relevant items");
    println!("   • NDCG: Ranking quality");
    println!("   • Diversity: Variety in recommendations");
    println!("   • Click-through rate: User engagement");

    Ok(())
}
