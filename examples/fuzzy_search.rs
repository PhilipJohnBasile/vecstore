//! Fuzzy Search with Typo Tolerance Example
//!
//! This example demonstrates fuzzy string matching capabilities for handling
//! typos, misspellings, and approximate text matching.
//!
//! ## Features Demonstrated
//!
//! - Levenshtein distance calculation
//! - Damerau-Levenshtein distance (handles transpositions)
//! - Fuzzy matching with configurable thresholds
//! - BK-tree for efficient fuzzy search
//! - Query correction suggestions
//!
//! ## Running
//!
//! ```bash
//! cargo run --example fuzzy_search
//! ```

use vecstore::fuzzy::{
    damerau_levenshtein_distance, levenshtein_distance, similarity_score, suggest_corrections,
    BKTree, FuzzyMatcher,
};

fn main() {
    println!("🔍 Fuzzy Search with Typo Tolerance Example\n");

    // ============================================================
    // 1. Basic Distance Calculations
    // ============================================================
    println!("📏 Distance Calculations:");
    println!("─────────────────────────");

    let dist1 = levenshtein_distance("kitten", "sitting");
    println!("  Levenshtein('kitten', 'sitting') = {}", dist1);

    let dist2 = levenshtein_distance("hello", "helo");
    println!("  Levenshtein('hello', 'helo') = {}", dist2);

    // Damerau-Levenshtein handles transpositions
    let dist3 = damerau_levenshtein_distance("teh", "the");
    println!(
        "  Damerau-Levenshtein('teh', 'the') = {} (transposition)",
        dist3
    );

    let dist4 = levenshtein_distance("teh", "the");
    println!("  Levenshtein('teh', 'the') = {} (no transposition)", dist4);

    // Similarity scores
    let sim = similarity_score("hello", "helo");
    println!("\n  Similarity('hello', 'helo') = {:.2}%", sim * 100.0);

    // ============================================================
    // 2. Fuzzy Matcher
    // ============================================================
    println!("\n\n🎯 Fuzzy Matcher:");
    println!("─────────────────");

    let matcher = FuzzyMatcher::new(2); // Allow up to 2 edits

    let query = "programming";
    let candidates = vec![
        "programming".to_string(), // exact match
        "programing".to_string(),  // 1 edit
        "progamming".to_string(),  // 1 edit
        "coding".to_string(),      // too different
    ];

    println!("  Query: '{}'", query);
    println!("  Max distance: 2 edits\n");

    for candidate in &candidates {
        let distance = matcher.distance(query, candidate);
        let is_match = matcher.is_match(query, candidate);
        let status = if is_match {
            "✓ MATCH"
        } else {
            "✗ no match"
        };

        println!("    {} '{}' (distance: {})", status, candidate, distance);
    }

    // Find best match
    if let Some((idx, dist)) = matcher.find_best_match(query, &candidates) {
        println!("\n  Best match: '{}' (distance: {})", candidates[idx], dist);
    }

    // ============================================================
    // 3. BK-Tree for Efficient Fuzzy Search
    // ============================================================
    println!("\n\n🌳 BK-Tree (Burkhard-Keller Tree):");
    println!("──────────────────────────────────");

    // Build a dictionary
    let mut tree = BKTree::new();

    let dictionary = vec![
        "apple",
        "application",
        "apply",
        "banana",
        "band",
        "can",
        "candy",
        "cat",
        "dog",
        "door",
        "example",
        "hello",
        "help",
        "programming",
        "python",
        "rust",
        "search",
        "world",
    ];

    println!("  Building BK-tree with {} words...", dictionary.len());

    for word in &dictionary {
        tree.insert(word.to_string());
    }

    println!("  ✓ Built tree with {} nodes\n", tree.len());

    // Search for typos
    let typos = vec![
        ("aple", 1),       // "apple" with 1 typo
        ("helo", 1),       // "hello" or "help"
        ("serch", 1),      // "search" with 1 typo
        ("progamming", 2), // "programming" with typos
    ];

    for (typo, max_dist) in typos {
        let matches = tree.search(typo, max_dist);
        println!("  Typo: '{}' (max distance: {})", typo, max_dist);

        if matches.is_empty() {
            println!("    No matches found");
        } else {
            println!("    Matches: {}", matches.join(", "));
        }

        // Find closest match
        if let Some((closest, dist)) = tree.find_closest(typo) {
            println!("    Closest: '{}' (distance: {})", closest, dist);
        }

        println!();
    }

    // ============================================================
    // 4. Query Correction
    // ============================================================
    println!("\n💡 Query Correction / Spell Checking:");
    println!("─────────────────────────────────────");

    let user_queries = vec![
        "progrmming", // should suggest "programming"
        "pyton",      // should suggest "python"
        "exmple",     // should suggest "example"
        "wrld",       // should suggest "world"
    ];

    let dict_strings: Vec<String> = dictionary.iter().map(|s| s.to_string()).collect();

    for query in user_queries {
        println!("  User query: '{}'", query);

        let suggestions = suggest_corrections(query, &dict_strings, 2, 3);

        if suggestions.is_empty() {
            println!("    No suggestions (too different)");
        } else {
            println!("    Suggestions: {}", suggestions.join(", "));

            // Show distances
            for sugg in &suggestions {
                let dist = levenshtein_distance(query, sugg);
                println!("      - '{}' (distance: {})", sugg, dist);
            }
        }

        println!();
    }

    // ============================================================
    // 5. Case Sensitivity
    // ============================================================
    println!("\n🔤 Case Sensitivity:");
    println!("───────────────────");

    // Case-insensitive (default)
    let matcher_ci = FuzzyMatcher::new(1).with_case_sensitive(false);
    println!("  Case-insensitive matching:");
    println!(
        "    'Hello' ~ 'hello': {}",
        matcher_ci.is_match("Hello", "hello")
    );
    println!(
        "    'RUST' ~ 'rust': {}",
        matcher_ci.is_match("RUST", "rust")
    );

    // Case-sensitive
    let matcher_cs = FuzzyMatcher::new(0).with_case_sensitive(true);
    println!("\n  Case-sensitive matching (exact only):");
    println!(
        "    'Hello' ~ 'hello': {}",
        matcher_cs.is_match("Hello", "hello")
    );
    println!(
        "    'Hello' ~ 'Hello': {}",
        matcher_cs.is_match("Hello", "Hello")
    );

    // ============================================================
    // 6. Transposition Handling
    // ============================================================
    println!("\n\n🔄 Transposition Handling:");
    println!("─────────────────────────");

    let common_typos = vec![
        ("teh", "the"),
        ("recieve", "receive"),
        ("occured", "occurred"),
    ];

    println!("  Common transposition typos:\n");

    for (typo, correct) in common_typos {
        let lev_dist = levenshtein_distance(typo, correct);
        let dam_dist = damerau_levenshtein_distance(typo, correct);

        println!("    '{}' → '{}'", typo, correct);
        println!("      Levenshtein: {} edits", lev_dist);
        println!("      Damerau-Levenshtein: {} edit(s)", dam_dist);

        if dam_dist < lev_dist {
            println!("      ✓ Handles transposition better!");
        }

        println!();
    }

    // ============================================================
    // 7. Performance Comparison
    // ============================================================
    println!("\n⚡ Performance:");
    println!("──────────────");

    let large_dict: Vec<String> = (0..1000).map(|i| format!("word{:04}", i)).collect();

    println!("  Dictionary size: {} words", large_dict.len());

    // Measure BK-tree search
    let mut tree_large = BKTree::from_words(large_dict.clone());

    let search_query = "word0500";
    let start = std::time::Instant::now();
    let matches = tree_large.search(search_query, 2);
    let elapsed = start.elapsed();

    println!("  BK-tree search for '{}' (max distance: 2)", search_query);
    println!("    Found {} matches in {:?}", matches.len(), elapsed);

    // Linear search comparison
    let matcher_perf = FuzzyMatcher::new(2);
    let start2 = std::time::Instant::now();
    let linear_matches = matcher_perf.find_all_matches(search_query, &large_dict);
    let elapsed2 = start2.elapsed();

    println!("\n  Linear search (for comparison):");
    println!(
        "    Found {} matches in {:?}",
        linear_matches.len(),
        elapsed2
    );

    if elapsed2 > elapsed {
        let speedup = elapsed2.as_micros() as f64 / elapsed.as_micros() as f64;
        println!("    BK-tree is {:.1}x faster!", speedup);
    }

    println!("\n✅ Fuzzy search example complete!");
    println!("\n💡 Use Cases:");
    println!("  - Spell checking and autocorrect");
    println!("  - Search with typo tolerance");
    println!("  - Duplicate detection");
    println!("  - Query suggestions");
    println!("  - Fuzzy autocomplete");
}
