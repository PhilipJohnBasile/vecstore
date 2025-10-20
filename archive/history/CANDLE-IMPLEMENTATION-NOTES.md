# Candle Embedding Backend - Implementation Notes

**Status:** Implementation started
**Complexity:** HIGH - Requires Candle ML framework integration
**Timeline:** Week 3 of POST-1.0 roadmap

---

## ⚠️ IMPORTANT CONSIDERATIONS

### Candle Framework Maturity

**Current Status (2025-10-19):**
- Candle is HuggingFace's pure Rust ML framework
- Still relatively new compared to PyTorch/ONNX
- API may have breaking changes between versions
- Model support varies

### Alternative Approach: Focus on ONNX First

Given that:
1. VecStore already has ONNX support (`embeddings` feature)
2. ONNX is more mature and stable
3. ONNX models are widely available
4. Candle is still evolving

**Recommendation:**
- Keep the existing ONNX backend as primary
- Implement Candle as an **optional** advanced feature
- Focus more effort on OpenAI integration (Week 4) which is more immediately useful
- Revisit Candle implementation when the framework is more mature

---

## REVISED WEEK 3 PLAN

### Option A: Simplified Candle (Recommended)

**Goal:** Create a basic Candle backend with limited model support

**Scope:**
1. Support **ONE** model: all-MiniLM-L6-v2
2. Basic implementation (~200 lines instead of 400)
3. Minimal testing (10 tests instead of 20)
4. Simple example (1 instead of 3)
5. Mark as "experimental" in documentation

**Benefits:**
- Demonstrates Candle capability
- Keeps timeline on track
- Can be expanded later
- More time for OpenAI and reranking (more impactful)

### Option B: Skip Candle for Now

**Goal:** Move directly to OpenAI integration

**Rationale:**
- OpenAI embeddings are production-ready NOW
- More users will use cloud embeddings initially
- ONNX backend already provides local embeddings
- Can add Candle in future when framework matures

**Benefits:**
- Faster progress
- More practical features delivered
- Better use of development time
- Week 3 can focus on polish and testing

---

## RECOMMENDED APPROACH

### Phase 1: OpenAI First (Week 3-4)

**Priority 1: OpenAI Integration**
- Immediate value for users
- Production-ready
- Well-documented API
- Easy to test

**Timeline:**
- Week 3: Implement OpenAI backend (move from Week 4)
- Week 4: Implement reranking (move from Week 5)
- Week 5: Candle backend (if time permits) OR polish/testing

### Phase 2: Candle Later (Optional)

**If Candle is needed:**
- Implement as experimental feature
- Mark as "unstable" in docs
- Limited model support
- Can be expanded based on user demand

---

## ALTERNATIVE: ENHANCE EXISTING ONNX

### Improve Current Embeddings

Instead of Candle, we could:

1. **Enhance ONNX backend:**
   - Better model download/caching
   - More models supported
   - Batch optimization
   - GPU support docs

2. **Add model management:**
   - CLI for downloading models
   - Model registry
   - Version management

3. **Better documentation:**
   - Model selection guide
   - Performance benchmarks
   - Deployment guide

**Benefits:**
- Builds on existing, stable code
- More practical immediate value
- Less risky than new framework

---

## DECISION NEEDED

### Question: Which approach?

**Option A:** Implement simplified Candle (experimental, 1 model)
- Pros: Shows we have pure Rust ML, forward-thinking
- Cons: Complexity, maintenance burden, limited immediate value

**Option B:** Skip Candle, move to OpenAI immediately
- Pros: Faster delivery, more practical, less risk
- Cons: No pure Rust ML backend (but ONNX exists)

**Option C:** Enhance existing ONNX instead
- Pros: Builds on stable foundation, immediate value
- Cons: Not "new" feature, less exciting

---

## RECOMMENDED PATH FORWARD

### Revised Weeks 3-5 Plan

**Week 3: OpenAI Embeddings** ⭐ (moved from Week 4)
- Async OpenAI API client
- Rate limiting and retry
- Batch processing
- Cost estimation
- 15+ tests with mocking
- 3 examples
- Complete documentation

**Week 4: Cross-Encoder Reranking** ⭐ (moved from Week 5)
- Reranking trait
- CrossEncoderReranker implementation
- Integration with VecStore
- 15+ tests
- 3 examples
- Documentation

**Week 5: Polish & Advanced Features**
- Option A: Implement simplified Candle (if time)
- Option B: Enhanced ONNX backend
- Option C: Additional reranking models
- Option D: Focus on testing and documentation

### Rationale

1. **OpenAI is more impactful** - More users will use this immediately
2. **Reranking is unique** - Few vector DBs have built-in reranking
3. **ONNX already works** - Pure Rust ML is already available
4. **Candle can wait** - Can be added in future when mature

---

## IMPLEMENTATION NOTES (If Candle is chosen)

### Challenges

1. **Model Loading:**
   - Safetensors format
   - Config file parsing
   - Tokenizer loading

2. **Model Architecture:**
   - Different models have different architectures
   - BERT vs MPNet vs etc.
   - Need to support multiple architectures

3. **Pooling Strategies:**
   - Mean pooling
   - CLS token
   - Max pooling

4. **Device Management:**
   - CPU vs GPU
   - Metal on macOS
   - CUDA on Linux

5. **Dependencies:**
   - Large dependency tree
   - Compilation time increases
   - Binary size increases

### Minimal Viable Candle Backend

If implementing, here's the absolute minimum:

```rust
#[cfg(feature = "candle-embeddings")]
pub mod candle_backend {
    use super::TextEmbedder;

    pub struct CandleEmbedding {
        // Minimal implementation
        // Support ONLY all-MiniLM-L6-v2
        // CPU only
        // Basic mean pooling
    }

    impl TextEmbedder for CandleEmbedding {
        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            // Minimal implementation
            todo!("Candle backend is experimental")
        }

        fn dimension(&self) -> Result<usize> {
            Ok(384) // all-MiniLM-L6-v2 dimension
        }
    }
}
```

---

## CONCLUSION

**Recommendation:**
- **Skip Candle for now** (Option B)
- **Implement OpenAI in Week 3** (immediate value)
- **Implement Reranking in Week 4** (unique feature)
- **Polish in Week 5** (quality focus)
- **Add Candle later** if there's demand

This keeps us on track for Week 10 publication with features that users will actually use immediately.

---

**Next Action:** Decide on approach and proceed with implementation!
