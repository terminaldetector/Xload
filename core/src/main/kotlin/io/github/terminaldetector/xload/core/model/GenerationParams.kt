package io.github.terminaldetector.xload.core.model

/** Sampling knobs for [io.github.terminaldetector.xload.core.engine.InferenceEngine.generate]. */
data class GenerationParams(
    val maxNewTokens: Int = 100,
    val temperature: Float = 0.8f,
    val topK: Int = 40,
) {
    init {
        // Upper bound matches xload_inference.py's MAX_NEW_TOKENS_CAP: without a KV-cache, every
        // generated token re-runs a full forward pass over the whole sequence so far (O(n^2)), so
        // a request beyond what the backend will actually honor would be misleading, not generous.
        require(maxNewTokens in 1..200) { "maxNewTokens must be in 1..200, was $maxNewTokens" }
        require(temperature in 0f..2f) { "temperature must be in 0..2, was $temperature" }
        require(topK >= 0) { "topK must be >= 0 (0 disables top-k filtering), was $topK" }
    }
}
