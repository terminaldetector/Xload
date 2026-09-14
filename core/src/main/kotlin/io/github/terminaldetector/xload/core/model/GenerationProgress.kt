package io.github.terminaldetector.xload.core.model

enum class GenerationState { GENERATING, COMPLETED, FAILED, CANCELLED }

data class GenerationProgress(
    val state: GenerationState,
    /** Full generated text so far (not just the newest piece) — safe to display as-is on every
     *  update rather than appended, since a backend may need to redecode from scratch to keep
     *  multi-byte text correct (see [io.github.terminaldetector.xload.core.engine.InferenceEngine]). */
    val text: String,
    val message: String? = null,
)
