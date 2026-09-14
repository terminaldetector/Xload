package io.github.terminaldetector.xload.core.engine

import io.github.terminaldetector.xload.core.model.DatasetSample
import io.github.terminaldetector.xload.core.model.GenerationParams
import io.github.terminaldetector.xload.core.model.GenerationProgress
import io.github.terminaldetector.xload.core.model.LoraConfig
import io.github.terminaldetector.xload.core.model.TrainingConfig
import kotlinx.coroutines.flow.Flow

/**
 * Generates text from a (trained or base) model and reports progress as it goes — the inference
 * counterpart to [TrainingEngine], and the same kind of pluggable backend boundary: a llama.cpp /
 * MLC-LLM / MediaPipe LlmInference-backed implementation can be dropped in behind this interface
 * later without touching the UI layer (see the README roadmap).
 */
interface InferenceEngine {
    /**
     * [lora] and [modelFilePath] must be the same values ([TrainingConfig.lora],
     * [TrainingConfig.modelFilePath]) that produced [checkpointPath] — an implementation may need
     * to rebuild the exact base architecture and LoRA layer shapes the checkpoint's adapter
     * weights were saved from before it can load them back in. Deliberately takes just these two
     * pieces of [TrainingConfig] rather than the whole thing: the rest (epochs, batch size, ...)
     * has no meaning for inference, and the caller doesn't keep the original [TrainingConfig]
     * instance around past [TrainingEngine.train] anyway.
     */
    fun generate(
        lora: LoraConfig,
        modelFilePath: String?,
        checkpointPath: String,
        prompt: String,
        params: GenerationParams = GenerationParams(),
    ): Flow<GenerationProgress>

    /**
     * Builds a short free-text "personality portrait" — communication style, recurring themes,
     * decision patterns — read straight off a handful of [dataset]'s own examples by the base
     * model's general summarization ability. Deliberately no LoRA/checkpoint involved (unlike
     * [generate]): describing how someone writes is a capability the base weights already have,
     * so this works even before training finishes. A first, small step towards the fuller
     * "personality map"/fractal-memory idea discussed for later — not RAPTOR-style clustering,
     * not a structured trait model, just a free-text summary (see README).
     */
    fun analyzePersonality(
        dataset: List<DatasetSample>,
        modelFilePath: String?,
        params: GenerationParams = GenerationParams(),
    ): Flow<GenerationProgress>

    /** Requests the current [generate]/[analyzePersonality] run to stop at the next safe point. */
    fun cancel()
}
