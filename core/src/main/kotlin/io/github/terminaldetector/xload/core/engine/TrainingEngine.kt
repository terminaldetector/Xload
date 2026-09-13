package io.github.terminaldetector.xload.core.engine

import io.github.terminaldetector.xload.core.model.DatasetSample
import io.github.terminaldetector.xload.core.model.TrainingConfig
import io.github.terminaldetector.xload.core.model.TrainingProgress
import kotlinx.coroutines.flow.Flow

/**
 * Runs LoRA fine-tuning and reports progress as it goes.
 *
 * This is the pluggable backend boundary from the plan's architecture stage:
 * a termux-train (Python/NumPy), MobileFineTuner (NDK/C++) or ExecuTorch-backed
 * implementation can be dropped in behind this interface without touching the UI
 * layer. [io.github.terminaldetector.xload.core.engine.reference.ReferenceLoraEngine]
 * is the only implementation shipped so far — see its KDoc for what it does and
 * does not prove.
 */
interface TrainingEngine {
    fun train(config: TrainingConfig, dataset: List<DatasetSample>): Flow<TrainingProgress>

    /** Requests the current [train] run to stop at the next safe point. */
    fun cancel()
}
