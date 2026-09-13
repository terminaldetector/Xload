package io.github.terminaldetector.xload.app.engine

import io.github.terminaldetector.xload.core.model.TrainingProgress
import io.github.terminaldetector.xload.core.model.TrainingState
import kotlinx.serialization.Serializable

/** Mirrors the JSON shape `xload_trainer.py`'s `emit()` sends over the Chaquopy boundary. */
@Serializable
internal data class PyTrainingProgress(
    val state: String,
    val epoch: Int,
    val totalEpochs: Int,
    val step: Int,
    val totalSteps: Int,
    val loss: Float,
    val etaSeconds: Long? = null,
    val message: String? = null,
    val checkpointPath: String? = null,
) {
    fun toTrainingProgress(totalEpochsFallback: Int) = TrainingProgress(
        state = runCatching { TrainingState.valueOf(state) }.getOrDefault(TrainingState.FAILED),
        epoch = epoch,
        totalEpochs = if (totalEpochs > 0) totalEpochs else totalEpochsFallback,
        step = step,
        totalSteps = totalSteps,
        loss = loss,
        etaSeconds = etaSeconds,
        message = message,
        checkpointPath = checkpointPath,
    )
}
