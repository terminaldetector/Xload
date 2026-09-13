package io.github.terminaldetector.xload.core.model

enum class TrainingState { PREPARING, TRAINING, PAUSED_THERMAL, COMPLETED, FAILED, CANCELLED }

data class TrainingProgress(
    val state: TrainingState,
    val epoch: Int,
    val totalEpochs: Int,
    val step: Int,
    val totalSteps: Int,
    val loss: Float,
    val etaSeconds: Long? = null,
    val message: String? = null,
    /** Set only on the terminal [TrainingState.COMPLETED] emission. */
    val checkpoint: AdapterCheckpoint? = null,
)
