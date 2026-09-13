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
    /** Set only on the terminal [TrainingState.COMPLETED] emission by engines that
     *  keep the trained weights in memory (e.g. [io.github.terminaldetector.xload.core.engine.reference.ReferenceLoraEngine]). */
    val checkpoint: AdapterCheckpoint? = null,
    /** Set only on the terminal [TrainingState.COMPLETED] emission by engines that
     *  write a multi-layer checkpoint straight to disk (e.g. a real termux-train
     *  backend saving a SafeTensors adapter with one lora_A/lora_B pair per
     *  layer) instead of returning it as an in-memory [AdapterCheckpoint]. */
    val checkpointPath: String? = null,
)
