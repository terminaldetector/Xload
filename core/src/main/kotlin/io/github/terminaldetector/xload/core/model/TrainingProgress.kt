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
    /** Machine-readable failure reason (e.g. "MODEL_FILE_NOT_FOUND",
     *  "UNSUPPORTED_ARCHITECTURE") set alongside [message] on [TrainingState.FAILED]
     *  -- lets the UI (or a future retry/diagnostic flow) branch on the failure kind
     *  without parsing the human-readable message. Null for every other state. */
    val errorCode: String? = null,
    /** Micro-batches completed / total within the *current* effective (optimizer)
     *  step -- [step]/[totalSteps] alone don't distinguish "finished" from
     *  "accumulating gradient, optimizer hasn't stepped yet" once
     *  gradientAccumulationSteps > 1. Null for engines that don't report it
     *  (e.g. [io.github.terminaldetector.xload.core.engine.reference.ReferenceLoraEngine]). */
    val microStep: Int? = null,
    val totalMicroSteps: Int? = null,
    /** Set only on the terminal [TrainingState.COMPLETED] emission by engines that
     *  keep the trained weights in memory (e.g. [io.github.terminaldetector.xload.core.engine.reference.ReferenceLoraEngine]). */
    val checkpoint: AdapterCheckpoint? = null,
    /** Set only on the terminal [TrainingState.COMPLETED] emission by engines that
     *  write a multi-layer checkpoint straight to disk (e.g. a real termux-train
     *  backend saving a SafeTensors adapter with one lora_A/lora_B pair per
     *  layer) instead of returning it as an in-memory [AdapterCheckpoint]. */
    val checkpointPath: String? = null,
)
