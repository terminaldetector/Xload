package io.github.terminaldetector.xload.app.viewmodel

/**
 * Editable draft of the numeric training hyperparameters, kept separate from
 * [io.github.terminaldetector.xload.core.model.TrainingConfig] so the UI has
 * somewhere to hold slider values while the user is adjusting them, before
 * they're assembled into a real (validated) TrainingConfig at "start training"
 * time.
 */
data class TrainingHyperparams(
    val batchSize: Int = 2,
    val gradientAccumulationSteps: Int = 4,
    val epochs: Int = 3,
    val learningRate: Float = 1.5e-4f,
)
