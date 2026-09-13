package io.github.terminaldetector.xload.core.model

import kotlinx.serialization.Serializable

/** Hyperparameters for a LoRA adapter (the plan's "Настройка LoRA" pipeline stage). */
@Serializable
data class LoraConfig(
    val rank: Int = 16,
    val alpha: Int = 32,
    val dropout: Float = 0.05f,
    val targetModules: Set<String> = setOf("q_proj", "k_proj", "v_proj", "o_proj"),
) {
    init {
        require(rank in 1..256) { "LoRA rank must be in 1..256, was $rank" }
        require(alpha in 1..512) { "LoRA alpha must be in 1..512, was $alpha" }
        require(dropout in 0f..1f) { "Dropout must be in 0..1, was $dropout" }
        require(targetModules.isNotEmpty()) { "At least one target module is required" }
    }

    /** Standard LoRA output scaling factor alpha/rank. */
    val scaling: Float get() = alpha.toFloat() / rank.toFloat()
}
