package io.github.terminaldetector.xload.core.model

import kotlinx.serialization.Serializable

/**
 * A trained LoRA adapter's weights plus enough metadata to describe how it was
 * produced. [io.github.terminaldetector.xload.core.engine.reference.ReferenceLoraEngine]
 * fills this in on completion; a production backend would populate the same shape
 * from the real adapter it trained.
 */
@Serializable
data class AdapterCheckpoint(
    val baseModel: BaseModel,
    val lora: LoraConfig,
    val dim: Int,
    /** Flattened rank x dim matrix, row-major. */
    val matrixA: FloatArray,
    /** Flattened dim x rank matrix, row-major. */
    val matrixB: FloatArray,
    val finalLoss: Float,
    val trainedSamples: Int,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is AdapterCheckpoint) return false
        return baseModel == other.baseModel &&
            lora == other.lora &&
            dim == other.dim &&
            matrixA.contentEquals(other.matrixA) &&
            matrixB.contentEquals(other.matrixB) &&
            finalLoss == other.finalLoss &&
            trainedSamples == other.trainedSamples
    }

    override fun hashCode(): Int {
        var result = baseModel.hashCode()
        result = 31 * result + lora.hashCode()
        result = 31 * result + dim
        result = 31 * result + matrixA.contentHashCode()
        result = 31 * result + matrixB.contentHashCode()
        result = 31 * result + finalLoss.hashCode()
        result = 31 * result + trainedSamples
        return result
    }
}
