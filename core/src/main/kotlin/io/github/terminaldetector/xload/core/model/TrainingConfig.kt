package io.github.terminaldetector.xload.core.model

data class TrainingConfig(
    val baseModel: BaseModel,
    val lora: LoraConfig = LoraConfig(),
    val batchSize: Int = 2,
    val gradientAccumulationSteps: Int = 4,
    val epochs: Int = 3,
    val learningRate: Float = 1.5e-4f,
) {
    init {
        require(batchSize in 1..64) { "batchSize must be in 1..64, was $batchSize" }
        require(gradientAccumulationSteps in 1..64) {
            "gradientAccumulationSteps must be in 1..64, was $gradientAccumulationSteps"
        }
        require(epochs in 1..100) { "epochs must be in 1..100, was $epochs" }
        require(learningRate in 1e-6f..1e-2f) {
            "learningRate must be in 1e-6..1e-2, was $learningRate"
        }
    }

    val effectiveBatchSize: Int get() = batchSize * gradientAccumulationSteps
}
