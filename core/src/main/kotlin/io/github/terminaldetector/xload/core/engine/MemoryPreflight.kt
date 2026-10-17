package io.github.terminaldetector.xload.core.engine

import io.github.terminaldetector.xload.core.model.BaseModel
import io.github.terminaldetector.xload.core.model.LoraConfig
import io.github.terminaldetector.xload.core.model.ModelMode

/**
 * Estimates whether a training run is likely to fit in available RAM,
 * before starting it -- "warn and suggest reducing seq/microbatch... rather
 * than hitting OOM first" (the P0.5 requirement this implements). A
 * deliberately rough, honest estimate built only from numbers this project
 * actually has ([BaseModel.paramCountBillions], [LoraConfig.rank]/
 * [LoraConfig.targetModules]) -- a precise per-tensor accounting needs the
 * real GGUF header (architecture-specific d_model/numLayers/dFf), which
 * isn't available in pure Kotlin without parsing the file a second time
 * outside Python. A rough, clearly-labeled-as-rough number that points in
 * the right direction beats a precise-looking one for a shape this code
 * doesn't actually compute -- see [BaseModel]'s `PHI_3_MINI` doc comment
 * for how far off a quantized-file-size-shaped guess can be once real
 * dequantization-to-float32 (confirmed by reading
 * gguf_llama_family_model.py's `_dequantized_numpy`) is accounted for.
 */
object MemoryPreflight {
    private const val BYTES_PER_PARAM_F32 = 4L
    private const val BYTES_PER_BILLION = 1_000_000_000L

    /** Fixed reserve for the Android OS + launcher/UI + the JVM/Chaquopy
     *  runtime itself, on top of whatever the model needs -- not part of
     *  "the model" but still real memory a training run competes with on a
     *  phone that's also, e.g., still rendering the training screen. */
    const val OS_RUNTIME_RESERVE_MB = 512L

    data class Estimate(
        /** Dequantized-to-float32 frozen base weights. For [ModelMode.REAL_GGUF]
         *  this is the model's *actual* resident footprint, not its on-disk
         *  (possibly 4-bit quantized) file size -- every GGUF tensor becomes a
         *  float32 NumPy array at load time regardless of source quantization. */
        val baseWeightsBytes: Long,
        /** LoRA A/B trainable parameters plus AdamW's two float32 moment
         *  buffers per trainable param -- small relative to [baseWeightsBytes]
         *  for any rank [LoraConfig] actually allows (<=256), which is the
         *  point: it's deliberately not the dominant term being warned about. */
        val loraAndOptimizerBytes: Long,
        val osAndRuntimeReserveBytes: Long,
    ) {
        val totalBytes: Long get() = baseWeightsBytes + loraAndOptimizerBytes + osAndRuntimeReserveBytes
        val totalMb: Long get() = totalBytes / (1024 * 1024)
    }

    fun estimate(baseModel: BaseModel, lora: LoraConfig, modelMode: ModelMode): Estimate {
        val baseWeightsBytes = if (modelMode == ModelMode.REAL_GGUF) {
            (baseModel.paramCountBillions * BYTES_PER_BILLION).toLong() * BYTES_PER_PARAM_F32
        } else {
            16L * 1024 * 1024 // TinyTransformerLM: a few hundred thousand params, negligible either way
        }

        // Rough per-module width proxy: spread the model's total params
        // evenly across its targeted modules and take a square root, so
        // this scales with model size (a 3.8B model's LoRA layers are
        // wider than a 0.5B model's) without needing the real d_model/
        // numLayers the GGUF header alone has.
        val targetCount = maxOf(1, lora.targetModules.size)
        val approxModuleWidth = Math.sqrt(
            (baseModel.paramCountBillions * BYTES_PER_BILLION) / targetCount,
        ).toLong()
        val loraParams = targetCount.toLong() * 2L * lora.rank * approxModuleWidth
        // param itself + AdamW's m and v moment buffers = 3x.
        val loraAndOptimizerBytes = loraParams * BYTES_PER_PARAM_F32 * 3L

        return Estimate(
            baseWeightsBytes = baseWeightsBytes,
            loraAndOptimizerBytes = loraAndOptimizerBytes,
            osAndRuntimeReserveBytes = OS_RUNTIME_RESERVE_MB * 1024 * 1024,
        )
    }

    /** `availableMemBytes`: the device's actual currently-available memory
     *  (e.g. `ActivityManager.MemoryInfo.availMem` on Android) -- passed in
     *  rather than read here so this stays a plain, unit-testable function
     *  with no Android dependency. `safetyFraction` < 1 leaves headroom
     *  instead of warning only once the estimate exactly consumes every
     *  free byte: 0.8 by default warns once the estimate would use more
     *  than 80% of what's currently free. */
    fun likelyExceedsAvailableMemory(
        estimate: Estimate,
        availableMemBytes: Long,
        safetyFraction: Double = 0.8,
    ): Boolean = availableMemBytes > 0 && estimate.totalBytes > (availableMemBytes * safetyFraction).toLong()
}
