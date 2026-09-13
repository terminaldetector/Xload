package io.github.terminaldetector.xload.core.model

enum class Quantization(val label: String) {
    Q4_0("4-bit (Q4_0)"),
    Q4_K_M("4-bit (Q4_K_M)"),
    Q8_0("8-bit (Q8_0)"),
}

/**
 * A base LLM Xload knows how to fine-tune. Weights are never bundled with the app;
 * the user imports a GGUF/SafeTensors file matching one of these entries (see the
 * "Целевые модели" list in the project plan).
 */
enum class BaseModel(
    val displayName: String,
    val paramCountBillions: Double,
    val defaultQuantization: Quantization,
    val minRamMb: Int,
    val targetModules: List<String>,
) {
    QWEN2_5_0_5B(
        displayName = "Qwen2.5-0.5B",
        paramCountBillions = 0.5,
        defaultQuantization = Quantization.Q4_K_M,
        minRamMb = 2048,
        targetModules = listOf("q_proj", "k_proj", "v_proj", "o_proj"),
    ),
    QWEN2_5_1_5B(
        displayName = "Qwen2.5-1.5B",
        paramCountBillions = 1.5,
        defaultQuantization = Quantization.Q4_K_M,
        minRamMb = 4096,
        targetModules = listOf("q_proj", "k_proj", "v_proj", "o_proj"),
    ),
    GEMMA_3_1B(
        displayName = "Gemma-3-1B",
        paramCountBillions = 1.0,
        defaultQuantization = Quantization.Q4_0,
        minRamMb = 4096,
        targetModules = listOf("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    ),
    LLAMA_3_2_1B(
        displayName = "Llama-3.2-1B",
        paramCountBillions = 1.0,
        defaultQuantization = Quantization.Q4_K_M,
        minRamMb = 4096,
        targetModules = listOf("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    ),
    PHI_3_MINI(
        displayName = "Phi-3-mini",
        paramCountBillions = 3.8,
        defaultQuantization = Quantization.Q4_K_M,
        minRamMb = 6144,
        targetModules = listOf("qkv_proj", "o_proj", "gate_up_proj", "down_proj"),
    ),
}
