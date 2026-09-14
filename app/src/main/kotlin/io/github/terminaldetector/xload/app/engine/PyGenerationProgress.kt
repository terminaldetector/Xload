package io.github.terminaldetector.xload.app.engine

import io.github.terminaldetector.xload.core.model.GenerationProgress
import io.github.terminaldetector.xload.core.model.GenerationState
import kotlinx.serialization.Serializable

/** Mirrors the JSON shape `xload_inference.py`'s `emit()` sends over the Chaquopy boundary. */
@Serializable
internal data class PyGenerationProgress(
    val state: String,
    val text: String = "",
    val message: String? = null,
) {
    fun toGenerationProgress() = GenerationProgress(
        state = runCatching { GenerationState.valueOf(state) }.getOrDefault(GenerationState.FAILED),
        text = text,
        message = message,
    )
}
