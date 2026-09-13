package io.github.terminaldetector.xload.core.checkpoint

import io.github.terminaldetector.xload.core.model.AdapterCheckpoint
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

/**
 * Serializes an [AdapterCheckpoint] to JSON for local export.
 *
 * This is Xload's own reference checkpoint format, not a real SafeTensors or
 * GGUF file — a production backend (termux-train / MobileFineTuner / ExecuTorch)
 * would export a HuggingFace-compatible `.safetensors` adapter instead. This
 * format exists so the save/export pipeline stage has something real to do
 * end-to-end ahead of that integration.
 */
object AdapterCheckpointJson {
    private val json = Json {
        prettyPrint = true
        encodeDefaults = true
    }

    fun encode(checkpoint: AdapterCheckpoint): String = json.encodeToString(checkpoint)

    fun decode(content: String): AdapterCheckpoint = json.decodeFromString(content)
}
