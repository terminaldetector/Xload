package io.github.terminaldetector.xload.app.engine

import com.chaquo.python.Python
import io.github.terminaldetector.xload.core.engine.InferenceEngine
import io.github.terminaldetector.xload.core.model.DatasetSample
import io.github.terminaldetector.xload.core.model.GenerationParams
import io.github.terminaldetector.xload.core.model.GenerationProgress
import io.github.terminaldetector.xload.core.model.GenerationState
import io.github.terminaldetector.xload.core.model.LoraConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.withContext
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject

/**
 * Generates text through termux-train via Chaquopy, delegating to
 * app/src/main/python/xload_inference.py — the inference counterpart to
 * [TermuxTrainEngine]. Rebuilds the exact same base architecture and LoRA
 * layer shapes [TermuxTrainEngine] trained ([lora] and [modelFilePath] must
 * be the same values that produced [checkpointPath]) and loads the saved
 * adapter weights into it before generating, or generates from the freshly
 * LoRA-injected but untrained model if [checkpointPath] doesn't point at a
 * real file.
 */
class TermuxInferenceEngine : InferenceEngine {
    private val json = Json { ignoreUnknownKeys = true }

    @Volatile private var cancelRequested: (() -> Unit)? = null

    override fun cancel() {
        cancelRequested?.invoke()
    }

    override fun generate(
        lora: LoraConfig,
        modelFilePath: String?,
        checkpointPath: String,
        prompt: String,
        params: GenerationParams,
    ): Flow<GenerationProgress> = callbackFlow {
        val module = Python.getInstance().getModule("xload_inference")
        cancelRequested = { module.callAttr("cancel") }

        val configJson = buildJsonObject {
            putJsonObject("lora") {
                put("rank", lora.rank)
                put("alpha", lora.alpha)
            }
            put("maxNewTokens", params.maxNewTokens)
            put("temperature", params.temperature)
            put("topK", params.topK)
        }.toString()

        val callback = PyProgressCallback { progressJson ->
            runCatching { json.decodeFromString<PyGenerationProgress>(progressJson) }
                .onSuccess { trySend(it.toGenerationProgress()) }
        }

        withContext(Dispatchers.IO) {
            runCatching {
                module.callAttr(
                    "generate", configJson, prompt, checkpointPath, callback,
                    modelFilePath ?: "",
                )
            }.onFailure { e ->
                trySend(
                    GenerationProgress(
                        state = GenerationState.FAILED,
                        text = "",
                        message = e.message ?: e.toString(),
                    ),
                )
            }
        }

        close()
        awaitClose { cancelRequested = null }
    }

    override fun analyzePersonality(
        dataset: List<DatasetSample>,
        modelFilePath: String?,
        params: GenerationParams,
    ): Flow<GenerationProgress> = callbackFlow {
        val module = Python.getInstance().getModule("xload_inference")
        cancelRequested = { module.callAttr("cancel") }

        val datasetJson = json.encodeToString(ListSerializer(DatasetSample.serializer()), dataset)
        val configJson = buildJsonObject {
            put("maxNewTokens", params.maxNewTokens)
            put("temperature", params.temperature)
            put("topK", params.topK)
        }.toString()

        val callback = PyProgressCallback { progressJson ->
            runCatching { json.decodeFromString<PyGenerationProgress>(progressJson) }
                .onSuccess { trySend(it.toGenerationProgress()) }
        }

        withContext(Dispatchers.IO) {
            runCatching {
                module.callAttr(
                    "analyze_personality", datasetJson, configJson, callback,
                    modelFilePath ?: "",
                )
            }.onFailure { e ->
                trySend(
                    GenerationProgress(
                        state = GenerationState.FAILED,
                        text = "",
                        message = e.message ?: e.toString(),
                    ),
                )
            }
        }

        close()
        awaitClose { cancelRequested = null }
    }
}
