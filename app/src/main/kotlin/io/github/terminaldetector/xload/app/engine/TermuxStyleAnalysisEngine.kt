package io.github.terminaldetector.xload.app.engine

import com.chaquo.python.Python
import io.github.terminaldetector.xload.core.engine.StyleAnalysisEngine
import io.github.terminaldetector.xload.core.model.DatasetSample
import io.github.terminaldetector.xload.core.model.StyleAnalysisParams
import io.github.terminaldetector.xload.core.model.StyleAnalysisProgress
import io.github.terminaldetector.xload.core.model.StyleAnalysisState
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

/**
 * Runs the FBDP stylometric analysis through Chaquopy, delegating to
 * app/src/main/python/fbdp_engine.py — unlike [TermuxTrainEngine]/[TermuxInferenceEngine], this
 * touches no model at all (no GGUF file, no LoRA, no checkpoint): it's pure numpy/statistics over
 * the dataset's own text, so it works with just a loaded dataset.
 */
class TermuxStyleAnalysisEngine : StyleAnalysisEngine {
    private val json = Json { ignoreUnknownKeys = true }

    @Volatile private var cancelRequested: (() -> Unit)? = null

    override fun cancel() {
        cancelRequested?.invoke()
    }

    override fun analyze(
        dataset: List<DatasetSample>,
        params: StyleAnalysisParams,
    ): Flow<StyleAnalysisProgress> = callbackFlow {
        val module = Python.getInstance().getModule("fbdp_engine")
        cancelRequested = { module.callAttr("cancel") }

        val datasetJson = json.encodeToString(ListSerializer(DatasetSample.serializer()), dataset)
        val configJson = buildJsonObject {
            put("rounds", params.rounds)
        }.toString()

        val callback = PyProgressCallback { progressJson ->
            runCatching { json.decodeFromString<PyStyleAnalysisProgress>(progressJson) }
                .onSuccess { trySend(it.toStyleAnalysisProgress()) }
        }

        withContext(Dispatchers.IO) {
            runCatching {
                module.callAttr("analyze", datasetJson, configJson, callback)
            }.onFailure { e ->
                trySend(
                    StyleAnalysisProgress(
                        state = StyleAnalysisState.FAILED,
                        message = e.message ?: e.toString(),
                    ),
                )
            }
        }

        close()
        awaitClose { cancelRequested = null }
    }
}
