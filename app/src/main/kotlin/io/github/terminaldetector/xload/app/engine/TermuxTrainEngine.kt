package io.github.terminaldetector.xload.app.engine

import android.content.Context
import com.chaquo.python.Python
import io.github.terminaldetector.xload.core.engine.TrainingEngine
import io.github.terminaldetector.xload.core.model.DatasetSample
import io.github.terminaldetector.xload.core.model.TrainingConfig
import io.github.terminaldetector.xload.core.model.TrainingProgress
import io.github.terminaldetector.xload.core.model.TrainingState
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.withContext
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject

/**
 * Runs LoRA fine-tuning through the real termux-train library (pinned to
 * 1.1.3 in app/build.gradle.kts — see the comment there for why) via
 * Chaquopy, delegating to app/src/main/python/xload_trainer.py. When
 * [TrainingConfig.modelFilePath] points at an importable GGUF file, that
 * file's real (dequantized) weights and its own embedded tokenizer are used
 * (qwen2, llama, phi3, and gemma3 GGUF architecture families only so far); otherwise xload_trainer.py falls
 * back to termux-train's small bundled demo transformer. See that file's
 * docstring for exactly what each path proves.
 */
class TermuxTrainEngine(private val context: Context) : TrainingEngine {
    private val json = Json { ignoreUnknownKeys = true }

    @Volatile private var cancelRequested: (() -> Unit)? = null

    override fun cancel() {
        cancelRequested?.invoke()
    }

    override fun train(config: TrainingConfig, dataset: List<DatasetSample>): Flow<TrainingProgress> = callbackFlow {
        val module = Python.getInstance().getModule("xload_trainer")
        cancelRequested = { module.callAttr("cancel") }

        val configJson = buildJsonObject {
            putJsonObject("lora") {
                put("rank", config.lora.rank)
                put("alpha", config.lora.alpha)
            }
            put("epochs", config.epochs)
            put("batchSize", config.batchSize)
            put("learningRate", config.learningRate)
        }.toString()

        val datasetJson = json.encodeToString(ListSerializer(DatasetSample.serializer()), dataset)

        val adaptersDir = File(context.filesDir, "adapters").apply { mkdirs() }
        val checkpointPath = File(adaptersDir, "adapter-${System.currentTimeMillis()}.safetensors").absolutePath

        val callback = PyProgressCallback { progressJson ->
            runCatching { json.decodeFromString<PyTrainingProgress>(progressJson) }
                .onSuccess { trySend(it.toTrainingProgress(config.epochs)) }
        }

        withContext(Dispatchers.IO) {
            runCatching {
                module.callAttr(
                    "train", configJson, datasetJson, checkpointPath, callback,
                    config.modelFilePath ?: "",
                )
            }.onFailure { e ->
                trySend(
                    TrainingProgress(
                        state = TrainingState.FAILED,
                        epoch = 0,
                        totalEpochs = config.epochs,
                        step = 0,
                        totalSteps = 0,
                        loss = 0f,
                        message = e.message ?: e.toString(),
                    ),
                )
            }
        }

        close()
        awaitClose { cancelRequested = null }
    }
}
