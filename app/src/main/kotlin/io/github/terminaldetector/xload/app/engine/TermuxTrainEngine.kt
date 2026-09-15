package io.github.terminaldetector.xload.app.engine

import android.content.Context
import android.os.PowerManager
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
 *
 * GPU is deliberately not wired in: termux-train's own VulkanBackend
 * (offloads only matmul(), everything else stays on NumPy/CPU regardless)
 * depends on `ameva-vulkan-runtime`, whose sole PyPI distribution is a
 * `py3-none-any` wheel -- pure Python, no compiled library for any platform
 * -- and its own `load_native_lib()` only *searches* fixed filesystem paths
 * for a prebuilt `libameva_vulkan.so` that nothing in the pip install
 * produces. Confirmed empirically (`AmevaVulkanLib().is_loaded()` ->
 * `False` right after a fresh `pip install ameva-vulkan-runtime`, in a
 * throwaway venv): switching backends would silently no-op back to the
 * same NumPy execution already in use, not accelerate anything. Real
 * on-device compute (this class's actual lever for "device overheats") is
 * throttled instead, below.
 */
class TermuxTrainEngine(private val context: Context) : TrainingEngine {
    private val json = Json { ignoreUnknownKeys = true }
    private val powerManager = context.getSystemService(Context.POWER_SERVICE) as? PowerManager

    @Volatile private var cancelRequested: (() -> Unit)? = null
    @Volatile private var cancelledForThermalWait = false

    override fun cancel() {
        cancelledForThermalWait = true
        cancelRequested?.invoke()
    }

    /**
     * PowerManager.currentThermalStatus has been available unconditionally
     * since API 29 (minSdk here is 30) -- no version guard needed. Sleeping
     * *inside* PyProgressCallback.onProgress works because that call is
     * synchronous Python->Kotlin (Chaquopy blocks the Python interpreter
     * thread on it): xload_trainer.py's training loop cannot advance past
     * `callback.onProgress(...)` until this returns, so blocking here
     * throttles the *actual* CPU-bound step loop at step boundaries without
     * touching a single line of Python. Slept in short chunks (not one long
     * sleep) so Cancel, requested mid-pause, takes effect within ~500ms
     * instead of waiting out the whole pause.
     */
    private fun maybeThrottleForThermal(last: TrainingProgress?, emit: (TrainingProgress) -> Unit) {
        val pm = powerManager ?: return
        val totalPauseMs = when (pm.currentThermalStatus) {
            PowerManager.THERMAL_STATUS_MODERATE -> 3_000L
            PowerManager.THERMAL_STATUS_SEVERE -> 8_000L
            PowerManager.THERMAL_STATUS_NONE, PowerManager.THERMAL_STATUS_LIGHT -> 0L
            else -> 20_000L // CRITICAL and worse
        }
        if (totalPauseMs <= 0L) return

        emit(
            (last ?: TrainingProgress(TrainingState.TRAINING, 0, 0, 0, 0, 0f)).copy(
                state = TrainingState.PAUSED_THERMAL,
                message = "Устройство греется — пауза ${totalPauseMs / 1000}с, чтобы не сажать CPU в троттлинг.",
            ),
        )
        var remaining = totalPauseMs
        val chunkMs = 500L
        while (remaining > 0 && !cancelledForThermalWait) {
            Thread.sleep(minOf(chunkMs, remaining))
            remaining -= chunkMs
        }
    }

    override fun train(config: TrainingConfig, dataset: List<DatasetSample>): Flow<TrainingProgress> = callbackFlow {
        val module = Python.getInstance().getModule("xload_trainer")
        cancelledForThermalWait = false
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

        var lastProgress: TrainingProgress? = null
        val callback = PyProgressCallback { progressJson ->
            maybeThrottleForThermal(lastProgress) { trySend(it) }
            runCatching { json.decodeFromString<PyTrainingProgress>(progressJson) }
                .onSuccess {
                    val mapped = it.toTrainingProgress(config.epochs)
                    lastProgress = mapped
                    trySend(mapped)
                }
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
