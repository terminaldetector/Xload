package io.github.terminaldetector.xload.core.engine.reference

import io.github.terminaldetector.xload.core.engine.TrainingEngine
import io.github.terminaldetector.xload.core.model.AdapterCheckpoint
import io.github.terminaldetector.xload.core.model.DatasetSample
import io.github.terminaldetector.xload.core.model.TrainingConfig
import io.github.terminaldetector.xload.core.model.TrainingProgress
import io.github.terminaldetector.xload.core.model.TrainingState
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.yield
import kotlin.random.Random

/**
 * CPU-only reference implementation of the LoRA update rule: for a frozen weight
 * matrix `W` (dim x dim) it learns low-rank adapters `A` (rank x dim) and `B`
 * (dim x rank) by SGD so that `y = W*x + scaling*(B*(A*x))` fits target vectors,
 * with `B` initialized to zero per standard LoRA practice.
 *
 * Inputs and targets are derived deterministically from dataset text (see
 * [TextVectorizer]) against a random fixed "teacher" matrix, rather than run
 * through a real tokenizer and transformer — this is not a language model and
 * does not produce a usable adapter. It exists to exercise the real LoRA math
 * and the config -> engine -> progress pipeline end-to-end (including cancellation
 * and loss going down) on any device with no GPU/NDK/Python dependency, ahead of
 * wiring in a production backend (termux-train, MobileFineTuner, ExecuTorch...)
 * behind the same [TrainingEngine] interface.
 */
class ReferenceLoraEngine(
    private val dim: Int = 32,
    private val seed: Long = 42L,
) : TrainingEngine {
    @Volatile private var cancelled = false

    override fun cancel() {
        cancelled = true
    }

    override fun train(config: TrainingConfig, dataset: List<DatasetSample>) = flow {
        cancelled = false
        if (dataset.isEmpty()) {
            emit(TrainingProgress(TrainingState.FAILED, 0, config.epochs, 0, 0, 0f, message = "Dataset is empty"))
            return@flow
        }

        val random = Random(seed)
        val rank = config.lora.rank
        val w = Matrix.random(dim, dim, random, scale = 0.1f) // frozen base projection
        val a = Matrix.random(rank, dim, random, scale = 1f / dim) // trainable
        val b = Matrix.zeros(dim, rank) // trainable, starts at zero so initial delta = 0
        val teacher = Matrix.random(dim, dim, Random(seed xor 0x5DEECE66DL), scale = 0.1f)
        val scaling = config.lora.scaling

        val vectorizer = TextVectorizer(dim)
        val inputs = dataset.map { vectorizer.vectorize(it) }
        val targets = inputs.map { teacher.timesVector(it) }

        val stepsPerEpoch = (dataset.size + config.batchSize - 1) / config.batchSize
        val totalSteps = stepsPerEpoch * config.epochs
        val startTime = System.nanoTime()
        var globalStep = 0
        var lastLoss = 0f

        emit(TrainingProgress(TrainingState.PREPARING, 0, config.epochs, 0, totalSteps, 0f))

        for (epoch in 1..config.epochs) {
            for (batchStart in dataset.indices step config.batchSize) {
                if (cancelled) {
                    emit(TrainingProgress(TrainingState.CANCELLED, epoch, config.epochs, globalStep, totalSteps, 0f))
                    return@flow
                }

                val batchEnd = minOf(batchStart + config.batchSize, dataset.size)
                val gradA = Matrix.zeros(rank, dim)
                val gradB = Matrix.zeros(dim, rank)
                var batchLoss = 0f

                for (i in batchStart until batchEnd) {
                    val x = inputs[i]
                    val target = targets[i]
                    val u = a.timesVector(x) // rank
                    val v = b.timesVector(u) // dim
                    val base = w.timesVector(x) // dim

                    val error = FloatArray(dim) { d -> (base[d] + scaling * v[d]) - target[d] }
                    batchLoss += error.sumOf { (it * it).toDouble() }.toFloat() / dim

                    // dLoss/dv = (2/dim)*scaling*error ; backprop through B then A.
                    val g = FloatArray(dim) { d -> (2f / dim) * scaling * error[d] }
                    for (d in 0 until dim) {
                        for (k in 0 until rank) gradB[d, k] += g[d] * u[k]
                    }
                    val btError = FloatArray(rank) { k ->
                        var sum = 0f
                        for (d in 0 until dim) sum += b[d, k] * g[d]
                        sum
                    }
                    for (k in 0 until rank) {
                        for (c in 0 until dim) gradA[k, c] += btError[k] * x[c]
                    }
                }

                val n = (batchEnd - batchStart).toFloat()
                val lr = config.learningRate
                for (idx in a.data.indices) a.data[idx] -= lr * (gradA.data[idx] / n)
                for (idx in b.data.indices) b.data[idx] -= lr * (gradB.data[idx] / n)
                batchLoss /= n
                lastLoss = batchLoss
                globalStep++

                val elapsedS = (System.nanoTime() - startTime) / 1_000_000_000.0
                val etaS = (elapsedS / globalStep * (totalSteps - globalStep)).toLong()

                emit(
                    TrainingProgress(
                        state = TrainingState.TRAINING,
                        epoch = epoch,
                        totalEpochs = config.epochs,
                        step = globalStep,
                        totalSteps = totalSteps,
                        loss = batchLoss,
                        etaSeconds = etaS,
                    ),
                )
                yield()
            }
        }

        val checkpoint = AdapterCheckpoint(
            baseModel = config.baseModel,
            lora = config.lora,
            dim = dim,
            matrixA = a.data.copyOf(),
            matrixB = b.data.copyOf(),
            finalLoss = lastLoss,
            trainedSamples = dataset.size,
        )
        emit(
            TrainingProgress(
                state = TrainingState.COMPLETED,
                epoch = config.epochs,
                totalEpochs = config.epochs,
                step = totalSteps,
                totalSteps = totalSteps,
                loss = lastLoss,
                checkpoint = checkpoint,
            ),
        )
    }
}
