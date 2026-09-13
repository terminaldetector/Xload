package io.github.terminaldetector.xload.core.engine.reference

import io.github.terminaldetector.xload.core.model.BaseModel
import io.github.terminaldetector.xload.core.model.DatasetSample
import io.github.terminaldetector.xload.core.model.LoraConfig
import io.github.terminaldetector.xload.core.model.TrainingConfig
import io.github.terminaldetector.xload.core.model.TrainingState
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ReferenceLoraEngineTest {
    private val dataset = listOf(
        DatasetSample("Translate to French", "Hello", "Bonjour"),
        DatasetSample("Translate to French", "Goodbye", "Au revoir"),
        DatasetSample("Translate to French", "Thank you", "Merci"),
        DatasetSample("Translate to French", "Yes", "Oui"),
    )

    @Test
    fun `training loss decreases and run completes`() = runBlocking {
        val config = TrainingConfig(
            baseModel = BaseModel.QWEN2_5_0_5B,
            lora = LoraConfig(rank = 4),
            batchSize = 2,
            gradientAccumulationSteps = 1,
            epochs = 100,
            learningRate = 1e-2f,
        )
        val engine = ReferenceLoraEngine(dim = 16)

        val progress = engine.train(config, dataset).toList()

        val trainingSteps = progress.filter { it.state == TrainingState.TRAINING }
        assertTrue("expected training steps to be emitted", trainingSteps.isNotEmpty())

        // Compare whole-epoch averages, not individual steps: batches aren't
        // shuffled, so different positions within an epoch have different
        // baseline loss and aren't comparable step-to-step.
        val stepsPerEpoch = trainingSteps.count { it.epoch == 1 }
        val firstEpochAvgLoss = trainingSteps.take(stepsPerEpoch).map { it.loss }.average()
        val lastEpochAvgLoss = trainingSteps.takeLast(stepsPerEpoch).map { it.loss }.average()
        assertTrue(
            "expected average epoch loss to drop ($firstEpochAvgLoss -> $lastEpochAvgLoss)",
            lastEpochAvgLoss < firstEpochAvgLoss,
        )
        assertEquals(TrainingState.COMPLETED, progress.last().state)
    }

    @Test
    fun `empty dataset fails fast instead of hanging`() = runBlocking {
        val config = TrainingConfig(baseModel = BaseModel.QWEN2_5_0_5B)
        val progress = ReferenceLoraEngine().train(config, emptyList()).toList()

        assertEquals(1, progress.size)
        assertEquals(TrainingState.FAILED, progress.single().state)
    }

    @Test
    fun `cancel stops the run before completion`() = runBlocking {
        val config = TrainingConfig(
            baseModel = BaseModel.QWEN2_5_0_5B,
            epochs = 100,
            batchSize = 1,
        )
        val engine = ReferenceLoraEngine(dim = 8)

        val states = mutableListOf<TrainingState>()
        var seen = 0
        engine.train(config, dataset).collect {
            states += it.state
            seen++
            if (seen == 3) engine.cancel()
        }

        assertTrue(states.contains(TrainingState.CANCELLED))
        // 100 epochs x 4 samples = 400 possible steps; cancelling after the 3rd
        // emission must cut this off almost immediately, not run to completion.
        assertTrue("run should stop well before all 400 steps, got ${states.size}", states.size < 20)
    }
}
