package io.github.terminaldetector.xload.core.checkpoint

import io.github.terminaldetector.xload.core.engine.reference.ReferenceLoraEngine
import io.github.terminaldetector.xload.core.model.BaseModel
import io.github.terminaldetector.xload.core.model.DatasetSample
import io.github.terminaldetector.xload.core.model.TrainingConfig
import io.github.terminaldetector.xload.core.model.TrainingState
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class AdapterCheckpointJsonTest {
    @Test
    fun `completed run produces a checkpoint that round-trips through json`() = runBlocking {
        val config = TrainingConfig(baseModel = BaseModel.QWEN2_5_0_5B, epochs = 2, batchSize = 2)
        val dataset = listOf(
            DatasetSample("Greet", "", "Hello!"),
            DatasetSample("Farewell", "", "Bye!"),
        )

        val progress = ReferenceLoraEngine(dim = 8).train(config, dataset).toList()
        val completed = progress.last()
        assertEquals(TrainingState.COMPLETED, completed.state)
        val checkpoint = completed.checkpoint
        assertNotNull(checkpoint)

        val encoded = AdapterCheckpointJson.encode(checkpoint!!)
        val decoded = AdapterCheckpointJson.decode(encoded)

        assertEquals(checkpoint, decoded)
    }
}
