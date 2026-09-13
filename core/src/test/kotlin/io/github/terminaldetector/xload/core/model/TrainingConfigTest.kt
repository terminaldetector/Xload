package io.github.terminaldetector.xload.core.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class TrainingConfigTest {
    @Test
    fun `effective batch size multiplies accumulation steps`() {
        val config = TrainingConfig(baseModel = BaseModel.QWEN2_5_0_5B, batchSize = 2, gradientAccumulationSteps = 4)
        assertEquals(8, config.effectiveBatchSize)
    }

    @Test
    fun `rejects invalid learning rate`() {
        assertThrows(IllegalArgumentException::class.java) {
            TrainingConfig(baseModel = BaseModel.QWEN2_5_0_5B, learningRate = 1f)
        }
    }

    @Test
    fun `rejects zero epochs`() {
        assertThrows(IllegalArgumentException::class.java) {
            TrainingConfig(baseModel = BaseModel.QWEN2_5_0_5B, epochs = 0)
        }
    }
}
