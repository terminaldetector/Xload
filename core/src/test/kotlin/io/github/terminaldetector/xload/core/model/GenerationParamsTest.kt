package io.github.terminaldetector.xload.core.model

import org.junit.Assert.assertThrows
import org.junit.Test

class GenerationParamsTest {
    @Test
    fun `accepts default values`() {
        GenerationParams()
    }

    @Test
    fun `rejects maxNewTokens above the no-KV-cache cap`() {
        assertThrows(IllegalArgumentException::class.java) {
            GenerationParams(maxNewTokens = 201)
        }
    }

    @Test
    fun `rejects zero maxNewTokens`() {
        assertThrows(IllegalArgumentException::class.java) {
            GenerationParams(maxNewTokens = 0)
        }
    }

    @Test
    fun `rejects negative temperature`() {
        assertThrows(IllegalArgumentException::class.java) {
            GenerationParams(temperature = -0.1f)
        }
    }

    @Test
    fun `rejects negative topK`() {
        assertThrows(IllegalArgumentException::class.java) {
            GenerationParams(topK = -1)
        }
    }

    @Test
    fun `allows topK of zero to disable top-k filtering`() {
        GenerationParams(topK = 0)
    }
}
