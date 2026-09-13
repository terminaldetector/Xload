package io.github.terminaldetector.xload.core.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class LoraConfigTest {
    @Test
    fun `scaling is alpha over rank`() {
        val config = LoraConfig(rank = 16, alpha = 32)
        assertEquals(2f, config.scaling, 1e-6f)
    }

    @Test
    fun `rejects out-of-range rank`() {
        assertThrows(IllegalArgumentException::class.java) { LoraConfig(rank = 0) }
        assertThrows(IllegalArgumentException::class.java) { LoraConfig(rank = 300) }
    }

    @Test
    fun `rejects empty target modules`() {
        assertThrows(IllegalArgumentException::class.java) { LoraConfig(targetModules = emptySet()) }
    }

    @Test
    fun `rejects dropout outside 0 to 1`() {
        assertThrows(IllegalArgumentException::class.java) { LoraConfig(dropout = 1.5f) }
    }
}
