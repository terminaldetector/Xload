package io.github.terminaldetector.xload.core.engine

import io.github.terminaldetector.xload.core.model.BaseModel
import io.github.terminaldetector.xload.core.model.LoraConfig
import io.github.terminaldetector.xload.core.model.ModelMode
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class MemoryPreflightTest {
    @Test
    fun `real_gguf base weights match paramCountBillions times 4 bytes (dequantized to float32)`() {
        val estimate = MemoryPreflight.estimate(
            BaseModel.QWEN2_5_0_5B, LoraConfig(), ModelMode.REAL_GGUF,
        )
        val expected = (0.5 * 1_000_000_000L * 4L).toLong()
        assertEquals(expected, estimate.baseWeightsBytes)
    }

    @Test
    fun `demo mode base weights are negligible regardless of the selected BaseModel`() {
        val estimate = MemoryPreflight.estimate(
            BaseModel.PHI_3_MINI, LoraConfig(), ModelMode.DEMO,
        )
        assertTrue(
            "demo-mode base weights should be tiny, was ${estimate.baseWeightsBytes}",
            estimate.baseWeightsBytes < 100L * 1024 * 1024,
        )
    }

    @Test
    fun `a larger BaseModel estimates a larger real_gguf footprint`() {
        val small = MemoryPreflight.estimate(BaseModel.QWEN2_5_0_5B, LoraConfig(), ModelMode.REAL_GGUF)
        val large = MemoryPreflight.estimate(BaseModel.PHI_3_MINI, LoraConfig(), ModelMode.REAL_GGUF)
        assertTrue(large.totalBytes > small.totalBytes)
        // Phi-3-mini (3.8B) vs Qwen2.5-0.5B (0.5B) -- base weights should
        // scale with the ~7.6x param-count ratio, not just "somewhat more".
        val ratio = large.baseWeightsBytes.toDouble() / small.baseWeightsBytes.toDouble()
        assertTrue("expected ~7.6x, was ${ratio}x", ratio in 7.0..8.0)
    }

    @Test
    fun `a higher LoRA rank increases loraAndOptimizerBytes`() {
        val lowRank = MemoryPreflight.estimate(BaseModel.QWEN2_5_0_5B, LoraConfig(rank = 4), ModelMode.REAL_GGUF)
        val highRank = MemoryPreflight.estimate(BaseModel.QWEN2_5_0_5B, LoraConfig(rank = 64), ModelMode.REAL_GGUF)
        assertTrue(highRank.loraAndOptimizerBytes > lowRank.loraAndOptimizerBytes)
    }

    @Test
    fun `flags a tiny available-memory budget against a large real model`() {
        val estimate = MemoryPreflight.estimate(BaseModel.PHI_3_MINI, LoraConfig(), ModelMode.REAL_GGUF)
        val tinyAvailable = 512L * 1024 * 1024 // a phone with 512MB free -- Phi-3-mini obviously won't fit
        assertTrue(MemoryPreflight.likelyExceedsAvailableMemory(estimate, tinyAvailable))
    }

    @Test
    fun `does not flag a generous available-memory budget`() {
        val estimate = MemoryPreflight.estimate(BaseModel.QWEN2_5_0_5B, LoraConfig(), ModelMode.REAL_GGUF)
        val generousAvailable = 64L * 1024 * 1024 * 1024 // 64GB free
        assertFalse(MemoryPreflight.likelyExceedsAvailableMemory(estimate, generousAvailable))
    }

    @Test
    fun `an unknown (non-positive) available memory reading never flags a warning`() {
        val estimate = MemoryPreflight.estimate(BaseModel.PHI_3_MINI, LoraConfig(), ModelMode.REAL_GGUF)
        assertFalse(MemoryPreflight.likelyExceedsAvailableMemory(estimate, 0L))
        assertFalse(MemoryPreflight.likelyExceedsAvailableMemory(estimate, -1L))
    }
}
