package io.github.terminaldetector.xload.core.model

import org.junit.Assert.assertThrows
import org.junit.Test

class StyleAnalysisParamsTest {
    @Test
    fun `accepts default values`() {
        StyleAnalysisParams()
    }

    @Test
    fun `rejects zero rounds`() {
        assertThrows(IllegalArgumentException::class.java) {
            StyleAnalysisParams(rounds = 0)
        }
    }

    @Test
    fun `rejects rounds above the cap`() {
        assertThrows(IllegalArgumentException::class.java) {
            StyleAnalysisParams(rounds = 11)
        }
    }
}
