package io.github.terminaldetector.xload.core.dataset

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class JsonlDatasetParserTest {
    @Test
    fun `parses valid instruction-response lines`() {
        val content = """
            {"instruction": "Say hi", "response": "Hi!"}
            {"instruction": "Add 2+2", "input": "2+2", "response": "4"}
        """.trimIndent()

        val result = JsonlDatasetParser.parse(content)

        assertTrue(result.isValid)
        assertEquals(2, result.samples.size)
        assertEquals("Say hi", result.samples[0].instruction)
        assertEquals("2+2", result.samples[1].input)
        assertTrue(result.approxTokenCount > 0)
    }

    @Test
    fun `skips blank lines`() {
        val content = "{\"instruction\": \"a\", \"response\": \"b\"}\n\n   \n"
        val result = JsonlDatasetParser.parse(content)
        assertEquals(1, result.samples.size)
        assertTrue(result.errors.isEmpty())
    }

    @Test
    fun `reports malformed json with its line number`() {
        val content = "{\"instruction\": \"a\", \"response\": \"b\"}\nnot json\n"
        val result = JsonlDatasetParser.parse(content)

        assertEquals(1, result.samples.size)
        assertEquals(1, result.errors.size)
        assertEquals(2, result.errors[0].lineNumber)
        assertTrue(!result.isValid)
    }

    @Test
    fun `reports blank required fields as errors`() {
        val content = "{\"instruction\": \"\", \"response\": \"b\"}"
        val result = JsonlDatasetParser.parse(content)

        assertEquals(0, result.samples.size)
        assertEquals(1, result.errors.size)
        assertEquals(1, result.errors[0].lineNumber)
    }
}
