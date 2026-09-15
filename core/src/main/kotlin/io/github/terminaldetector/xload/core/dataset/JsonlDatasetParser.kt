package io.github.terminaldetector.xload.core.dataset

import io.github.terminaldetector.xload.core.model.DatasetSample
import kotlinx.serialization.json.Json

data class DatasetLineError(val lineNumber: Int, val reason: String)

data class DatasetParseResult(
    val samples: List<DatasetSample>,
    val errors: List<DatasetLineError>,
    val approxTokenCount: Long,
) {
    val isValid: Boolean get() = samples.isNotEmpty() && errors.isEmpty()
}

/**
 * Parses instruction/response JSONL datasets, one JSON object per line, e.g.
 * `{"instruction": "...", "input": "...", "response": "..."}`.
 *
 * Blank lines are skipped. Malformed or incomplete lines are collected as errors
 * (with line numbers) rather than thrown, so the dataset screen can point the user
 * at exactly what to fix instead of rejecting the whole file on the first typo.
 */
object JsonlDatasetParser {
    private val json = Json { ignoreUnknownKeys = true }

    // Rough heuristic (~4 chars/token) used only to size progress estimates before
    // training; the real backend's tokenizer produces the authoritative count.
    private const val CHARS_PER_TOKEN = 4.0

    fun parse(content: String): DatasetParseResult {
        val samples = mutableListOf<DatasetSample>()
        val errors = mutableListOf<DatasetLineError>()
        var approxTokens = 0L

        content.lineSequence().forEachIndexed { index, rawLine ->
            val line = rawLine.trim()
            if (line.isEmpty()) return@forEachIndexed
            val lineNumber = index + 1

            runCatching { json.decodeFromString(DatasetSample.serializer(), line) }
                .onSuccess { sample ->
                    if (sample.instruction.isBlank() || sample.response.isBlank()) {
                        errors += DatasetLineError(lineNumber, "\"instruction\" and \"response\" must not be blank")
                    } else {
                        samples += sample
                        val chars = sample.instruction.length + sample.input.length + sample.response.length
                        approxTokens += (chars / CHARS_PER_TOKEN).toLong()
                    }
                }
                .onFailure { e ->
                    // kotlinx.serialization's own message (e.g. "Fields [instruction,
                    // response] are required ... but they were missing at path: $")
                    // never shows what WAS on the line, which is the one thing that
                    // actually explains a missing-field error -- the line parsed as
                    // valid JSON, it just doesn't have those keys at the root (wrong
                    // schema, e.g. "prompt"/"completion" instead of
                    // "instruction"/"response"). Append a preview so that's visible
                    // directly in the error list instead of requiring a hex-editor.
                    val preview = if (line.length > 80) line.take(80) + "…" else line
                    val reason = "${e.message ?: "Invalid JSON"} | line content: $preview"
                    errors += DatasetLineError(lineNumber, reason)
                }
        }

        return DatasetParseResult(samples, errors, approxTokens)
    }
}
