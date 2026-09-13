package io.github.terminaldetector.xload.core.model

import kotlinx.serialization.Serializable

/** One instruction-tuning example, as read from a dataset JSONL line. */
@Serializable
data class DatasetSample(
    val instruction: String,
    val input: String = "",
    val response: String,
)
