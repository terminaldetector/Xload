package io.github.terminaldetector.xload.core.engine.reference

import io.github.terminaldetector.xload.core.model.DatasetSample
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Deterministically maps a dataset sample's text to a fixed-size float vector via
 * a byte/character histogram, so [ReferenceLoraEngine] has real numeric input to
 * train on without needing a real tokenizer or embedding table.
 */
internal class TextVectorizer(private val dim: Int) {
    fun vectorize(sample: DatasetSample): FloatArray {
        val text = sample.instruction + " " + sample.input + " " + sample.response
        val out = FloatArray(dim)
        for ((i, ch) in text.withIndex()) {
            val bucket = (ch.code + i) % dim
            out[bucket] += sin(ch.code.toDouble()).toFloat()
        }
        val norm = sqrt(out.sumOf { (it * it).toDouble() }).toFloat().coerceAtLeast(1e-6f)
        for (i in out.indices) out[i] /= norm
        return out
    }
}
