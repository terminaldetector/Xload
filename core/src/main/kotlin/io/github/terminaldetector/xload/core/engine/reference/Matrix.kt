package io.github.terminaldetector.xload.core.engine.reference

import kotlin.random.Random

/** Minimal row-major float matrix — just enough linear algebra for [ReferenceLoraEngine]. */
internal class Matrix(val rows: Int, val cols: Int, val data: FloatArray = FloatArray(rows * cols)) {
    operator fun get(r: Int, c: Int): Float = data[r * cols + c]

    operator fun set(r: Int, c: Int, v: Float) {
        data[r * cols + c] = v
    }

    fun timesVector(x: FloatArray): FloatArray {
        require(x.size == cols) { "Expected vector of size $cols, got ${x.size}" }
        val out = FloatArray(rows)
        for (r in 0 until rows) {
            var sum = 0f
            val base = r * cols
            for (c in 0 until cols) sum += data[base + c] * x[c]
            out[r] = sum
        }
        return out
    }

    companion object {
        fun random(rows: Int, cols: Int, random: Random, scale: Float): Matrix {
            val m = Matrix(rows, cols)
            for (i in m.data.indices) m.data[i] = (random.nextFloat() * 2f - 1f) * scale
            return m
        }

        fun zeros(rows: Int, cols: Int) = Matrix(rows, cols)
    }
}
