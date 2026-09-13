package io.github.terminaldetector.xload.app.engine

/** Called from Python (xload_trainer.py) through Chaquopy with a JSON-encoded progress update. */
fun interface PyProgressCallback {
    @Suppress("unused") // invoked by Python via Chaquopy's reflection bridge, not from Kotlin
    fun onProgress(json: String)
}
