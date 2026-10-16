package io.github.terminaldetector.xload.core.model

/**
 * Explicit choice between training on a real imported GGUF model and
 * termux-train's small bundled demo transformer -- see [TrainingConfig]'s
 * `modelMode` field. Deliberately not inferred from whether
 * [TrainingConfig.modelFilePath] happens to be non-null: a null path could
 * mean "user never imported a file" (demo is correct) or "user imported one
 * but the copy hadn't finished / had failed" (demo would be a silent, wrong
 * fallback) -- those two cases must not be indistinguishable to the trainer.
 */
enum class ModelMode { REAL_GGUF, DEMO }
