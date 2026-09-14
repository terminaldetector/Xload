package io.github.terminaldetector.xload.core.engine

import io.github.terminaldetector.xload.core.model.DatasetSample
import io.github.terminaldetector.xload.core.model.StyleAnalysisParams
import io.github.terminaldetector.xload.core.model.StyleAnalysisProgress
import kotlinx.coroutines.flow.Flow

/**
 * Builds a structured stylometric "personality map" from [DatasetSample.response] texts —
 * content clusters, per-cluster style centroids, lateral style-similarity edges linking
 * different topics in the same voice, calibrated feature weights, and a held-out reconstruction
 * score — the structured successor to [InferenceEngine.analyzePersonality]'s free-text summary.
 *
 * Ported from the user's own prior work (fbdp_engine_v2.py — a real, already-debugged
 * stylometric analysis engine, not FractalMind/RAPTOR itself) rather than built from scratch;
 * see app/src/main/python/fbdp_engine.py's module docstring for exactly what changed in the
 * port (numpy TF-IDF/KMeans instead of scikit-learn, this app's own dataset shape as input) and
 * what's still a known gap (no real per-cluster LLM summary yet, same gap the original engine's
 * own upgrade notes flagged).
 */
interface StyleAnalysisEngine {
    fun analyze(
        dataset: List<DatasetSample>,
        params: StyleAnalysisParams = StyleAnalysisParams(),
    ): Flow<StyleAnalysisProgress>

    /** Requests the current [analyze] run to stop at the next safe point (between calibration
     *  rounds — each round's numpy computation itself isn't interruptible mid-round). */
    fun cancel()
}
