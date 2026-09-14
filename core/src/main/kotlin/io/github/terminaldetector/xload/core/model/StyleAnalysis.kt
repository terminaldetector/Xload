package io.github.terminaldetector.xload.core.model

/** Sampling/compute knobs for [io.github.terminaldetector.xload.core.engine.StyleAnalysisEngine.analyze]. */
data class StyleAnalysisParams(val rounds: Int = 5) {
    init {
        // Matches fbdp_engine.py's MAX_ROUNDS -- calibration cost is cheap (numpy matrix
        // ops, not LLM generation) but still capped so a large dataset can't run forever.
        require(rounds in 1..10) { "rounds must be in 1..10, was $rounds" }
    }
}

enum class StyleAnalysisState { CALIBRATING, COMPLETED, FAILED, CANCELLED }

/** One content cluster (a group of dataset examples about the same topic) from the latest
 *  calibration round, with its own style fingerprint. [styleCentroid] maps each of the 15
 *  stylometric feature names (see fbdp_engine.py's STYLE_FEATURES) to its normalized value for
 *  this cluster. [topTerms] is a cheap, LLM-free label -- the cluster's highest-TF-IDF-weight
 *  vocabulary words, not a real summary (see README on why that gap is still open). */
data class StyleCluster(
    val id: String,
    val size: Int,
    val topTerms: List<String>,
    val styleCentroid: Map<String, Float>,
)

/** A lateral link between two [StyleCluster]s from *different* topics whose calibrated style
 *  centroids are similar (cosine >= 0.85) — "these branches talk about different things but in
 *  the same voice." Never connects two clusters that share a topic (that's implicit in them
 *  being the same cluster already). */
data class StyleEdge(val from: String, val to: String, val weight: Float)

data class StyleFeatureWeight(val feature: String, val weight: Float)

/** The terminal (COMPLETED) result of a full calibration run — see
 *  [io.github.terminaldetector.xload.core.engine.StyleAnalysisEngine] for what this proves and
 *  doesn't. [topStyleFeatures] are the highest-weighted of the 15 stylometric features after
 *  calibration (features used more *consistently* across the dataset get weighted higher).
 *  [reconstructionScore] is the latest round's held-out validation score (0..1, higher = the
 *  calibrated style fingerprint predicts withheld examples better). */
data class StyleAnalysisResult(
    val clusters: List<StyleCluster>,
    val edges: List<StyleEdge>,
    val topStyleFeatures: List<StyleFeatureWeight>,
    val reconstructionScore: Float,
    val roundsCompleted: Int,
)

data class StyleAnalysisProgress(
    val state: StyleAnalysisState,
    val round: Int = 0,
    val totalRounds: Int = 0,
    val reconstructionScore: Float? = null,
    val edgeCount: Int? = null,
    /** Set only on the terminal [StyleAnalysisState.COMPLETED] emission. */
    val result: StyleAnalysisResult? = null,
    val message: String? = null,
)
