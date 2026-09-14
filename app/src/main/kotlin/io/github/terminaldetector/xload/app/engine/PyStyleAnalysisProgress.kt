package io.github.terminaldetector.xload.app.engine

import io.github.terminaldetector.xload.core.model.StyleAnalysisProgress
import io.github.terminaldetector.xload.core.model.StyleAnalysisResult
import io.github.terminaldetector.xload.core.model.StyleAnalysisState
import io.github.terminaldetector.xload.core.model.StyleCluster
import io.github.terminaldetector.xload.core.model.StyleEdge
import io.github.terminaldetector.xload.core.model.StyleFeatureWeight
import kotlinx.serialization.Serializable

@Serializable
internal data class PyStyleCluster(
    val id: String,
    val size: Int,
    val topTerms: List<String> = emptyList(),
    val styleCentroid: Map<String, Float> = emptyMap(),
) {
    fun toStyleCluster() = StyleCluster(id = id, size = size, topTerms = topTerms, styleCentroid = styleCentroid)
}

@Serializable
internal data class PyStyleEdge(val from: String, val to: String, val weight: Float) {
    fun toStyleEdge() = StyleEdge(from = from, to = to, weight = weight)
}

@Serializable
internal data class PyStyleFeatureWeight(val feature: String, val weight: Float) {
    fun toStyleFeatureWeight() = StyleFeatureWeight(feature = feature, weight = weight)
}

/** Mirrors the JSON shape `fbdp_engine.py`'s `emit()` sends over the Chaquopy boundary — a
 *  CALIBRATING event carries only [round]/[totalRounds]/[reconstructionScore]/[edgeCount], a
 *  COMPLETED event carries [clusters]/[edges]/[topStyleFeatures]/[reconstructionScore]/
 *  [roundsCompleted] (the Python side spreads its summary dict's keys in alongside "state"), and
 *  FAILED/CANCELLED carry just [message] (or nothing) — every non-`state` field is defaulted so
 *  one data class decodes all four shapes. */
@Serializable
internal data class PyStyleAnalysisProgress(
    val state: String,
    val round: Int = 0,
    val totalRounds: Int = 0,
    val reconstructionScore: Float? = null,
    val edgeCount: Int? = null,
    val clusters: List<PyStyleCluster> = emptyList(),
    val edges: List<PyStyleEdge> = emptyList(),
    val topStyleFeatures: List<PyStyleFeatureWeight> = emptyList(),
    val roundsCompleted: Int = 0,
    val message: String? = null,
) {
    fun toStyleAnalysisProgress(): StyleAnalysisProgress {
        val parsedState = runCatching { StyleAnalysisState.valueOf(state) }.getOrDefault(StyleAnalysisState.FAILED)
        val result = if (parsedState == StyleAnalysisState.COMPLETED) {
            StyleAnalysisResult(
                clusters = clusters.map { it.toStyleCluster() },
                edges = edges.map { it.toStyleEdge() },
                topStyleFeatures = topStyleFeatures.map { it.toStyleFeatureWeight() },
                reconstructionScore = reconstructionScore ?: 0f,
                roundsCompleted = roundsCompleted,
            )
        } else {
            null
        }
        return StyleAnalysisProgress(
            state = parsedState,
            round = round,
            totalRounds = totalRounds,
            reconstructionScore = reconstructionScore,
            edgeCount = edgeCount,
            result = result,
            message = message,
        )
    }
}
