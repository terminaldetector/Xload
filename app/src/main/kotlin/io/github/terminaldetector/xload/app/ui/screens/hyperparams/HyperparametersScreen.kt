package io.github.terminaldetector.xload.app.ui.screens.hyperparams

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import io.github.terminaldetector.xload.app.ui.components.StepScaffold
import io.github.terminaldetector.xload.app.viewmodel.TrainingSessionViewModel
import kotlin.math.abs
import kotlin.math.roundToInt

private val LEARNING_RATE_PRESETS =
    listOf(1e-5f, 5e-5f, 1e-4f, 1.5e-4f, 2e-4f, 3e-4f, 5e-4f, 1e-3f, 5e-3f, 1e-2f)

@Composable
fun HyperparametersScreen(viewModel: TrainingSessionViewModel, onBack: () -> Unit, onNext: () -> Unit) {
    val loraConfig by viewModel.loraConfig.collectAsState()
    val trainingParams by viewModel.trainingParams.collectAsState()

    StepScaffold(title = "3. Гиперпараметры", stepNumber = 3) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .padding(16.dp)
                .fillMaxSize()
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text("LoRA", style = MaterialTheme.typography.titleMedium)
            LabeledIntSlider(
                label = "Rank",
                value = loraConfig.rank,
                range = 4..64,
                onValueChange = { viewModel.updateLoraConfig(loraConfig.copy(rank = it)) },
            )
            LabeledIntSlider(
                label = "Alpha",
                value = loraConfig.alpha,
                range = 8..128,
                onValueChange = { viewModel.updateLoraConfig(loraConfig.copy(alpha = it)) },
            )
            Text(
                "Целевые модули (по умолчанию для модели): ${loraConfig.targetModules.joinToString()}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            Text("Обучение", style = MaterialTheme.typography.titleMedium)
            LabeledIntSlider(
                label = "Batch size",
                value = trainingParams.batchSize,
                range = 1..8,
                onValueChange = { viewModel.updateTrainingParams(trainingParams.copy(batchSize = it)) },
            )
            LabeledIntSlider(
                label = "Gradient accumulation",
                value = trainingParams.gradientAccumulationSteps,
                range = 1..16,
                onValueChange = {
                    viewModel.updateTrainingParams(trainingParams.copy(gradientAccumulationSteps = it))
                },
            )
            LabeledIntSlider(
                label = "Epochs",
                value = trainingParams.epochs,
                range = 1..20,
                onValueChange = { viewModel.updateTrainingParams(trainingParams.copy(epochs = it)) },
            )
            LearningRateSlider(
                value = trainingParams.learningRate,
                onValueChange = { viewModel.updateTrainingParams(trainingParams.copy(learningRate = it)) },
            )

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                OutlinedButton(onClick = onBack, modifier = Modifier.weight(1f)) { Text("Назад") }
                Button(onClick = onNext, modifier = Modifier.weight(1f)) { Text("Далее") }
            }
        }
    }
}

@Composable
private fun LabeledIntSlider(label: String, value: Int, range: IntRange, onValueChange: (Int) -> Unit) {
    Column {
        Text("$label: $value", style = MaterialTheme.typography.bodyMedium)
        Slider(
            value = value.toFloat(),
            onValueChange = { onValueChange(it.roundToInt()) },
            valueRange = range.first.toFloat()..range.last.toFloat(),
            steps = (range.last - range.first - 1).coerceAtLeast(0),
        )
    }
}

@Composable
private fun LearningRateSlider(value: Float, onValueChange: (Float) -> Unit) {
    val closestIndex = LEARNING_RATE_PRESETS.indices.minByOrNull { abs(LEARNING_RATE_PRESETS[it] - value) } ?: 0
    Column {
        Text("Learning rate: %.5f".format(value), style = MaterialTheme.typography.bodyMedium)
        Slider(
            value = closestIndex.toFloat(),
            onValueChange = { onValueChange(LEARNING_RATE_PRESETS[it.roundToInt()]) },
            valueRange = 0f..(LEARNING_RATE_PRESETS.size - 1).toFloat(),
            steps = LEARNING_RATE_PRESETS.size - 2,
        )
    }
}
