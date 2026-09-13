package io.github.terminaldetector.xload.app.ui.screens.training

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import io.github.terminaldetector.xload.app.ui.components.StepScaffold
import io.github.terminaldetector.xload.app.viewmodel.TrainingSessionViewModel
import io.github.terminaldetector.xload.core.model.TrainingState

@Composable
fun TrainingScreen(viewModel: TrainingSessionViewModel, onBack: () -> Unit, onNext: () -> Unit) {
    val progress by viewModel.trainingProgress.collectAsState()
    val state = progress?.state

    StepScaffold(title = "4. Обучение", stepNumber = 4) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .padding(16.dp)
                .fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text(
                when (state) {
                    null -> "Готово к запуску."
                    TrainingState.PREPARING -> "Подготовка..."
                    TrainingState.TRAINING -> "Обучение..."
                    TrainingState.PAUSED_THERMAL -> "Пауза: перегрев устройства."
                    TrainingState.COMPLETED -> "Обучение завершено."
                    TrainingState.FAILED -> "Ошибка: ${progress?.message}"
                    TrainingState.CANCELLED -> "Обучение остановлено."
                },
                style = MaterialTheme.typography.titleMedium,
            )

            val current = progress
            if (current != null && current.totalSteps > 0) {
                LinearProgressIndicator(
                    progress = { current.step / current.totalSteps.toFloat() },
                    modifier = Modifier.fillMaxWidth(),
                )
                Text("Шаг ${current.step} / ${current.totalSteps} · эпоха ${current.epoch}/${current.totalEpochs}")
                Text("Loss: %.4f".format(current.loss))
                current.etaSeconds?.let { Text("Осталось: ~${it} с") }
            } else {
                Text(
                    "Справочный движок обучается на синтетических данных за миллисекунды — " +
                        "это демонстрация пайплайна, а не реальное дообучение LLM (см. README).",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            Spacer(Modifier.weight(1f))

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                OutlinedButton(onClick = onBack, modifier = Modifier.weight(1f)) { Text("Назад") }
                if (state == TrainingState.TRAINING || state == TrainingState.PREPARING) {
                    Button(onClick = { viewModel.cancelTraining() }, modifier = Modifier.weight(1f)) {
                        Text("Остановить")
                    }
                } else {
                    Button(onClick = { viewModel.startTraining() }, modifier = Modifier.weight(1f)) {
                        Text(if (state == TrainingState.COMPLETED) "Обучить заново" else "Начать обучение")
                    }
                }
            }

            if (state == TrainingState.COMPLETED) {
                Button(onClick = onNext, modifier = Modifier.fillMaxWidth()) {
                    Text("К тестированию и экспорту")
                }
            }
        }
    }
}
