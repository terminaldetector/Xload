package io.github.terminaldetector.xload.app.ui.screens.testexport

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import io.github.terminaldetector.xload.app.ui.components.StepScaffold
import io.github.terminaldetector.xload.app.viewmodel.TrainingSessionViewModel
import io.github.terminaldetector.xload.core.checkpoint.AdapterCheckpointJson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun TestExportScreen(viewModel: TrainingSessionViewModel, onRestart: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val progress by viewModel.trainingProgress.collectAsState()
    val checkpoint = progress?.checkpoint
    var exportMessage by remember { mutableStateOf<String?>(null) }

    val exportLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("application/json"),
    ) { uri ->
        if (uri == null || checkpoint == null) return@rememberLauncherForActivityResult
        scope.launch {
            withContext(Dispatchers.IO) {
                context.contentResolver.openOutputStream(uri)?.use { out ->
                    out.write(AdapterCheckpointJson.encode(checkpoint).toByteArray())
                }
            }
            exportMessage = "Адаптер сохранён."
        }
    }

    StepScaffold(title = "5. Тест и экспорт", stepNumber = 5) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .padding(16.dp)
                .fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            if (checkpoint != null) {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text("Итог обучения", style = MaterialTheme.typography.titleMedium)
                        Text("Модель: ${checkpoint.baseModel.displayName}")
                        Text("LoRA: rank=${checkpoint.lora.rank}, alpha=${checkpoint.lora.alpha}")
                        Text("Финальный loss: %.4f".format(checkpoint.finalLoss))
                        Text("Примеров использовано: ${checkpoint.trainedSamples}")
                    }
                }

                OutlinedButton(
                    onClick = {
                        exportLauncher.launch("xload-adapter-${checkpoint.baseModel.name.lowercase()}.json")
                    },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text("Экспортировать чекпоинт адаптера (JSON)")
                }
                exportMessage?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
            } else {
                Text("Сначала завершите обучение на предыдущем шаге.", style = MaterialTheme.typography.bodyMedium)
            }

            Card(modifier = Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("Проверка в чате", style = MaterialTheme.typography.titleMedium)
                    Text(
                        "Появится вместе с реальным движком инференса (llama.cpp / MLC-LLM / " +
                            "MediaPipe LlmInference) — справочный движок обучения в этой сборке " +
                            "не выполняет генерацию текста, см. README.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }

            Spacer(Modifier.weight(1f))

            Button(onClick = onRestart, modifier = Modifier.fillMaxWidth()) {
                Text("Начать новый проект")
            }
        }
    }
}
