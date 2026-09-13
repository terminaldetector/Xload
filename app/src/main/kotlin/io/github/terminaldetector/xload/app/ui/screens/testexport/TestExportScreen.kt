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
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun TestExportScreen(viewModel: TrainingSessionViewModel, onRestart: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val progress by viewModel.trainingProgress.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()
    val loraConfig by viewModel.loraConfig.collectAsState()
    val checkpointPath = progress?.checkpointPath
    var exportMessage by remember { mutableStateOf<String?>(null) }

    val exportLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("application/octet-stream"),
    ) { uri ->
        if (uri == null || checkpointPath == null) return@rememberLauncherForActivityResult
        scope.launch {
            withContext(Dispatchers.IO) {
                File(checkpointPath).inputStream().use { input ->
                    context.contentResolver.openOutputStream(uri)?.use { out -> input.copyTo(out) }
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
            if (checkpointPath != null) {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text("Итог обучения", style = MaterialTheme.typography.titleMedium)
                        Text("Модель (заявленная): ${selectedModel?.displayName ?: "?"}")
                        Text("LoRA: rank=${loraConfig.rank}, alpha=${loraConfig.alpha}")
                        Text("Финальный loss: %.4f".format(progress?.loss ?: 0f))
                        Text(
                            "Адаптер обучен движком termux-train на встроенной демо-архитектуре " +
                                "(не на реальных весах выбранной модели) — см. README.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }

                OutlinedButton(
                    onClick = {
                        val name = selectedModel?.name?.lowercase() ?: "adapter"
                        exportLauncher.launch("xload-adapter-$name.safetensors")
                    },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text("Экспортировать адаптер (SafeTensors)")
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
                            "MediaPipe LlmInference) — обучение сейчас не подключено к инференсу " +
                            "реальной базовой модели, см. README.",
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
