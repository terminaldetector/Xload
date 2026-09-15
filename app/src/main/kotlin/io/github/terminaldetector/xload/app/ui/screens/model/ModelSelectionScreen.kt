package io.github.terminaldetector.xload.app.ui.screens.model

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import io.github.terminaldetector.xload.app.ui.components.StepScaffold
import io.github.terminaldetector.xload.app.viewmodel.TrainingSessionViewModel
import io.github.terminaldetector.xload.core.model.BaseModel

@Composable
fun ModelSelectionScreen(viewModel: TrainingSessionViewModel, onNext: () -> Unit) {
    val selectedModel by viewModel.selectedModel.collectAsState()
    val importedModelFileName by viewModel.importedModelFileName.collectAsState()
    val importedModelFilePath by viewModel.importedModelFilePath.collectAsState()

    val filePicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { viewModel.onModelFilePicked(it) }
    }

    StepScaffold(title = "1. Базовая модель", stepNumber = 1) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .padding(16.dp)
                .fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text(
                "Выберите базовую модель для дообучения. Веса модели не входят в приложение " +
                    "— их нужно импортировать отдельно (GGUF или SafeTensors).",
                style = MaterialTheme.typography.bodyMedium,
            )

            LazyColumn(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                items(BaseModel.entries) { model ->
                    val isSelected = model == selectedModel
                    Card(
                        onClick = { viewModel.selectModel(model) },
                        colors = if (isSelected) {
                            CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
                        } else {
                            CardDefaults.cardColors()
                        },
                    ) {
                        Column(Modifier.padding(16.dp)) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                RadioButton(selected = isSelected, onClick = null)
                                Text(model.displayName, style = MaterialTheme.typography.titleMedium)
                            }
                            Text(
                                "${model.paramCountBillions}B · ${model.defaultQuantization.label} " +
                                    "· от ${model.minRamMb} МБ ОЗУ",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                modifier = Modifier.padding(start = 48.dp),
                            )
                        }
                    }
                }
            }

            OutlinedButton(
                onClick = { filePicker.launch(arrayOf("*/*")) },
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(importedModelFileName ?: "Импортировать файл модели (GGUF)")
            }
            when {
                importedModelFileName == null -> Unit
                importedModelFilePath != null -> Text(
                    "Импортирован и будет использован для реального дообучения, если архитектура " +
                        "файла — qwen2, llama, phi3 или gemma3 (иначе автоматически используется демо-модель).",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                else -> Text(
                    "Копирование файла…",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            Button(
                onClick = onNext,
                enabled = selectedModel != null,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("Далее")
            }
        }
    }
}
