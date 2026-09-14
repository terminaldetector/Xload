package io.github.terminaldetector.xload.app.ui.screens.testexport

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import io.github.terminaldetector.xload.app.ui.components.StepScaffold
import io.github.terminaldetector.xload.app.viewmodel.ChatMessage
import io.github.terminaldetector.xload.app.viewmodel.ChatRole
import io.github.terminaldetector.xload.app.viewmodel.TrainingSessionViewModel
import io.github.terminaldetector.xload.core.model.StyleAnalysisState
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
    val importedModelFilePath by viewModel.importedModelFilePath.collectAsState()
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
            PersonalityPortraitCard(viewModel)
            StyleAnalysisCard(viewModel)

            if (checkpointPath != null) {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text("Итог обучения", style = MaterialTheme.typography.titleMedium)
                        Text("Модель (заявленная): ${selectedModel?.displayName ?: "?"}")
                        Text("LoRA: rank=${loraConfig.rank}, alpha=${loraConfig.alpha}")
                        Text("Финальный loss: %.4f".format(progress?.loss ?: 0f))
                        Text(
                            if (importedModelFilePath != null) {
                                "Адаптер обучен движком termux-train на реальных весах импортированного " +
                                    "GGUF-файла (если его архитектура поддержана — см. README) — либо, если " +
                                    "нет, на встроенной демо-архитектуре как fallback."
                            } else {
                                "Адаптер обучен движком termux-train на встроенной демо-архитектуре " +
                                    "(файл модели не был импортирован) — см. README."
                            },
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

                ChatCard(viewModel)
            } else {
                Text("Сначала завершите обучение на предыдущем шаге.", style = MaterialTheme.typography.bodyMedium)
            }

            Spacer(Modifier.weight(1f))

            Button(onClick = onRestart, modifier = Modifier.fillMaxWidth()) {
                Text("Начать новый проект")
            }
        }
    }
}

@Composable
private fun PersonalityPortraitCard(viewModel: TrainingSessionViewModel) {
    val datasetResult by viewModel.datasetResult.collectAsState()
    val portrait by viewModel.personalityPortrait.collectAsState()
    val error by viewModel.personalityError.collectAsState()
    val isAnalyzing by viewModel.isAnalyzingPersonality.collectAsState()
    val samples = datasetResult?.samples ?: emptyList()

    if (samples.isEmpty()) return

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Портрет личности (пробник)", style = MaterialTheme.typography.titleMedium)
            Text(
                "Пробует описать стиль общения и характерные темы по примерам из датасета — " +
                    "свободный текст от той же on-device модели, не структурированная карта и не " +
                    "кластеризация RAPTOR/FractalMind (см. README). Не требует завершённого обучения. " +
                    "Если портрет построен, он подмешивается в контекст чата ниже.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            when {
                error != null -> Text(
                    "Ошибка: $error",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
                portrait != null -> Text(portrait!!, style = MaterialTheme.typography.bodyMedium)
            }

            if (isAnalyzing) {
                OutlinedButton(onClick = { viewModel.cancelPersonalityAnalysis() }, modifier = Modifier.fillMaxWidth()) {
                    Text("Стоп")
                }
            } else {
                OutlinedButton(
                    onClick = { viewModel.analyzePersonality() },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(if (portrait != null) "Построить заново" else "Построить портрет")
                }
            }
        }
    }
}

@Composable
private fun StyleAnalysisCard(viewModel: TrainingSessionViewModel) {
    val datasetResult by viewModel.datasetResult.collectAsState()
    val progress by viewModel.styleAnalysisProgress.collectAsState()
    val isAnalyzing by viewModel.isAnalyzingStyle.collectAsState()
    val samples = datasetResult?.samples ?: emptyList()

    if (samples.isEmpty()) return

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Стилевой анализ (FBDP, пробник)", style = MaterialTheme.typography.titleMedium)
            Text(
                "Настоящая статистика по примерам датасета вместо текстовой сводки выше: " +
                    "TF-IDF/KMeans-кластеры по темам, стилевой центроид (15 метрик — длина " +
                    "фраз, пунктуация, хеджи и т.д.) у каждого кластера, латеральные связи " +
                    "между разнотемными кластерами, написанными одним и тем же голосом, и " +
                    "оценка реконструкции на отложенных примерах. Портировано из уже " +
                    "существовавшей у пользователя наработки (см. README) — не FractalMind " +
                    "и не RAPTOR как есть, кластеры пока без настоящего LLM-резюме.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            when (progress?.state) {
                StyleAnalysisState.CALIBRATING -> Text(
                    "Калибровка: раунд ${progress?.round} из ${progress?.totalRounds}…",
                    style = MaterialTheme.typography.bodySmall,
                )
                StyleAnalysisState.FAILED -> Text(
                    "Ошибка: ${progress?.message}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
                StyleAnalysisState.COMPLETED -> progress?.result?.let { result ->
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text(
                            "Реконструкция: %.4f · связей между кластерами: %d · раундов: %d"
                                .format(result.reconstructionScore, result.edges.size, result.roundsCompleted),
                            style = MaterialTheme.typography.bodySmall,
                        )
                        result.clusters.forEach { cluster ->
                            Text(
                                "· ${cluster.id}: ${cluster.size} примеров — ${cluster.topTerms.joinToString(", ")}",
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                        if (result.topStyleFeatures.isNotEmpty()) {
                            Text(
                                "Ведущие стилевые черты: " +
                                    result.topStyleFeatures.joinToString(", ") { "${it.feature} (${"%.2f".format(it.weight)})" },
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                    }
                }
                StyleAnalysisState.CANCELLED, null -> Unit
            }

            if (isAnalyzing) {
                OutlinedButton(onClick = { viewModel.cancelStyleAnalysis() }, modifier = Modifier.fillMaxWidth()) {
                    Text("Стоп")
                }
            } else {
                OutlinedButton(
                    onClick = { viewModel.analyzeStyle() },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(if (progress?.result != null) "Построить заново" else "Построить стилевой анализ")
                }
            }
        }
    }
}

@Composable
private fun ChatCard(viewModel: TrainingSessionViewModel) {
    val messages by viewModel.chatMessages.collectAsState()
    val isGenerating by viewModel.isGenerating.collectAsState()
    val portrait by viewModel.personalityPortrait.collectAsState()
    var input by remember { mutableStateOf("") }
    val scrollState = rememberScrollState()

    LaunchedEffect(messages.size, messages.lastOrNull()?.text) {
        scrollState.animateScrollTo(scrollState.maxValue)
    }

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Проверка в чате", style = MaterialTheme.typography.titleMedium)
            Text(
                "Генерация через termux-train поверх обученного адаптера — без KV-кэша, поэтому " +
                    "ответ короткий и не мгновенный (см. README про xload_inference.py). Не " +
                    "полноценный чат-движок (llama.cpp / MLC-LLM / MediaPipe LlmInference) — та " +
                    "задача ещё впереди, это лишь проверка, что дообучение вообще что-то поменяло." +
                    if (portrait != null) " Портрет личности выше подмешивается в каждое сообщение." else "",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            if (messages.isNotEmpty()) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(max = 240.dp)
                        .verticalScroll(scrollState),
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    messages.forEach { message -> ChatBubble(message) }
                }
            }

            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                OutlinedTextField(
                    value = input,
                    onValueChange = { input = it },
                    modifier = Modifier.weight(1f),
                    placeholder = { Text("Спросите что-нибудь…") },
                    enabled = !isGenerating,
                    singleLine = true,
                )
                if (isGenerating) {
                    OutlinedButton(onClick = { viewModel.cancelGeneration() }) {
                        Text("Стоп")
                    }
                } else {
                    Button(
                        onClick = {
                            viewModel.sendChatMessage(input)
                            input = ""
                        },
                        enabled = input.isNotBlank(),
                    ) {
                        Text("Отправить")
                    }
                }
            }
        }
    }
}

@Composable
private fun ChatBubble(message: ChatMessage) {
    val isUser = message.role == ChatRole.USER
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start,
    ) {
        Box(
            modifier = Modifier
                .background(
                    color = if (isUser) {
                        MaterialTheme.colorScheme.primaryContainer
                    } else {
                        MaterialTheme.colorScheme.surfaceVariant
                    },
                    shape = RoundedCornerShape(12.dp),
                )
                .padding(horizontal = 12.dp, vertical = 8.dp),
        ) {
            Text(
                text = message.error?.let { "Ошибка: $it" }
                    ?: message.text.ifEmpty { "…" },
                color = if (message.error != null) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurface,
                style = MaterialTheme.typography.bodyMedium,
            )
        }
    }
}
