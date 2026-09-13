package io.github.terminaldetector.xload.app.ui.screens.dataset

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun DatasetScreen(viewModel: TrainingSessionViewModel, onBack: () -> Unit, onNext: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val datasetResult by viewModel.datasetResult.collectAsState()
    var isLoading by remember { mutableStateOf(false) }

    val filePicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        isLoading = true
        scope.launch {
            val text = withContext(Dispatchers.IO) {
                context.contentResolver.openInputStream(uri)?.bufferedReader()?.use { it.readText() }
            }
            isLoading = false
            if (text != null) viewModel.loadDataset(text)
        }
    }

    StepScaffold(title = "2. Датасет", stepNumber = 2) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .padding(16.dp)
                .fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text(
                "Загрузите JSONL-файл: одна строка — один JSON-объект вида " +
                    "{\"instruction\": \"...\", \"input\": \"...\", \"response\": \"...\"}.",
                style = MaterialTheme.typography.bodyMedium,
            )

            OutlinedButton(
                onClick = { filePicker.launch(arrayOf("*/*")) },
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("Выбрать JSONL-файл")
            }

            if (isLoading) {
                CircularProgressIndicator()
            }

            val result = datasetResult
            Column(modifier = Modifier.weight(1f)) {
                if (result != null) {
                    val summary = buildString {
                        append("Найдено примеров: ${result.samples.size} · ~${result.approxTokenCount} токенов")
                        if (result.errors.isNotEmpty()) append(" · ошибок: ${result.errors.size}")
                    }
                    Text(summary, style = MaterialTheme.typography.bodyMedium)

                    if (result.errors.isNotEmpty()) {
                        LazyColumn(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                            items(result.errors) { error ->
                                Card(modifier = Modifier.fillMaxWidth()) {
                                    Text(
                                        "Строка ${error.lineNumber}: ${error.reason}",
                                        modifier = Modifier.padding(12.dp),
                                        style = MaterialTheme.typography.bodySmall,
                                    )
                                }
                            }
                        }
                    }
                } else {
                    Spacer(Modifier.fillMaxWidth())
                }
            }

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                OutlinedButton(onClick = onBack, modifier = Modifier.weight(1f)) {
                    Text("Назад")
                }
                Button(
                    onClick = onNext,
                    enabled = result?.isValid == true,
                    modifier = Modifier.weight(1f),
                ) {
                    Text("Далее")
                }
            }
        }
    }
}
