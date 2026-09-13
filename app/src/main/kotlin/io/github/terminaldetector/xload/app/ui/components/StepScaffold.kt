package io.github.terminaldetector.xload.app.ui.components

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier

private const val TOTAL_STEPS = 5

/** Shared chrome for each wizard step: a title bar plus an overall step progress bar. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun StepScaffold(
    title: String,
    stepNumber: Int,
    content: @Composable (PaddingValues) -> Unit,
) {
    Scaffold(
        topBar = {
            Column {
                TopAppBar(title = { Text(title) })
                LinearProgressIndicator(
                    progress = { stepNumber / TOTAL_STEPS.toFloat() },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        },
        content = content,
    )
}
