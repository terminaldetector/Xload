package io.github.terminaldetector.xload.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import io.github.terminaldetector.xload.app.ui.XloadApp
import io.github.terminaldetector.xload.app.ui.theme.XloadTheme
import io.github.terminaldetector.xload.app.viewmodel.TrainingSessionViewModel

class MainActivity : ComponentActivity() {
    private val viewModel: TrainingSessionViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            XloadTheme {
                XloadApp(viewModel = viewModel)
            }
        }
    }
}
