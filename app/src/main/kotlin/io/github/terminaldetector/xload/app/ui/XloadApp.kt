package io.github.terminaldetector.xload.app.ui

import androidx.compose.runtime.Composable
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import io.github.terminaldetector.xload.app.ui.navigation.Screen
import io.github.terminaldetector.xload.app.ui.screens.dataset.DatasetScreen
import io.github.terminaldetector.xload.app.ui.screens.hyperparams.HyperparametersScreen
import io.github.terminaldetector.xload.app.ui.screens.model.ModelSelectionScreen
import io.github.terminaldetector.xload.app.ui.screens.testexport.TestExportScreen
import io.github.terminaldetector.xload.app.ui.screens.training.TrainingScreen
import io.github.terminaldetector.xload.app.viewmodel.TrainingSessionViewModel

@Composable
fun XloadApp(viewModel: TrainingSessionViewModel) {
    val navController = rememberNavController()

    NavHost(navController = navController, startDestination = Screen.ModelSelection.route) {
        composable(Screen.ModelSelection.route) {
            ModelSelectionScreen(
                viewModel = viewModel,
                onNext = { navController.navigate(Screen.Dataset.route) },
            )
        }
        composable(Screen.Dataset.route) {
            DatasetScreen(
                viewModel = viewModel,
                onBack = { navController.popBackStack() },
                onNext = { navController.navigate(Screen.Hyperparameters.route) },
            )
        }
        composable(Screen.Hyperparameters.route) {
            HyperparametersScreen(
                viewModel = viewModel,
                onBack = { navController.popBackStack() },
                onNext = { navController.navigate(Screen.Training.route) },
            )
        }
        composable(Screen.Training.route) {
            TrainingScreen(
                viewModel = viewModel,
                onBack = { navController.popBackStack() },
                onNext = { navController.navigate(Screen.TestExport.route) },
            )
        }
        composable(Screen.TestExport.route) {
            TestExportScreen(
                viewModel = viewModel,
                onRestart = {
                    viewModel.reset()
                    navController.popBackStack(Screen.ModelSelection.route, inclusive = false)
                },
            )
        }
    }
}
