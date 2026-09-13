package io.github.terminaldetector.xload.app.ui.navigation

/** The training wizard's five steps, in order (see the project plan's pipeline stages). */
enum class Screen(val route: String, val stepNumber: Int) {
    ModelSelection("model_selection", 1),
    Dataset("dataset", 2),
    Hyperparameters("hyperparameters", 3),
    Training("training", 4),
    TestExport("test_export", 5),
}
