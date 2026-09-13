package io.github.terminaldetector.xload.app.viewmodel

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.github.terminaldetector.xload.app.engine.TermuxTrainEngine
import io.github.terminaldetector.xload.core.dataset.DatasetParseResult
import io.github.terminaldetector.xload.core.dataset.JsonlDatasetParser
import io.github.terminaldetector.xload.core.engine.TrainingEngine
import io.github.terminaldetector.xload.core.model.BaseModel
import io.github.terminaldetector.xload.core.model.LoraConfig
import io.github.terminaldetector.xload.core.model.TrainingConfig
import io.github.terminaldetector.xload.core.model.TrainingProgress
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * Holds the whole training wizard's state (model -> dataset -> hyperparameters ->
 * training -> export) so it survives navigating back and forth between steps.
 *
 * Uses [TermuxTrainEngine] (real termux-train via Chaquopy) rather than the
 * pure-Kotlin [io.github.terminaldetector.xload.core.engine.reference.ReferenceLoraEngine] —
 * see that class and app/src/main/python/xload_trainer.py for what each does
 * and doesn't prove. AndroidViewModel (not plain ViewModel) because the
 * engine needs a Context for Chaquopy and for where to write the checkpoint.
 */
class TrainingSessionViewModel(application: Application) : AndroidViewModel(application) {
    private val trainingEngine: TrainingEngine = TermuxTrainEngine(application)

    private val _selectedModel = MutableStateFlow<BaseModel?>(null)
    val selectedModel: StateFlow<BaseModel?> = _selectedModel.asStateFlow()

    private val _importedModelFileName = MutableStateFlow<String?>(null)
    val importedModelFileName: StateFlow<String?> = _importedModelFileName.asStateFlow()

    private val _loraConfig = MutableStateFlow(LoraConfig())
    val loraConfig: StateFlow<LoraConfig> = _loraConfig.asStateFlow()

    private val _trainingParams = MutableStateFlow(TrainingHyperparams())
    val trainingParams: StateFlow<TrainingHyperparams> = _trainingParams.asStateFlow()

    private val _datasetResult = MutableStateFlow<DatasetParseResult?>(null)
    val datasetResult: StateFlow<DatasetParseResult?> = _datasetResult.asStateFlow()

    private val _trainingProgress = MutableStateFlow<TrainingProgress?>(null)
    val trainingProgress: StateFlow<TrainingProgress?> = _trainingProgress.asStateFlow()

    private var trainingJob: Job? = null

    fun selectModel(model: BaseModel) {
        _selectedModel.value = model
        _loraConfig.value = _loraConfig.value.copy(targetModules = model.targetModules.toSet())
    }

    fun onModelFilePicked(uri: Uri) {
        _importedModelFileName.value = uri.lastPathSegment ?: uri.toString()
    }

    fun loadDataset(content: String) {
        _datasetResult.value = JsonlDatasetParser.parse(content)
    }

    fun updateLoraConfig(config: LoraConfig) {
        _loraConfig.value = config
    }

    fun updateTrainingParams(params: TrainingHyperparams) {
        _trainingParams.value = params
    }

    fun startTraining() {
        val model = _selectedModel.value ?: return
        val samples = _datasetResult.value?.samples ?: return
        if (samples.isEmpty()) return

        val params = _trainingParams.value
        val config = TrainingConfig(
            baseModel = model,
            lora = _loraConfig.value,
            batchSize = params.batchSize,
            gradientAccumulationSteps = params.gradientAccumulationSteps,
            epochs = params.epochs,
            learningRate = params.learningRate,
        )

        trainingJob?.cancel()
        trainingJob = viewModelScope.launch {
            trainingEngine.train(config, samples).collect { progress ->
                _trainingProgress.value = progress
            }
        }
    }

    fun cancelTraining() {
        trainingEngine.cancel()
    }

    fun reset() {
        trainingEngine.cancel()
        trainingJob?.cancel()
        _selectedModel.value = null
        _importedModelFileName.value = null
        _loraConfig.value = LoraConfig()
        _trainingParams.value = TrainingHyperparams()
        _datasetResult.value = null
        _trainingProgress.value = null
    }

    override fun onCleared() {
        trainingEngine.cancel()
    }
}
