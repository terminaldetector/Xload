# Xload

**On-Device LoRA Trainer** — Android-приложение для локального дообучения LoRA-адаптеров
компактных LLM (0.5–3B параметров) прямо на смартфоне, без выгрузки данных в облако.

Целевые модели: Qwen2.5-0.5B/1.5B, Gemma-3-1B, Llama-3.2-1B, Phi-3-mini (в 4-bit квантизации).

## Статус проекта

Это начальная стадия разработки (Этап 1 из плана — инфраструктура — завершён). Реализовано:

- Полный каркас Android-приложения на Kotlin + Jetpack Compose, разбитый на модули
  `:core` (чистая Kotlin/JVM логика, без зависимости от Android SDK) и `:app` (UI).
- Мастер из 5 экранов по стадиям пайплайна из плана: выбор модели → датасет →
  гиперпараметры → обучение → тест/экспорт.
- Реальный, протестированный пайплайн `:core`: валидация конфигурации, парсер
  JSONL-датасетов, интерфейс `TrainingEngine` и его справочная (reference) реализация
  на чистом Kotlin, экспорт чекпоинта адаптера в JSON.
- Импорт файлов модели/датасета через Storage Access Framework — приложение не просит
  разрешение `INTERNET` и не имеет сетевых вызовов: всё офлайн, как и требует план.

**Чего пока нет и почему** — см. [«Что дальше»](#что-дальше-дорожная-карта): настоящего
бэкенда обучения для реальных LLM (termux-train / MobileFineTuner / QVAC Fabric /
ExecuTorch), инференса для проверки адаптера в чате, GPU/квантизации, термотроттлинга.
Это отдельные, крупные этапы работы (см. риски в исходном плане — каждый бэкенд это
недели-месяцы работы сам по себе), и в рамках «начать разработку» было честнее
заложить для них чистую архитектуру и довести первый сквозной пайплайн до конца,
чем имитировать работающее обучение реальной LLM.

## Архитектура

```
Xload/
├── core/   — чистый Kotlin/JVM модуль, без Android-зависимостей
│   └── src/main/kotlin/io/github/terminaldetector/xload/core/
│       ├── model/      — BaseModel, LoraConfig, TrainingConfig, TrainingProgress,
│       │                 DatasetSample, AdapterCheckpoint (все с валидацией/сериализацией)
│       ├── dataset/    — JsonlDatasetParser: построчный разбор JSONL с отчётом об ошибках
│       ├── engine/     — интерфейс TrainingEngine — точка подключения бэкенда обучения
│       │   └── reference/ — ReferenceLoraEngine: честная реализация LoRA-обновления
│       │                    (W*x + scaling*B*A*x, SGD, B инициализируется нулём) на
│       │                    синтетических данных — демонстрирует весь пайплайн и
│       │                    математику LoRA без GPU/NDK/Python, но НЕ является
│       │                    языковой моделью и не производит пригодный адаптер
│       └── checkpoint/ — AdapterCheckpointJson: сериализация чекпоинта в JSON
│                          (свой формат, не настоящий SafeTensors/GGUF)
└── app/    — Android-приложение (Jetpack Compose, Material 3, Navigation Compose)
    └── src/main/kotlin/io/github/terminaldetector/xload/app/
        ├── ui/screens/  — 5 экранов мастера
        ├── ui/theme/    — тема Material 3 (с поддержкой Dynamic Color)
        └── viewmodel/   — TrainingSessionViewModel — состояние всего мастера
```

`TrainingEngine` — единственная граница, которую должен реализовать реальный бэкенд
(termux-train, MobileFineTuner, QVAC Fabric, ExecuTorch...), не трогая UI:

```kotlin
interface TrainingEngine {
    fun train(config: TrainingConfig, dataset: List<DatasetSample>): Flow<TrainingProgress>
    fun cancel()
}
```

## Сборка и запуск

`:core` — обычный Kotlin/JVM модуль, собирается и тестируется где угодно:

```bash
./gradlew :core:test
```

`:app` требует Android SDK и доступ к Google Maven (`dl.google.com`) для Android
Gradle Plugin и AndroidX/Compose-зависимостей — откройте проект в Android Studio
(Koala+), она сама подтянет SDK/AGP. minSdk 30 (Android 11+), т.к. плану нужны
NNAPI/Vulkan; тестового окружения с эмулятором/устройством у меня в этой сессии не было,
поэтому `:app` собран по всем актуальным практикам Compose, но не прогонялся через
реальную сборку — при первом открытии в Android Studio возможны мелкие правки версий.

## Что дальше (дорожная карта)

По этапам из исходного плана:

1. **Бэкенд обучения для настоящих LLM** — реализовать `TrainingEngine` поверх
   termux-train (Python/NumPy, проще для MVP) или MobileFineTuner (NDK/C++,
   производительнее); загрузка GGUF/SafeTensors, токенизация, настоящий backprop.
2. **Инференс для проверки адаптера** — llama.cpp / MLC-LLM / MediaPipe LlmInference,
   чат-экран на шаге 5 (сейчас там честная заглушка "недоступно").
3. **Экспорт в реальный формат** — настоящий `.safetensors`/GGUF вместо
   JSON-чекпоинта справочного движка.
4. **GPU и энергоэффективность** — Vulkan/OpenCL для Adreno/Mali, мониторинг
   температуры с паузой при перегреве (см. риски ниже).
5. **UI-полировка** — выбор целевых LoRA-модулей вручную (сейчас берутся по
   умолчанию из `BaseModel.targetModules`), сохранение прогресса/настроек между
   запусками, локализация строк через ресурсы (сейчас тексты захардкожены на русском).

## Риски (из исходного плана)

- **Память**: обучение 3B-модели может потребовать >6 ГБ ОЗУ — часть устройств отсеется.
- **Скорость**: на CPU обучение 1B-модели — часы; GPU ускоряет в 5–10×.
- **Термотроттлинг**: длительное обучение греет устройство — нужен мониторинг и паузы.
- **Фрагментация**: Snapdragon/MediaTek/Exynos по-разному поддерживают Vulkan/OpenCL.
