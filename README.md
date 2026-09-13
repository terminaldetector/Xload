# Xload

**On-Device LoRA Trainer** — Android-приложение для локального дообучения LoRA-адаптеров
компактных LLM (0.5–3B параметров) прямо на смартфоне, без выгрузки данных в облако.

Целевые модели: Qwen2.5-0.5B/1.5B, Gemma-3-1B, Llama-3.2-1B, Phi-3-mini (в 4-bit квантизации).

## Статус проекта

Инфраструктура (Этап 1) готова, и в `:app` подключён настоящий бэкенд обучения —
[termux-train](https://pypi.org/project/termux-train/) через Chaquopy (Python-рантайм
внутри Android-приложения). Подробности и что это означает на практике — ниже.

Реализовано:

- Полный каркас Android-приложения на Kotlin + Jetpack Compose, разбитый на модули
  `:core` (чистая Kotlin/JVM логика, без зависимости от Android SDK) и `:app` (UI + бэкенд).
- Мастер из 5 экранов по стадиям пайплайна из плана: выбор модели → датасет →
  гиперпараметры → обучение → тест/экспорт.
- `:core`: валидация конфигурации, парсер JSONL-датасетов, интерфейс `TrainingEngine`
  (точка подключения бэкенда) и `ReferenceLoraEngine` — независимая от Android
  справочная реализация LoRA-обновления на чистом Kotlin (14 тестов, `./gradlew :core:test`).
- `:app` → `TermuxTrainEngine`: реальное обучение через termux-train (см. ниже) —
  настоящий ByteTokenizer, настоящий backprop через их autograd-движок, настоящий
  SafeTensors-чекпоинт, который их же загрузчик может прочитать обратно.
- Импорт файлов модели/датасета через Storage Access Framework — приложение не просит
  разрешение `INTERNET` и не имеет сетевых вызовов: всё офлайн, как и требует план.

### Про termux-train: что проверено и что нет

`termux-train` — реальная, работающая библиотека, но **последняя версия на PyPI (1.1.5)
не устанавливается**: она требует `ameva-component-sdk`, которого нет в PyPI
(проверено: `404`). Зависимость появилась между 1.1.3 и 1.1.5; версия **1.1.3** — самая
свежая, которая ставится и работает. `app/build.gradle.kts` намеренно пинит именно её.

Всё, что описано ниже, **проверено вручную в локальном venv** (`pip install
termux-train==1.1.3`, Python 3.11) — не только прочитано в документации:

- Инъекция LoRA в attention-слои `termux_train.nn.transformer.TinyTransformerLM`
  (у библиотеки нет готового «apply LoRA to model» хелпера — пришлось разобраться
  в устройстве `Module`/`LoRALinear` и написать это самому).
- Обучение через `ByteTokenizer` + `optim.AdamW` + автоград библиотеки — loss
  реально и стабильно падает (проверено на 60 эпохах: 5.73 → 5.15).
- Сохранение и загрузка адаптера через `checkpoint.lora_io.save_lora_adapter` /
  `load_lora_adapter` (не `checkpoint.safetensors.save_safetensors` напрямик — он
  ожидает только тензоры, а `adapter_state_dict()` подмешивает метаданные; это
  легко перепутать, наступил на эти грабли и задокументировал в коде).

Итоговый скрипт лежит в `app/src/main/python/xload_trainer.py` — **он и есть тот
код, что был протестирован**, не переписан načisto после.

**Что НЕ проверено** (и не могло быть в этой изолированной среде): сама сборка
`:app` через Chaquopy. Google Maven (`dl.google.com`) и собственный Maven Chaquopy
(`chaquo.com`) заблокированы политикой прокси в этой сессии — тот же блокер, что
и для остального `:app` (см. ниже). Версия плагина `com.chaquo.python:gradle:17.0.0`
подтверждена как реально существующая (через зеркало Gradle Plugin Portal), но сам
плагин и загрузку Python-рантайма для Android я не прогонял. Мост Kotlin↔Python
(`TermuxTrainEngine`) — сам паттерн `callbackFlow` + блокирующий вызов + `trySend`
я тоже отдельно проверил на чистом JVM-тесте (без Chaquopy), а вот именно
Java-объект-как-колбэк-в-Python — стандартный, годами стабильный механизм
Chaquopy, но не тестировался здесь напрямую.

**Архитектурное упрощение**: `TinyTransformerLM` — это лёгкая демо-архитектура
из самой termux-train, **не** архитектура Qwen/Gemma/Llama. То есть сейчас
обучается LoRA-адаптер для этой демо-модели, а не для реально выбранного на
первом экране базового файла — загрузка настоящих весов GGUF/SafeTensors с
точным соответствием слоёв конкретной модели осталась в дорожной карте
(«Что дальше», п. 1). Зато весь остальной путь — токенизация, LoRA, автоград,
оптимизатор, чекпоинт — теперь настоящий, а не заглушка.

### FBD/фрактальная память — отложено

Второй присланный план (интеграция FractalMind + RAPTOR поверх той же LoRA-ветки)
пока не реализован осознанно: FractalMind по умолчанию ходит в Ollama Cloud для
REM-фазы (суммаризация/эмбеддинги) — облачный вызов, напрямую конфликтующий с
целью «без выгрузки данных». Плюс сам FractalMind — Rust+SurrealDB+Axum сервер
(REST API на 12000), а не встраиваемая Android-библиотека, так что его запуск
на телефоне — отдельная задача (Termux, либо кросс-компиляция Rust под
`aarch64-linux-android`). Решено сначала докрутить LoRA-ветку; ФБД возвращается
в работу отдельным заходом, когда будет решено, чем заменить Ollama Cloud.

## Архитектура

```
Xload/
├── core/   — чистый Kotlin/JVM модуль, без Android-зависимостей
│   └── src/main/kotlin/io/github/terminaldetector/xload/core/
│       ├── model/      — BaseModel, LoraConfig, TrainingConfig, TrainingProgress,
│       │                 DatasetSample, AdapterCheckpoint (все с валидацией/сериализацией)
│       ├── dataset/    — JsonlDatasetParser: построчный разбор JSONL с отчётом об ошибках
│       ├── engine/     — интерфейс TrainingEngine — точка подключения бэкенда обучения
│       │   └── reference/ — ReferenceLoraEngine: справочная LoRA-математика на чистом
│       │                    Kotlin (W*x + scaling*B*A*x, SGD) — демонстрирует пайплайн
│       │                    без Python/NDK, но не языковая модель
│       └── checkpoint/ — AdapterCheckpointJson: сериализация чекпоинта ReferenceLoraEngine
│                          в JSON (свой формат, не для termux-train)
└── app/    — Android-приложение (Jetpack Compose, Material 3, Navigation Compose)
    └── src/main/
        ├── python/xload_trainer.py — обучение через termux-train (см. выше)
        └── kotlin/io/github/terminaldetector/xload/app/
            ├── engine/      — TermuxTrainEngine: Kotlin↔Python мост (Chaquopy)
            ├── ui/screens/  — 5 экранов мастера
            ├── ui/theme/    — тема Material 3 (с поддержкой Dynamic Color)
            └── viewmodel/   — TrainingSessionViewModel — состояние всего мастера
```

`TrainingEngine` — граница между UI и бэкендом обучения, которую сейчас реализуют
`ReferenceLoraEngine` (в `:core`, чистый Kotlin) и `TermuxTrainEngine` (в `:app`,
активный по умолчанию); MobileFineTuner/QVAC Fabric/ExecuTorch встанут сюда же:

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

`:app` требует Android SDK, а также доступ к **Google Maven** (`dl.google.com`,
для Android Gradle Plugin и AndroidX/Compose) и к **Chaquopy Maven** (`chaquo.com`,
для плагина и Python-рантайма Android) — откройте проект в Android Studio (Koala+),
она сама подтянет SDK/AGP/Chaquopy. minSdk 30 (Android 11+), т.к. плану нужны
NNAPI/Vulkan. Ни то, ни другое не было доступно в изолированной среде, где писался
этот код (см. подробности выше) — `:core` и Python-логика `xload_trainer.py`
реально протестированы, `:app` в целом — нет; при первом открытии в Android
Studio возможны мелкие правки версий.

## Что дальше (дорожная карта)

1. **Реальные веса базовой модели** — загрузка GGUF/SafeTensors выбранного файла
   вместо демо-архитектуры `TinyTransformerLM`, с сопоставлением слоёв под
   Qwen2.5/Gemma-3/Llama-3.2; либо/дополнительно MobileFineTuner (NDK/C++,
   быстрее term termux-train на чистом Python) или QVAC Fabric (GPU).
2. **Ускорение** — extra `accelerated` (NumPy) или `vulkan` (`ameva-runtime`,
   существует на PyPI) у termux-train; сейчас обучение идёт на чистом Python
   и заметно медленнее, чем будет с этими бэкендами.
3. **Инференс для проверки адаптера** — llama.cpp / MLC-LLM / MediaPipe LlmInference,
   чат-экран на шаге 5 (сейчас там честная заглушка "недоступно").
4. **ФБД-ветка** — FractalMind/RAPTOR, после решения вопроса с Ollama Cloud
   (локальная замена, либо явное согласие пользователя на этот один вызов).
5. **GPU и энергоэффективность** — мониторинг температуры с паузой при перегреве.
6. **UI-полировка** — выбор целевых LoRA-модулей вручную (сейчас берутся по
   умолчанию из `BaseModel.targetModules`), сохранение прогресса/настроек между
   запусками, локализация строк через ресурсы (сейчас тексты захардкожены на русском).

## Риски (из исходного плана)

- **Память**: обучение 3B-модели может потребовать >6 ГБ ОЗУ — часть устройств отсеется.
- **Скорость**: чистый Python (без NumPy/Vulkan extras) — счёт на секунды за шаг даже
  для маленькой демо-модели; для 1B-модели на CPU это часы, как и предупреждал план.
  GPU/accelerated-бэкенды ускоряют на порядок.
- **Термотроттлинг**: длительное обучение греет устройство — нужен мониторинг и паузы.
- **Фрагментация**: Snapdragon/MediaTek/Exynos по-разному поддерживают Vulkan/OpenCL.
