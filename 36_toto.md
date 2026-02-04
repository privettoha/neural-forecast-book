# Toto: time series optimized transformer от Databricks

## Ключевая идея

Toto[^toto] — foundation model от Databricks (2024), оптимизированная специально для временных рядов. В отличие от Chronos (использует LLM-архитектуру T5) или Lag-Llama (использует LLaMA), Toto разрабатывалась **с нуля** для задач прогнозирования.

Ключевая инновация — **exogenous variables из коробки**. Toto принимает не только историю целевой переменной, но и ковариаты (праздники, погода, цены), что критично для многих бизнес-задач.

**Результаты:** Toto показывает state-of-the-art результаты на нескольких бенчмарках, включая GiftEval[^gifteval] — новый benchmark для foundation models временных рядов.

## Зачем специализированная архитектура?

Большинство foundation models заимствуют архитектуры из NLP:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ЗАИМСТВОВАННЫЕ vs НАТИВНЫЕ АРХИТЕКТУРЫ               │
│                                                                         │
│  ЗАИМСТВОВАННЫЕ ИЗ NLP:                                                │
│  ──────────────────────                                                 │
│  • Chronos → T5 (encoder-decoder для перевода)                         │
│  • Lag-Llama → LLaMA (decoder для генерации текста)                    │
│  • TimesFM → Decoder-only (GPT-style)                                  │
│                                                                         │
│  Проблемы:                                                              │
│  • Токенизация теряет информацию                                       │
│  • Attention не оптимизирован для временной структуры                  │
│  • Нет встроенной поддержки ковариат                                   │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  НАТИВНАЯ АРХИТЕКТУРА (Toto):                                          │
│  ────────────────────────────                                           │
│  • Разработана специально для временных рядов                          │
│  • Нативная поддержка exogenous variables                              │
│  • Оптимизированный attention для temporal patterns                    │
│  • Efficient patching для длинных контекстов                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Архитектура

Toto использует **encoder-decoder** архитектуру с несколькими ключевыми модификациями:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        АРХИТЕКТУРА Toto                                 │
│                                                                         │
│  Входы:                                                                 │
│  • Target history: [y₁, y₂, ..., yₜ]                                   │
│  • Exogenous past: [x₁, x₂, ..., xₜ]                                   │
│  • Exogenous future: [xₜ₊₁, xₜ₊₂, ..., xₜ₊ₕ]                           │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  1. MULTI-RESOLUTION PATCHING                                   │   │
│  │                                                                  │   │
│  │  Target:    [...........] [...........] [...........]           │   │
│  │  Exog past: [...........] [...........] [...........]           │   │
│  │                 patch₁        patch₂        patch₃               │   │
│  │                                                                  │   │
│  │  Разные patch sizes для разных масштабов:                       │   │
│  │  • Мелкие патчи: локальные паттерны                             │   │
│  │  • Крупные патчи: глобальные тренды                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  2. TEMPORAL ENCODER                                            │   │
│  │                                                                  │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │  Layer 1:                                                │   │   │
│  │  │  • Temporal Self-Attention (между патчами во времени)   │   │   │
│  │  │  • Cross-Attention (target ↔ exogenous)                 │   │   │
│  │  │  • FFN                                                   │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  │  ...                                                            │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │  Layer N:                                                │   │   │
│  │  │  • Temporal Self-Attention                               │   │   │
│  │  │  • Cross-Attention                                       │   │   │
│  │  │  • FFN                                                   │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  3. FUTURE DECODER                                              │   │
│  │                                                                  │   │
│  │  • Future exogenous embedding                                   │   │
│  │  • Cross-attention к encoder output                             │   │
│  │  • Autoregressive или direct prediction (configurable)         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  4. DISTRIBUTION HEAD                                           │   │
│  │                                                                  │   │
│  │  Несколько вариантов output:                                    │   │
│  │  • Point forecast: Linear → ŷ                                   │   │
│  │  • Probabilistic: Linear → (μ, σ) для Gaussian                 │   │
│  │  • Quantile: Linear → [q₀.₁, q₀.₅, q₀.₉]                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Выход: Ŷ ∈ ℝ^H или распределение p(y_{t+1:t+H})                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Ключевые компоненты

**1. Multi-Resolution Patching**

Toto использует патчи нескольких размеров одновременно:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MULTI-RESOLUTION PATCHING                            │
│                                                                         │
│  Входной ряд: [x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈, x₉, x₁₀, x₁₁, x₁₂]    │
│                                                                         │
│  Patch size = 2:  [1,2] [3,4] [5,6] [7,8] [9,10] [11,12]              │
│                     ↓     ↓     ↓     ↓      ↓      ↓                   │
│                    p₁    p₂    p₃    p₄     p₅     p₆  (6 патчей)     │
│                                                                         │
│  Patch size = 4:  [1,2,3,4] [5,6,7,8] [9,10,11,12]                     │
│                      ↓          ↓          ↓                            │
│                     P₁         P₂         P₃  (3 патча)               │
│                                                                         │
│  Patch size = 6:  [1,2,3,4,5,6] [7,8,9,10,11,12]                       │
│                        ↓               ↓                                │
│                       PP₁            PP₂  (2 патча)                    │
│                                                                         │
│  Все патчи обрабатываются вместе в attention                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Преимущества:**
- Мелкие патчи захватывают краткосрочные флуктуации
- Крупные патчи захватывают долгосрочные тренды
- Модель сама учится комбинировать масштабы

**2. Exogenous Cross-Attention**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EXOGENOUS CROSS-ATTENTION                            │
│                                                                         │
│  Query: Target embeddings (что прогнозируем)                           │
│  Key, Value: Exogenous embeddings (внешние факторы)                    │
│                                                                         │
│  Attention(Q, K, V) = softmax(QK^T / √d) V                             │
│                                                                         │
│  Пример:                                                                │
│  ─────────                                                              │
│  Target: продажи мороженого                                            │
│  Exogenous: температура воздуха                                         │
│                                                                         │
│  Cross-attention позволяет модели:                                     │
│  • Понять корреляцию продаж с температурой                            │
│  • Использовать прогноз температуры для прогноза продаж               │
│  • Автоматически определить важность каждого фактора                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**3. Flexible Output Modes**

Toto поддерживает несколько режимов вывода:

| Режим | Описание | Use case |
|-------|----------|----------|
| **Point** | Одно значение | Простые задачи |
| **Probabilistic** | μ, σ для Gaussian | Uncertainty estimation |
| **Quantile** | Несколько квантилей | Risk-aware forecasting |
| **Sample** | Monte Carlo samples | Full distribution |

## Обработка ковариат

Одно из главных преимуществ Toto — **нативная поддержка exogenous variables**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ТИПЫ КОВАРИАТ В Toto                                 │
│                                                                         │
│  1. PAST-ONLY COVARIATES (известны только для прошлого)                │
│  ─────────────────────────────────────────────────────────              │
│  Примеры: фактические продажи конкурентов, реальная погода             │
│                                                                         │
│  Past:   [x₁, x₂, ..., xₜ]                                             │
│  Future: [?, ?, ..., ?]  ← неизвестны                                  │
│                                                                         │
│  Использование: только в encoder                                        │
│                                                                         │
│  2. KNOWN FUTURE COVARIATES (известны для будущего)                    │
│  ──────────────────────────────────────────────────                     │
│  Примеры: праздники, день недели, запланированные промо                │
│                                                                         │
│  Past:   [x₁, x₂, ..., xₜ]                                             │
│  Future: [xₜ₊₁, xₜ₊₂, ..., xₜ₊ₕ]  ← известны заранее                  │
│                                                                         │
│  Использование: в encoder и decoder                                     │
│                                                                         │
│  3. STATIC COVARIATES (не меняются во времени)                         │
│  ──────────────────────────────────────────                             │
│  Примеры: категория товара, регион, тип клиента                        │
│                                                                         │
│  Значение: c = const                                                    │
│                                                                         │
│  Использование: добавляется ко всем embeddings                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Pre-training данные

Toto обучена на **масштабном корпусе** временных рядов:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ДАННЫЕ PRE-TRAINING                                  │
│                                                                         │
│  Источники:                                                             │
│  ───────────                                                            │
│  • Публичные датасеты (Monash, ETT, Weather, Traffic)                  │
│  • Синтетические данные (GP, ARIMA, сезонные паттерны)                 │
│  • Данные из Databricks Data Intelligence Platform                     │
│                                                                         │
│  Объём:                                                                 │
│  ───────                                                                │
│  • ~1 миллиард точек                                                   │
│  • Разные частоты: от минутных до годовых                              │
│  • Разные домены: финансы, ритейл, энергетика, IoT                     │
│                                                                         │
│  Augmentations:                                                         │
│  ──────────────                                                         │
│  • Масштабирование (scale jittering)                                   │
│  • Сдвиг (temporal shift)                                              │
│  • Добавление шума                                                     │
│  • Случайное маскирование                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Размеры модели

| Вариант | Параметры | Layers | Hidden dim | Context |
|---------|-----------|--------|------------|---------|
| Toto-Base | ~100M | 12 | 768 | 1024 |
| Toto-Large | ~300M | 24 | 1024 | 2048 |

## Код: Toto

### С Databricks

```python
# Через Databricks Model Serving
import requests
import json

# API endpoint
endpoint = "https://<workspace>.databricks.com/serving-endpoints/toto/invocations"

# Данные
data = {
    "inputs": {
        "target": [100, 102, 105, 103, 108, 110, ...],
        "exogenous": {
            "temperature": [20, 22, 25, 23, 28, 30, ...],
            "is_holiday": [0, 0, 0, 1, 0, 0, ...]
        },
        "future_exogenous": {
            "temperature": [32, 30, 28, 25, ...],  # прогноз погоды
            "is_holiday": [0, 0, 1, 0, ...]        # будущие праздники
        },
        "prediction_length": 24
    }
}

# Прогноз
response = requests.post(
    endpoint,
    headers={"Authorization": f"Bearer {token}"},
    json=data
)

forecast = response.json()["predictions"]
```

### С NeuralForecast (если доступно)

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import Toto

# Создание модели
model = Toto(
    h=24,                    # горизонт
    input_size=512,          # длина контекста
    futr_exog_list=['temperature', 'is_holiday'],  # future known
    hist_exog_list=['competitor_sales'],           # past only
    stat_exog_list=['product_category']            # static
)

# NeuralForecast wrapper
nf = NeuralForecast(models=[model], freq='H')

# Прогноз
forecasts = nf.predict(df, futr_df=future_exog_df)
```

### Fine-tuning

```python
from toto import TotoModel, TotoConfig

# Загрузка pretrained
model = TotoModel.from_pretrained("databricks/toto-base")

# Конфигурация fine-tuning
config = TotoConfig(
    learning_rate=1e-5,
    batch_size=32,
    epochs=10,
    freeze_encoder=True  # заморозить encoder, обучать только decoder
)

# Fine-tuning
model.fine_tune(
    train_data=train_dataset,
    val_data=val_dataset,
    config=config
)

# Сохранение
model.save_pretrained("my-toto-finetuned")
```

## GiftEval Benchmark

Databricks представила **GiftEval**[^gifteval] — новый benchmark для foundation models:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GIFTEVAL: НОВЫЙ BENCHMARK                            │
│                                                                         │
│  Почему новый benchmark?                                                │
│  ─────────────────────────                                              │
│  • Существующие бенчмарки (Monash) слишком просты для FM               │
│  • Не тестируют ковариаты                                              │
│  • Не тестируют few-shot и fine-tuning                                 │
│                                                                         │
│  GiftEval включает:                                                     │
│  ───────────────────                                                    │
│  • 23 датасета из разных доменов                                       │
│  • Тесты с ковариатами и без                                           │
│  • Zero-shot, few-shot, full fine-tuning режимы                        │
│  • Разные горизонты (короткие, средние, длинные)                       │
│  • Multivariate и univariate задачи                                    │
│                                                                         │
│  Метрики:                                                               │
│  ─────────                                                              │
│  • MASE (Mean Absolute Scaled Error)                                   │
│  • CRPS (для probabilistic)                                            │
│  • Coverage (калибровка интервалов)                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Результаты на GiftEval

| Модель | MASE (zero-shot) | MASE (fine-tuned) | With Covariates |
|--------|------------------|-------------------|-----------------|
| Seasonal Naive | 1.000 | — | N/A |
| Chronos-Large | 0.812 | 0.756 | ✗ |
| TimesFM | 0.798 | 0.742 | ✗ |
| Moirai-Large | 0.785 | 0.731 | ✗ |
| **Toto-Base** | **0.756** | **0.698** | **✓** |
| **Toto-Large** | **0.723** | **0.672** | **✓** |

**Важно:** Toto — единственная модель в сравнении с нативной поддержкой ковариат.

## Сравнение с другими foundation models

| Аспект | Toto | Chronos | TimesFM | Moirai | TimeGPT |
|--------|------|---------|---------|--------|---------|
| **Exogenous** | ✓ Native | ✗ | ✗ | ✗ | ✓ |
| **Multivariate** | ✓ | ✗ | ✗ | ✓ | ✗ |
| **Probabilistic** | ✓ | ✓ | ✗ | ✓ | ✓ |
| **Открытая** | Частично | Да | Да | Да | Нет |
| **Fine-tuning** | Да | Да | Частично | Да | Через API |
| **Архитектура** | Native TS | T5 | GPT-style | Encoder | Unknown |

## Интеграция с Databricks

Toto глубоко интегрирована с экосистемой Databricks:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DATABRICKS ECOSYSTEM                                 │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Delta Lake                                                     │   │
│  │  • Хранение временных рядов                                    │   │
│  │  • Версионирование данных                                      │   │
│  │  • Time travel для reproducibility                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  MLflow                                                         │   │
│  │  • Tracking экспериментов                                      │   │
│  │  • Model registry для Toto                                     │   │
│  │  • A/B testing разных версий                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Model Serving                                                  │   │
│  │  • REST API для inference                                      │   │
│  │  • Auto-scaling                                                │   │
│  │  • GPU/CPU endpoints                                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Сильные стороны

**Exogenous variables.** Единственная открытая foundation model с нативной поддержкой ковариат.

**Native architecture.** Архитектура разработана специально для временных рядов, а не адаптирована из NLP.

**Multi-resolution.** Патчи разных размеров для захвата паттернов на разных масштабах.

**Enterprise-ready.** Глубокая интеграция с Databricks для production deployment.

**GiftEval benchmark.** Новый стандарт оценки foundation models.

## Ограничения

**Vendor lock-in.** Оптимально работает в экосистеме Databricks. Standalone использование ограничено.

**Частично открытая.** Веса доступны, но training code и некоторые детали не публикуются.

**Computational cost.** 300M параметров требует GPU для эффективного inference.

**Новая модель.** Меньше community support и готовых интеграций по сравнению с Chronos.

## Когда выбирать Toto

✅ **Используйте Toto, если:**
- Важны ковариаты (праздники, промо, погода)
- Используете Databricks
- Нужен production-ready deployment
- Важна точность на enterprise данных

❌ **Рассмотрите альтернативы, если:**
- Нет ковариат → Chronos, TimesFM (проще)
- Ограничены ресурсы → Lag-Llama (компактнее)
- Нужна полная открытость → Chronos
- Privacy-sensitive данные → локальные модели

## Выводы

1. **Native architecture** — разработка с нуля для временных рядов даёт преимущество над адаптированными NLP-архитектурами.

2. **Exogenous variables** — критичная возможность для бизнес-задач. Toto — первая открытая FM с нативной поддержкой.

3. **Multi-resolution patching** — элегантное решение для захвата паттернов на разных временных масштабах.

4. **Enterprise focus** — глубокая интеграция с Databricks упрощает production deployment.

5. **GiftEval** — новый benchmark устанавливает более высокую планку для foundation models.

---

## Ссылки

[^toto]: Databricks. (2024). Introducing Toto: The First Foundation Model for Time Series. *Databricks Blog*. https://www.databricks.com/blog/introducing-toto-first-multimodal-foundation-model-time-series

[^gifteval]: Databricks. (2024). GiftEval: A Benchmark for Time Series Foundation Models. *GitHub*. https://github.com/databricks/gifteval

[^databricks]: Databricks. (2024). Databricks Documentation: Time Series Forecasting. https://docs.databricks.com/machine-learning/time-series-forecasting.html

[^patchsize]: Nie, Y., et al. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *ICLR 2023*. https://arxiv.org/abs/2211.14730
