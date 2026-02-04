# TimesFM: decoder-only foundation model от Google

## Ключевая идея

TimesFM[^timesfm] — foundation model от Google Research (2024), использующая **decoder-only** архитектуру (как GPT, LLaMA) вместо encoder-decoder (как T5 в Chronos). Модель обучена на **100 миллиардах точек** из внутренних данных Google и публичных источников.

Ключевое отличие TimesFM — **patching без токенизации**. Вместо квантования значений в дискретные токены (как Chronos), TimesFM работает с непрерывными патчами напрямую. Это сохраняет точность и упрощает архитектуру.

**Результаты:** TimesFM показывает state-of-the-art zero-shot результаты на Monash benchmark и конкурирует с task-specific моделями[^timesfm].

## Decoder-only vs Encoder-Decoder

```
┌─────────────────────────────────────────────────────────────────────────┐
│              ENCODER-DECODER vs DECODER-ONLY                            │
│                                                                         │
│  ENCODER-DECODER (T5, Chronos):                                        │
│  ──────────────────────────────                                         │
│                                                                         │
│  [История] ──→ Encoder ──→ [Context]                                   │
│                               │                                         │
│                               ▼                                         │
│  [BOS] ──→ Decoder ──→ [Прогноз]                                       │
│                               │                                         │
│  Cross-attention между encoder и decoder                               │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  DECODER-ONLY (GPT, TimesFM):                                          │
│  ─────────────────────────────                                          │
│                                                                         │
│  [История | Прогноз] ──→ Decoder ──→ [Следующий токен]                 │
│        ↑                                    │                           │
│        └────────────────────────────────────┘                           │
│                 (autoregressive)                                        │
│                                                                         │
│  Causal attention: каждый токен видит только предыдущие               │
│                                                                         │
│  Преимущества decoder-only:                                            │
│  • Проще архитектура (нет cross-attention)                             │
│  • Лучше масштабируется                                                │
│  • Наследует инсайты из LLM                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Архитектура TimesFM

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        АРХИТЕКТУРА TimesFM                              │
│                                                                         │
│  Вход: x = [x₁, x₂, ..., xₗ]  (L точек)                                │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  1. INPUT PATCHING                                              │   │
│  │                                                                  │   │
│  │  [x₁...x₃₂] [x₃₃...x₆₄] [x₆₅...x₉₆] ...                        │   │
│  │      ↓           ↓           ↓                                   │   │
│  │   patch₁      patch₂      patch₃     (размер 32)               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  2. PATCH EMBEDDING                                             │   │
│  │                                                                  │   │
│  │  Residual block: Linear → ReLU → Linear                        │   │
│  │  patch ∈ ℝ³² → embedding ∈ ℝᵈ                                  │   │
│  │                                                                  │   │
│  │  + Positional encoding (learnable)                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  3. STACKED TRANSFORMER DECODER                                 │   │
│  │                                                                  │   │
│  │  Layer 1: Masked Self-Attention → FFN → LayerNorm              │   │
│  │  Layer 2: Masked Self-Attention → FFN → LayerNorm              │   │
│  │  ...                                                            │   │
│  │  Layer N: Masked Self-Attention → FFN → LayerNorm              │   │
│  │                                                                  │   │
│  │  Causal mask: каждый патч видит только предыдущие              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  4. OUTPUT PROJECTION                                           │   │
│  │                                                                  │   │
│  │  embedding ∈ ℝᵈ → Linear → output_patch ∈ ℝ¹²⁸                 │   │
│  │                                                                  │   │
│  │  Output patch size = 128 (больше input patch для эффективности)│   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Выход: ŷ = [ŷ₁, ŷ₂, ..., ŷₕ]                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Ключевые особенности

**1. Input patch size = 32**

Входные патчи фиксированного размера 32. Это позволяет работать с рядами разной длины без изменения архитектуры.

**2. Output patch size = 128**

Выходные патчи **больше** входных. Это ускоряет inference: вместо генерации 1 точки за шаг, модель генерирует 128 точек.

```
Сравнение:
• Chronos: 1 токен = 1 точка → H forward passes для горизонта H
• TimesFM: 1 патч = 128 точек → ⌈H/128⌉ forward passes
```

**3. Непрерывные значения**

В отличие от Chronos, TimesFM **не квантует** значения. Патчи содержат реальные числа, а loss function — MSE.

$$\mathcal{L} = \frac{1}{H} \sum_{t=1}^{H} (y_t - \hat{y}_t)^2$$

## Pre-training

TimesFM обучается на **100 миллиардах точек** из разных источников:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ДАННЫЕ ДЛЯ PRE-TRAINING                              │
│                                                                         │
│  Публичные данные (~40%):                                               │
│  • Google Trends                                                        │
│  • Wiki page views                                                      │
│  • Electricity datasets                                                 │
│  • Weather data                                                         │
│                                                                         │
│  Внутренние данные Google (~60%):                                       │
│  • Google Search trends                                                 │
│  • YouTube metrics                                                      │
│  • Ads data                                                             │
│  • Cloud monitoring                                                     │
│                                                                         │
│  Синтетические данные:                                                  │
│  • Gaussian Processes                                                   │
│  • ARIMA simulations                                                    │
│  • Seasonal patterns                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Frequency tokens

TimesFM использует **frequency tokens** для адаптации к разным частотам данных:

```python
# Frequency tokens добавляются к входу
freq_token = get_frequency_token(freq='H')  # часовые данные
# Модель учится различать часовые, дневные, месячные паттерны
```

Это позволяет одной модели работать с данными от минутной до годовой гранулярности.

## Размеры моделей

| Модель | Параметры | Layers | Hidden dim | Heads |
|--------|-----------|--------|------------|-------|
| TimesFM-200M | 200M | 20 | 1280 | 16 |

На момент публикации доступна только одна версия (200M). Google планирует выпустить модели большего размера.

## Код: TimesFM

### Установка

```bash
pip install timesfm
```

### Базовый пример

```python
import timesfm
import numpy as np

# Инициализация модели
tfm = timesfm.TimesFm(
    context_len=512,      # максимальная длина контекста
    horizon_len=128,      # максимальный горизонт
    input_patch_len=32,   # размер входного патча
    output_patch_len=128, # размер выходного патча
    num_layers=20,
    model_dims=1280,
)

# Загрузка весов
tfm.load_from_checkpoint(
    "google/timesfm-1.0-200m"
)

# Подготовка данных
context = np.array([100, 102, 105, 103, 108, ...])  # история

# Прогноз
forecast = tfm.forecast(
    inputs=context,
    freq='H',          # частота данных
    horizon=24         # горизонт
)

# forecast.shape: [24]
```

### Batch inference

```python
# Множество рядов одновременно
contexts = [
    np.array([100, 102, 105, ...]),
    np.array([50, 52, 48, ...]),
    np.array([200, 210, 205, ...])
]

forecasts = tfm.forecast(
    inputs=contexts,
    freq='H',
    horizon=24
)
# forecasts.shape: [3, 24]
```

### С NeuralForecast

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import TimesFM

model = TimesFM(
    h=24,
    input_size=512,
    freq='H'
)

nf = NeuralForecast(models=[model], freq='H')
forecasts = nf.predict(df)
```

## Результаты

### Zero-shot на Monash benchmark

| Модель | MASE (mean) | Rank |
|--------|-------------|------|
| Seasonal Naive | 1.000 | — |
| AutoARIMA | 0.891 | 5 |
| DeepAR | 0.812 | 4 |
| PatchTST | 0.789 | 3 |
| Chronos-Base | 0.756 | 2 |
| **TimesFM-200M** | **0.734** | **1** |

TimesFM показывает лучшие zero-shot результаты среди всех моделей.

### In-domain vs Zero-shot

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TimesFM: КОГДА РАБОТАЕТ ЛУЧШЕ                        │
│                                                                         │
│  Сильные стороны:                                                       │
│  ─────────────────                                                      │
│  • Данные похожи на Google Trends / Search                             │
│  • Дневная/часовая гранулярность                                       │
│  • Короткие/средние горизонты (до 128)                                 │
│                                                                         │
│  Слабые стороны:                                                        │
│  ────────────────                                                       │
│  • Финансовые данные (мало в pre-training)                             │
│  • Очень длинные горизонты (>128)                                      │
│  • Данные с сильной сезонностью (лучше task-specific)                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Сравнение с Chronos

| Аспект | TimesFM | Chronos |
|--------|---------|---------|
| **Архитектура** | Decoder-only | Encoder-decoder (T5) |
| **Токенизация** | Непрерывные патчи | Дискретные токены |
| **Output granularity** | 128 точек/шаг | 1 токен/шаг |
| **Inference speed** | Быстрее | Медленнее |
| **Probabilistic** | Нет (point forecast) | Да (sampling) |
| **Размеры** | 200M | 8M - 710M |
| **Открытость** | Частично (веса, не код) | Полностью открытая |

## Ограничения

**Только point forecasts.** TimesFM не поддерживает probabilistic forecasting из коробки. Для uncertainty нужны дополнительные методы (conformal prediction).

**Фиксированный output patch.** Горизонт должен быть кратен 128 или требуется padding.

**Закрытый training code.** Веса доступны, но код обучения не опубликован.

**Нет ковариат.** TimesFM работает только с историей целевой переменной.

## Когда выбирать TimesFM

✅ **Используйте TimesFM, если:**
- Нужен быстрый zero-shot прогноз
- Горизонт до 128 точек
- Point forecast достаточен
- Данные похожи на web/search metrics

❌ **Рассмотрите альтернативы, если:**
- Нужен probabilistic output → Chronos, Lag-Llama
- Очень длинные горизонты → Chronos
- Важны ковариаты → TimeGPT
- Multivariate → Moirai

## Выводы

1. **Decoder-only архитектура** показывает сильные результаты, наследуя инсайты из LLM.

2. **Непрерывные патчи** вместо токенизации упрощают архитектуру и сохраняют точность.

3. **Большие выходные патчи (128)** ускоряют inference по сравнению с autoregressive моделями.

4. **100B точек pre-training** — масштаб данных критичен для качества.

5. **Практика:** TimesFM — хороший выбор для быстрого zero-shot с акцентом на скорость.

В следующей главе рассмотрим **Moirai** — foundation model от Salesforce с поддержкой любого числа переменных.

---

## Ссылки

[^timesfm]: Das, A., Kong, W., Leber-Yao, A., Karwa, S., & Mathias, S. (2024). A Decoder-only Foundation Model for Time-series Forecasting. *ICML 2024*. https://arxiv.org/abs/2310.10688

[^google]: Google Research. TimesFM: Time Series Foundation Model. https://research.google/blog/timesfm/

[^patchtst]: Nie, Y., et al. (2023). A Time Series is Worth 64 Words. *ICLR 2023*. https://arxiv.org/abs/2211.14730
