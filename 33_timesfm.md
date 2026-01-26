# TimesFM. Decoder-only foundation model от Google Research

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/privettoha/neural-forecast-book/blob/main/notebooks/33_timesfm.ipynb)

TimesFM ([статья](https://arxiv.org/abs/2310.10688), ICML 2024) — decoder-only foundation model для временных рядов от Google Research. Модель с 200M параметрами, обученная на 100+ миллиардах временных точек, демонстрирует zero-shot производительность, сопоставимую с supervised моделями, обученными специально на тестовых данных[^timesfm-paper]

[^timesfm-paper]: Das, A., et al. "A decoder-only foundation model for time-series forecasting." ICML 2024. https://arxiv.org/abs/2310.10688

TimesFM также доступен как **официальный продукт Google** в BigQuery через функцию `AI.FORECAST` — без необходимости управлять моделями или эндпоинтами[^timesfm-bigquery]

[^timesfm-bigquery]: Google Cloud Blog. "TimesFM models in BigQuery and AlloyDB." November 2025. "TimesFM is a powerful time-series foundation model... pre-trained on a vast dataset of over 400 billion real-world time-points." https://cloud.google.com/blog/products/data-analytics/timesfm-models-in-bigquery-and-alloydb

## Идея

Почему **decoder-only**, а не encoder-decoder?[^timesfm-decoder]

[^timesfm-decoder]: Das, A., et al. "TimesFM." ICML 2024. Section 3. https://arxiv.org/abs/2310.10688

Decoder-only архитектура (как в GPT) обучается предсказывать следующий элемент на основе всех предыдущих, используя **каузальное внимание** (токены не могут «заглядывать в будущее»). При inference это даёт гибкость: произвольная длина контекста → произвольный горизонт прогноза, патч за патчем.

Ключевая инновация TimesFM — **асимметричный патчинг**:
➖ Input patch = **32** точки (мелкие порции для «чтения» истории)
➖ Output patch = **128** точек (крупные блоки для «выдачи» прогноза)

Это позволяет делать меньше авторегрессионных шагов: для прогноза на 512 точек нужно всего 4 шага вместо 16[^timesfm-patching]

[^timesfm-patching]: Das, A., et al. "TimesFM." ICML 2024. Section 6.2: "By keeping the output_patch_len longer than input_patch_len one can ensure fewer autoregressive steps." https://arxiv.org/abs/2310.10688

## Архитектура

```
Input time series
    ↓
Разбиение на патчи (input_patch_len=32)
    ↓
Input Residual Block (MLP) → vector [model_dim]
    ↓
Positional Encoding
    ↓
Stacked Transformer Layers (num_layers=20)
  - Multi-head causal self-attention
  - Feed-forward network
    ↓
Output Residual Block (MLP) → forecast [output_patch_len=128]
```

🔢 **Параметры TimesFM 1.0-200M:**
➖ num_layers = 20
➖ model_dims = 1280
➖ input_patch_len = 32
➖ output_patch_len = 128
➖ ~200M параметров[^timesfm-params]

[^timesfm-params]: HuggingFace: google/timesfm-1.0-200m. "input_patch_len=32, output_patch_len=128, num_layers=20, model_dims=1280." https://huggingface.co/google/timesfm-1.0-200m

## Данные обучения

Корпус ~100B временных точек из трёх источников[^timesfm-data]:

[^timesfm-data]: Das, A., et al. "TimesFM." ICML 2024. Section 4, Table 1: "Composition of TimesFM pretraining dataset." https://arxiv.org/abs/2310.10688

🔢 **Google Trends** — популярность поисковых запросов (cutoff: EoY 2022)
🔢 **Wikipedia Pageviews** — просмотры страниц Википедии (cutoff: Nov 2023)
🔢 **Синтетические данные** — искусственные ряды с контролируемыми паттернами

Это создаёт domain bias: модель обучена преимущественно на веб-аналитике

## Версии модели

| Версия | Параметры | Контекст | Особенности |
|--------|-----------|----------|-------------|
| **1.0-200M** | 200M | 512 | Первый релиз, только point forecasts[^timesfm-10] |
| **2.0-500M** | 500M | 2048 | Finetuning, quantile heads, covariates (XReg)[^timesfm-20] |
| **2.5-200M** | 200M | **16K** | #1 на GIFT-Eval, continuous quantile head[^timesfm-25] |

[^timesfm-10]: HuggingFace: google/timesfm-1.0-200m. "Context lengths up to 512 time points... focuses on point forecasts." https://huggingface.co/google/timesfm-1.0-200m
[^timesfm-20]: PyPI: timesfm. "500m checkpoint... up to 25% better than v1.0... Launched finetuning support... Launched ~zero-shot covariate support." https://pypi.org/project/timesfm/
[^timesfm-25]: MarkTechPost. "TimesFM-2.5 runs with 200M parameters... 16K context length... now tops the leaderboard across accuracy metrics (MASE, CRPS) among zero-shot foundation models." September 2025. https://www.marktechpost.com/2025/09/16/google-ai-ships-timesfm-2-5-smaller-longer-context-foundation-model-that-now-leads-gift-eval-zero-shot-forecasting/

**TimesFM 2.5** — первая foundation model, которая побила AutoTheta на second-level frequency (исторически слабое место FM моделей)[^timesfm-autotheta]

[^timesfm-autotheta]: AI Horizon Forecast. "TimesFM-2.5 became the first foundation model to beat AutoTheta on second-level frequency." October 2025. https://aihorizonforecast.substack.com/p/timesfm-25-hands-on-tutorial-with

## Индикатор частоты

TimesFM использует категориальный индикатор {0, 1, 2}[^timesfm-freq]:

[^timesfm-freq]: HuggingFace: google/timesfm-1.0-200m. "TimesFM expects a categorical indicator valued in {0, 1, 2}." https://huggingface.co/google/timesfm-1.0-200m

| Значение | Частота | Рекомендация |
|----------|---------|--------------|
| 0 | Высокая | До daily (default) |
| 1 | Средняя | Weekly, monthly |
| 2 | Низкая | Quarterly, yearly |

Это не жёсткое ограничение — можно экспериментировать

## Ковариаты через XReg

TimesFM не поддерживает ковариаты нативно, но версии 2.0+ предлагают **XReg** — external regressors через linear model[^timesfm-xreg]:

[^timesfm-xreg]: GitHub: google-research/timesfm. "Added back the covariate support through XReg for TimesFM 2.5." https://github.com/google-research/timesfm

Поддерживаемые типы:
➖ Static Categorical (например, Category)
➖ Static Numerical (например, Base_price)
➖ Dynamic Categorical (например, Weekday, Has_promotion)
➖ Dynamic Numerical (только known future, не past observed)

Механизм: fit linear model на covariates → forecast residuals с TimesFM → combine

## Что умеет

➖ **Zero-shot forecasting** — без обучения на ваших данных
➖ **Произвольный контекст и горизонт** — decoder-only flexibility
➖ **Probabilistic forecasts** (2.0+) — квантили 10th-90th
➖ **Finetuning** (2.0+) — дообучение на своих данных
➖ **Covariates** (2.0+) — через XReg
➖ **BigQuery integration** — `AI.FORECAST` в SQL[^timesfm-sql]
➖ **Открытые веса** — Apache-2.0 license

[^timesfm-sql]: Google Cloud Documentation. "AI.FORECAST... using BigQuery ML's built-in TimesFM model." https://docs.cloud.google.com/bigquery/docs/timesfm-model

## Когда использовать

👍 **Хорошо работает:**
➖ Zero-shot baseline за минуты
➖ Univariate forecasting
➖ Длинный контекст (до 16K в 2.5)
➖ Интеграция с BigQuery/AlloyDB
➖ Нужен открытый код и веса

👎 **Проблемы:**
➖ **Univariate only** — multivariate не поддерживается[^timesfm-univariate]
➖ **Ковариаты ограничены** — только через XReg (linear model)
➖ **Непрерывный контекст** — нет поддержки «дырок» в данных
➖ **Domain bias** — обучен на веб-аналитике
➖ **Apple Silicon не поддерживается** — зависимость lingvo[^timesfm-apple]

[^timesfm-univariate]: Das, A., et al. "TimesFM." ICML 2024. Section 7: "TimesFM is trained for univariate forecasting." https://arxiv.org/abs/2310.10688
[^timesfm-apple]: HuggingFace: google/timesfm-1.0-200m. "The dependency lingvo does not support ARM architectures and the inference code is not working for machines with Apple silicon." https://huggingface.co/google/timesfm-1.0-200m

## TimesFM vs другие модели

| Критерий | TimesFM 2.5 | Chronos-2 | TiRex | TimeGPT |
|----------|-------------|-----------|-------|---------|
| Архитектура | Decoder-only | T5 encoder | sLSTM | Enc-Dec |
| Параметры | 200M | 120M | 35M | Closed |
| Контекст | 16K | 2048 | 2048+ | Unknown |
| Multivariate | ✗ | ✓ | ✗ | ✓ (2.1) |
| Covariates | XReg | ✓ | ✗ | ✓ |
| Открытые веса | ✓ | ✓ | ✓ | ✗ |
| GIFT-Eval | #1 open-source | SOTA | #1 overall | N/A |

## Реализации

| Ресурс | Ссылка |
|--------|--------|
| Официальный код | [google-research/timesfm](https://github.com/google-research/timesfm) |
| TimesFM 2.5 (PyTorch) | [google/timesfm-2.5-200m-pytorch](https://huggingface.co/google/timesfm-2.5-200m-pytorch) |
| TimesFM 2.0 (JAX) | [google/timesfm-2.0-500m-jax](https://huggingface.co/google/timesfm-2.0-500m-jax) |
| BigQuery ML | [AI.FORECAST documentation](https://docs.cloud.google.com/bigquery/docs/timesfm-model) |
| PyPI | [timesfm](https://pypi.org/project/timesfm/) |
| Google Research Blog | [A decoder-only foundation model](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/) |

## Что дальше

TimesFM демонстрирует, что decoder-only архитектура с асимметричным патчингом эффективна для zero-shot forecasting. Модель занимает #1 среди open-source на GIFT-Eval и доступна как официальный продукт Google в BigQuery.

В следующих главах мы рассмотрим другие foundation models и научимся комбинировать их с классическими подходами.

:::{seealso}
**Источники:**
- Das, A., Kong, W., Leblond, R., Sen, R. (2024). [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688). ICML 2024
- [TimesFM GitHub](https://github.com/google-research/timesfm) — Google Research
- [TimesFM HuggingFace Collection](https://huggingface.co/collections/google/timesfm-702f2d66e0591107c6758508)
- [Google Research Blog](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
- [TimesFM in BigQuery](https://cloud.google.com/blog/products/data-analytics/timesfm-models-in-bigquery-and-alloydb)
:::