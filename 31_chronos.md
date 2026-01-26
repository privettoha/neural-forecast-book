# Chronos. Временной ряд как текст

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/privettoha/neural-forecast-book/blob/main/notebooks/31_chronos.ipynb)

Chronos ([статья](https://arxiv.org/abs/2403.07815), TMLR 2024) — семейство моделей от Amazon, которое буквально превращает временной ряд в текст: значения ряда токенизируются как слова, и стандартная языковая модель обучается предсказывать следующие токены. С более чем **600 миллионами скачиваний** на HuggingFace, Chronos стал одним из самых популярных foundation models для временных рядов[^chronos-downloads]

[^chronos-downloads]: Amazon Science Blog. "Chronos and Chronos-Bolt have been collectively downloaded over 600 million times from Hugging Face." https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting

## Идея

Языковые модели научились улавливать сложные зависимости в последовательностях: грамматику, контекст, долгосрочные связи. Chronos задаёт вопрос: если правильно закодировать временной ряд в последовательность токенов, может быть, эти способности перенесутся на прогнозирование?[^chronos-idea]

[^chronos-idea]: Ansari, A.F., et al. "Chronos: Learning the Language of Time Series." TMLR 2024. Section 1: "What are the fundamental differences between a language model that predicts the next token, and a time series forecasting model that predicts the next values?" https://arxiv.org/abs/2403.07815

Amazon выпустил три поколения:

➖ **Chronos (v1)** — оригинальная модель на базе T5 (encoder-decoder), март 2024[^chronos-v1]
➖ **Chronos-Bolt** — оптимизированная версия, **250x быстрее**, ноябрь 2024[^chronos-bolt]
➖ **Chronos-2** — поддержка **multivariate и covariates**, октябрь 2025[^chronos2]

[^chronos-v1]: Ansari, A.F., et al. "Chronos: Learning the Language of Time Series." TMLR 2024. https://arxiv.org/abs/2403.07815
[^chronos-bolt]: AWS Blog. "Fast and accurate zero-shot forecasting with Chronos-Bolt." December 2024. https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/
[^chronos2]: Ansari, A.F., et al. "Chronos-2: From Univariate to Universal Forecasting." arXiv:2510.15821. October 2025. https://arxiv.org/abs/2510.15821

## Токенизация: от чисел к словам

Ключевая инновация Chronos — превращение непрерывных значений в дискретные токены[^chronos-tokenization]:

[^chronos-tokenization]: Ansari, A.F., et al. "Chronos." TMLR 2024. Section 3.1. https://arxiv.org/abs/2403.07815

🔢 **Scaling (масштабирование)**
Каждый ряд нормализуется на его среднее абсолютное значение:

$$\tilde{x}_t = \frac{x_t}{\frac{1}{T}\sum_{i=1}^{T}|x_i| + \epsilon}$$

Это делает модель инвариантной к масштабу: ряд 0–100 и ряд 0–1000000 после нормализации выглядят похоже

🔢 **Quantization (квантизация)**
Нормализованные значения квантуются в один из **4096 бинов**[^chronos-bins]:

[^chronos-bins]: GitHub: amazon-science/chronos-forecasting. "Chronos-T5 models use 4096 different tokens, compared to 32128 of the original T5 models." https://github.com/amazon-science/chronos-forecasting

$$\text{token}(x) = \text{round}\left(\frac{x - x_{\min}}{x_{\max} - x_{\min}} \cdot (B - 1)\right)$$

Каждый бин — один токен в словаре. Временной ряд буквально становится последовательностью «слов»

🔢 **Специальные токены**
➖ PAD — padding/missing values
➖ EOS — end-of-sequence[^chronos-special]

[^chronos-special]: Amazon Science Blog. "In addition to these bin tokens, we add two special tokens, PAD and EOS, to denote padding/missing values and end-of-sequence." https://www.amazon.science/blog/adapting-language-model-architectures-for-time-series-forecasting

## Архитектура Chronos v1: T5

Оригинальный Chronos использует [T5](https://arxiv.org/abs/1910.10683) (Text-to-Text Transfer Transformer) без модификаций архитектуры — только размер словаря уменьшен до 4096[^chronos-t5]:

[^chronos-t5]: Ansari, A.F., et al. "Chronos." TMLR 2024. Section 3.2: "Chronos follows a minimalist approach by tokenizing time series values into a fixed vocabulary and training existing language model architectures on these tokens without any time-series-specific design." https://arxiv.org/abs/2403.07815

```
Encoder:
  Токены истории → Token embeddings + Positional encoding
    → Transformer Encoder (self-attention)
    → Encoded representation

Decoder (авторегрессионный):
  Для каждого шага прогноза:
    Encoded history + Ранее сгенерированные токены
      → Masked self-attention
      → Cross-attention на encoder output
      → Softmax → распределение над словарём
      → Sample токен
```

Вероятностный прогноз через многократное сэмплирование траекторий

🔢 **Размеры моделей v1:**

| Модель | Параметры | Контекст |
|--------|-----------|----------|
| chronos-t5-tiny | 8M | 512 |
| chronos-t5-mini | 20M | 512 |
| chronos-t5-small | 46M | 512 |
| chronos-t5-base | 200M | 512 |
| chronos-t5-large | 710M | 512 |

## Chronos-Bolt: 250x быстрее

Chronos-Bolt (ноябрь 2024) переосмысливает архитектуру для скорости[^bolt-arch]:

[^bolt-arch]: HuggingFace: amazon/chronos-bolt-base. "Chronos-Bolt is based on the T5 encoder-decoder architecture and has been trained on nearly 100 billion time series observations." https://huggingface.co/amazon/chronos-bolt-base

🔢 **Patching вместо токенизации**
История разбивается на патчи из нескольких наблюдений, которые подаются в encoder

🔢 **Direct multi-step forecasting**
Вместо авторегрессионной генерации токен-за-токеном, decoder выдаёт **весь горизонт за один проход** — квантили для всех шагов сразу[^bolt-direct]

[^bolt-direct]: AWS Blog. "The decoder then uses these representations to directly generate quantile forecasts across multiple future steps—a method known as direct multi-step forecasting." https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/

🔢 **Результаты:**
➖ **В 250 раз быстрее** на GPU (в 20 раз на CPU)[^bolt-speed]
➖ **В 20 раз меньше памяти**
➖ **На 5% точнее** (меньше WQL)
➖ Контекст до **2048** (vs 512 у v1)
➖ Bolt-Base превосходит v1-Large, будучи в 600 раз быстрее[^bolt-vs-large]

[^bolt-speed]: GitHub: amazon-science/chronos-forecasting. "Chronos-Bolt models are more accurate (5% lower error), up to 250x faster and 20x more memory efficient." https://github.com/amazon-science/chronos-forecasting
[^bolt-vs-large]: HuggingFace: amazon/chronos-bolt-base. "Chronos-Bolt (Base) surpasses the original Chronos (Large) model while being over 600 times faster." https://huggingface.co/amazon/chronos-bolt-base

🔢 **Размеры Bolt:**

| Модель | Параметры | Контекст |
|--------|-----------|----------|
| chronos-bolt-tiny | 9M | 2048 |
| chronos-bolt-mini | 21M | 2048 |
| chronos-bolt-small | 48M | 2048 |
| chronos-bolt-base | 205M | 2048 |

## Chronos-2: Multivariate и Covariates

Chronos-2 (октябрь 2025) — ключевое обновление, которое решает главное ограничение предыдущих версий[^chronos2-paper]:

[^chronos2-paper]: Ansari, A.F., et al. "Chronos-2: From Univariate to Universal Forecasting." arXiv:2510.15821. October 2025. https://arxiv.org/abs/2510.15821

🔢 **Новые возможности:**
➖ **Univariate** — как раньше
➖ **Multivariate** — несколько связанных рядов одновременно
➖ **Covariates** — внешние переменные (past-only и known future)

🔢 **Архитектура:**
➖ **Encoder-only** (120M параметров) на основе T5 encoder
➖ **Group attention** — механизм для обмена информацией между группами рядов (variates, covariates)[^chronos2-group]
➖ Multi-step quantile forecasts за один проход

[^chronos2-group]: Ansari, A.F., et al. "Chronos-2." arXiv:2510.15821. Abstract: "Chronos-2 employs a group attention mechanism that facilitates in-context learning through efficient information sharing across multiple time series within a group." https://arxiv.org/abs/2510.15821

🔢 **Результаты:**
➖ **SOTA** на fev-bench, GIFT-Eval, Chronos Benchmark II
➖ **Win rate >90%** против Chronos-Bolt в head-to-head сравнениях[^chronos2-winrate]
➖ **300+ прогнозов в секунду** на одном A10G GPU

[^chronos2-winrate]: HuggingFace: amazon/chronos-2. "Chronos-2 achieves a win rate of over 90% against Chronos-Bolt in head-to-head comparisons." https://huggingface.co/amazon/chronos-2

🔢 **Синтетические данные:**
Для обучения multivariate capabilities используются синтетические данные, где multivariate структура накладывается на univariate ряды[^chronos2-synthetic]

[^chronos2-synthetic]: Ansari, A.F., et al. "Chronos-2." arXiv:2510.15821. Section 3: "To enable its ICL capabilities, we rely on synthetic time series data generated by imposing multivariate structure on time series sampled from base univariate generators." https://arxiv.org/abs/2510.15821

## Что умеет

➖ **Universal tokenization** — работает на данных любого масштаба и частоты
➖ **Zero-shot** из коробки — не нужно обучение на своих данных
➖ **Вероятностный выход** — сэмплы (v1) или квантили (Bolt, v2)
➖ **Covariates** (только Chronos-2) — внешние факторы
➖ **Multivariate** (только Chronos-2) — связанные ряды
➖ **Открытые веса и код** — Apache-2.0 license
➖ **CPU inference** — особенно Bolt-tiny[^chronos-cpu]

[^chronos-cpu]: GitHub: amazon-science/chronos-forecasting. "device_map='cpu' for CPU inference." https://github.com/amazon-science/chronos-forecasting

## Когда использовать

👍 **Хорошо работает:**
➖ Zero-shot прогноз без обучения
➖ Нужны ковариаты → **Chronos-2**
➖ Multivariate данные → **Chronos-2**
➖ Продакшн, нужна скорость → **Chronos-Bolt**
➖ CPU inference → **Bolt-tiny**
➖ Произвольные квантили (0.95) → **v1** (сэмплирование)

👎 **Проблемы:**
➖ **Потеря точности при квантизации** — discretization error
➖ **Чувствительность к выбросам** — влияют на scaling
➖ **v1 медленный** — авторегрессионная генерация
➖ **Bolt/v2: фиксированные квантили** — нельзя запросить произвольный

## Какую версию выбрать?

| Сценарий | Рекомендация |
|----------|--------------|
| Есть ковариаты | **Chronos-2** |
| Multivariate данные | **Chronos-2** |
| Нужна скорость, univariate | **Chronos-Bolt** |
| CPU inference | **Bolt-tiny** |
| Произвольные квантили | v1 (сэмплирование) |
| Только начинаете | **Chronos-2** |

## Chronos vs другие модели

| Критерий | Chronos-2 | Chronos-Bolt | Chronos v1 | TiRex | FlowState |
|----------|-----------|--------------|------------|-------|-----------|
| Архитектура | T5 encoder | T5 enc-dec | T5 enc-dec | sLSTM | SSM+FBD |
| Параметры | 120M | 9M–205M | 8M–710M | 35M | 9.1M |
| Multivariate | ✓ | ✗ | ✗ | ✗ | ✗ |
| Covariates | ✓ | ✗* | ✗ | ✗ | ✗ |
| Скорость | Быстро | Очень быстро | Медленно | Средне | Быстро |
| GIFT-Eval | SOTA | Top-5 | Top-10 | #1 | #2 |

*Bolt можно комбинировать с external covariate regressors через AutoGluon

## Реализации

| Ресурс | Ссылка |
|--------|--------|
| Официальный код | [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) |
| Chronos-2 | [amazon/chronos-2](https://huggingface.co/amazon/chronos-2) |
| Chronos-Bolt | [amazon/chronos-bolt-base](https://huggingface.co/amazon/chronos-bolt-base) |
| Chronos v1 | [amazon/chronos-t5-large](https://huggingface.co/amazon/chronos-t5-large) |
| AutoGluon интеграция | [AutoGluon-TimeSeries](https://auto.gluon.ai/stable/tutorials/timeseries/) |
| AWS SageMaker | [JumpStart tutorial](https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/deploy-chronos-bolt-to-amazon-sagemaker.ipynb) |

## Что дальше

Chronos показал, что идея «ряд как текст» работает. Квантизация + трансформер + cross-entropy loss — это работающий пайплайн, который масштабируется до multivariate и covariates.

В следующем посте мы рассмотрим TimeGPT — закрытую модель от Nixtla, доступную только через API. Это другой подход к foundation models: вместо открытых весов — сервис с гарантированным качеством.

:::{seealso}
**Источники:**
- Ansari, A.F., et al. (2024). [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815). Transactions on Machine Learning Research (TMLR)
- Ansari, A.F., et al. (2025). [Chronos-2: From Univariate to Universal Forecasting](https://arxiv.org/abs/2510.15821). arXiv
- [Официальный код Chronos](https://github.com/amazon-science/chronos-forecasting) — Amazon Science GitHub
- [AWS Blog: Chronos-Bolt](https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/)
- [Amazon Science Blog: Chronos-2](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting)
:::