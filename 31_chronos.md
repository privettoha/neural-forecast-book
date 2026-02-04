# Chronos: временной ряд как текст

## Ключевая идея

Chronos[^chronos] — foundation model от Amazon (2024), которая превращает **временной ряд в последовательность токенов** и использует language model архитектуру (T5) для прогнозирования. Идея проста: если LLM умеют продолжать текст, почему бы не научить их продолжать числовые последовательности?

Вместо того чтобы предсказывать непрерывные значения напрямую, Chronos **квантует** числа в дискретные токены (как BPE токенизация для текста). После этого задача прогнозирования становится задачей **language modeling**: предсказать следующий токен по контексту.

**Результаты:** Chronos показывает лучшие zero-shot результаты среди открытых моделей. На 27 бенчмарках (не входивших в обучение) Chronos превосходит task-specific модели в **87%** случаев[^chronos].

## Зачем квантование?

Почему нельзя просто подать числа в трансформер?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ПРОБЛЕМА НЕПРЕРЫВНЫХ ЗНАЧЕНИЙ                        │
│                                                                         │
│  ПОДХОД 1: Регрессия (MSE loss)                                        │
│  ─────────────────────────────────                                      │
│  [100, 105, 110] → Transformer → ŷ ∈ ℝ                                 │
│                                                                         │
│  Проблемы:                                                              │
│  • Не моделирует uncertainty (один point estimate)                     │
│  • MSE плохо работает для multimodal distributions                     │
│  • Трудно использовать pre-trained LLM архитектуры                     │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  ПОДХОД 2: Токенизация (cross-entropy loss)                            │
│  ──────────────────────────────────────────                             │
│  [100, 105, 110] → [tok_500, tok_525, tok_550] → Transformer → p(tok)  │
│                                                                         │
│  Преимущества:                                                          │
│  • Cross-entropy — хорошо изученный loss для LLM                       │
│  • Можно использовать T5, GPT, LLaMA архитектуры напрямую              │
│  • Автоматически моделирует uncertainty (распределение по токенам)     │
│  • Probabilistic output из коробки (сэмплирование из p(tok))           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Архитектура Chronos

Chronos состоит из трёх компонентов: **токенизация**, **T5 backbone**, **детокенизация**.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        АРХИТЕКТУРА CHRONOS                              │
│                                                                         │
│  Вход: x = [100.5, 102.3, 105.1, 103.8]                                │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  1. SCALING + TOKENIZATION                                      │   │
│  │                                                                  │   │
│  │  Mean = 102.9, Std = 1.7                                        │   │
│  │  Normalized: [-1.4, -0.4, 1.3, 0.5]                             │   │
│  │  Quantized:  [token_23, token_48, token_82, token_62]           │   │
│  │              (из словаря 4096 токенов)                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  2. T5 ENCODER-DECODER                                          │   │
│  │                                                                  │   │
│  │  Encoder: [tok_23, tok_48, tok_82, tok_62] → context            │   │
│  │                                                                  │   │
│  │  Decoder: [BOS] → p(tok_1|ctx) → tok_73                         │   │
│  │           [BOS, tok_73] → p(tok_2|ctx) → tok_81                 │   │
│  │           ...                                                    │   │
│  │                                                                  │   │
│  │  Autoregressive generation                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  3. DE-TOKENIZATION + UNSCALING                                 │   │
│  │                                                                  │   │
│  │  [tok_73, tok_81, ...] → [0.8, 1.5, ...] (normalized)          │   │
│  │  × Std + Mean → [104.3, 105.5, ...] (original scale)           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Выход: ŷ = [104.3, 105.5, ...]                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1. Scaling

Перед токенизацией значения **нормализуются** для каждого ряда независимо:

$$x'_t = \frac{x_t - \text{mean}(x)}{\text{std}(x) + \epsilon}$$

Это критически важно: без нормализации ряд с температурой (20-30) и ряд с продажами (1000-10000) попали бы в разные области словаря.

### 2. Tokenization

Chronos использует **quantile-based tokenization**:

1. Определяем бины на основе квантилей стандартного нормального распределения
2. Каждый бин соответствует одному токену
3. Значение попадает в токен по принадлежности к бину

```python
# Упрощённая версия токенизации
def tokenize(values, n_tokens=4096):
    # Бины на основе квантилей N(0,1)
    quantiles = np.linspace(0.0001, 0.9999, n_tokens + 1)
    bin_edges = scipy.stats.norm.ppf(quantiles)

    # Каждое значение → номер бина (токен)
    tokens = np.digitize(values, bin_edges)
    return tokens
```

**Размер словаря:** По умолчанию 4096 токенов. Это обеспечивает точность квантования ~0.05% от диапазона.

### 3. T5 Backbone

Chronos использует **T5** (Text-to-Text Transfer Transformer)[^t5] — encoder-decoder архитектуру:

- **Encoder:** Обрабатывает историю (токенизированный контекст)
- **Decoder:** Автрегрессивно генерирует прогноз

```
Encoder input:  [tok_23, tok_48, tok_82, tok_62]
                     ↓
                 [context]
                     ↓
Decoder output: [tok_73, tok_81, tok_77, ...]  (autoregressive)
```

**Почему T5, а не decoder-only (GPT)?** Авторы экспериментировали с обоими вариантами. T5 показал лучшие результаты на коротких контекстах, типичных для временных рядов[^chronos].

### 4. De-tokenization

Обратное преобразование: токен → число.

```python
def detokenize(tokens, n_tokens=4096):
    # Центр каждого бина
    quantiles = np.linspace(0.0001, 0.9999, n_tokens)
    bin_centers = scipy.stats.norm.ppf(quantiles)

    # Токен → центр соответствующего бина
    values = bin_centers[tokens]
    return values
```

## Pre-training

Chronos обучается на **смеси реальных и синтетических данных**:

### Реальные данные

- **Источники:** Monash Time Series Repository, LibCity traffic, Electric, M1-M5 competitions
- **Объём:** ~3 миллиона уникальных рядов
- **Домены:** Ритейл, энергетика, финансы, метео, трафик

### Синтетические данные (TSMix)

Chronos использует **data augmentation** через генерацию синтетических рядов:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TSMix: СИНТЕТИЧЕСКИЕ ДАННЫЕ                     │
│                                                                         │
│  1. Gaussian Processes с разными kernel'ами:                           │
│     • RBF (гладкие ряды)                                               │
│     • Matern (более шумные)                                            │
│     • Periodic (сезонность)                                            │
│                                                                         │
│  2. Комбинации компонент:                                              │
│     trend + seasonality + noise                                        │
│                                                                         │
│  3. Аугментации реальных рядов:                                        │
│     • Scaling, shifting                                                 │
│     • Adding noise                                                      │
│     • Concatenation                                                     │
│                                                                         │
│  Результат: +10% качества на zero-shot бенчмарках                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Loss function

Стандартный **cross-entropy** для language modeling:

$$\mathcal{L} = -\sum_{t=1}^{H} \log p(y_t | y_{<t}, x)$$

где $y_t$ — токен прогноза, $x$ — токенизированная история.

## Probabilistic Forecasting

Chronos генерирует **probabilistic forecasts** через сэмплирование:

```python
def predict_probabilistic(model, history, horizon, n_samples=20):
    trajectories = []

    for _ in range(n_samples):
        # Сэмплируем из p(y|x) на каждом шаге
        trajectory = model.generate(
            context=history,
            length=horizon,
            temperature=1.0,  # стохастическое сэмплирование
            top_k=50
        )
        trajectories.append(trajectory)

    # Квантили из сэмплов
    p10 = np.percentile(trajectories, 10, axis=0)
    p50 = np.percentile(trajectories, 50, axis=0)  # медиана
    p90 = np.percentile(trajectories, 90, axis=0)

    return p10, p50, p90
```

**Temperature:** Контролирует стохастичность сэмплирования.
- `temperature=0` → greedy decoding (point forecast)
- `temperature=1` → сэмплирование из полного распределения

## Размеры моделей

Chronos доступен в 5 размерах:

| Модель | Параметры | Context | GPU память |
|--------|-----------|---------|------------|
| Chronos-T5-Tiny | 8M | 512 | ~1 GB |
| Chronos-T5-Mini | 20M | 512 | ~2 GB |
| Chronos-T5-Small | 46M | 512 | ~4 GB |
| Chronos-T5-Base | 200M | 512 | ~8 GB |
| Chronos-T5-Large | 710M | 512 | ~16 GB |

**Рекомендация:** Начните с `chronos-t5-small` — хороший баланс качества и скорости. Для production с GPU — `chronos-t5-base`.

## Код: Chronos

### Установка

```bash
pip install chronos-forecasting
```

### Zero-shot прогноз

```python
import torch
from chronos import ChronosPipeline
import pandas as pd

# Загрузка модели
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16  # для экономии памяти
)

# Подготовка данных
context = torch.tensor(df["y"].values)

# Прогноз
forecast = pipeline.predict(
    context,
    prediction_length=24,  # горизонт
    num_samples=20         # число сэмплов для probabilistic
)

# forecast.shape: [num_samples, prediction_length]
median = forecast.median(dim=0).values
p10 = forecast.quantile(0.1, dim=0).values
p90 = forecast.quantile(0.9, dim=0).values
```

### С NeuralForecast

```python
from neuralforecast import NeuralForecast
from neuralforecast.models.chronos import Chronos

model = Chronos(
    h=24,
    input_size=512,
    model_path="amazon/chronos-t5-small",
    num_samples=20
)

nf = NeuralForecast(models=[model], freq='H')
# Для zero-shot не нужен fit()
forecasts = nf.predict(df)
```

### Fine-tuning

```python
from chronos.training import ChronosTrainer

trainer = ChronosTrainer(
    model_name="amazon/chronos-t5-small",
    output_dir="./finetuned-chronos",
    learning_rate=1e-5,
    num_train_epochs=3,
    per_device_train_batch_size=32
)

trainer.train(train_dataset)
```

## Результаты

### Zero-shot бенчмарки

На 27 датасетах, **не входивших** в обучение:

| Модель | Probabilistic (WQL) | Point (MASE) |
|--------|---------------------|--------------|
| Seasonal Naive | 0.427 | 1.000 |
| DeepAR | 0.389 | 0.891 |
| PatchTST | — | 0.878 |
| **Chronos-Small** | **0.347** | **0.815** |
| **Chronos-Base** | **0.327** | **0.789** |
| **Chronos-Large** | **0.315** | **0.776** |

Chronos превосходит как statistical baselines, так и task-specific модели **без обучения на целевых данных**.

### In-domain vs Zero-shot

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    IN-DOMAIN vs ZERO-SHOT                               │
│                                                                         │
│  In-domain (датасеты из обучения):                                      │
│  ────────────────────────────────                                       │
│  Chronos ≈ Task-specific (PatchTST, DeepAR)                            │
│  Вывод: На знакомых данных разницы мало                                │
│                                                                         │
│  Zero-shot (новые датасеты):                                            │
│  ───────────────────────────                                            │
│  Chronos >> Task-specific                                              │
│  Вывод: Chronos обобщается лучше                                       │
│                                                                         │
│  Практика: Если ваши данные похожи на M4/ETT — task-specific ок.       │
│            Если новый домен — Chronos выигрывает.                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Ограничения

**Квантование снижает точность.** 4096 токенов — это ~12 бит точности. Для большинства задач достаточно, но для научных данных с высокой точностью — может быть проблемой.

**Медленный inference.** Autoregressive generation: $O(H)$ forward passes. Для $H=720$ — медленно.

**Фиксированный context length.** 512 токенов максимум. Для длинных рядов — ограничение.

**GPU желателен.** Модель 200M+ параметров. На CPU работает, но медленно.

## Когда выбирать Chronos

✅ **Используйте Chronos, если:**
- Нужен zero-shot прогноз без обучения
- Мало данных для task-specific модели
- Данные из нового домена
- Нужен probabilistic output

❌ **Рассмотрите альтернативы, если:**
- Много однородных данных → task-specific (N-HiTS, PatchTST)
- Критична скорость inference → DLinear, N-HiTS
- Multivariate важен → Moirai
- Нет GPU → TimesFM (оптимизирован для CPU)

## Выводы

1. **Токенизация** позволяет использовать language model архитектуры для временных рядов.

2. **T5 backbone** + cross-entropy loss — проверенная комбинация из NLP.

3. **Синтетические данные (TSMix)** критичны для zero-shot обобщения.

4. **Probabilistic output** из коробки через сэмплирование.

5. **Практика:** Chronos — отличный default для zero-shot. Начните с `chronos-t5-small`, масштабируйте если нужно.

В следующей главе рассмотрим **TimeGPT** — первую коммерческую foundation model для временных рядов.

---

## Ссылки

[^chronos]: Ansari, A., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., ... & Wang, B. (2024). Chronos: Learning the Language of Time Series. *arXiv preprint*. https://arxiv.org/abs/2403.07815

[^t5]: Raffel, C., et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. *JMLR*. https://arxiv.org/abs/1910.10683

[^monash]: Godahewa, R., et al. (2021). Monash Time Series Forecasting Archive. *NeurIPS Datasets and Benchmarks*. https://forecastingdata.org/
