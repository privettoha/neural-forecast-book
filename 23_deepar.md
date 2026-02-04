# DeepAR: вероятностное прогнозирование

## Ключевая идея

DeepAR[^deepar] — архитектура 2020 года от Amazon, которая решает фундаментально другую задачу: вместо **точечного прогноза** (одно число на каждую точку) модель выдаёт **распределение вероятностей**. Это позволяет оценить uncertainty — насколько модель уверена в своём предсказании.

В отличие от N-BEATS, PatchTST и других encoder-only моделей, DeepAR использует **авторегрессионный подход**: каждое следующее предсказание зависит от предыдущего. Это делает модель медленнее при inference, но позволяет естественно моделировать uncertainty.

**Результаты:** DeepAR стал стандартом для probabilistic forecasting в индустрии. Модель используется в Amazon Forecast, GluonTS и многих production-системах[^deepar].

## Зачем нужно вероятностное прогнозирование

Point forecast (точечный прогноз) — это одно число. Но для бизнес-решений этого недостаточно.

```
┌─────────────────────────────────────────────────────────────────────────┐
│              POINT FORECAST vs PROBABILISTIC FORECAST                   │
│                                                                         │
│  POINT FORECAST:                                                        │
│  «Продажи завтра будут 1000 единиц»                                    │
│                                                                         │
│  Проблема: Насколько уверена модель?                                   │
│  • 1000 ± 10? Можно планировать точно                                  │
│  • 1000 ± 500? Нужен запас                                             │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  PROBABILISTIC FORECAST:                                                │
│  «Продажи завтра: медиана 1000, P10=700, P90=1400»                     │
│                                                                         │
│  • 10% вероятность, что продажи ниже 700                               │
│  • 10% вероятность, что продажи выше 1400                              │
│  • 80% вероятность, что продажи между 700 и 1400                       │
│                                                                         │
│                    ▂▃▄▅▆▇█▇▆▅▄▃▂                                       │
│                 700    1000    1400                                     │
│                  P10   median   P90                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Примеры применения:**

1. **Inventory management:** Заказать товар на P90, чтобы удовлетворить спрос в 90% случаев
2. **Capacity planning:** Планировать мощности на P99 для критичных систем
3. **Risk assessment:** Оценить вероятность экстремальных событий
4. **Decision making:** Разные решения при разных уровнях uncertainty

## Авторегрессионный подход

В отличие от encoder-only моделей (N-BEATS, PatchTST), DeepAR генерирует прогноз **последовательно**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ENCODER-ONLY vs AUTOREGRESSIVE                       │
│                                                                         │
│  ENCODER-ONLY (N-BEATS, PatchTST, DLinear):                            │
│  ─────────────────────────────────────────                              │
│  [x₁, x₂, ..., xₜ] ──→ [Encoder] ──→ [ŷ₁, ŷ₂, ..., ŷₕ]               │
│                                                                         │
│  • Весь прогноз за один forward pass                                   │
│  • Быстрый inference                                                    │
│  • Фиксированный горизонт                                               │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  AUTOREGRESSIVE (DeepAR):                                               │
│  ──────────────────────────                                             │
│  [x₁, x₂, ..., xₜ] ──→ [Encoder] ──→ h                                 │
│                                        │                                │
│                                        ▼                                │
│  ŷ₁ ← sample(h) ──→ [Decoder] ──→ h₁                                   │
│                                    │                                    │
│                                    ▼                                    │
│  ŷ₂ ← sample(h₁) ──→ [Decoder] ──→ h₂                                  │
│                                    │                                    │
│                                    ▼                                    │
│  ŷ₃ ← sample(h₂) ──→ ...                                               │
│                                                                         │
│  • Каждый шаг зависит от предыдущего                                   │
│  • Медленнее, но моделирует uncertainty                                │
│  • Гибкий горизонт                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Почему авторегрессия для uncertainty

При авторегрессионном inference модель **сэмплирует** из предсказанного распределения на каждом шаге. Разные сэмплы дают разные траектории. Множество траекторий образует **prediction intervals**.

```
Траектория 1: [1000, 1050, 1100, 1080, 1120]  (оптимистичный сценарий)
Траектория 2: [1000,  980,  950,  930,  910]  (пессимистичный сценарий)
Траектория 3: [1000, 1020, 1010, 1030, 1025]  (средний сценарий)
...
Траектория N: [...]

P10 = 10-й перцентиль всех траекторий
P50 = медиана (point forecast)
P90 = 90-й перцентиль всех траекторий
```

## Архитектура

DeepAR использует **LSTM** (Long Short-Term Memory) — классическую рекуррентную архитектуру:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        АРХИТЕКТУРА DeepAR                               │
│                                                                         │
│  ОБУЧЕНИЕ (Teacher Forcing):                                            │
│  ─────────────────────────────                                          │
│                                                                         │
│  История: [x₁, x₂, ..., xₜ]    Цель: [y₁, y₂, ..., yₕ]                 │
│                                                                         │
│  ┌─────┐    ┌─────┐    ┌─────┐         ┌─────┐    ┌─────┐              │
│  │LSTM │───→│LSTM │───→│LSTM │───→ ... │LSTM │───→│LSTM │              │
│  └──┬──┘    └──┬──┘    └──┬──┘         └──┬──┘    └──┬──┘              │
│     │          │          │               │          │                  │
│     ↑          ↑          ↑               ↑          ↑                  │
│  [x₁,cov]  [x₂,cov]  [x₃,cov]  ...    [y₁*,cov] [y₂*,cov]             │
│                                           ↓          ↓                  │
│                                        (μ₁,σ₁)   (μ₂,σ₂)               │
│                                                                         │
│  * При обучении используются РЕАЛЬНЫЕ значения (teacher forcing)       │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  INFERENCE (Autoregressive Sampling):                                   │
│  ────────────────────────────────────                                   │
│                                                                         │
│  ┌─────┐    ┌─────┐    ┌─────┐         ┌─────┐    ┌─────┐              │
│  │LSTM │───→│LSTM │───→│LSTM │───→ ... │LSTM │───→│LSTM │              │
│  └──┬──┘    └──┬──┘    └──┬──┘         └──┬──┘    └──┬──┘              │
│     │          │          │               │          │                  │
│     ↑          ↑          ↑               ↑          ↑                  │
│  [x₁,cov]  [x₂,cov]  [x₃,cov]  ...    [ŷ₁,cov]  [ŷ₂,cov]              │
│                                           ↓          ↓                  │
│                                        (μ₁,σ₁) → ŷ₁ ← sample           │
│                                                  │                      │
│                                                  └──→ (μ₂,σ₂) → ŷ₂     │
│                                                                         │
│  * При inference используются СЭМПЛИРОВАННЫЕ значения                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Компоненты

**1. Encoder (LSTM)**

Рекуррентная сеть обрабатывает историю и генерирует скрытое состояние:

$$h_t = \text{LSTM}(h_{t-1}, [x_t, \text{cov}_t])$$

где $\text{cov}_t$ — ковариаты (день недели, праздники, и т.д.).

**2. Output Layer**

Скрытое состояние преобразуется в **параметры распределения**:

$$\mu_t = W_\mu h_t + b_\mu, \quad \sigma_t = \text{softplus}(W_\sigma h_t + b_\sigma)$$

**3. Distribution**

DeepAR поддерживает разные распределения:

| Распределение | Параметры | Когда использовать |
|---------------|-----------|-------------------|
| **Gaussian** | $\mu, \sigma$ | Непрерывные данные (температура, цены) |
| **Negative Binomial** | $\mu, \alpha$ | Счётные данные (продажи, клики) |
| **Student's t** | $\mu, \sigma, \nu$ | Данные с тяжёлыми хвостами |

### Teacher Forcing

При обучении используется **teacher forcing**: вместо сэмплированных значений модель получает **реальные** значения из обучающей выборки.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TEACHER FORCING                                 │
│                                                                         │
│  БЕЗ teacher forcing (autoregressive training):                        │
│  ─────────────────────────────────────────────                          │
│  ŷ₁ = sample(p(y₁|x)) → ŷ₂ = sample(p(y₂|ŷ₁)) → ...                   │
│                                                                         │
│  Проблема: ошибки накапливаются                                        │
│  Если ŷ₁ далеко от y₁, то ŷ₂ будет ещё хуже                            │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  С teacher forcing:                                                     │
│  ───────────────────                                                    │
│  loss₁ = -log p(y₁|x)                                                  │
│  loss₂ = -log p(y₂|y₁*)    ← используем РЕАЛЬНОЕ y₁                    │
│  loss₃ = -log p(y₃|y₂*)    ← используем РЕАЛЬНОЕ y₂                    │
│                                                                         │
│  Преимущество: стабильное обучение                                     │
│  Недостаток: exposure bias (train/test mismatch)                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Loss Function

DeepAR оптимизирует **negative log-likelihood** (NLL):

$$\mathcal{L} = -\sum_{t=1}^{H} \log p(y_t | \mu_t, \sigma_t)$$

Для Gaussian:
$$\mathcal{L} = \sum_{t=1}^{H} \left[ \frac{(y_t - \mu_t)^2}{2\sigma_t^2} + \log \sigma_t \right]$$

**Важно:** Модель учится предсказывать не только среднее ($\mu$), но и **неопределённость** ($\sigma$). Если модель уверена — $\sigma$ будет маленьким. Если не уверена — большим.

## Inference: Monte Carlo Sampling

При inference DeepAR генерирует **множество траекторий** (обычно 100-1000):

```python
def predict_probabilistic(model, history, horizon, n_samples=100):
    trajectories = []

    for _ in range(n_samples):
        h = model.encode(history)
        trajectory = []

        for t in range(horizon):
            mu, sigma = model.decode(h)
            y_t = sample_gaussian(mu, sigma)  # сэмплируем
            trajectory.append(y_t)
            h = model.update_state(h, y_t)    # обновляем состояние

        trajectories.append(trajectory)

    # Вычисляем квантили
    p10 = np.percentile(trajectories, 10, axis=0)
    p50 = np.percentile(trajectories, 50, axis=0)  # медиана
    p90 = np.percentile(trajectories, 90, axis=0)

    return p10, p50, p90
```

**Время inference:** $O(H \times N_{samples})$ forward passes вместо $O(1)$ для encoder-only моделей. При $H=100$ и $N=100$ — это 10,000 forward passes.

## Глобальная модель

DeepAR — **глобальная модель**: одна модель обучается на всех временных рядах одновременно.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ЛОКАЛЬНАЯ vs ГЛОБАЛЬНАЯ МОДЕЛЬ                       │
│                                                                         │
│  ЛОКАЛЬНАЯ (ARIMA, ETS):                                               │
│  ────────────────────────                                               │
│  Ряд 1 → Модель 1 → Прогноз 1                                          │
│  Ряд 2 → Модель 2 → Прогноз 2                                          │
│  Ряд 3 → Модель 3 → Прогноз 3                                          │
│  ...                                                                    │
│  Ряд N → Модель N → Прогноз N                                          │
│                                                                         │
│  • N отдельных моделей                                                 │
│  • Не используют информацию между рядами                               │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  ГЛОБАЛЬНАЯ (DeepAR, N-BEATS):                                         │
│  ──────────────────────────────                                         │
│  Ряд 1 ┐                                                               │
│  Ряд 2 ├──→ Одна общая модель ──→ Прогнозы для всех                    │
│  Ряд 3 ┤                                                               │
│  ...   │                                                               │
│  Ряд N ┘                                                               │
│                                                                         │
│  • Одна модель на все ряды                                             │
│  • Учится на общих паттернах                                           │
│  • Требует меньше данных на ряд                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Преимущества глобальной модели:**
- Может прогнозировать ряды с короткой историей (cold start)
- Переносит паттерны между похожими рядами
- Лучше масштабируется (одна модель vs тысячи)

## Код: DeepAR

### GluonTS (официальная реализация)

```python
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.trainer import Trainer

# Подготовка данных
dataset = PandasDataset.from_long_dataframe(
    df,
    item_id="unique_id",
    timestamp="ds",
    target="y"
)

# Создание модели
estimator = DeepAREstimator(
    freq="D",
    prediction_length=16,
    context_length=64,

    # Архитектура
    num_layers=2,
    hidden_size=40,
    dropout_rate=0.1,

    # Распределение
    distr_output=GaussianOutput(),  # или NegativeBinomialOutput()

    # Обучение
    trainer=Trainer(
        epochs=50,
        learning_rate=1e-3,
        batch_size=32
    )
)

# Обучение
predictor = estimator.train(dataset)

# Прогноз с квантилями
forecasts = predictor.predict(dataset)
for forecast in forecasts:
    print(f"Median: {forecast.median}")
    print(f"P10: {forecast.quantile(0.1)}")
    print(f"P90: {forecast.quantile(0.9)}")
```

### NeuralForecast

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import DeepAR
from neuralforecast.losses.pytorch import DistributionLoss

HORIZON = 16

model = DeepAR(
    h=HORIZON,
    input_size=64,

    # Архитектура LSTM
    lstm_n_layers=2,
    lstm_hidden_size=128,
    lstm_dropout=0.1,

    # Probabilistic output
    loss=DistributionLoss(
        distribution='Normal',  # или 'NegativeBinomial', 'StudentT'
        level=[80, 95]          # prediction intervals
    ),

    # Обучение
    max_steps=1000,
    learning_rate=1e-3,
    batch_size=32,

    random_seed=42
)

nf = NeuralForecast(models=[model], freq='D')
nf.fit(df=train_df)

# Прогноз с интервалами
forecasts = nf.predict()
# Колонки: DeepAR-median, DeepAR-lo-80, DeepAR-hi-80, DeepAR-lo-95, DeepAR-hi-95
```

## Гиперпараметры

| Параметр | Типичные значения | Комментарий |
|----------|-------------------|-------------|
| `context_length` | 2H — 5H | Длина lookback |
| `num_layers` | 2, 3 | Число LSTM слоёв |
| `hidden_size` | 40, 64, 128 | Размер скрытого состояния |
| `dropout_rate` | 0.1, 0.2 | Регуляризация |
| `num_samples` | 100, 200 | Число траекторий при inference |

**Выбор распределения:**
- **Gaussian:** непрерывные данные (температура, цены, метрики)
- **Negative Binomial:** счётные данные с overdispersion (продажи, клики)
- **Student's t:** данные с выбросами

## Сильные стороны

**Probabilistic output.** Главное преимущество — оценка uncertainty. Критично для принятия решений.

**Глобальная модель.** Одна модель для всех рядов. Хорошо работает при cold start.

**Ковариаты.** Нативная поддержка внешних признаков (праздники, промо, погода).

**Гибкий горизонт.** Авторегрессивный подход позволяет генерировать прогноз любой длины.

**Production-ready.** Используется в Amazon Forecast, хорошо задокументирован.

## Ограничения

**Медленный inference.** $O(H \times N_{samples})$ forward passes. При большом $H$ и $N$ — секунды вместо миллисекунд.

**LSTM ограничения.** Сложность с очень длинными зависимостями (>500 точек). Трансформеры справляются лучше.

**Exposure bias.** Расхождение между training (teacher forcing) и inference (sampling).

**Калибровка.** Prediction intervals могут быть плохо калиброваны (P90 покрывает не 90% случаев).

## Когда выбирать DeepAR

✅ **Используйте DeepAR, если:**
- Нужны prediction intervals, а не только point forecasts
- Много рядов с короткой историей (cold start)
- Есть важные ковариаты (праздники, промо)
- Допустим медленный inference

❌ **Рассмотрите альтернативы, если:**
- Нужен только point forecast → N-HiTS, PatchTST (быстрее)
- Очень длинные зависимости → Transformers, foundation models
- Критична скорость inference → encoder-only модели
- Probabilistic + современная архитектура → Lag-Llama (глава 35)

## Выводы

1. **Probabilistic forecasting** критичен для бизнес-решений. Point forecast недостаточен.

2. **Авторегрессионный подход** позволяет естественно моделировать uncertainty через sampling.

3. **Teacher forcing** ускоряет обучение, но создаёт exposure bias.

4. **Глобальная модель** решает проблему cold start и переносит паттерны между рядами.

5. **Trade-off:** богатый probabilistic output за счёт медленного inference.

6. **Наследие:** DeepAR заложил основу для probabilistic forecasting в deep learning. Современные модели (Lag-Llama, Moirai) развивают эти идеи.

В следующей главе рассмотрим foundation models для временных рядов — pretrained модели с zero-shot возможностями.

---

## Ссылки

[^deepar]: Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. *International Journal of Forecasting*, 36(3), 1181-1191. https://arxiv.org/abs/1704.04110

[^gluonts]: Alexandrov, A., et al. (2020). GluonTS: Probabilistic and Neural Time Series Modeling in Python. *JMLR*, 21(116), 1-6. https://arxiv.org/abs/1906.05264

[^nbeats]: Oreshkin, B. N., et al. (2020). N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. *ICLR 2020*. https://arxiv.org/abs/1905.10437
