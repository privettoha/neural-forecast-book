# TimeGPT: foundation model как сервис

## Ключевая идея

TimeGPT[^timegpt] — первая **коммерческая** foundation model для временных рядов, выпущенная компанией Nixtla в 2023 году. В отличие от Chronos и других открытых моделей, TimeGPT доступен только через **API** — веса модели не публикуются.

Главное преимущество TimeGPT — **простота использования**. Не нужно разбираться с GPU, устанавливать зависимости, выбирать гиперпараметры. Один API-вызов — и прогноз готов. Для бизнес-пользователей это часто важнее, чем возможность fine-tuning.

**Результаты:** TimeGPT показывает конкурентоспособные zero-shot результаты на стандартных бенчмарках. На момент выхода (октябрь 2023) это была первая публичная foundation model для временных рядов[^timegpt].

## Почему API, а не открытая модель?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ОТКРЫТАЯ МОДЕЛЬ vs API                               │
│                                                                         │
│  ОТКРЫТАЯ МОДЕЛЬ (Chronos, Lag-Llama):                                 │
│  ─────────────────────────────────────                                  │
│  + Полный контроль над inference                                        │
│  + Fine-tuning на своих данных                                         │
│  + Работает offline                                                     │
│  + Нет зависимости от провайдера                                       │
│  - Нужен GPU (для больших моделей)                                     │
│  - Нужно разбираться с deployment                                       │
│  - Нужно следить за обновлениями                                       │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  API (TimeGPT):                                                         │
│  ──────────────                                                         │
│  + Работает сразу, без setup                                           │
│  + Автоматические обновления модели                                     │
│  + Не нужен GPU                                                         │
│  + Поддержка и SLA                                                      │
│  - Данные отправляются на сервер                                       │
│  - Зависимость от провайдера                                           │
│  - Стоимость за inference                                              │
│  - Нет fine-tuning (только через API)                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Позиционирование Nixtla:** TimeGPT — это «GPT для временных рядов». Как OpenAI API упростил доступ к LLM, так TimeGPT упрощает доступ к foundation models для прогнозирования.

## Архитектура

Детали архитектуры TimeGPT **не раскрываются полностью**. Из публикации известно:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ИЗВЕСТНОЕ ОБ АРХИТЕКТУРЕ TimeGPT                     │
│                                                                         │
│  • Transformer-based (encoder-decoder или decoder-only — не уточняется)│
│  • Обучен на >100 миллиардов точек данных                              │
│  • Данные из открытых источников (энергетика, финансы, метео, ритейл)  │
│  • Поддерживает разные частоты (от минут до лет)                       │
│  • Conformal prediction для uncertainty quantification                  │
│                                                                         │
│  Что НЕ известно:                                                       │
│  • Точная архитектура (число слоёв, размерности)                       │
│  • Токенизация (непрерывная или дискретная)                            │
│  • Детали pre-training (loss function, augmentations)                  │
│  • Число параметров                                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Conformal Prediction

TimeGPT использует **conformal prediction** для оценки uncertainty — это статистический метод, гарантирующий калиброванные prediction intervals.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CONFORMAL PREDICTION                                 │
│                                                                         │
│  Идея: использовать ошибки на calibration set для оценки интервалов   │
│                                                                         │
│  1. Делаем прогноз на calibration данных                               │
│  2. Вычисляем residuals: r_i = |y_i - ŷ_i|                             │
│  3. Находим квантиль residuals для нужного уровня доверия              │
│  4. Применяем этот квантиль к новым прогнозам                          │
│                                                                         │
│  Пример (90% интервал):                                                │
│  • Residuals: [0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.9, ...]                 │
│  • 90-й перцентиль: 1.8                                                │
│  • Интервал для нового прогноза: [ŷ - 1.8, ŷ + 1.8]                   │
│                                                                         │
│  Гарантия: интервал покрывает истинное значение в ≥90% случаев        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Преимущество:** В отличие от Monte Carlo sampling (DeepAR, Chronos), conformal prediction даёт **статистические гарантии** на coverage.

## Возможности API

TimeGPT API предоставляет несколько режимов работы:

### 1. Zero-shot прогноз

```python
from nixtla import NixtlaClient

client = NixtlaClient(api_key="your_api_key")

forecast = client.forecast(
    df=df,
    h=24,  # горизонт
    freq='H',
    time_col='ds',
    target_col='y'
)
```

### 2. Прогноз с ковариатами

```python
forecast = client.forecast(
    df=df,
    h=24,
    freq='H',
    time_col='ds',
    target_col='y',
    X_df=future_exog,  # будущие значения ковариат
)
```

### 3. Anomaly detection

```python
anomalies = client.detect_anomalies(
    df=df,
    time_col='ds',
    target_col='y',
    freq='H'
)
```

### 4. Cross-validation

```python
cv_results = client.cross_validation(
    df=df,
    h=24,
    freq='H',
    n_windows=5  # число fold'ов
)
```

## Код: TimeGPT

### Установка

```bash
pip install nixtla
```

### Базовый пример

```python
import pandas as pd
from nixtla import NixtlaClient

# Инициализация клиента
client = NixtlaClient(api_key="YOUR_API_KEY")

# Загрузка данных
df = pd.read_csv("sales.csv")
# Формат: ds (datetime), y (target), unique_id (optional)

# Zero-shot прогноз
forecast = client.forecast(
    df=df,
    h=24,           # горизонт
    freq='H',       # частота
    time_col='ds',
    target_col='y',
    level=[80, 95]  # уровни доверия для интервалов
)

# Результат содержит:
# - TimeGPT: point forecast
# - TimeGPT-lo-80, TimeGPT-hi-80: 80% интервал
# - TimeGPT-lo-95, TimeGPT-hi-95: 95% интервал
```

### С ковариатами

```python
# Исторические данные с ковариатами
df = pd.DataFrame({
    'ds': dates,
    'y': sales,
    'unique_id': 'product_1',
    'price': prices,
    'is_holiday': holidays
})

# Будущие значения ковариат (известны заранее)
future_df = pd.DataFrame({
    'ds': future_dates,
    'unique_id': 'product_1',
    'price': future_prices,
    'is_holiday': future_holidays
})

forecast = client.forecast(
    df=df,
    X_df=future_df,  # ковариаты
    h=24,
    freq='H',
    time_col='ds',
    target_col='y'
)
```

### Fine-tuning через API

```python
# TimeGPT поддерживает fine-tuning через API
forecast = client.forecast(
    df=df,
    h=24,
    freq='H',
    time_col='ds',
    target_col='y',
    finetune_steps=100,      # число шагов fine-tuning
    finetune_loss='mae'      # loss function
)
```

## Сравнение с открытыми моделями

| Аспект | TimeGPT | Chronos | Lag-Llama |
|--------|---------|---------|-----------|
| **Доступ** | API only | Открытая | Открытая |
| **Веса** | Закрытые | Публичные | Публичные |
| **Fine-tuning** | Через API | Локальный | Локальный |
| **GPU нужен** | Нет | Да (для больших) | Да |
| **Ковариаты** | Да | Нет | Нет |
| **Anomaly detection** | Да | Нет | Нет |
| **Стоимость** | Pay-per-use | Бесплатно | Бесплатно |
| **Privacy** | Данные на сервере | Локально | Локально |

## Ценообразование

TimeGPT использует модель pay-per-use:

| Tier | Цена | Включено |
|------|------|----------|
| Free | $0 | 1,000 calls/month |
| Pro | $99/month | 100,000 calls/month |
| Enterprise | Custom | Unlimited + SLA |

**Примечание:** Цены могут меняться. Актуальную информацию см. на nixtla.io.

## Когда выбирать TimeGPT

### ✅ Хороший выбор

**Быстрый старт.** Если нужен прогноз за 5 минут — TimeGPT работает из коробки.

**Нет GPU.** API работает без локальных вычислительных ресурсов.

**Бизнес-пользователи.** Аналитики без ML-экспертизы могут использовать сразу.

**Anomaly detection.** Встроенная функциональность из коробки.

**Ковариаты важны.** TimeGPT нативно поддерживает exogenous переменные.

### ❌ Плохой выбор

**Privacy-sensitive данные.** Медицинские, финансовые данные — отправка на сервер может быть неприемлема.

**Высокий объём.** При миллионах прогнозов стоимость может быть значительной.

**Нужен fine-tuning.** Fine-tuning через API ограничен по сравнению с локальным.

**Offline-сценарии.** Edge deployment, air-gapped системы — требуется интернет.

**Исследования.** Нельзя экспериментировать с архитектурой.

## Интеграция с экосистемой Nixtla

TimeGPT интегрируется с другими инструментами Nixtla:

```python
# StatsForecast для statistical models
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, ETS

sf = StatsForecast(models=[AutoARIMA(), ETS()], freq='H')
sf_forecast = sf.forecast(df, h=24)

# NeuralForecast для neural models
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

nf = NeuralForecast(models=[NHITS(h=24)], freq='H')
nf_forecast = nf.predict(df)

# TimeGPT для foundation model
from nixtla import NixtlaClient
client = NixtlaClient()
tg_forecast = client.forecast(df, h=24)

# Сравнение всех моделей
```

## Ограничения

**Закрытая модель.** Нельзя изучить архитектуру, воспроизвести результаты.

**Зависимость от провайдера.** Если Nixtla изменит API или цены — нужно адаптироваться.

**Latency.** Сетевой round-trip добавляет задержку (~100-500ms).

**Rate limits.** Ограничения на количество запросов в зависимости от тарифа.

## Выводы

1. **TimeGPT — первая коммерческая foundation model** для временных рядов. API-first подход для простоты использования.

2. **Conformal prediction** обеспечивает калиброванные prediction intervals с гарантиями.

3. **Ковариаты из коробки** — преимущество перед Chronos и Lag-Llama.

4. **Trade-off:** Простота vs контроль. Для быстрого прототипа — отлично. Для production с privacy требованиями — нужны альтернативы.

5. **Позиционирование:** «OpenAI API для временных рядов». Удобно для бизнес-пользователей.

В следующей главе рассмотрим **TimesFM** — открытую foundation model от Google с decoder-only архитектурой.

---

## Ссылки

[^timegpt]: Garza, A., & Mergenthaler-Canseco, M. (2023). TimeGPT-1. *arXiv preprint*. https://arxiv.org/abs/2310.03589

[^nixtla]: Nixtla. TimeGPT Documentation. https://docs.nixtla.io/

[^conformal]: Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World. *Springer*.
