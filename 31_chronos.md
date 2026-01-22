# Chronos. Временной ряд как текст

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/privettoha/neural-forecast-book/blob/main/notebooks/31_chronos.ipynb)

## Ключевая идея

Chronos задаёт провокационный вопрос: а что если временной ряд — это просто текст? Не метафорически, а буквально: возьмём значения ряда, превратим их в токены (как слова), и применим языковую модель для генерации следующих токенов (прогноза).

Это звучит как хак, но за ним стоит глубокая интуиция. Языковые модели научились улавливать сложные зависимости в последовательностях: грамматику, контекст, долгосрочные связи. Если правильно закодировать временной ряд в последовательность токенов, может быть, эти же способности перенесутся на прогнозирование?

Amazon выпустил две версии Chronos:

- **Chronos (v1)** — оригинальная модель на базе [T5](https://arxiv.org/abs/1910.10683) (encoder-decoder), март 2024
- **Chronos Bolt (v2)** — оптимизированная версия на базе T5-Efficient, декабрь 2024

Chronos Bolt — не просто инкрементальное улучшение. Это переработанная архитектура, которая в 250 раз быстрее оригинала при сопоставимом или лучшем качестве. Если вы начинаете работу с Chronos сегодня — начинайте с Bolt.

## Две версии: что изменилось

### Chronos v1 (оригинал)

Оригинальный Chronos использует классическую [T5-архитектуру](https://arxiv.org/abs/1910.10683):

- **Encoder-decoder** трансформер
- **Авторегрессионная генерация** — токен за токеном
- **Сэмплирование** для вероятностного прогноза
- Размеры: Tiny (8M), Mini (20M), Small (46M), Base (200M), Large (710M)

Главный недостаток — скорость. Авторегрессионная генерация означает, что для прогноза на H шагов нужно H последовательных вызовов decoder. Это медленно, особенно для длинных горизонтов.

### Chronos Bolt (v2)

Bolt переосмысливает архитектуру:

- **T5-Efficient** с оптимизированными attention-паттернами
- **Direct multi-step forecasting** — весь горизонт за один проход
- **Детерминированный выход квантилей** вместо сэмплирования
- Размеры: Tiny (9M), Mini (21M), Small (48M), Base (205M)

Ключевые улучшения:

- **В 250 раз быстрее** на GPU (в 20 раз на CPU)
- **Лучше качество** на бенчмарках — #1 на Benchmark I и II в zero-shot
- **Проще использовать** — не нужно сэмплирование для квантилей

## Токенизация: от чисел к словам

Обе версии используют одинаковый подход к токенизации — это ядро идеи Chronos.

### Scaling (масштабирование)

Каждый ряд нормализуется на его среднее абсолютное значение в контексте:

$$\tilde{x}_t = \frac{x_t}{\frac{1}{T}\sum_{i=1}^{T}|x_i| + \epsilon}$$

Это делает модель инвариантной к масштабу: ряд с значениями 0–100 и ряд с значениями 0–1000000 после нормализации выглядят похоже.

### Quantization (квантизация)

После нормализации значения квантуются в один из $B$ бинов. Chronos использует $B = 4096$ бинов, равномерно распределённых в диапазоне, покрывающем типичные нормализованные значения.

$$\text{token}(x) = \text{round}\left(\frac{x - x_{\min}}{x_{\max} - x_{\min}} \cdot (B - 1)\right)$$

Каждый бин — один токен в словаре модели. Временной ряд буквально превращается в последовательность «слов».

### Обратное преобразование

При декодировании прогноза токены превращаются обратно в числа, затем применяется обратное масштабирование.

## Архитектура Chronos v1: T5

Оригинальный Chronos использует [T5](https://arxiv.org/abs/1910.10683) (Text-to-Text Transfer Transformer).

### Encoder

```
Токены истории: [t_1, t_2, ..., t_T]
    ↓
Token embeddings + Positional encoding
    ↓
Transformer Encoder (self-attention)
    ↓
Encoded representation
```

### Decoder (авторегрессионный)

```
Для каждого шага прогноза:
    Encoded history + Ранее сгенерированные токены
        ↓
    Masked self-attention
        ↓
    Cross-attention на encoder output
        ↓
    Softmax → распределение над словарём
        ↓
    Sample токен
```

Вероятностный прогноз получается через многократное сэмплирование разных траекторий.

## Архитектура Chronos Bolt: T5-Efficient

Bolt использует переработанную архитектуру.

### Direct forecasting

Вместо генерации токен-за-токеном, Bolt предсказывает весь горизонт за один forward pass:

```
Токены истории
    ↓
T5-Efficient Encoder
    ↓
Decoder (один проход)
    ↓
Output: квантили для всех шагов горизонта сразу
```

### Предсказание квантилей напрямую

Вместо сэмплирования траекторий Bolt напрямую выдаёт квантили (p10, p50, p90 и др.) как выходы модели. Это:

- Быстрее (один проход вместо многих сэмплов)
- Стабильнее (нет variance от сэмплирования)
- Проще в использовании

### Оптимизации

- Efficient attention patterns — меньше вычислений при сохранении качества
- KV-cache оптимизации
- Лучший batching

## Размеры моделей

### Chronos v1

|Модель|Параметры|Контекст|Рекомендации|
|---|---|---|---|
|chronos-t5-tiny|8M|512|CPU, быстрый прототип|
|chronos-t5-mini|20M|512|Баланс|
|chronos-t5-small|46M|512|Хорошее качество|
|chronos-t5-base|200M|512|Высокое качество|
|chronos-t5-large|710M|512|Максимальное качество|

### Chronos Bolt (v2)

|Модель|Параметры|Контекст|Рекомендации|
|---|---|---|---|
|chronos-bolt-tiny|9M|2048|CPU, продакшн|
|chronos-bolt-mini|21M|2048|Отличный баланс|
|chronos-bolt-small|48M|2048|Рекомендуется|
|chronos-bolt-base|205M|2048|Максимальное качество|

Обратите внимание: Bolt поддерживает контекст до 2048 токенов — в 4 раза больше, чем v1.

## Сравнение производительности

Amazon приводит следующие цифры для Bolt vs v1:

|Метрика|Chronos v1|Chronos Bolt|Улучшение|
|---|---|---|---|
|GPU throughput|1x|250x|В 250 раз|
|CPU throughput|1x|20x|В 20 раз|
|Качество (WQL)|Baseline|Лучше|+5-15%|
|Контекст|512|2048|В 4 раза|

На практике это означает: прогноз, который на v1 занимал минуту, на Bolt занимает доли секунды.

## Сильные стороны

**Chronos v1:**

- Полностью вероятностный выход через сэмплирование
- Можно получить произвольные квантили
- Гибкость генерации (temperature, top-k, top-p)
- Хорошо изученная архитектура T5

**Chronos Bolt:**

- Радикально быстрее — пригоден для продакшена
- Больший контекст (2048 vs 512)
- Лучшее качество на бенчмарках
- Стабильные квантили без variance сэмплирования
- Проще в использовании

**Общие:**

- Универсальность — работает на данных любого масштаба и частоты
- Zero-shot из коробки
- Открытые веса и код
- Работа на CPU (особенно Bolt-tiny)

## Ограничения

**Chronos v1:**

- Очень медленный inference
- Ограниченный контекст (512)
- Много сэмплов для стабильных квантилей

**Chronos Bolt:**

- Фиксированный набор квантилей (нельзя запросить произвольный)
- Нет контроля temperature/sampling

**Общие:**

- Потеря точности при квантизации
- Нет поддержки ковариат
- Чувствительность к выбросам в масштабировании

## Код: Chronos

### Chronos Bolt (рекомендуется)

python

```python
import torch
import numpy as np
import pandas as pd
from chronos import BaseChronosPipeline

# Загрузка Bolt модели
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-small",
    device_map="cuda",  # или "cpu"
    torch_dtype=torch.bfloat16,
)

# Подготовка данных
context = torch.tensor([
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    [10.0, 20.0, 15.0, 25.0, 30.0, 20.0, 35.0, 40.0, 45.0, 50.0],
])

# Прогнозирование
# Bolt возвращает квантили напрямую, без сэмплирования
quantiles, mean = pipeline.predict_quantiles(
    context,
    prediction_length=12,
    quantile_levels=[0.1, 0.5, 0.9],  # какие квантили нужны
)

# quantiles shape: (batch, num_quantiles, prediction_length)
# mean shape: (batch, prediction_length)
print(f"Quantiles shape: {quantiles.shape}")
print(f"Mean shape: {mean.shape}")

# Доступ к конкретным квантилям
p10 = quantiles[:, 0, :]  # 10-й перцентиль
p50 = quantiles[:, 1, :]  # медиана
p90 = quantiles[:, 2, :]  # 90-й перцентиль
```

### Chronos v1 (если нужно сэмплирование)

python

```python
from chronos import ChronosPipeline

# Загрузка v1 модели
pipeline_v1 = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

# Прогнозирование с сэмплированием
forecast = pipeline_v1.predict(
    context,
    prediction_length=12,
    num_samples=100,  # количество траекторий
    temperature=1.0,
    top_k=50,
    top_p=1.0,
)

# forecast shape: (batch, num_samples, prediction_length)
print(f"Forecast shape: {forecast.shape}")

# Квантили из сэмплов
p10 = np.percentile(forecast.numpy(), 10, axis=1)
p50 = np.percentile(forecast.numpy(), 50, axis=1)
p90 = np.percentile(forecast.numpy(), 90, axis=1)
```

### Сравнение скорости v1 vs Bolt

python

```python
import time

def benchmark_chronos_versions(context, prediction_length=24):
    """Сравнение скорости Chronos v1 и Bolt."""
    
    results = {}
    
    # Bolt
    bolt = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-small",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    
    # Warmup
    _ = bolt.predict_quantiles(context, prediction_length=10, quantile_levels=[0.5])
    
    # Benchmark Bolt
    start = time.time()
    for _ in range(100):
        _ = bolt.predict_quantiles(
            context, 
            prediction_length=prediction_length,
            quantile_levels=[0.1, 0.5, 0.9]
        )
    bolt_time = (time.time() - start) / 100
    results['bolt'] = bolt_time
    
    del bolt
    torch.cuda.empty_cache()
    
    # v1
    v1 = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    
    # Warmup
    _ = v1.predict(context, prediction_length=10, num_samples=10)
    
    # Benchmark v1
    start = time.time()
    for _ in range(10):  # меньше итераций, потому что медленнее
        _ = v1.predict(
            context,
            prediction_length=prediction_length,
            num_samples=20
        )
    v1_time = (time.time() - start) / 10
    results['v1'] = v1_time
    
    print(f"Chronos v1: {v1_time:.3f}s per forecast")
    print(f"Chronos Bolt: {bolt_time:.3f}s per forecast")
    print(f"Speedup: {v1_time / bolt_time:.1f}x")
    
    return results

# Тест
test_context = torch.randn(10, 256)  # 10 рядов
benchmark_chronos_versions(test_context.cuda())
```

### Работа с pandas DataFrame

python

```python
def chronos_bolt_forecast_df(df, pipeline, prediction_length, 
                              quantile_levels=[0.1, 0.5, 0.9]):
    """
    Прогнозирование Chronos Bolt для DataFrame.
    """
    results = []
    
    for uid in df['unique_id'].unique():
        series = df[df['unique_id'] == uid].sort_values('ds')['y'].values
        context = torch.tensor(series).unsqueeze(0).float()
        
        # Прогноз
        quantiles, mean = pipeline.predict_quantiles(
            context.to(pipeline.device),
            prediction_length=prediction_length,
            quantile_levels=quantile_levels
        )
        
        quantiles_np = quantiles[0].cpu().numpy()
        mean_np = mean[0].cpu().numpy()
        
        for h in range(prediction_length):
            result = {
                'unique_id': uid,
                'horizon': h + 1,
                'forecast': mean_np[h],
            }
            for i, q in enumerate(quantile_levels):
                result[f'p{int(q*100)}'] = quantiles_np[i, h]
            results.append(result)
    
    return pd.DataFrame(results)

# Использование
forecast_df = chronos_bolt_forecast_df(
    train, pipeline, 
    prediction_length=16,
    quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9]
)
```

### Визуализация

python

```python
import matplotlib.pyplot as plt

def plot_chronos_forecast(context, mean, quantiles, quantile_levels, title=''):
    """Визуализация прогноза Chronos с интервалами."""
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # История
    history_idx = range(len(context))
    ax.plot(history_idx, context, 'b-', linewidth=2, label='История')
    
    # Прогноз
    forecast_start = len(context)
    horizon = len(mean)
    forecast_idx = range(forecast_start, forecast_start + horizon)
    
    # Находим индексы квантилей для интервалов
    q_dict = {q: i for i, q in enumerate(quantile_levels)}
    
    # 80% интервал (если есть p10 и p90)
    if 0.1 in q_dict and 0.9 in q_dict:
        ax.fill_between(
            forecast_idx,
            quantiles[q_dict[0.1]],
            quantiles[q_dict[0.9]],
            alpha=0.2, color='red', label='80% интервал'
        )
    
    # 50% интервал (если есть p25 и p75)
    if 0.25 in q_dict and 0.75 in q_dict:
        ax.fill_between(
            forecast_idx,
            quantiles[q_dict[0.25]],
            quantiles[q_dict[0.75]],
            alpha=0.3, color='red', label='50% интервал'
        )
    
    # Медиана или mean
    ax.plot(forecast_idx, mean, 'r-', linewidth=2, label='Прогноз')
    
    ax.axvline(x=forecast_start, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.set_title(title or 'Chronos Bolt: прогноз')
    ax.set_xlabel('Время')
    ax.set_ylabel('Значение')
    
    plt.tight_layout()
    plt.show()
```

## Какую версию выбрать?

|Сценарий|Рекомендация|
|---|---|
|Продакшн, нужна скорость|**Bolt**|
|Длинный контекст (>512 точек)|**Bolt**|
|Нужен произвольный квантиль (например, 0.95)|v1 (сэмплирование)|
|Исследования, эксперименты с generation|v1|
|CPU inference|**Bolt-tiny**|
|Максимальное качество|**Bolt-base**|
|Только начинаете с Chronos|**Bolt**|

**Практический совет:** начинайте с Chronos Bolt. Переходите на v1 только если нужны специфичные возможности сэмплирования (temperature control, произвольные квантили, анализ распределения траекторий).

## Chronos vs другие модели

|Критерий|Chronos Bolt|Chronos v1|[TiRex](https://arxiv.org/abs/2402.02868)|[FlowState](https://arxiv.org/abs/2403.08280)|
|---|---|---|---|---|
|Скорость|Быстро|Медленно|Средне|Быстро|
|Контекст|2048|512|2048+|2048+|
|Вероятностный выход|Квантили|Сэмплы|Квантили|Квантили|
|Параметры|9M–205M|8M–710M|35M|<10M|
|CPU inference|✓ (tiny)|~ (медленно)|✗|~|
|Произвольные квантили|✗|✓|✗|✗|

## Что дальше

Chronos показал, что идея «ряд как текст» работает, а Bolt доказал, что её можно сделать практичной для продакшена. Квантизация, трансформер, direct forecasting — это работающий пайплайн.

В следующем посте мы рассмотрим TimeGPT — закрытую модель от Nixtla, которая доступна только через API. Это другой подход к foundation models: вместо открытых весов — сервис с гарантированным качеством и простотой использования.

:::{seealso}
**Источники и ссылки:**
- Ansari, A.F., et al. (2024). [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815). TMLR 2024.
- Ansari, A.F., et al. (2025). [Chronos-Bolt: Efficient and Accurate Foundation Models for Time Series Forecasting](https://arxiv.org/abs/2510.15821). arXiv.
- [Официальный код Chronos](https://github.com/amazon-science/chronos-forecasting) — Amazon Science GitHub
- [Amazon Science Blog: Chronos](https://www.amazon.science/blog/adapting-language-model-architectures-for-time-series-forecasting)
:::