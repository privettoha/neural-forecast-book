# DLinear: критика сложности

## Ключевая идея

В 2022 году группа исследователей из CUHK и Microsoft задала неудобный вопрос: **действительно ли трансформеры эффективны для прогнозирования временных рядов?**[^dlinear]

Их статья получила три оценки strong accept и была принята как **Oral на AAAI 2023**. Ответ оказался обескураживающим: простейшая линейная модель с одним слоем превзошла Informer[^informer], Autoformer[^autoformer], FEDformer[^fedformer] и другие сложные архитектуры на **девяти стандартных бенчмарках**. Не на некоторых — практически на всех, и с заметным отрывом.

Эта работа не просто предложила модель — она поставила под сомнение целое направление исследований и заставила сообщество пересмотреть подходы к оценке прогнозных моделей.

**Результаты:** DLinear требует в **3000+ раз меньше вычислений** и работает в **70-140 раз быстрее**, показывая лучшее или сравнимое качество[^dlinear].

## Почему трансформеры могут не работать

Авторы выделяют фундаментальную проблему: механизм self-attention по своей природе **перестановочно-инвариантен** (permutation-invariant).

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PERMUTATION INVARIANCE                           │
│                                                                     │
│  Self-attention вычисляет:  Attention(Q, K, V) = softmax(QK^T/√d)V │
│                                                                     │
│  Проблема: если переставить элементы входа,                        │
│  attention-веса изменятся только из-за positional encoding,        │
│  НЕ из-за самого механизма внимания.                               │
│                                                                     │
│  Для текста:   "Кот сидит на коврике" ≈ "На коврике сидит кот"    │
│                 (смысл примерно сохраняется)                        │
│                                                                     │
│  Для рядов:    [100, 105, 110] ≠ [110, 100, 105]                   │
│                 (порядок — это ВСЁ!)                                │
└─────────────────────────────────────────────────────────────────────┘
```

Для текста порядок не критичен — смысл предложения определяется словами. Но для временных рядов **порядок — это всё**. Значение в момент $t$ связано именно с $t-1$, $t-2$, а не с произвольными точками.

Positional encoding частично компенсирует проблему, но авторы показывают: этого недостаточно. Трансформеры переобучаются на шум, воспринимая резкие изменения как значимые паттерны.

## Семейство LTSF-Linear: три модели

Авторы предложили три линейные модели, каждая решает свою задачу:

### Linear — чистый бейзлайн

Один полносвязный слой, отображающий вход длины $L$ в прогноз длины $H$:

$$\hat{\mathbf{X}} = \mathbf{X} \mathbf{W}, \quad \mathbf{X} \in \mathbb{R}^{L}, \; \mathbf{W} \in \mathbb{R}^{L \times H}$$

Никаких активаций, никакой нелинейности. Параметров: $L \times H$.

```
Input [x₁, x₂, ..., xₗ]
         │
         ▼
    ┌─────────┐
    │ Linear  │  W ∈ ℝ^(L×H)
    │  Layer  │
    └─────────┘
         │
         ▼
Output [ŷ₁, ŷ₂, ..., ŷₕ]
```

Казалось бы, такая модель не может ничего интересного. Но именно она показала, что сложность трансформеров не оправдана.

### NLinear — нормализация для distribution shift

Одна из главных проблем — различие распределений между train и test. Если модель обучалась на данных со средним 100, а тестируется на данных со средним 150, будет систематическая ошибка.

NLinear решает элегантно:

1. Вычитаем последнее значение: $\mathbf{X}' = \mathbf{X} - x_L$
2. Пропускаем через линейный слой: $\hat{\mathbf{X}}' = \mathbf{X}' \mathbf{W}$
3. Добавляем обратно: $\hat{\mathbf{X}} = \hat{\mathbf{X}}' + x_L$

```
Input [x₁, x₂, ..., xₗ]
         │
         ▼
    Subtract xₗ  ──────────────────────┐
         │                              │
         ▼                              │
    ┌─────────┐                         │
    │ Linear  │                         │
    └─────────┘                         │
         │                              │
         ▼                              │
    Add xₗ back  ◄─────────────────────┘
         │
         ▼
Output [ŷ₁, ŷ₂, ..., ŷₕ]
```

Это простейшая нормализация, «привязывающая» прогноз к последнему известному значению. На датасетах с distribution shift (ETTh1, ETTh2, ILI) NLinear показывает значительное улучшение.

### DLinear — декомпозиция для трендов и сезонности

DLinear использует идею декомпозиции из Autoformer[^autoformer], но применяет к линейной модели:

1. **Разделяем** ряд на тренд и сезонность через скользящее среднее:
   - $\mathbf{X}_{\text{trend}} = \text{AvgPool}(\text{Padding}(\mathbf{X}))$
   - $\mathbf{X}_{\text{seasonal}} = \mathbf{X} - \mathbf{X}_{\text{trend}}$

2. **Применяем** отдельный линейный слой к каждой компоненте:
   - $\hat{\mathbf{X}}_{\text{trend}} = \mathbf{X}_{\text{trend}} \mathbf{W}_{\text{trend}}$
   - $\hat{\mathbf{X}}_{\text{seasonal}} = \mathbf{X}_{\text{seasonal}} \mathbf{W}_{\text{seasonal}}$

3. **Суммируем:** $\hat{\mathbf{X}} = \hat{\mathbf{X}}_{\text{trend}} + \hat{\mathbf{X}}_{\text{seasonal}}$

```
Input [x₁, x₂, ..., xₗ]
         │
         ▼
┌─────────────────────────┐
│   Moving Average        │
│   (kernel_size=25)      │
└─────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
  Trend    Seasonal
  (smooth)  (X - Trend)
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│Linear │ │Linear │
│W_trend│ │W_seas │
└───────┘ └───────┘
    │         │
    └────┬────┘
         ▼
        Sum
         │
         ▼
Output [ŷ₁, ŷ₂, ..., ŷₕ]
```

Параметров: $2 \times L \times H$ — вдвое больше vanilla Linear, но на порядки меньше любого трансформера.

**Ядро скользящего среднего:** По умолчанию 25 (как в Autoformer). Можно подбирать под сезонность данных.

## Результаты: масштаб превосходства

Эксперименты на 9 бенчмарках: ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Exchange-Rate, Traffic, Weather, ILI.

### Multivariate forecasting

DLinear превзошёл FEDformer (лучший трансформер на тот момент):

| Датасет | Улучшение MSE |
|---------|---------------|
| Exchange-Rate | **>40%** |
| Traffic, Electricity, Weather | ~30% |
| ETTm1 | ~25% |

### Особый случай: Exchange-Rate

Удивительный факт: даже **наивный Repeat** (повторение последнего значения) превосходит все трансформеры на ~45%.

**Причина:** Трансформеры переобучаются на резкие изменения в обучающих данных, воспринимая их как значимые паттерны. На тесте они предсказывают несуществующие развороты тренда.

### Вычислительная эффективность

| Модель | MACs (×10⁶) | Время (мс) | MSE |
|--------|-------------|------------|-----|
| Informer | 1,720 | 61.2 | 0.304 |
| Autoformer | 1,630 | 44.6 | 0.227 |
| FEDformer | 4,130 | 83.6 | 0.214 |
| **DLinear** | **0.51** | **0.6** | **0.212** |

DLinear: **3000× меньше вычислений**, **70-140× быстрее**, **лучшее качество**.

## Интерпретируемость весов

Неожиданный бонус — интерпретируемость. Веса $\mathbf{W}$ напрямую показывают, какие лаги влияют на какие точки прогноза.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Визуализация весов DLinear                       │
│                        (ETTh1, часовые данные)                      │
│                                                                     │
│  Прогноз    ┌──────────────────────────────────────┐               │
│  точка 1    │ ░░▓▓░░▓▓░░▓▓░░▓▓░░▓▓░░▓▓░░▓▓░░▓▓░░▓▓ │  ← период 24 │
│  точка 2    │ ▓░░▓▓░░▓▓░░▓▓░░▓▓░░▓▓░░▓▓░░▓▓░░▓▓░░▓ │               │
│  ...        │ ...                                   │               │
│             └──────────────────────────────────────┘               │
│                        Входные лаги (1...336)                       │
│                                                                     │
│  Модель автоматически выучила суточный цикл (период=24)            │
│  без явного указания на сезонность!                                │
└─────────────────────────────────────────────────────────────────────┘
```

Такая прозрачность невозможна для трансформеров с многослойными attention-механизмами.

## Код: DLinear

### PyTorch реализация (~30 строк)

```python
import torch
import torch.nn as nn

class MovingAvg(nn.Module):
    """Скользящее среднее для декомпозиции"""
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Паддинг для сохранения длины
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, kernel_size=25):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Декомпозиция
        self.decomposition = MovingAvg(kernel_size)
        
        # Отдельные линейные слои для тренда и сезонности
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [batch, seq_len, channels]
        trend = self.decomposition(x)
        seasonal = x - trend
        
        # Channel-independent прогноз
        trend_out = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal_out = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
        
        return trend_out + seasonal_out
```

### NeuralForecast

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import DLinear
from neuralforecast.losses.pytorch import MAE

model = DLinear(
    h=96,               # горизонт
    input_size=336,     # входное окно
    loss=MAE(),
    max_steps=1000,
    learning_rate=1e-3,
)

nf = NeuralForecast(models=[model], freq='H')
nf.fit(df=train_df)
forecasts = nf.predict()
```

## Ограничения и критика

После публикации статья вызвала активную дискуссию.

### 1. Сравнение в разных условиях

Hugging Face показали[^hfblog]: при **равных условиях** (одинаковый lookback window) трансформеры могут превосходить DLinear. Авторы использовали lookback 336 для DLinear и 96 для трансформеров.

### 2. Channel-independence

DLinear обрабатывает каждый канал независимо — не улавливает корреляции между переменными. Для задач с важными межканальными зависимостями лучше TSMixer, iTransformer.

### 3. Нелинейные зависимости

Линейная модель по определению не моделирует нелинейные паттерны. Если они есть — DLinear их пропустит.

### 4. Ограниченность бенчмарков

Все 9 бенчмарков «хорошо структурированы» — с явными трендами и сезонностью. На хаотичных данных результаты могут отличаться.

## Место в экосистеме

После «Are Transformers Effective?» произошёл сдвиг в исследованиях:

| Модель | Год | Ответ на DLinear |
|--------|-----|------------------|
| **PatchTST**[^patchtst] | 2023 | Patching решает часть проблем трансформеров |
| **iTransformer**[^itransformer] | 2024 | Инверсия измерений для cross-variate |
| **TSMixer**[^tsmixer] | 2023 | All-MLP с channel mixing |

**DLinear остаётся важным бейзлайном:** если ваша модель не превосходит DLinear, стоит задуматься о её ценности.

## Когда использовать DLinear

✅ **Хороший выбор:**
- Быстрый бейзлайн для оценки сложности задачи
- Данные с явной трендовой и сезонной структурой
- Ограниченные вычислительные ресурсы
- Потребность в интерпретируемости
- Продакшен с жёсткими требованиями к latency

❌ **Лучше другая модель:**
- Важны корреляции между переменными → TSMixer, iTransformer
- Сложные нелинейные паттерны → N-BEATS, N-HiTS
- Zero-shot на новых данных → Chronos, TimesFM
- Probabilistic forecasting → DeepAR, Lag-Llama

## Выводы

DLinear — важный урок для всего сообщества:

1. **Сложность ≠ качество.** Миллионы параметров не гарантируют результат.

2. **Бейзлайны имеют значение.** Сравнение только со сложными моделями может быть обманчивым.

3. **Специфика задачи важна.** Трансформеры работают на тексте, но временные ряды — другая задача.

4. **Простота имеет ценность.** Интерпретируемость, скорость, надёжность.

5. **Историческое значение:** Эта работа «закончила эру доминирования трансформеров» и открыла ренессанс исследований альтернативных архитектур[^survey].

Перед применением сложной архитектуры всегда проверяйте: не справится ли простая линейная модель?

В следующей главе рассмотрим PatchTST — частичный ответ на критику DLinear, где patching позволяет трансформерам работать эффективнее.

---

## Ссылки

[^dlinear]: Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? *AAAI 2023 (Oral)*. https://arxiv.org/abs/2205.13504

[^informer]: Zhou, H., et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. *AAAI 2021 Best Paper*. https://arxiv.org/abs/2012.07436

[^autoformer]: Wu, H., et al. (2021). Autoformer: Decomposition Transformers with Auto-Correlation. *NeurIPS 2021*. https://arxiv.org/abs/2106.13008

[^fedformer]: Zhou, T., et al. (2022). FEDformer: Frequency Enhanced Decomposed Transformer. *ICML 2022*. https://arxiv.org/abs/2201.12740

[^patchtst]: Nie, Y., et al. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *ICLR 2023*. https://arxiv.org/abs/2211.14730

[^itransformer]: Liu, Y., et al. (2024). iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. *ICLR 2024*. https://arxiv.org/abs/2310.06625

[^tsmixer]: Chen, S., et al. (2023). TSMixer: An All-MLP Architecture for Time Series Forecasting. *TMLR*. https://arxiv.org/abs/2303.06053

[^hfblog]: Hugging Face Blog. Yes, Transformers are Effective for Time Series Forecasting (+ Autoformer). https://huggingface.co/blog/autoformer

[^survey]: Kim, J., et al. (2025). A comprehensive survey of deep learning for time series forecasting. *Applied Intelligence*. https://doi.org/10.1007/s10462-025-11123-9