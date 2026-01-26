## DLinear

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/privettoha/neural-forecast-book/blob/main/notebooks/14_dlinear.ipynb)

DLinear — простейшая линейная модель с одним слоем, которая превзошла Informer, Autoformer, FEDformer на девяти стандартных бенчмарках[^dlinear]. Статья «Are Transformers Effective for Time Series Forecasting?» получила Oral на AAAI 2023 и поставила под сомнение целое направление исследований.

[^dlinear]: Zeng, A., et al. "Are Transformers Effective for Time Series Forecasting?" AAAI 2023. Table 1. https://arxiv.org/abs/2205.13504

### Идея

Механизм self-attention **перестановочно-инвариантен**: если поменять местами элементы последовательности, выход изменится только из-за позиционного кодирования. Для текста это не критично — смысл определяется словами, не порядком. Для временных рядов порядок — это всё.

Позиционное кодирование частично компенсирует проблему, но недостаточно. Трансформеры переобучаются на шум, особенно на резкие изменения трендов, которые могут быть просто выбросами.

### Семейство LTSF-Linear

**🔢 Linear — чистый бейзлайн**

Один полносвязный слой: входное окно $L$ → прогноз $T$:

$$\hat{X} = XW, \quad W \in \mathbb{R}^{L \times T}$$

Никаких активаций, никакой нелинейности. Параметров: $L \times T$.

**🔢 NLinear — борьба с distribution shift**

Проблема: модель обучалась на данных со средним 100, тестируется на данных со средним 150.

Решение:
1. Вычитаем последнее значение: $X' = X - X_L$
2. Линейный слой: $\hat{X}' = X'W$
3. Добавляем обратно: $\hat{X} = \hat{X}' + X_L$

«Привязывает» прогноз к последнему известному значению. На датасетах с distribution shift (ETTh1, ETTh2, ILI) — значительное улучшение.

**🔢 DLinear — декомпозиция**

Идея из Autoformer[^autoformer], но для линейной модели:

[^autoformer]: Wu, H., et al. "Autoformer: Decomposition Transformers with Auto-Correlation." NeurIPS 2021. https://arxiv.org/abs/2106.13008

1. Разделяем на тренд и сезонность через скользящее среднее:
   - $X_{trend} = \text{AvgPool}(\text{Padding}(X))$
   - $X_{seasonal} = X - X_{trend}$

2. Отдельный линейный слой к каждой компоненте:
   - $\hat{X}_{trend} = X_{trend} W_{trend}$
   - $\hat{X}_{seasonal} = X_{seasonal} W_{seasonal}$

3. Суммируем: $\hat{X} = \hat{X}_{trend} + \hat{X}_{seasonal}$

Параметров: $2 \times L \times T$ — на порядки меньше любого трансформера.

### Результаты

**Качество на multivariate forecasting:**

| Датасет | Улучшение DLinear vs FEDformer (MSE) |
|---------|-------------------------------------|
| Exchange-Rate | >40% |
| Traffic, Electricity, Weather | ~30% |
| ETTm1 | ~25% |

**Особый случай — Exchange-Rate:** даже наивный Repeat (повторение последнего значения) превосходит все трансформеры на ~45%. Причина: трансформеры переобучаются на резкие изменения, воспринимая шум как паттерны.

**Эффективность:**

| Модель | MACs (×10⁶) | Время (мс) | MSE |
|--------|-------------|------------|-----|
| Informer | 1,720 | 61.2 | 0.304 |
| Autoformer | 1,630 | 44.6 | 0.227 |
| FEDformer | 4,130 | 83.6 | 0.214 |
| **DLinear** | **0.51** | **0.6** | **0.212** |

В 3000+ раз меньше вычислений, в 70–140 раз быстрее, качество лучше.

### Интерпретируемость

Веса матрицы $W$ показывают, какие лаги влияют на какие точки прогноза. Визуализация на ETTh1 выявляет периодическую структуру с периодом 24 — модель «выучила» суточный цикл без явного указания.

Такая прозрачность невозможна для трансформеров с многослойным attention.

### Ограничения и критика

➖ **Сравнение в разных условиях**: lookback 336 для DLinear vs 96 для трансформеров — преимущество линейным моделям[^hf-response]

[^hf-response]: Hugging Face Blog. "Yes, Transformers are Effective for Time Series Forecasting." https://huggingface.co/blog/autoformer

➖ **Channel-independence**: каждый канал обрабатывается отдельно — не улавливает корреляции между переменными

➖ **Нелинейные зависимости**: линейная модель по определению их не моделирует

➖ **Ограниченность бенчмарков**: все девять датасетов «хорошо структурированы» — с явными трендами и сезонностью

### Влияние на сообщество

После публикации — сдвиг в исследованиях:

➖ **PatchTST**[^patchtst] (2023): patching решает часть проблем трансформеров, превосходит DLinear

[^patchtst]: Nie, Y., et al. "A Time Series is Worth 64 Words." ICLR 2023. https://arxiv.org/abs/2211.14730

➖ **iTransformer**[^itransformer] (2024): инверсия измерений (переменная = токен) для cross-variate информации

[^itransformer]: Liu, Y., et al. "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting." ICLR 2024. https://arxiv.org/abs/2310.06625

➖ **TSMixer** (2023): all-MLP с cross-channel зависимостями

DLinear остаётся критическим бейзлайном: если новая модель его не превосходит — стоит задуматься.

### Когда использовать

**Да:**
➖ Быстрый бейзлайн для оценки сложности задачи
➖ Данные с явной трендовой и сезонной структурой
➖ Жёсткие требования к latency
➖ Нужна интерпретируемость

**Нет:**
➖ Важны корреляции между переменными → TSMixer, iTransformer
➖ Сложные нелинейные паттерны → N-BEATS, N-HiTS
➖ Zero-shot → foundation models
➖ Probabilistic forecasting → DeepAR, Lag-Llama

### Выводы

1. **Сложность ≠ качество**
2. **Бейзлайны имеют значение** — сравнение только со сложными моделями может быть обманчивым
3. **Специфика задачи важна** — трансформеры для текста ≠ трансформеры для рядов
4. **Простота имеет ценность** — интерпретируемость, скорость, надёжность

Перед сложной архитектурой всегда проверяйте: не справится ли простая линейная модель?

### Ссылки

| Ресурс | URL |
|--------|-----|
| Статья | https://arxiv.org/abs/2205.13504 |
| Код | https://github.com/cure-lab/LTSF-Linear |
| NeuralForecast | https://nixtlaverse.nixtla.io/neuralforecast/models.html#dlinear |
| Ответ Hugging Face | https://huggingface.co/blog/autoformer |
