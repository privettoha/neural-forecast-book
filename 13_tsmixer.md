## TSMixer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/privettoha/neural-forecast-book/blob/main/notebooks/13_tsmixer.ipynb)

TSMixer (Time Series Mixer) — адаптация MLP-Mixer из компьютерного зрения для временных рядов. Модель «перемешивает» информацию по двум осям — времени и признакам — без рекуррентности и механизма внимания. Первая MLP-архитектура, достигшая уровня SOTA на multivariate бенчмарках[^tsmixer].

[^tsmixer]: Chen, S., et al. "TSMixer: An All-MLP Architecture for Time Series Forecasting." TMLR, 2023. https://arxiv.org/abs/2303.06053

### Контекст: почему не трансформеры?

В 2022 году статья «Are Transformers Effective for Time Series Forecasting?»[^dlinear] показала неудобную правду: простая линейная модель (DLinear) обходит Informer, Autoformer, FEDformer на стандартных бенчмарках.

[^dlinear]: Zeng, A., et al. "Are Transformers Effective for Time Series Forecasting?" AAAI 2023. https://arxiv.org/abs/2205.13504

Почему трансформеры буксуют на временных рядах:

➖ **Permutation invariance**: self-attention инвариантен к порядку без позиционного кодирования, а для рядов порядок — это всё

➖ **Point-wise attention**: одна точка ряда — просто число, не токен с семантикой; паттерны проявляются в последовательностях

➖ **Квадратичная сложность**: $O(L^2)$ по длине — узкое место для длинных рядов

TSMixer предлагает отказаться от attention в пользу более простых операций.

### Идея

Исходник — MLP-Mixer[^mlpmixer] от Google для изображений. Вместо self-attention используются два типа MLP:

[^mlpmixer]: Tolstikhin, I., et al. "MLP-Mixer: An all-MLP Architecture for Vision." NeurIPS 2021. https://arxiv.org/abs/2105.01601

➖ **Token-mixing**: перемешивает информацию между пространственными позициями
➖ **Channel-mixing**: перемешивает информацию между признаками

TSMixer переносит это на временные ряды. Данные — матрица $T \times C$ (время × каналы):

➖ **Time-mixing**: перемешивание вдоль оси времени (для каждого канала независимо) — учит временные паттерны
➖ **Feature-mixing**: перемешивание вдоль оси признаков (для каждого момента независимо) — учит зависимости между переменными

### Архитектура

**🔢 Time-mixing подблок**

```
Input X ∈ ℝ^(T×C)
    ↓ Transpose → ℝ^(C×T)
    ↓ LayerNorm
    ↓ MLP (к каждой строке независимо)
    ↓ Transpose → ℝ^(T×C)
    ↓ + Residual
Output ∈ ℝ^(T×C)
```

Транспонирование нужно, чтобы MLP работал вдоль временной оси. Каждая строка — история одного канала.

**🔢 Feature-mixing подблок**

```
Input X ∈ ℝ^(T×C)
    ↓ LayerNorm
    ↓ MLP (к каждой строке независимо)
    ↓ + Residual
Output ∈ ℝ^(T×C)
```

Без транспонирования — MLP сразу работает по признакам. Каждая строка — срез всех каналов в один момент.

**🔢 Полный блок**

Один блок = Time-mixing → Feature-mixing. Модель — стек таких блоков.

$$X' = X + \text{MLP}_{\text{time}}(\text{LayerNorm}(X)^T)^T$$
$$X'' = X' + \text{MLP}_{\text{feature}}(\text{LayerNorm}(X'))$$

**🔢 Temporal Projection**

После mixer-блоков — линейная проекция $T \rightarrow H$ (горизонт):

$$\hat{Y} = W \cdot X_{\text{final}} + b, \quad W \in \mathbb{R}^{H \times T}$$

Для каждого канала независимо проецирует $T$ исторических значений в $H$ будущих.

### Почему mixing работает

➖ **Индуктивное смещение**: time-mixing предполагает, что временные паттерны похожи для разных переменных; feature-mixing — что зависимости между переменными стабильны во времени. Для большинства рядов это разумно

➖ **Меньше параметров**: $O(T \times d)$ для time-mixing, $O(C \times d)$ для feature-mixing — линейно, не квадратично

➖ **Устойчивость к шуму**: MLP учит фиксированное преобразование, менее чувствительное к случайным флуктуациям, чем attention на основе сходства

### Что умеет

➖ Нативная работа с многомерными рядами — моделирует зависимости между каналами через feature-mixing

➖ Простота: линейные слои, LayerNorm, GELU — никаких сложных механизмов

➖ Конкурентное качество на multivariate бенчмарках

➖ Линейная сложность по длине ряда

➖ TSMixer-Ext: расширение для работы с ковариатами (статические признаки, известное будущее, календарные переменные)

### Ограничения

➖ Feature-mixing имеет смысл только при нескольких связанных переменных — для univariate избыточен

➖ Фиксированная структура зависимостей — модель не может адаптивно «решать», на что обратить внимание

➖ Чувствительность к порядку каналов — при изменении порядка нужно переобучать

➖ Относительная новизна (2023) — меньше практического опыта, чем у N-BEATS

### Когда использовать

**Да:**
➖ Группа связанных переменных (категории товаров, сенсоры одной системы)
➖ Нужно моделировать межканальные зависимости
➖ Есть ковариаты (TSMixer-Ext)

**Нет:**
➖ Независимые univariate ряды → N-HiTS
➖ Нужна интерпретируемость (trend/seasonality) → N-BEATS interpretable
➖ Cold start → foundation models

### TSMixer vs N-BEATS/N-HiTS

| Критерий | N-BEATS/N-HiTS | TSMixer |
|----------|----------------|---------|
| Univariate | ✓ Оптимально | ~ Избыточно |
| Multivariate связанные | ~ Независимо | ✓ Моделирует зависимости |
| Интерпретируемость | ✓ trend/seasonality | ✗ Чёрный ящик |
| Ковариаты | ✗ (только N-BEATSx) | ✓ TSMixer-Ext |

### Ссылки

| Ресурс | URL |
|--------|-----|
| Статья | https://arxiv.org/abs/2303.06053 |
| MLP-Mixer (предшественник) | https://arxiv.org/abs/2105.01601 |
| Google Blog | https://blog.research.google/2023/09/tsmixer-all-mlp-architecture-for-time.html |
| Код | https://github.com/google-research/google-research/tree/master/tsmixer |
