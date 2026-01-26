# iTransformer. Внимание наоборот

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/privettoha/neural-forecast-book/blob/main/notebooks/22_itransformer.ipynb)

iTransformer ([статья](https://arxiv.org/abs/2310.06625), ICLR 2024 Spotlight) задаёт провокационный вопрос: а что если мы неправильно применяли трансформеры к временным рядам всё это время? Традиционный подход — attention вдоль оси времени, [PatchTST](https://arxiv.org/abs/2211.14730) улучшил это через патчинг, но iTransformer делает радикальный шаг: переворачивает оси местами и применяет attention вдоль оси переменных (каналов), а не времени[^itrans-core]

[^itrans-core]: Liu, Y., et al. "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting." ICLR 2024 Spotlight. Section 3, Figure 2. https://arxiv.org/abs/2310.06625

## Идея

Ключевой момент отличия от традиционных трансформеров — **инверсия измерений**

Название расшифровывается как Inverted Transformer, и инверсия здесь не метафора: модель буквально транспонирует входные данные, превращая каждую переменную (канал) в токен, а временные точки — в признаки этого токена[^itrans-transpose]. Авторы формулируют разделение обязанностей так: **attention для межканальных корреляций, FFN для временных представлений**[^itrans-responsibilities]

[^itrans-transpose]: Liu, Y., et al. "iTransformer." ICLR 2024. Section 3.1: "We embed the whole time series of each variate independently into a (variate) token." https://arxiv.org/abs/2310.06625

[^itrans-responsibilities]: Liu, Y., et al. "iTransformer." ICLR 2024. Section 3.2: "The attention mechanism captures multivariate correlations; meanwhile, the feed-forward network is applied for each variate token to learn nonlinear representations." https://arxiv.org/abs/2310.06625

Результат парадоксален: модель, которая «не смотрит» на временные зависимости через attention, достигает SOTA на 7 реальных многомерных бенчмарках[^itrans-sota]

[^itrans-sota]: Liu, Y., et al. "iTransformer." ICLR 2024. Table 1, Table 2. https://arxiv.org/abs/2310.06625

## Проблема традиционного подхода

В стандартном трансформере входная матрица $(T, C)$ — время × каналы, каждый временной шаг становится токеном. Авторы идентифицируют три проблемы[^itrans-problems]:

[^itrans-problems]: Liu, Y., et al. "iTransformer." ICLR 2024. Section 2. https://arxiv.org/abs/2310.06625

➖ **Бессмысленность точечного сравнения**: attention сравнивает токены через dot product, но похожесть значений в понедельник и четверг не означает связь между этими днями. [DLinear](https://arxiv.org/abs/2205.13504) показал, что линейная модель конкурирует с трансформерами именно потому, что те тратят мощность на бессмысленные зависимости[^dlinear-critique]

[^dlinear-critique]: Zeng, A., et al. "Are Transformers Effective for Time Series Forecasting?" AAAI 2023. Section 4.2. https://arxiv.org/abs/2205.13504

➖ **Квадратичная сложность**: матрица внимания $T \times T$ становится узким местом для длинных рядов

➖ **Потеря индивидуальности переменных**: когда все переменные сливаются в один токен, модели сложно выучить, что температура и влажность ведут себя по-разному[^itrans-variate-problem]

[^itrans-variate-problem]: Liu, Y., et al. "iTransformer." ICLR 2024. Abstract: "The embedding for each temporal token fuses multiple variates that represent potential delayed events and distinct physical measurements." https://arxiv.org/abs/2310.06625

## Архитектура

Модель состоит из стандартного Transformer Encoder, но с инвертированными входами:

🔢 **Transpose и Variate Embedding**
Входная матрица $(T, C)$ транспонируется в $(C, T)$. Каждая переменная $\mathbf{x}_i \in \mathbb{R}^T$ проецируется в токен размерности $D$:
$$\mathbf{h}_i^{(0)} = W_e \cdot \mathbf{x}_i + b_e$$
где $W_e \in \mathbb{R}^{D \times T}$. Это «экстремальный случай патчинга», где один патч покрывает всю историю[^itrans-extreme-patch]. Нет позиционного кодирования по времени — вся временная информация упакована в эмбеддинг

[^itrans-extreme-patch]: Liu, Y., et al. "iTransformer." ICLR 2024. Section 3.1: "the extreme case of Patching that enlarges local receptive field." https://arxiv.org/abs/2310.06625

🔢 **Self-Attention между переменными**
Теперь $C$ токенов размерности $D$, и attention работает между переменными:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$
Матрица внимания $C \times C$ — количество переменных, не временных шагов. Если 100 переменных и 1000 точек: $10{,}000$ весов вместо $1{,}000{,}000$

🔢 **Feed-Forward для временных представлений**
FFN применяется к каждому токену независимо — это единственное место обработки временной информации. Авторы используют стандартный Transformer[^attention-orig] без модификаций

[^attention-orig]: Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017. https://arxiv.org/abs/1706.03762

🔢 **Layer Normalization**
Особая роль: уменьшает расхождения от несогласованных единиц измерения между переменными[^itrans-layernorm]

[^itrans-layernorm]: Liu, Y., et al. "iTransformer." ICLR 2024. Section 3.2. https://arxiv.org/abs/2310.06625

🔢 **Projection**
Каждый токен проецируется в прогноз: $\hat{\mathbf{y}}_i = W_p \cdot \mathbf{h}_i^{(L)} + b_p$, где $W_p \in \mathbb{R}^{H \times D}$

```
Input: X ∈ ℝ^(T×C)
    ↓
Transpose → X^T ∈ ℝ^(C×T)
    ↓
Variate Embedding → C токенов ∈ ℝ^D
    ↓
Transformer Encoder (L слоёв):
  - Multi-head Self-Attention между переменными
  - FFN для каждой переменной
    ↓
Projection → Ŷ ∈ ℝ^(H×C)
```

## Почему это работает

➖ **Временные паттерны проще межканальных**: автокорреляция, сезонность, тренд — относительно простые паттерны для MLP. Зависимости между переменными (как отказ сервера влияет на нагрузку другого) — сложные, контекстно-зависимые связи, для которых attention полезен[^itrans-rationale]

[^itrans-rationale]: Liu, Y., et al. "iTransformer." ICLR 2024. Section 3.1. https://arxiv.org/abs/2310.06625

➖ **Меньше шума в attention**: моделировать какие переменные влияют друг на друга проще, чем какие исторические моменты релевантны

➖ **Лучшее использование длинного lookback**: в отличие от традиционных трансформеров, iTransformer улучшает качество при увеличении входного окна[^itrans-lookback]

[^itrans-lookback]: Liu, Y., et al. "iTransformer." ICLR 2024. Figure 4: "iTransformers show a surprising improvement with increasing lookback window." https://arxiv.org/abs/2310.06625

## Ablation study (Table 3)

Авторы проверяют разные комбинации компонентов[^itrans-ablation]:

[^itrans-ablation]: Liu, Y., et al. "iTransformer." ICLR 2024. Table 3. https://arxiv.org/abs/2310.06625

| Variate (корреляции) | Temporal (представления) | Результат |
|---------------------|-------------------------|-----------|
| Attention | FFN | **лучший** |
| FFN | Attention | хуже |
| Attention | Attention | средне |

🔢 **Важно**: vanilla Transformer (Attention по времени, FFN по переменным) показывает **худший результат** среди всех вариантов — это подтверждает «несовместимость ответственностей» в традиционной архитектуре[^itrans-vanilla-worst]

[^itrans-vanilla-worst]: Liu, Y., et al. "iTransformer." ICLR 2024. Table 3: "The performance of vanilla Transformer performs the worst among these designs." https://arxiv.org/abs/2310.06625

## Что умеет

➖ **SOTA на многомерных бенчмарках**: ECL, ETT, Traffic (862 переменных), Weather, PEMS[^itrans-table1]
➖ **Генерализация на новые переменные**: можно прогнозировать переменные, которых не было при обучении[^itrans-generalization]
➖ **Интерпретируемость**: матрица внимания показывает связи между переменными[^itrans-interpret]
➖ **Простота**: стандартный Transformer без модификаций, просто инвертированные входы[^itrans-simplicity]

[^itrans-table1]: Liu, Y., et al. "iTransformer." ICLR 2024. Table 1, Table 2. https://arxiv.org/abs/2310.06625
[^itrans-generalization]: Liu, Y., et al. "iTransformer." ICLR 2024. Section 4.4. https://arxiv.org/abs/2310.06625
[^itrans-interpret]: Liu, Y., et al. "iTransformer." ICLR 2024. Figure 5. https://arxiv.org/abs/2310.06625
[^itrans-simplicity]: GitHub: thuml/iTransformer. https://github.com/thuml/iTransformer

## Когда использовать

👍 **Хорошо работает:**
➖ Многомерные данные с сильными зависимостями между переменными (связанные сенсоры, группа товаров-комплементов, метрики одной системы, транспортный трафик)
➖ Высокоразмерные датасеты (десятки-сотни переменных)
➖ Длинные входные окна (линейная сложность по времени)
➖ Когда нужна интерпретируемость межканальных связей

👎 **Проблемы:**
➖ Не работает для одномерных рядов (вырождается в один токен)
➖ Линейное время-embedding может терять сложные нелинейные паттерны[^itrans-embed-limit]
➖ Предполагает стабильность межканальных связей во времени
➖ Если переменные независимы — лучше [PatchTST](https://arxiv.org/abs/2211.14730) или [N-HiTS](https://arxiv.org/abs/2201.12886)

[^itrans-embed-limit]: Liu, Y., et al. "iTransformer." ICLR 2024. Appendix D. https://arxiv.org/abs/2310.06625

## iTransformer vs PatchTST vs TSMixer

| Критерий | iTransformer | PatchTST | TSMixer |
|----------|--------------|----------|---------|
| Межканальные связи | ✓ Явно | ✗ Независимо | ~ Feature mixing |
| Независимые каналы | ✗ Избыточно | ✓ Оптимально | ~ Работает |
| Много каналов (>100) | ~ $C^2$ attention | ✓ Независимо | ~ |
| Интерпретируемость связей | ✓ | ✗ | ✗ |
| Одномерные ряды | ✗ | ✓ | ✗ |
| Transfer learning | ~ | ✓✓ | ~ |

## Реализации

| Библиотека | Ссылка |
|------------|--------|
| Официальный код | [thuml/iTransformer](https://github.com/thuml/iTransformer) |
| Time-Series-Library | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) |
| GluonTS (probabilistic head) | [awslabs/gluonts](https://github.com/awslabs/gluonts) |
| neuralforecast | [Nixtla/neuralforecast](https://github.com/Nixtla/neuralforecast) |

:::{seealso}
**Источники:**
- Liu, Y., et al. (2024). [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625). ICLR 2024 Spotlight
- [Официальный код](https://github.com/thuml/iTransformer) — Tsinghua ML Group
- [OpenReview](https://openreview.net/forum?id=JePfAI8fah) — рецензии и ответы авторов
:::