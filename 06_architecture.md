# Карта архитектур

## Зачем нужна карта

Прежде чем погружаться в конкретные модели, полезно увидеть общую картину: какие подходы существуют, как они соотносятся друг с другом и почему одни архитектуры сменяли другие.

Без этой карты легко потеряться в зоопарке названий — N-BEATS, PatchTST, Chronos, TiRex, Toto, FlowState — и не понимать, это конкуренты, эволюция одной идеи или принципиально разные инструменты для разных задач.

## Три оси классификации

Любую нейросетевую архитектуру для временных рядов можно расположить в трёхмерном пространстве[^kim2025]:

### Ось 1: Тип вычислительного блока

**MLP-based (полносвязные)**

Обычные fully connected слои: окно истории на вход, прогноз на выход. Никакой рекуррентности, никакого attention — только матричные умножения и нелинейности.

Звучит примитивно, но N-BEATS[^nbeats], N-HiTS[^nhits], DLinear[^dlinear] показали: при правильном дизайне MLP работают не хуже сложных альтернатив.

| Преимущества | Недостатки |
|--------------|------------|
| Полная параллелизация | Фиксированный контекст |
| Простота реализации | Нет явного моделирования зависимостей |
| Интерпретируемость | |

**RNN/SSM-based (последовательные)**

Модели, обрабатывающие ряд шаг за шагом с «памятью» о предыдущих наблюдениях. Классические LSTM/GRU и современные State Space Models (Mamba[^mamba], S4[^s4]).

После забвения в 2020–2022 последовательные архитектуры возвращаются: xLSTM[^xlstm] (основа TiRex[^tirex]) и SSM-encoder (основа FlowState[^flowstate]) показывают SOTA в 2025.

| Преимущества | Недостатки |
|--------------|------------|
| Естественное моделирование времени | Сложность параллелизации (классика) |
| Гибкий контекст | Vanishing gradients (классика) |
| $O(1)$ память на шаг | Современные SSM решают эти проблемы |

**Transformer-based (на основе attention)**

Self-attention[^vaswani] позволяет каждому моменту «смотреть» на любой другой напрямую, без цепочки скрытых состояний. В теории — решение проблемы долгосрочных зависимостей. На практике: временные ряды — не текст, и прямое применение трансформеров даёт неоднозначные результаты.

PatchTST[^patchtst], iTransformer[^itransformer] — попытки адаптировать трансформеры к специфике рядов.

| Преимущества | Недостатки |
|--------------|------------|
| Глобальный контекст | $O(L^2)$ сложность |
| Параллельное обучение | Permutation invariance |
| Масштабируемость | Не всегда лучше простых альтернатив[^dlinear] |

**Почему permutation invariance — проблема:** Self-attention по умолчанию инвариантен к перестановке входов — результат не зависит от порядка токенов. Для текста это частично компенсируется тем, что смысл предложения сохраняется при небольших перестановках («кошка съела мышку» ≈ «мышку съела кошка» — смысл понятен). Для временных рядов порядок — это сама суть данных. Перемешайте точки — и ряд уничтожен, паттерны исчезли. Positional encoding[^vaswani] частично решает проблему, но это «костыль», а не естественное свойство архитектуры.

**Decoder-only (LLM-style)**

Подкласс трансформеров с архитектурой языковых моделей: авторегрессионная генерация, causal attention, часто — токенизация входов. Chronos[^chronos], TimeGPT[^timegpt], TimesFM[^timesfm], Lag-Llama[^lagllama].

Философия: если работает для текста — может сработать для рядов при достаточном масштабе.

### Ось 2: Режим обучения

**Local models (локальные)**

Отдельная модель для каждого ряда. 1000 рядов = 1000 моделей. Каждая полностью адаптирована к своему ряду, но не использует информацию из других.

*Примеры:* Классический ARIMA, ETS.

**Global models (глобальные)**

Одна модель на все ряды. Параметры общие, модель учится на паттернах из всех рядов одновременно.

*Примеры:* N-BEATS[^nbeats], N-HiTS[^nhits], DeepAR[^deepar], PatchTST[^patchtst].

**Foundation models (предобученные)**

Модель предобучена на огромном корпусе данных, применяется к новым рядам без дообучения (zero-shot) или с минимальной адаптацией (fine-tuning).

*Примеры:* Chronos[^chronos], TimesFM[^timesfm], Moirai[^moirai], Lag-Llama[^lagllama], Toto[^toto].

**Жёсткий критерий разграничения:**

| Характеристика | Global | Foundation |
|----------------|--------|------------|
| Данные обучения | Один домен (только ритейл) | Cross-domain (ритейл + погода + финансы + датчики) |
| Цель | Уловить зависимости между рядами домена | Выучить «универсальную математику» колебаний |
| Применение | К рядам того же домена | К любым рядам (zero-shot) |
| Примеры | DeepAR на данных компании | Chronos, TimesFM |

DeepAR[^deepar], обученный на всех SKU ритейлера — это global model. Chronos[^chronos], обученный на Monash[^monash] + синтетике + разных доменах — foundation model. Граница проходит по **разнообразию доменов в обучающих данных**, а не по размеру модели.

### Ось 3: Тип выхода

**Point forecast**

Одно значение на каждый шаг горизонта: $\hat{y}_{t+h}$.

*Примеры:* DLinear[^dlinear], N-BEATS[^nbeats] (default), PatchTST[^patchtst].

**Probabilistic forecast**

Распределение или квантили: $p(y_{t+h})$ или $[q_{0.1}, q_{0.5}, q_{0.9}]$.

*Примеры:* DeepAR[^deepar], Chronos[^chronos], Lag-Llama[^lagllama], Moirai[^moirai], Toto[^toto].

**Тренд:** Почти все foundation models выдают probabilistic output. Это становится стандартом.

## Визуальная карта моделей

Расположим модели по двум основным осям — тип блока и режим обучения:

```
                        │ Global              │ Foundation
────────────────────────┼─────────────────────┼─────────────────────
MLP-based               │ N-BEATS             │
                        │ N-HiTS              │
                        │ TSMixer             │
                        │ DLinear             │
────────────────────────┼─────────────────────┼─────────────────────
RNN/SSM-based           │ DeepAR              │ TiRex (xLSTM)
                        │                     │ FlowState (SSM)
────────────────────────┼─────────────────────┼─────────────────────
Transformer-based       │ PatchTST            │ Moirai
(Encoder)               │ iTransformer        │ MOMENT
────────────────────────┼─────────────────────┼─────────────────────
Transformer-based       │                     │ Chronos (T5)
(Decoder / Enc-Dec)     │                     │ TimeGPT
                        │                     │ TimesFM
                        │                     │ Lag-Llama
                        │                     │ Toto
```

**Комментарии к карте:**

- **Границы размыты.** TiRex[^tirex] и FlowState[^flowstate] — foundation models на последовательных архитектурах. Toto[^toto] — трансформер с фокусом на observability. Chronos[^chronos] — гибрид трансформера и LLM-подхода через токенизацию.

- **MLP-based нет в foundation.** Пока не существует foundation model на чистых MLP. Это интересное направление для исследований.

- **Decoder-only доминирует в FM.** TimesFM[^timesfm], Lag-Llama[^lagllama], Toto[^toto] — все decoder-only. Это не случайность: scaling laws работают, унифицированное обучение проще.

## Хронология: как мы сюда пришли

```
2017    DeepAR          Глобальные probabilistic модели
  │
2020    N-BEATS         MLP достаточно для SOTA
  │
2021    Informer        Трансформеры для long-term forecasting
        Autoformer
  │
2022    FEDformer       Frequency-enhanced attention
  │
2023    DLinear         Линейный слой побеждает трансформеры (!)
        PatchTST        Patching решает проблемы трансформеров
        TSMixer         MLP-Mixer от Google
        N-HiTS          Иерархический N-BEATS
        Chronos         Токенизация + T5
        TimeGPT         Закрытая foundation model
  │
2024    iTransformer    Инверсия измерений
        TimesFM         Decoder-only от Google
        Moirai          Any-variate attention
        Lag-Llama       Open-source probabilistic FM
  │
2025    TiRex           xLSTM возвращается
        FlowState       SSM + time-scale invariance
        Toto            Domain-specific FM (observability)
```

**Ключевые переломные моменты:**

1. **2020: N-BEATS**[^nbeats] показал, что MLP конкурентоспособны
2. **2023: DLinear**[^dlinear] продемонстрировал, что линейный слой побеждает сложные трансформеры
3. **2023-2024: Foundation models** изменили парадигму с «обучаем на своих данных» на «применяем предобученную модель»
4. **2025: Возвращение RNN/SSM** — xLSTM[^xlstm] и SSM показывают SOTA

## Что мы рассматриваем в книге

### MLP-based: N-BEATS, N-HiTS, TSMixer, DLinear

Демонстрируют, что для хорошего прогнозирования не нужны ни рекуррентность, ни attention.

| Модель | Ключевая идея | Когда использовать |
|--------|---------------|-------------------|
| N-BEATS[^nbeats] | Basis expansion + residual stacking | Interpretable прогноз |
| N-HiTS[^nhits] | Иерархическая обработка масштабов | Long horizon |
| TSMixer[^tsmixer] | Time-mixing + feature-mixing | Multivariate с cross-channel |
| DLinear[^dlinear] | Decomposition + linear | Быстрый baseline |

### Transformer-based: PatchTST, iTransformer

Два подхода к адаптации трансформеров.

| Модель | Ключевая идея | Когда использовать |
|--------|---------------|-------------------|
| PatchTST[^patchtst] | Patching + channel independence | Long context univariate |
| iTransformer[^itransformer] | Attention по каналам | Multivariate с корреляциями |

### RNN/SSM-based: DeepAR, TiRex, FlowState

Последовательные архитектуры — классика и современность.

| Модель | Ключевая идея | Когда использовать |
|--------|---------------|-------------------|
| DeepAR[^deepar] | LSTM + probabilistic | Production с ковариатами |
| TiRex[^tirex] | xLSTM foundation model | Zero-shot probabilistic |
| FlowState[^flowstate] | SSM + time-scale invariance | Разные частоты данных |

### Foundation models: Chronos, TimesFM, Moirai, Lag-Llama, Toto

Шесть моделей с разными стратегиями предобучения.

| Модель | Архитектура | Особенность |
|--------|-------------|-------------|
| Chronos[^chronos] | T5 (enc-dec) | Токенизация значений |
| TimeGPT[^timegpt] | Enc-dec | Закрытый API |
| TimesFM[^timesfm] | Decoder-only | Patching + Google scale |
| Moirai[^moirai] | Masked → Decoder | Any-variate, mixture distribution |
| Lag-Llama[^lagllama] | Decoder-only | Open-source, лаги как входы |
| Toto[^toto] | Decoder-only | Domain-specific (observability) |

### Что мы НЕ рассматриваем

**Informer[^informer], Autoformer[^autoformer], FEDformer[^fedformer]** — исторически важные трансформерные архитектуры 2021-2022 годов. Не включаем, потому что DLinear[^dlinear] показал: они уступают даже линейному слою. Современные подходы (PatchTST[^patchtst], iTransformer[^itransformer]) решают их проблемы лучше.

**TFT (Temporal Fusion Transformer)**[^tft] — отличная модель для сценариев с богатыми признаками (десятки ковариат, статические метаданные, known future inputs). Не включаем не потому, что она плоха, а потому что представляет «старую школу» heavy-feature инженерии. TFT требует тщательной подготовки признаков, в то время как современные foundation models стремятся к zero-shot на сырых данных. Если у вас уже есть развитый feature engineering pipeline — TFT остаётся сильным выбором.

**Архитектуры для anomaly detection и classification** — отдельные задачи со своей спецификой, выходящие за рамки книги о прогнозировании.

## Как читать книгу дальше

### Линейный путь

Читайте последовательно:

1. **Часть 1: MLP-based** — как много можно достичь простыми средствами
2. **Часть 2: Transformer-based** — сложности адаптации attention
3. **Часть 3: RNN/SSM-based** — возвращение последовательных архитектур
4. **Часть 4: Foundation models** — как предобучение меняет правила игры

### Выборочный путь

Используйте таблицу ниже как навигатор:

| Ваша ситуация | Рекомендация | Глава |
|---------------|--------------|-------|
| Нужен быстрый baseline | DLinear[^dlinear] | 14 |
| Interpretable прогноз | N-BEATS[^nbeats] (interpretable) | 11 |
| Zero-shot без GPU | Chronos-Bolt[^chronos] | 31 |
| Multivariate с корреляциями | iTransformer[^itransformer] | 16 |
| Probabilistic + production | DeepAR[^deepar] | 23 |
| Observability метрики | Toto[^toto] | 36 |
| Разные частоты данных | FlowState[^flowstate] | 25 |
| Максимальное качество | Сравнение в главе 40 | 40 |

## Сводная таблица моделей

| Модель | Тип блока | Режим | Выход | Год | Params |
|--------|-----------|-------|-------|-----|--------|
| N-BEATS[^nbeats] | MLP | Global | Point | 2020 | ~5M |
| N-HiTS[^nhits] | MLP | Global | Point | 2023 | ~5M |
| DLinear[^dlinear] | Linear | Global | Point | 2023 | <1M |
| TSMixer[^tsmixer] | MLP | Global | Point | 2023 | ~1M |
| PatchTST[^patchtst] | Transformer | Global | Point | 2023 | ~2M |
| iTransformer[^itransformer] | Transformer | Global | Point | 2024 | ~5M |
| DeepAR[^deepar] | LSTM | Global | Prob | 2017 | ~5M |
| Chronos[^chronos] | T5 | Foundation | Prob | 2024 | 8M-710M |
| TimesFM[^timesfm] | Decoder | Foundation | Point/Prob | 2024 | 200M |
| Moirai[^moirai] | Transformer | Foundation | Prob | 2024 | 14M-311M |
| Lag-Llama[^lagllama] | Decoder | Foundation | Prob | 2024 | 8M |
| Toto[^toto] | Decoder | Foundation | Prob | 2025 | 151M |
| TiRex[^tirex] | xLSTM | Foundation | Prob | 2025 | ~50M |
| FlowState[^flowstate] | SSM | Foundation | Prob | 2025 | <10M |

## Выводы

1. **Простота часто побеждает.** DLinear[^dlinear] с одним слоем превосходит сложные трансформеры. N-BEATS[^nbeats] без attention конкурирует с SOTA.

2. **Foundation models меняют парадигму.** Zero-shot прогнозирование становится реальностью, но не везде работает одинаково хорошо[^gifteval].

3. **RNN/SSM возвращаются.** После доминирования трансформеров в 2021-2023 последовательные архитектуры снова показывают SOTA[^xlstm][^mamba].

4. **Probabilistic output — стандарт.** Почти все современные модели выдают распределения, а не точечные прогнозы.

5. **Единого победителя нет.** Выбор модели зависит от задачи, данных и ограничений. Эта книга поможет сделать осознанный выбор.

В следующей части мы начинаем с MLP-based архитектур — семейства моделей, которые показали, что иногда меньше действительно значит больше.

---

## Ссылки

### MLP-based архитектуры

[^nbeats]: Oreshkin, B. N., et al. (2020). N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. *ICLR 2020*. https://arxiv.org/abs/1905.10437

[^nhits]: Challu, C., et al. (2023). N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting. *AAAI 2023*. https://arxiv.org/abs/2201.12886

[^dlinear]: Zeng, A., et al. (2023). Are Transformers Effective for Time Series Forecasting? *AAAI 2023*. https://arxiv.org/abs/2205.13504

[^tsmixer]: Chen, S., et al. (2023). TSMixer: An All-MLP Architecture for Time Series Forecasting. *TMLR*. https://arxiv.org/abs/2303.06053

### Transformer-based архитектуры

[^vaswani]: Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*. https://arxiv.org/abs/1706.03762

[^informer]: Zhou, H., et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. *AAAI 2021 Best Paper*. https://arxiv.org/abs/2012.07436

[^autoformer]: Wu, H., et al. (2021). Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. *NeurIPS 2021*. https://arxiv.org/abs/2106.13008

[^fedformer]: Zhou, T., et al. (2022). FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting. *ICML 2022*. https://arxiv.org/abs/2201.12740

[^patchtst]: Nie, Y., et al. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *ICLR 2023*. https://arxiv.org/abs/2211.14730

[^itransformer]: Liu, Y., et al. (2024). iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. *ICLR 2024*. https://arxiv.org/abs/2310.06625

[^tft]: Lim, B., et al. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748-1764. https://arxiv.org/abs/1912.09363

### RNN/SSM-based архитектуры

[^deepar]: Salinas, D., et al. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. *International Journal of Forecasting*, 36(3), 1181-1191. https://arxiv.org/abs/1704.04110

[^xlstm]: Beck, M., et al. (2024). xLSTM: Extended Long Short-Term Memory. *arXiv*. https://arxiv.org/abs/2405.04517

[^mamba]: Gu, A., & Dao, T. (2024). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *ICLR 2024*. https://arxiv.org/abs/2312.00752

[^s4]: Gu, A., et al. (2022). Efficiently Modeling Long Sequences with Structured State Spaces. *ICLR 2022*. https://arxiv.org/abs/2111.00396

### Foundation models

[^chronos]: Ansari, A. F., et al. (2024). Chronos: Learning the Language of Time Series. *TMLR*. https://arxiv.org/abs/2403.07815

[^timesfm]: Das, A., et al. (2024). A decoder-only foundation model for time-series forecasting. *ICML 2024*. https://arxiv.org/abs/2310.10688

[^moirai]: Woo, G., et al. (2024). Unified Training of Universal Time Series Forecasting Transformers. *ICML 2024*. https://arxiv.org/abs/2402.02592

[^lagllama]: Rasul, K., et al. (2024). Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting. *TMLR*. https://arxiv.org/abs/2310.08278

[^toto]: Cohen, S., et al. (2025). Toto: Time Series Optimized Transformer for Observability. *Datadog*. https://www.datadoghq.com/blog/ai/toto-time-series-foundation-model/

[^timegpt]: Garza, A., & Mergenthaler-Canseco, M. (2023). TimeGPT-1. *arXiv*. https://arxiv.org/abs/2310.03589

[^tirex]: Ekambaram, V., et al. (2025). Scalable Time Series Foundation Model with Recurrence. *IBM Research*. (в процессе публикации)

[^flowstate]: FlowState: Time-Scale Invariant Foundation Model for Time Series. (2025). (в процессе публикации)

### Обзоры и бенчмарки

[^kim2025]: Kim, J., et al. (2025). A comprehensive survey of deep learning for time series forecasting. *Artificial Intelligence Review*, 58, 216. https://doi.org/10.1007/s10462-025-11223-9

[^gifteval]: Aksu, et al. (2024). GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation. https://arxiv.org/abs/2410.10393

[^monash]: Godahewa, R., et al. (2021). Monash Time Series Forecasting Archive. *NeurIPS Datasets and Benchmarks*. https://arxiv.org/abs/2105.06643