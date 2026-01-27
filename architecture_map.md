## Карта архитектур

Прежде чем погружаться в конкретные модели, полезно увидеть общую картину: какие подходы существуют и как они соотносятся друг с другом.

---

### Четыре оси классификации

#### Ось 1: тип вычислительного блока

| Тип | Как работает | Примеры |
|-----|--------------|---------|
| **Linear** | Линейная проекция входного окна на горизонт | DLinear[^dlinear], NLinear |
| **MLP-based** | Fully connected слои, окно истории → прогноз | N-BEATS[^nbeats], N-HiTS[^nhits], TSMixer[^tsmixer] |
| **CNN-based** | Свёртки для извлечения локальных паттернов | TimesNet[^timesnet] |
| **RNN/SSM-based** | Последовательная обработка с «памятью» | DeepAR[^deepar], TiRex (xLSTM), FlowState (SSM) |
| **Transformer-based** | Self-attention между временными точками | PatchTST[^patchtst], iTransformer[^itransformer], Moirai[^moirai] |
| **Decoder-only (LLM-style)** | Авторегрессионная генерация, causal attention | Chronos[^chronos], TimeGPT[^timegpt], TimesFM[^timesfm], Lag-Llama[^lagllama] |

[^dlinear]: Zeng, A., et al. "Are Transformers Effective for Time Series Forecasting?" AAAI 2023. https://arxiv.org/abs/2205.13504
[^nbeats]: Oreshkin, B., et al. "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting." ICLR 2020. https://arxiv.org/abs/1905.10437
[^nhits]: Challu, C., et al. "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting." AAAI 2023. https://arxiv.org/abs/2201.12886
[^tsmixer]: Chen, S., et al. "TSMixer: An All-MLP Architecture for Time Series Forecasting." TMLR, 2023. https://arxiv.org/abs/2303.06053
[^timesnet]: Wu, H., et al. "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis." ICLR 2023. https://arxiv.org/abs/2210.02186
[^deepar]: Salinas, D., et al. "DeepAR: Probabilistic forecasting with autoregressive recurrent networks." IJF, 2020. https://arxiv.org/abs/1704.04110
[^patchtst]: Nie, Y., et al. "A Time Series is Worth 64 Words." ICLR 2023. https://arxiv.org/abs/2211.14730
[^itransformer]: Liu, Y., et al. "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting." ICLR 2024. https://arxiv.org/abs/2310.06625
[^moirai]: Woo, G., et al. "Unified Training of Universal Time Series Forecasting Transformers." ICML, 2024. https://arxiv.org/abs/2402.02592
[^chronos]: Ansari, A., et al. "Chronos: Learning the Language of Time Series." TMLR, 2024. https://arxiv.org/abs/2403.07815
[^timegpt]: Garza, A., et al. "TimeGPT-1." 2023. https://arxiv.org/abs/2310.03589
[^timesfm]: Das, A., et al. "A decoder-only foundation model for time-series forecasting." ICML, 2024. https://arxiv.org/abs/2310.10688
[^lagllama]: Rasul, K., et al. "Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting." 2023. https://arxiv.org/abs/2310.08278

#### Ось 2: режим обучения

| Режим | Описание | Когда использовать |
|-------|----------|-------------------|
| **Local** | Отдельная модель для каждого ряда | Мало рядов, каждый уникален |
| **Global** | Одна модель на все ряды, ряды обрабатываются независимо, веса общие | Много рядов, общие паттерны |
| **Multivariate** | Одна модель, признаки разных рядов объединяются, зависимости моделируются явно | Сильные корреляции между рядами |
| **Foundation** | Предобучение на миллионах рядов, zero-shot или fine-tuning | Cold start, нет ресурсов на обучение |

Global часто работает лучше Multivariate — модель получает больше примеров при тех же данных[^tsururu].

[^tsururu]: Sber AI Lab. "Tsururu: Time Series Forecasting Framework." https://github.com/sb-ai-lab/tsururu

#### Ось 3: тип выхода

| Тип | Что выдаёт | Примеры |
|-----|------------|---------|
| **Point forecast** | Одно число на момент | N-BEATS, DLinear, TimesFM |
| **Probabilistic forecast** | Распределение, квантили или samples | DeepAR, Chronos, TiRex, Lag-Llama, Toto |

#### Ось 4: стратегия прогнозирования

| Стратегия | Как работает | Плюсы | Минусы |
|-----------|--------------|-------|--------|
| **Recursive** | Предсказываем t+1, используем как вход для t+2 | Любой горизонт | Накопление ошибок |
| **MIMO** | Один forward pass → все H значений | Нет накопления, быстро | Фиксированный горизонт |
| **Direct** | Отдельная модель для каждого шага | Нет накопления | H моделей |

Влияние стратегии на качество при разном уровне шума[^forecasting-strategies]:

[^forecasting-strategies]: Taieb, S.B., et al. "A review and comparison of strategies for multi-step ahead time series forecasting." Expert Systems with Applications, 2012. https://doi.org/10.1016/j.eswa.2012.01.039

| Шум (std) | Recursive MAE | MIMO MAE |
|-----------|---------------|----------|
| 0.1 | 0.028 | 0.041 |
| 0.7 | 0.463 | 0.320 |
| 1.0 | 0.638 | 0.498 |

---

### Визуальная карта моделей

```
                    │ Global         │ Foundation       │ Стратегия
────────────────────┼────────────────┼──────────────────┼───────────
Linear              │ DLinear        │                  │ MIMO
────────────────────┼────────────────┼──────────────────┼───────────
MLP-based           │ N-BEATS        │                  │ MIMO
                    │ N-HiTS         │                  │ MIMO
                    │ TSMixer        │                  │ MIMO
────────────────────┼────────────────┼──────────────────┼───────────
CNN-based           │ TimesNet       │                  │ MIMO
────────────────────┼────────────────┼──────────────────┼───────────
RNN/SSM-based       │ DeepAR         │ TiRex (xLSTM)    │ Recursive
                    │                │ FlowState (SSM)  │ Recursive
────────────────────┼────────────────┼──────────────────┼───────────
Transformer-based   │ PatchTST       │ Moirai           │ MIMO
                    │ iTransformer   │ Toto             │ MIMO
────────────────────┼────────────────┼──────────────────┼───────────
Decoder-only        │                │ Chronos          │ Recursive
(LLM-style)         │                │ TimeGPT          │ Recursive
                    │                │ TimesFM          │ Recursive
                    │                │ Lag-Llama        │ Recursive
```

---

### Что рассматриваем

**Linear: DLinear**

Критический бейзлайн. Простая линейная проекция, которая побеждает сложные трансформеры на многих бенчмарках[^dlinear]. Включён для калибровки ожиданий.

**MLP-based: N-BEATS, N-HiTS, TSMixer**

N-BEATS — basis expansion с интерпретируемой декомпозицией. N-HiTS — иерархическая обработка масштабов, эффективнее на длинных горизонтах. TSMixer — mixing операции от Google, первая MLP-модель на уровне SOTA multivariate.

**CNN-based: TimesNet**

Конвертирует 1D ряд в 2D представление для захвата multi-periodicity. Универсальная архитектура для forecasting, classification, anomaly detection[^timesnet].

**Transformer-based: PatchTST, iTransformer**

PatchTST — патчинг вместо point-wise attention. iTransformer — attention по переменным, не по времени. Informer, Autoformer, FEDformer не включаем — уступают более простым альтернативам[^dlinear].

**RNN/SSM-based: DeepAR, TiRex, FlowState**

DeepAR — классика глобальных вероятностных моделей. TiRex (2025) — xLSTM[^xlstm] конкурирует с трансформерами. FlowState (2025) — State Space Models с time-scale invariance.

[^xlstm]: Beck, M., et al. "xLSTM: Extended Long Short-Term Memory." 2024. https://arxiv.org/abs/2405.04517

**Foundation models: Chronos, TimeGPT, TimesFM, Moirai, Lag-Llama, Toto**

| Модель | Архитектура | Открытость | Особенность |
|--------|-------------|------------|-------------|
| Chronos | T5 + квантизация | Открытая | Ряд как текст |
| TimeGPT | Encoder-decoder | Закрытая (API) | Первая коммерческая FM |
| TimesFM | Decoder-only + патчинг | Открытая | 200M параметров |
| Moirai | Transformer + MoE | Открытая | Any-variate |
| Lag-Llama | LLaMA + лаги | Полностью открытая | Веса, код, данные |
| Toto[^toto] | Transformer | Открытая | Domain-specific (observability) |

[^toto]: Datadog. "Toto: A Time Series Foundation Model for Observability." 2025. https://www.datadoghq.com/blog/ai-forecasting/

---

### Эмпирические наблюдения

| Фактор | Лучший вариант | Комментарий |
|--------|----------------|-------------|
| Datetime features | Без них | Добавление часто ухудшает результат |
| ID features | С ними | Помогает global моделям |
| Режим | Global | Лучше чем Multivariate |
| Стратегия | MIMO | Лучший median MAE |

---

### Инструменты

| Фреймворк | Фокус | Ссылка |
|-----------|-------|--------|
| Tsururu | Эксперименты со стратегиями | [GitHub](https://github.com/sb-ai-lab/tsururu) |
| NeuralForecast | Unified API | [Docs](https://nixtlaverse.nixtla.io/neuralforecast/) |
| GluonTS | Probabilistic forecasting | [Docs](https://ts.gluon.ai/) |

