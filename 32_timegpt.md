# TimeGPT: foundation model как сервис

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/privettoha/neural-forecast-book/blob/main/notebooks/32_timegpt.ipynb)

TimeGPT ([статья](https://arxiv.org/abs/2310.03589), октябрь 2023) — foundation model от Nixtla, доступная исключительно через API. Это принципиально отличает её от всех моделей, которые мы рассматривали раньше: вы не скачиваете веса, не запускаете inference на своём железе — вместо этого отправляете данные на серверы Nixtla и получаете готовый прогноз[^timegpt-api]

[^timegpt-api]: Nixtla Documentation. "TimeGPT is the first foundation model for time series, providing state-of-the-art forecasting and anomaly detection capabilities." https://www.nixtla.io/docs/introduction/introduction

## Nixtla: компания за TimeGPT

Nixtla — не случайный стартап на хайпе foundation models, а команда с глубоким пониманием домена и проверенным track record в open-source[^nixtla-oss]:

[^nixtla-oss]: GitHub: Nixtla. https://github.com/Nixtla

➖ **[StatsForecast](https://nixtlaverse.nixtla.io/statsforecast/)** — молниеносные реализации классических методов (ARIMA, ETS, Theta)
➖ **[NeuralForecast](https://nixtlaverse.nixtla.io/neuralforecast/)** — нейросетевые модели (N-BEATS, N-HiTS, PatchTST)
➖ **[MLForecast](https://nixtlaverse.nixtla.io/mlforecast/)** — feature engineering для gradient boosting
➖ **[HierarchicalForecast](https://nixtlaverse.nixtla.io/hierarchicalforecast/)** — согласование иерархических прогнозов

TimeGPT — коммерческая сторона компании. И теперь Nixtla предлагает доступ не только к своей модели, но и к другим ведущим foundation models (Chronos, TiRex и др.) через единый API[^nixtla-multimodel]

[^nixtla-multimodel]: GitHub: Nixtla/nixtla. "NixtlaClient may be used to access services powered by technology from Google, Amazon, IBM, Datadog, and NXAI." https://github.com/Nixtla/nixtla

## Семейство TimeGPT

Nixtla развивает несколько поколений модели[^timegpt-versions]:

[^timegpt-versions]: Nixtla Blog. "TimeGPT 2: Next Generation Time Series Model." https://www.nixtla.io/blog/timegpt-2-announcement

🔢 **TimeGPT-1** (октябрь 2023)
➖ Первый релиз, encoder-decoder transformer
➖ Обучен на 100+ миллиардах точек данных[^timegpt-data]
➖ Zero-shot inference

[^timegpt-data]: Garza, A., et al. "TimeGPT-1." arXiv:2310.03589. https://arxiv.org/abs/2310.03589

🔢 **TimeGPT-2** (2025, private preview)
Модульное семейство с тремя вариантами[^timegpt2]:
➖ **timegpt-2-mini** — быстрый inference, resource-constrained environments
➖ **timegpt-2** — баланс между compute cost и accuracy
➖ **timegpt-2-pro** — максимальная точность для short и long horizons

[^timegpt2]: Nixtla Blog. "TimeGPT-2: Next Generation Time Series Model." https://www.nixtla.io/blog/timegpt-2-announcement

Ключевые улучшения TimeGPT-2:
➖ До **60% улучшения точности** vs TimeGPT-1
➖ Privacy-first approach
➖ Self-hosted и on-premises deployments

🔢 **TimeGPT 2.1** (2025, private preview)
Первая **multivariate** модель в семействе TimeGPT[^timegpt21]:
➖ Поддержка exogenous variables
➖ Cross-learned forecasts
➖ Self-hosted deployments

[^timegpt21]: Nixtla Blog. "TimeGPT 2.1: Next-Gen Time Series Forecasting." https://www.nixtla.io/blog/timegpt-2-1-announcement

🔢 **TimeGEN-1** (Azure)
Версия TimeGPT, оптимизированная для Azure AI[^timegen]:
➖ Model-as-a-Service через Azure AI Model Catalog
➖ Первый cloud provider с foundation model для временных рядов
➖ Microsoft Build 2024

[^timegen]: Microsoft Tech Community. "Announcing TimeGEN-1 in Azure AI." June 2024. https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/announcing-timegen-1-in-azure-ai-leap-forward-in-time-series-forecasting/4140446

## Архитектура

TimeGPT использует encoder-decoder трансформер, концептуально похожий на оригинальный [«Attention Is All You Need»](https://arxiv.org/abs/1706.03762)[^timegpt-arch]:

[^timegpt-arch]: Garza, A., et al. "TimeGPT-1." arXiv:2310.03589. Section 3: "TimeGPT is a Transformer-based time series model with self-attention mechanisms." https://arxiv.org/abs/2310.03589

```
Input: историческое окно временного ряда
    ↓
Local positional encoding
    ↓
Encoder:
  - Multiple layers
  - Self-attention
  - Residual connections
  - Layer normalization
    ↓
Decoder:
  - Cross-attention на encoder output
  - Residual connections
  - Layer normalization
    ↓
Linear layer → forecasting window
```

Конкретные детали (количество слоёв, размерность, число голов) Nixtla не раскрывает — это часть IP защиты[^timegpt-closed]

[^timegpt-closed]: GitHub: Nixtla/nixtla. "TimeGPT is closed source. However, this SDK is open source and available under the Apache 2.0 License." https://github.com/Nixtla/nixtla

## Conformal Prediction для интервалов

TimeGPT использует **conformal prediction** для доверительных интервалов — метод с теоретическими гарантиями[^timegpt-conformal]:

[^timegpt-conformal]: Nixtla Documentation. "Prediction intervals." https://docs.nixtla.io/

🔢 **Как работает:**
1. Делаем прогнозы на калибровочном наборе, собираем ошибки: $r_i = |y_i - \hat{y}_i|$
2. Для уровня $(1-\alpha)$ интервал: $[\hat{y} - q_{1-\alpha}, \hat{y} + q_{1-\alpha}]$
3. $q_{1-\alpha}$ — квантиль эмпирического распределения ошибок

Гарантия: если данные exchangeable, интервал покроет истинное значение с вероятностью ≥ $(1-\alpha)$

## Данные обучения

TimeGPT обучен на **100+ миллиардах точек данных** из разнообразных источников[^timegpt-training]:
➖ Финансы
➖ Транспорт
➖ Банки
➖ Веб-трафик
➖ Погода
➖ Энергетика
➖ Healthcare

[^timegpt-training]: Liao, W., et al. "TimeGPT in Load Forecasting." 2024. "TimeGPT is trained on massive and diverse time series datasets consisting of 100 billion data points." https://arxiv.org/abs/2404.04885

Конкретные датасеты не называются — типично для коммерческих моделей

## Что умеет

➖ **Zero-shot inference** — прогнозы без обучения на ваших данных[^timegpt-zeroshot]
➖ **Fine-tuning через API** — адаптация к вашему домену
➖ **Exogenous variables** — внешние факторы
➖ **Anomaly detection** — обнаружение аномалий
➖ **Multiple series** — одновременный прогноз нескольких рядов
➖ **Multivariate** (TimeGPT 2.1) — cross-learned forecasts
➖ **Custom loss functions** — выбор функции потерь при fine-tuning

[^timegpt-zeroshot]: Nixtla About. "TimeGPT's zero-shot inference capabilities outperform existing approaches." https://nixtlaverse.nixtla.io/nixtla/docs/getting-started/introduction.html

## Когда использовать

👍 **Хорошо работает:**
➖ Нет ML-инфраструктуры и нет желания её строить
➖ Нужен быстрый результат — прототип за день
➖ В команде нет глубокой ML-экспертизы
➖ Умеренные объёмы (тысячи, не миллионы рядов)
➖ Данные не конфиденциальны
➖ Важна поддержка ковариат из коробки

👎 **Проблемы:**
➖ **Данные конфиденциальны** — уходят на серверы Nixtla (кроме self-hosted TimeGPT-2)
➖ **Нужен офлайн-режим** — зависимость от внешнего сервиса
➖ **Большие объёмы** — стоимость масштабируется
➖ **Критична латентность <100ms** — сетевой round-trip
➖ **Нужен полный контроль** — модель закрыта

## TimeGPT vs открытые модели

| Критерий | TimeGPT | Chronos-2 | TiRex | Moirai |
|----------|---------|-----------|-------|--------|
| Доступ | Только API* | Открытые веса | Открытые веса | Открытые веса |
| Инфраструктура | Не нужна | Нужна | Нужна (GPU) | Нужна (GPU) |
| Контроль | Минимальный | Полный | Полный | Полный |
| Приватность | Данные уходят* | Полная | Полная | Полная |
| Ковариаты | ✓ | ✓ | ✗ | ~ |
| Multivariate | ✓ (2.1) | ✓ | ✗ | ✓ |
| Fine-tuning | ✓ API | ✗ | По запросу | ✓ |
| Офлайн | ✗* | ✓ | ✓ | ✓ |

*TimeGPT-2 поддерживает self-hosted deployments

## Реализации

| Ресурс | Ссылка |
|--------|--------|
| Python SDK | [github.com/Nixtla/nixtla](https://github.com/Nixtla/nixtla) |
| R SDK | [nixtlar](https://nixtla.github.io/nixtlar/) |
| Документация | [docs.nixtla.io](https://docs.nixtla.io/) |
| Dashboard (API keys) | [dashboard.nixtla.io](https://dashboard.nixtla.io/) |
| Azure AI (TimeGEN-1) | [Azure AI Model Catalog](https://ai.azure.com/) |

## Стоимость

➖ **Free tier** — ограниченное количество вызовов для экспериментов
➖ **Платные планы** — по количеству прогнозируемых точек
➖ **Enterprise** — индивидуальные условия, SLA, self-hosted

При оценке важно учитывать не только прямые расходы, но и косвенные: время разработчиков на настройку инфраструктуры может превысить стоимость API

## Что дальше

TimeGPT демонстрирует, что foundation models могут существовать в форме сервиса — вы отдаёте контроль и (частично) приватность в обмен на простоту и скорость выхода на рынок. TimeGPT-2 с self-hosted deployments размывает эту границу.

В следующем посте мы рассмотрим TimesFM от Google — открытую foundation model с 200M параметрами, обученную на Google Trends, публичных датасетах и синтетических данных.

:::{seealso}
**Источники:**
- Garza, A., Challu, C., Mergenthaler-Canseco, M. (2023). [TimeGPT-1](https://arxiv.org/abs/2310.03589). arXiv
- [Nixtla TimeGPT Documentation](https://docs.nixtla.io/)
- [Nixtla GitHub](https://github.com/Nixtla/nixtla)
- [TimeGPT-2 Announcement](https://www.nixtla.io/blog/timegpt-2-announcement)
- [TimeGPT 2.1 Announcement](https://www.nixtla.io/blog/timegpt-2-1-announcement)
- [TimeGEN-1 in Azure AI](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/announcing-timegen-1-in-azure-ai-leap-forward-in-time-series-forecasting/4140446)
:::