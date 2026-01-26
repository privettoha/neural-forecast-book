# DeepAR. Вероятностное прогнозирование с RNN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/privettoha/neural-forecast-book/blob/main/notebooks/23_deepar.ipynb)

DeepAR ([статья](https://arxiv.org/abs/1704.04110), International Journal of Forecasting 2020) — модель от Amazon, которая в 2017 году изменила представление о том, как нейросети могут прогнозировать временные ряды. До DeepAR глубокое обучение для рядов означало «обучить LSTM на одном ряде и надеяться на лучшее». DeepAR предложил три идеи, ставшие стандартом: глобальная модель на множестве рядов, авторегрессионная генерация прогноза и вероятностный выход вместо точечного[^deepar-core]

[^deepar-core]: Salinas, D., Flunkert, V., Gasthaus, J., Januschowski, T. "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks." International Journal of Forecasting, 2020. Section 1. https://arxiv.org/abs/1704.04110

## Идея

Ключевые отличия от классических методов и ранних нейросетей — разберём подробнее

**Глобальная модель** — вместо 10 000 моделей для 10 000 товаров обучаем одну RNN на всех рядах. Модель не запоминает конкретные ряды, а учится распознавать паттерны: как выглядит сезонный пик, как ведёт себя тренд, как ряд реагирует на праздники[^deepar-global]. Во время инференса модель получает историю нового ряда — возможно, такого, которого не было в обучении — и генерирует прогноз, опираясь на выученные паттерны

[^deepar-global]: Salinas, D., et al. "DeepAR." IJF 2020. Section 3: "We learn a global model from historical data of all time series in the data set." https://arxiv.org/abs/1704.04110

**Авторегрессионная генерация** — модель генерирует прогноз шаг за шагом: предсказывает следующий шаг, использует это предсказание как вход для следующего, и так далее до конца горизонта[^deepar-ar]

[^deepar-ar]: Salinas, D., et al. "DeepAR." IJF 2020. Section 3.1. https://arxiv.org/abs/1704.04110

**Вероятностный выход** — вместо точечного прогноза «продажи завтра = 150» выдаёт распределение: «продажи распределены с параметрами μ=150, σ=20». Из этого распределения получаем точечный прогноз, доверительные интервалы и полное распределение для оценки рисков[^deepar-prob]

[^deepar-prob]: Salinas, D., et al. "DeepAR." IJF 2020. Section 3.2: "Probabilistic forecasts enable optimal decision making under uncertainty by minimizing risk functions." https://arxiv.org/abs/1704.04110

## Контекст: что было до DeepAR

➖ **Классические методы** (ARIMA, ETS, Prophet) работали с каждым рядом независимо — для новых товаров без истории модели нет, для товаров с короткой историей модель переобучается

➖ **Ранние нейросети** (vanilla LSTM, GRU) обычно обучались на отдельных рядах (локальный подход) и выдавали точечный прогноз без оценки неопределённости

➖ **Cold start проблема** — новый товар, магазин, сервис не имеет истории для обучения модели

DeepAR решил все три проблемы одним архитектурным выбором и показал улучшение около 15% по сравнению с state-of-the-art методами того времени[^deepar-results]

[^deepar-results]: Salinas, D., et al. "DeepAR." IJF 2020. Abstract: "Accuracy improvements of around 15% compared to state-of-the-art methods." https://arxiv.org/abs/1704.04110

## Архитектура

Модель строится на LSTM с несколькими ключевыми компонентами:

🔢 **Входы на каждом шаге t**
➖ $y_{t-1}$: предыдущее значение (или 0 для первого шага)
➖ $x_t$: динамические ковариаты (день недели, праздник, цена)
➖ $e$: embedding статических признаков (категория товара, регион)

Формально:
$$h_t = \text{LSTM}(h_{t-1}, [y_{t-1}, x_t, e])$$
$$\theta_t = \text{Linear}(h_t)$$
$$y_t \sim p(y | \theta_t)$$

где $h_t$ — скрытое состояние, $\theta_t$ — параметры выходного распределения[^deepar-arch]

[^deepar-arch]: Salinas, D., et al. "DeepAR." IJF 2020. Section 3.1, Equation 1-3. https://arxiv.org/abs/1704.04110

🔢 **Масштабирование входов**
Временные ряды имеют разный масштаб — один в единицах, другой в миллионах. DeepAR нормализует каждый ряд на его среднее значение в context window[^deepar-scaling]. Авторы отмечают, что распределение масштабов следует степенному закону (power law) — это наблюдение на данных Amazon с миллионами товаров[^deepar-powerlaw]

[^deepar-scaling]: Salinas, D., et al. "DeepAR." IJF 2020. Section 3.3: "We scale the time series by the average over the conditioning range." https://arxiv.org/abs/1704.04110
[^deepar-powerlaw]: Salinas, D., et al. "DeepAR." IJF 2020. Figure 1: "The distribution is over a few orders of magnitude an approximate power-law." https://arxiv.org/abs/1704.04110

🔢 **Lagged features**
Помимо $y_{t-1}$, DeepAR может использовать значения с большим лагом: $y_{t-7}$ (неделю назад), $y_{t-30}$ (месяц назад). Это помогает модели видеть сезонные паттерны даже если RNN «забыла» далёкое прошлое[^deepar-lags]

[^deepar-lags]: Salinas, D., et al. "DeepAR." IJF 2020. Section 3.1: "Lagged observations as covariates help capture seasonality." https://arxiv.org/abs/1704.04110

🔢 **Выходные распределения**
DeepAR поддерживает несколько семейств:
➖ **Gaussian** — для непрерывных данных с симметричным шумом. Параметры: μ, σ
➖ **Negative Binomial** — для count data (количество заказов, число событий). Поддерживает overdispersion[^deepar-negbin]
➖ **Student-t** — для данных с тяжёлыми хвостами (выбросами). Параметры: μ, σ, ν

[^deepar-negbin]: Salinas, D., et al. "DeepAR." IJF 2020. Table 1: "DeepAR performs significantly better... The results also show the importance of modeling these data sets with a count distribution, as rnn-gaussian performs significantly worse." https://arxiv.org/abs/1704.04110

🔢 **Важно**: выбор распределения — это гиперпараметр. Для ритейла с целочисленными продажами обычно Negative Binomial. Для непрерывных метрик — Gaussian или Student-t

## Ковариаты

DeepAR умеет использовать дополнительные признаки, что отличает его от многих современных архитектур (N-BEATS в базовой версии ковариаты не поддерживает)[^deepar-covariates]:

[^deepar-covariates]: Salinas, D., et al. "DeepAR." IJF 2020. Section 3.1: "Time-varying covariates and static metadata can be incorporated." https://arxiv.org/abs/1704.04110

➖ **Статические признаки** — характеристики ряда, не меняющиеся во времени: категория товара, регион магазина, тип клиента. Кодируются через embedding

➖ **Динамические признаки** — признаки, меняющиеся во времени: день недели, месяц, праздник, цена, промо-флаг

🔢 **Важное требование**: динамические признаки должны быть известны на горизонте прогноза. День недели известен заранее, цена может быть запланирована, но фактическая погода на следующую неделю — нет

## Обучение и инференс

🔢 **Обучение**
Для каждого ряда случайно выбирается окно: context + prediction. Модель прогоняется через context (teacher forcing — используются истинные значения), затем через prediction. Loss — negative log-likelihood:
$$\mathcal{L} = -\sum_{t} \log p(y_t | \theta_t)$$

🔢 **Инференс (Monte Carlo sampling)**
Модель сэмплирует из предсказанного распределения, использует сэмпл как вход для следующего шага. Процесс повторяется несколько раз (например, 100 сэмплов), и из множества траекторий строятся квантили[^deepar-mc]

[^deepar-mc]: Salinas, D., et al. "DeepAR." IJF 2020. Section 3.4: "We draw sample paths from the joint predictive distribution." https://arxiv.org/abs/1704.04110

## Экспериментальные результаты

Авторы тестируют на нескольких датасетах: electricity, traffic, parts, ec (Amazon e-commerce)[^deepar-datasets]:

[^deepar-datasets]: Salinas, D., et al. "DeepAR." IJF 2020. Section 4.1, Table 3. https://arxiv.org/abs/1704.04110

➖ DeepAR значительно превосходит все другие методы на этих датасетах (Table 1)
➖ Важность count distribution: rnn-gaussian показывает значительно худшие результаты на count data
➖ Без масштабирования и weighted sampling точность падает на датасетах с power-law распределением (ec, ec-sub)
➖ На датасете parts, где нет power-law поведения, rnn-negbin показывает результаты близкие к DeepAR

## Что умеет

➖ **Вероятностный выход из коробки** — не нужно отдельно обучать модель для квантилей или строить conformal prediction поверх точечного прогноза
➖ **Работа с ковариатами** — праздники, промо, цены можно подать на вход и получить условный прогноз
➖ **Гибкость горизонта** — авторегрессионная генерация позволяет делать прогноз на любой горизонт без переобучения
➖ **Холодный старт** — новый ряд с минимальной историей получает разумный прогноз благодаря глобальной модели[^deepar-coldstart]

[^deepar-coldstart]: Salinas, D., et al. "DeepAR." IJF 2020. Section 1: "Learn global model that allows predicting new time series from scratch." https://arxiv.org/abs/1704.04110

## Когда использовать

👍 **Хорошо работает:**
➖ Нужен вероятностный прогноз с ковариатами
➖ Есть динамические признаки, известные на горизонте прогноза (праздники, промо, запланированные цены)
➖ Горизонт прогноза относительно короткий (до 30 шагов)
➖ Важна зрелость и поддержка (Amazon SageMaker, GluonTS)
➖ Count data с overdispersion (Negative Binomial)

👎 **Проблемы:**
➖ **Накопление ошибки** — авторегрессия означает, что ошибка на шаге 1 влияет на шаг 2, на длинных горизонтах прогноз может «уплыть»
➖ **Скорость инференса** — генерация последовательная + Monte Carlo sampling требует многократных прогонов
➖ **Чувствительность к гиперпараметрам** — количество слоёв, размер скрытого состояния, выбор распределения, лаги[^deepar-hyper]
➖ **Возраст архитектуры** — LSTM из 1997 года, появились более эффективные последовательные архитектуры (xLSTM, SSM)

[^deepar-hyper]: OpenReview. "A Worrying Analysis of Probabilistic Time-series Models for Sales Forecasting." NeurIPS 2021 Workshop. https://openreview.net/pdf?id=D7YBmfX_VQy

## DeepAR vs современные модели

| Критерий | DeepAR | N-HiTS | Foundation models |
|----------|--------|--------|-------------------|
| Вероятностный выход | ✓ Нативно | ~ Через quantile loss | ✓ Большинство |
| Ковариаты | ✓ Статические + динамические | ~ Только N-BEATSx | ~ Зависит от модели |
| Длинные горизонты | ✗ Накопление ошибки | ✓ Direct подход | ✓ Direct подход |
| Скорость инференса | ✗ Последовательно | ✓ Параллельно | ✓ Параллельно |
| Cold start | ✓ Глобальная модель | ✓ Глобальная модель | ✓✓ Zero-shot |
| Зрелость | ✓ 7+ лет в продакшене | ✓ 3+ года | ~ 1-2 года |

## Реализации

| Библиотека | Особенности |
|------------|-------------|
| [GluonTS](https://ts.gluon.ai/) | Оригинальная реализация от Amazon, MXNet и PyTorch backends[^gluonts] |
| [Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html) | Managed service, production-ready |
| [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.deepar.DeepAR.html) | PyTorch Lightning интеграция |
| [neuralforecast](https://github.com/Nixtla/neuralforecast) | Единый API с другими моделями |

[^gluonts]: Alexandrov, A., et al. "GluonTS: Probabilistic Time Series Modeling in Python." JMLR 2020. https://ts.gluon.ai/

## Что дальше

DeepAR показал, что RNN могут работать для временных рядов при правильном подходе: глобальное обучение, вероятностный выход, ковариаты. Но классические LSTM имеют известные ограничения: затухание градиентов, сложность параллелизации, ограниченная память.

В следующих постах мы рассмотрим, как современные последовательные архитектуры решают эти проблемы. [TiRex](https://arxiv.org/abs/2503.08656) использует [xLSTM](https://arxiv.org/abs/2405.04517) — расширенную версию LSTM с экспоненциальными гейтами и матричной памятью. FlowState строится на State Space Models, которые сочетают преимущества RNN и свёрточных сетей. Оба показывают state-of-the-art результаты в 2025 году, доказывая, что последовательные архитектуры рано списывать со счетов.

:::{seealso}
**Источники:**
- Salinas, D., Flunkert, V., Gasthaus, J., Januschowski, T. (2020). [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://arxiv.org/abs/1704.04110). International Journal of Forecasting
- [GluonTS: Probabilistic Time Series Modeling](https://ts.gluon.ai/) — Amazon AWS
- [PyTorch Forecasting: DeepAR](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.deepar.DeepAR.html)
- [AWS SageMaker DeepAR](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html)
:::