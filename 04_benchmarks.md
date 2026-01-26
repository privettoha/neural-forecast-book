## Как оценивать модели

Когда в статье написано «наша модель превосходит SOTA на 15%», возникают вопросы: на каком датасете? с какими бейзлайнами? по какой метрике? Без понимания методологии невозможно критически оценивать результаты.

---

### Уроки M-соревнований

M-соревнования — серия открытых турниров по прогнозированию, определяющих, какие методы работают на практике[^m-history].

[^m-history]: Makridakis, S. "The M Competitions: Background and Perspectives." IJF, 2020.

**M4 (2018):** 100,000 рядов, 61 метод[^m4].

[^m4]: Makridakis, S., et al. "The M4 Competition: 100,000 time series and 61 forecasting methods." IJF, 2020. https://doi.org/10.1016/j.ijforecast.2019.04.014

- 12 из 17 лучших методов — комбинации статистических подходов
- Все 6 чистых ML-методов хуже комбинационного бенчмарка
- Победитель (Smyl, Uber) — гибрид ETS + RNN, +10% к бенчмарку

**M5 (2020):** 42,000 рядов Walmart с экзогенными переменными[^m5].

[^m5]: Makridakis, S., et al. "The M5 Accuracy Competition: Results, Findings and Conclusions." IJF, 2022. https://doi.org/10.1016/j.ijforecast.2021.01.006

- Все топ-50 — чистый ML (LightGBM, нейросети)
- Победитель: +22.4% к бенчмарку

**Вывод:** ML побеждает при наличии экзогенных переменных и иерархической структуры. На разнородных univariate рядах статистика сильнее.

---

### Проблемы современных бенчмарков

#### Drop Last trick

При тестировании данные разбиваются на batch'и. Неполный последний batch часто отбрасывается. Проблема: при разных batch_size отбрасываются разные примеры[^tfb].

[^tfb]: Qiu, X., et al. "TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods." PVLDB, 2024. Section 4.3. https://arxiv.org/abs/2403.20150

Пример: ETTh2, тест 2880 точек, горизонт 336, окно 512.
- batch_size=32 → отбрасывается 17 примеров
- batch_size=64 → отбрасывается 49 примеров  
- batch_size=128 → отбрасывается 113 примеров

Модели сравниваются на разных подмножествах данных.

**Решение:** `drop_last=False` при тестировании.

#### Единые гиперпараметры для всех методов

Многие бенчмарки используют одинаковые настройки (lookback window, learning rate) для всех моделей. Это несправедливо: разные архитектуры имеют разные оптимальные параметры[^sota-variability].

[^sota-variability]: Bai, J., et al. "The Unreasonable Effectiveness of Hyperparameter Search for Time Series." arXiv, 2023. https://arxiv.org/abs/2310.06119

#### Узкое покрытие доменов

«Стандартные» датасеты deep learning — преимущественно traffic и electricity. Это специфические домены с сильной сезонностью. Модель, работающая на electricity, может провалиться на финансовых данных.

#### Отсутствие классических бейзлайнов

В статьях по deep learning редко сравнивают с AutoTheta, AutoETS, качественно настроенным ARIMA. Бейзлайн — обычно Seasonal Naive или предыдущие нейросети.

**Пример: Tourism Monthly**

| Модель | MAE |
|--------|-----|
| AutoTheta | 2484 |
| AutoETS | 2573 |
| Chronos Bolt (small) | 2610 |
| PatchTST | 2912 |
| TimesNet | 3009 |
| TimesFM | 3192 |
| SeasonalNaive | 3812 |
| LLMTime Qwen 3b | 4539 |

*Источник: [Kaggle TS Demo Notebook](https://www.kaggle.com/code/simakov/ts-demo-notebook)*

Классические статистические методы побеждают все нейросети на этом датасете.

---

### Современные бенчмарки

| Бенчмарк | Фокус | Характеристики | Ссылка |
|----------|-------|----------------|--------|
| **Monash Archive** | Univariate | 25+ датасетов, 13 baseline-методов, золотой стандарт | [forecastingdata.org](https://forecastingdata.org/) |
| **TFB** | Comprehensive | 10 доменов, устраняет Drop Last, включает статистические методы | [GitHub](https://github.com/decisionintelligence/TFB) |
| **GIFT-Eval** | Foundation models | 144K рядов, zero-shot оценка, non-leaking pretraining data | [GitHub](https://github.com/SalesforceAIResearch/gift-eval) |

---

### Метрики

#### Point forecast

| Метрика | Формула | Когда использовать |
|---------|---------|-------------------|
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Сравнение на одном датасете |
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Штраф за большие ошибки |
| **MAPE** | $\frac{100}{n}\sum\|\frac{y_i - \hat{y}_i}{y_i}\|$ | Бизнес-отчётность (не работает при $y_i=0$) |
| **sMAPE** | $\frac{100}{n}\sum\frac{\|y_i - \hat{y}_i\|}{(\|y_i\|+\|\hat{y}_i\|)/2}$ | Совместимость с M3/M4 |
| **MASE** | MAE / MAE(seasonal naive на train) | Сравнение между датасетами[^mase] |

[^mase]: Hyndman, R.J. & Koehler, A.B. "Another look at measures of forecast accuracy." IJF, 2006. https://doi.org/10.1016/j.ijforecast.2006.03.001

#### Probabilistic forecast

| Метрика | Что измеряет |
|---------|--------------|
| **CRPS** | Качество всего распределения (обобщение MAE) |
| **Quantile Loss** | Качество конкретного квантиля |
| **Coverage** | Доля значений в предсказанном интервале |

---

### Стратегии оценки

**Fixed horizon:** один прогноз на весь тестовый период от конца обучающей выборки.

**Rolling forecast:** последовательные прогнозы со сдвигом на шаг. Реалистичнее, но дороже вычислительно.

**Time series CV:** expanding или sliding window. Реализация: `sklearn.model_selection.TimeSeriesSplit`.

---

### Чек-лист для чтения статей

**Датасеты:**
- [ ] Разнообразие доменов (не только ETT/Electricity)?
- [ ] Нет утечки между train и test?

**Бейзлайны:**
- [ ] Есть AutoTheta, AutoETS, AutoARIMA?
- [ ] Есть DLinear/NLinear?

**Условия:**
- [ ] Одинаковый lookback window?
- [ ] `drop_last=False`?
- [ ] Несколько random seeds со std?

**Воспроизводимость:**
- [ ] Код и данные доступны?
- [ ] Все гиперпараметры указаны?

---

### Ключевые источники

| Статья | Год | Тема |
|--------|-----|------|
| [M4 Competition](https://doi.org/10.1016/j.ijforecast.2019.04.014) | 2020 | Статистика vs ML на univariate |
| [M5 Competition](https://doi.org/10.1016/j.ijforecast.2021.01.006) | 2022 | ML побеждает с экзогенными переменными |
| [TFB](https://arxiv.org/abs/2403.20150) | 2024 | Drop Last и честное сравнение |
| [GIFT-Eval](https://arxiv.org/abs/2410.10393) | 2024 | Бенчмарк foundation models |
| [Monash Archive](https://arxiv.org/abs/2105.06643) | 2021 | Стандартизированные датасеты |

---

Так лучше? Следующая глава?