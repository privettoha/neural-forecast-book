## Зачем нейросети для временных рядов

Этот материал — для тех, кто уже работает с прогнозированием (Prophet, ARIMA, бустинг с лаговыми фичами) и хочет понять, когда нейросети действительно дают преимущество, а когда это избыточное усложнение. Мы не будем обещать магию — будем честно показывать, где нейросети выигрывают, где проигрывают, и почему.

---

### Словарь терминов

| Термин | Определение |
|--------|-------------|
| **Local model** | Отдельная модель для каждого ряда. Если 1000 рядов — 1000 моделей. ARIMA, ETS работают так по умолчанию. |
| **Global model** | Одна модель на все ряды. Учится находить общие паттерны и переносить их между рядами. DeepAR, N-BEATS, LightGBM с `item_id` — примеры. |
| **Foundation model** | Модель, предобученная на миллионах рядов из разных доменов. Может прогнозировать новые ряды без дообучения. Chronos, TimesFM, Moirai — примеры. |
| **Zero-shot** | Использование модели без обучения на целевых данных. Подаёшь историю — получаешь прогноз. |
| **Cold start** | Ситуация, когда исторических данных мало или нет совсем: новый продукт, новый магазин, новый клиент. |

---

### Когда нейросети дают преимущество

**Ситуация 1: много рядов, мало данных на каждый**

Тысяча товаров в сотне магазинов — это 100 000 рядов. Для многих комбинаций истории недостаточно, чтобы локальная модель выучила сезонность. Global models решают эту проблему: одна модель учится на всех рядах и переносит паттерны между ними[^deepar].

[^deepar]: Salinas, D., et al. "DeepAR: Probabilistic forecasting with autoregressive recurrent networks." International Journal of Forecasting, 2020. https://arxiv.org/abs/1704.04110 — Одна глобальная модель на 300+ рядах Amazon превосходит локальные ARIMA в 75% случаев.

Бустинг тоже умеет работать глобально — LightGBM с категориальными фичами `store_id`, `item_id` обучается на всех рядах сразу[^m5]. Разница в том, *как* модели обобщают: бустинг использует one-hot или target encoding, нейросети — обучаемые embeddings, которые могут улавливать более сложные связи между рядами[^global-principles].

[^m5]: Makridakis, S., et al. "The M5 Accuracy Competition: Results, Findings and Conclusions." International Journal of Forecasting, 2022. https://doi.org/10.1016/j.ijforecast.2021.01.006 — На M5 (42,000 рядов Walmart) все топ-50 решений использовали ML, улучшив бенчмарк на 20%.

[^global-principles]: Montero-Manso, P. & Hyndman, R.J. "Principles and Algorithms for Forecasting Groups of Time Series: Locality and Globality." International Journal of Forecasting, 2021. https://doi.org/10.1016/j.ijforecast.2021.03.013 — Теоретическое обоснование преимуществ глобальных моделей.

**Ситуация 2: паттерны, которые сложно описать явно**

Классические методы требуют, чтобы вы знали, что искать. ARIMA — выбор порядка (p, d, q), пусть и автоматизированный через `auto_arima`[^autoarima]. Prophet — указание типа сезонности. Бустинг — конструирование лаговых фичей, скользящих средних, календарных признаков.

[^autoarima]: Hyndman, R.J. & Khandakar, Y. "Automatic Time Series Forecasting: The forecast Package for R." Journal of Statistical Software, 2008. https://doi.org/10.18637/jss.v027.i03 — Автоматический подбор ограничен заранее заданным пространством поиска.

Нейросетевые архитектуры (N-BEATS[^nbeats], PatchTST[^patchtst], TSMixer[^tsmixer]) работают с сырыми входами и могут находить закономерности, которые человек не догадается закодировать: нелинейные взаимодействия между переменными, сложную зависимость от экзогенных факторов, меняющуюся со временем сезонность.

[^nbeats]: Oreshkin, B., et al. "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting." ICLR 2020. https://arxiv.org/abs/1905.10437 — MLP-архитектура без явного указания структуры ряда, +11% над статистическими бенчмарками M3/M4.

[^patchtst]: Nie, Y., et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." ICLR 2023. https://arxiv.org/abs/2211.14730 — Патчинг + channel independence для трансформеров.

[^tsmixer]: Chen, S., et al. "TSMixer: An All-MLP Architecture for Time Series Forecasting." TMLR, 2023. https://arxiv.org/abs/2303.06053 — Первая мультивариативная MLP-модель на уровне SOTA univariate моделей.

**Ситуация 3: cold start**

Новый продукт, две недели данных — слишком мало для любой статистической модели. Foundation models предлагают альтернативу: они обучены на миллионах рядов и могут делать zero-shot прогнозы для рядов, которых никогда не видели[^chronos].

[^chronos]: Ansari, A., et al. "Chronos: Learning the Language of Time Series." TMLR, 2024. https://arxiv.org/abs/2403.07815 — Zero-shot качество, сравнимое с обученным AutoARIMA на 27 из 42 датасетов бенчмарка (Table 2).

На практике: TimesFM без какой-либо настройки показывает RMSLE 0.398 на Kaggle Store Sales, превосходя бустинг с ручным feature engineering[^timesfm].

[^timesfm]: Das, A., et al. "A decoder-only foundation model for time-series forecasting." ICML, 2024. https://arxiv.org/abs/2310.10688

Важная оговорка: foundation models не решают cold start полностью — им всё равно нужна *какая-то* история для контекста. Но требуемый минимум значительно меньше, чем для обучения локальной модели с нуля.

---

### Когда нейросети НЕ нужны

**Короткие ряды с чистой сезонностью.** ETS или Prophet справятся не хуже, а настроить их проще. Нейросети с тысячами параметров на 50 наблюдениях скорее выучат шум, чем сигнал.

**Высокий шум, слабый сигнал.** На финансовых рядах Seasonal Naive может оказаться лучше любой сложной модели — он не пытается учить паттерны там, где их нет[^m4naive].

[^m4naive]: Makridakis, S., et al. "The M4 Competition: 100,000 time series and 61 forecasting methods." International Journal of Forecasting, 2020. https://doi.org/10.1016/j.ijforecast.2019.04.014, Table 3 — Seasonal Naive занял 16-е место из 61, обойдя большинство ML-методов.

**Один ряд с понятной структурой.** Если ряд один, структура ясна (тренд + годовая сезонность), данных достаточно — преимущество глобальных моделей исчезает, а сложность остаётся.

---

### Структура материала

Методичка организована по типам архитектур:

1. **Бейзлайны и метрики** — Seasonal Naive, MASE, правила честного сравнения
2. **MLP-архитектуры** — N-BEATS, N-HiTS, TSMixer
3. **Трансформеры** — PatchTST, iTransformer
4. **Последовательные модели** — DeepAR, современные альтернативы на базе SSM
5. **Foundation models** — Chronos, TimesFM, Moirai, Lag-Llama

Все модели тестируем на одних данных: Store Sales (Kaggle) как основной датасет, ETT как дополнительный бенчмарк. Код — на `neuralforecast` и `statsforecast` от Nixtla.
