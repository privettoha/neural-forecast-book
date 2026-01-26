# Lag-Llama: лаги как универсальный язык временных рядов

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/privettoha/neural-forecast-book/blob/main/notebooks/35_lag_llama.ipynb)

## Ключевая идея

Lag-Llama[^lagllama-main], представленная в октябре 2023 года коллаборацией исследователей из Morgan Stanley, ServiceNow, Mila, Université de Montréal и McGill University, занимает особое место среди foundation-моделей для временных рядов. В то время как другие модели экспериментируют с патч-токенизацией (TimesFM, Chronos) или мультимасштабными представлениями (Moirai), Lag-Llama возвращается к классической идее статистического прогнозирования — использованию лагов (запаздывающих значений) как основных входных признаков.

[^lagllama-main]: Rasul, K., Ashok, A., Williams, A.R., Ghonia, H., Bhagwatkar, R., Khorasani, A., Bayazi, M.J.D., Adamopoulos, G., Riachi, R., Hassen, N., Biloš, M., Garg, S., Schneider, A., Chapados, N., Drouin, A., Zantedeschi, V., Nevmyvaka, Y., Rish, I. (2024). "Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting." arXiv:2310.08278v3. https://arxiv.org/abs/2310.08278

Эта идея может показаться удивительно простой для foundation-модели, но в ней заложена глубокая интуиция: лаговые признаки представляют собой универсальный язык для описания временных рядов. Независимо от домена, частоты или масштаба данных, зависимость текущего значения от предыдущих — это то, что объединяет все временные ряды. Модель, научившаяся эффективно использовать лаги на разнообразном корпусе данных, может обобщаться на произвольные ряды.

## Архитектура: decoder-only трансформер с distribution head

Lag-Llama использует decoder-only архитектуру трансформера, вдохновлённую LLaMA[^llama] — семейством языковых моделей от Meta. Важно отметить: название Lag-Llama отражает только архитектурное сходство, модель не использует предобученные веса LLaMA[^lagllama-name].

[^llama]: Touvron, H., et al. (2023). "LLaMA: Open and Efficient Foundation Language Models." arXiv:2302.13971.

[^lagllama-name]: Rasul et al. (2024), footnote 1: "the name of our model Lag-Llama, although inspired by Llama, is used to signify architectural similarities only, and not the use of any pretrained language model weights."

Как и в LLaMA, модель включает[^lagllama-arch]:

- **RMSNorm** вместо классического LayerNorm для нормализации — более вычислительно эффективный и стабильный при обучении[^rmsnorm]
- **Rotary Positional Encoding (RoPE)** — позиционные кодировки, встроенные в механизм внимания через поворот векторов[^rope]
- **SwiGLU activation** — улучшенная версия GLU, показывающая лучшие результаты в языковых моделях
- **Causal attention** — маска, запрещающая модели «смотреть в будущее»

[^lagllama-arch]: Rasul et al. (2024), Section 4.1, Figure 2: архитектура модели.

[^rmsnorm]: Zhang, B., Sennrich, R. (2019). "Root Mean Square Layer Normalization." NeurIPS 2019.

[^rope]: Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv:2104.09864.

Ключевое отличие от языковых моделей: выход трансформера подаётся не на softmax для выбора следующего токена, а на **distribution head** — специальный слой, который предсказывает параметры вероятностного распределения для следующего значения ряда.

## Токенизация через лаги

Центральная идея Lag-Llama — универсальная токенизация временных рядов через набор лагов[^lagllama-tokenization]. Для каждой временной точки модель формирует входной вектор, содержащий:

[^lagllama-tokenization]: Rasul et al. (2024), Section 4.2: "We construct a lag vector for each time value, where each element corresponds to the value at a specific lag."

1. **Значение в текущей точке** (если оно известно, то есть для контекста)
2. **Лаговые признаки** — значения ряда в предыдущие моменты времени
3. **Временные ковариаты** — производные признаки: second-of-minute, minute-of-hour, hour-of-day, day-of-week, day-of-month, day-of-year, week-of-year, month-of-year, quarter-of-year

Набор используемых лагов фиксирован и охватывает характерные периоды для разных частот[^lagllama-lags]:

[^lagllama-lags]: Rasul et al. (2024), Section 4.2: полный список лаговых индексов.

```python
# Примеры лаговых индексов (неполный список)
LAG_INDICES = [
    1, 2, 3, 4, 5, 6, 7,           # краткосрочные (до недели)
    14, 21, 28,                     # двух-, трёх-, четырёхнедельные
    30, 60, 90,                     # месячные кратные
    365, 730,                       # годовые
    # ... включая квартальные, часовые, минутные лаги
]
```

Когда модель получает временной ряд, она определяет его частоту и выбирает из полного набора подходящие лаги. Для дневных данных будут использованы лаги 1, 7, 14, 30, 365, а для часовых — 1, 24, 168 (неделя в часах) и так далее.

Важный момент: такой подход имеет обратную сторону — входные токены могут быть очень длинными. Например, для месячной сезонности при часовых данных нужен лаг 730, что требует минимум 730 исторических точек[^lagllama-context].

[^lagllama-context]: Rasul et al. (2024), Section 4.2: "This means that the input token has a length of at least 730, in addition to all static covariates."

## Probabilistic forecasting: Student's t-distribution

Lag-Llama — это **вероятностная** модель: вместо одного точечного прогноза она генерирует распределение возможных будущих значений. Это фундаментально важно для практических приложений.

Рассмотрим пример: вы прогнозируете спрос на товар для определения объёма заказа. Детерминистическая модель скажет «закажите 100 единиц». Но что если реальный спрос окажется 150? Вы потеряете продажи. А если 50? Останутся излишки на складе. Вероятностная модель даёт распределение: «с вероятностью 90% спрос будет между 60 и 140 единицами» — теперь вы можете принять обоснованное решение, балансируя риски.

Lag-Llama использует **распределение Стьюдента** (Student's t-distribution) для моделирования неопределённости[^lagllama-student]. Выбор не случаен:

[^lagllama-student]: Rasul et al. (2024), Section 4.3: "In our experiments, we use the Student's t-distribution."

- Оно имеет «тяжёлые хвосты», что позволяет лучше моделировать редкие экстремальные события
- При большом числе степеней свободы приближается к нормальному распределению
- Параметр degrees of freedom позволяет модели адаптивно регулировать «толщину хвостов»

Distribution head предсказывает три параметра для каждой будущей точки:

- **μ (mu)** — параметр положения (аналог среднего)
- **σ (sigma)** — параметр масштаба (аналог стандартного отклонения), с гарантией положительности через softplus
- **ν (nu)** — число степеней свободы, также с гарантией положительности

Функция потерь — отрицательное логарифмическое правдоподобие (negative log-likelihood) распределения Стьюдента.

## Обучение: данные и аугментация

Lag-Llama обучалась на корпусе из **27 датасетов** различных доменов: энергетика, транспорт, экономика, климат и другие[^lagllama-data]. Корпус включает 7965 унивариативных временных рядов, что в сумме составляет около **352 миллиона токенов (окон)**[^lagllama-tokens].

[^lagllama-data]: Rasul et al. (2024), Section 5 и Appendix A: полный список датасетов.

[^lagllama-tokens]: Rasul et al. (2024), Section 5: "Our pretraining corpus comprises a total of 7,965 different univariate time series... comprising a total of around 352 million data windows (tokens)."

Это значительно меньше, чем у TimesFM (сотни миллиардов точек) или Moirai, но авторы показывают, что при правильной токенизации через лаги можно достичь сильного zero-shot обобщения на относительно скромном объёме данных.

Для борьбы с переобучением и улучшения обобщения используются техники аугментации[^lagllama-aug]:

[^lagllama-aug]: Rasul et al. (2024), Section 5.1: Freq-Mix и Freq-Mask аугментации.

- **Freq-Mix** — смешивание частотных компонент из разных рядов
- **Freq-Mask** — маскирование случайных частотных компонент

Для обработки разных масштабов данных каждое окно стандартизируется: вычисляются среднее и дисперсия внутри окна, ряд нормализуется, а статистики передаются как дополнительные ковариаты.

## Результаты: zero-shot и fine-tuning

### Zero-shot производительность

В zero-shot режиме Lag-Llama сравнивается с supervised-бейзлайнами, обученными на целевых датасетах: AutoARIMA, AutoETS, AutoTheta, DeepAR, TFT, AutoGluon[^lagllama-results].

[^lagllama-results]: Rasul et al. (2024), Table 1: CRPS на unseen datasets.

По метрике CRPS (Continuous Ranked Probability Score) Lag-Llama в zero-shot режиме показывает средний ранг **6.714**, сравнимый со supervised-бейзлайнами. Для foundation-модели без дообучения это сильный результат.

### Fine-tuning производительность

После дообучения на целевых датасетах Lag-Llama достигает **среднего ранга 2.786** — лучший результат среди всех сравниваемых методов, включая state-of-the-art на трёх из тестовых датасетов[^lagllama-finetune].

[^lagllama-finetune]: Rasul et al. (2024), Table 1: "Fine-tuning further enhances its capabilities, leading to state-of-the-art performance in three datasets."

Важный эксперимент: авторы проверили, как модель справляется при ограниченном объёме исторических данных (20%, 40%, 60%, 80% от полной истории). Lag-Llama стабильно показывает лучший средний ранг на всех уровнях — это подтверждает сильные capabilities для few-shot обучения[^lagllama-fewshot].

[^lagllama-fewshot]: Rasul et al. (2024), Section 6.2: эксперименты с ограниченной историей.

### Scaling laws

Авторы исследуют scaling laws для Lag-Llama, показывая, как zero-shot производительность улучшается с ростом размера модели[^lagllama-scaling]. Это даёт возможность экстраполировать и предсказывать качество для более крупных моделей.

[^lagllama-scaling]: Rasul et al. (2024), Section 7 и Figure 6: neural scaling laws.

## Сравнение с другими foundation models

|Характеристика|Lag-Llama|TimesFM|Moirai|Chronos|
|---|---|---|---|---|
|Архитектура|Decoder-only|Decoder-only|Masked encoder|Encoder-decoder (T5)|
|Токенизация|Лаги + временные ковариаты|Патчи|Патчи (multi-size)|Квантование + языковая модель|
|Многомерность|Univariate|Channel-independent|Any-variate|Univariate|
|Вероятностный вывод|Да (Student's t)|Точечный (опционально квантили)|Да (mixture)|Да (квантили из сэмплов)|
|Размер модели|~2.5M параметров|200M параметров|14M-300M|20M-710M|
|Открытость|Полностью|Полностью|Полностью|Полностью|

Главные преимущества Lag-Llama:

- **Компактность** — модель с ~2.5M параметров может работать на CPU
- **Нативное вероятностное прогнозирование** — Student's t-distribution даёт интервалы «из коробки»
- **Эффективный fine-tuning** — стандартный PyTorch/GluonTS workflow, хорошие результаты даже на малых данных

Главные ограничения:

- **Только унивариативные ряды** — нельзя моделировать взаимосвязи между переменными
- **Фиксированный набор лагов** — может быть субоптимальным для необычных частот или паттернов
- **Потенциально длинные входные последовательности** — при включении годовых лагов для высокочастотных данных

## Когда выбирать Lag-Llama

**Хороший выбор:**

- Нужны интервалы предсказания для принятия бизнес-решений под неопределённостью
- Ограниченные вычислительные ресурсы — можно запускать на CPU
- Планируется fine-tuning на небольшом объёме данных
- Данные имеют чёткую периодичность (дневная, недельная, годовая сезонность)

**Рассмотрите альтернативы:**

- Multivariate данные с важными взаимосвязями между переменными → Moirai
- Максимальная zero-shot производительность без вероятностного вывода → TimesFM
- Очень короткие ряды без явной периодичности → Chronos

## Практика: zero-shot и fine-tuning с GluonTS

Lag-Llama реализована на базе GluonTS[^gluonts]. Рассмотрим практический пример.

[^gluonts]: Alexandrov, A., et al. (2020). "GluonTS: Probabilistic and Neural Time Series Modeling in Python." JMLR.

```{code-cell} python
:tags: [remove-output]

# Установка (если необходимо)
!pip install gluonts torch lightning -q
!pip install git+https://github.com/time-series-foundation-models/lag-llama -q
```

```{code-cell} python
import torch
from gluonts.dataset.repository import get_dataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from lag_llama.gluon.estimator import LagLlamaEstimator

# Загрузка датасета
dataset = get_dataset("australian_electricity_demand")

# Параметры
prediction_length = dataset.metadata.prediction_length
context_length = prediction_length * 3  # рекомендуется 2-4x от prediction_length
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Prediction length: {prediction_length}")
print(f"Context length: {context_length}")
print(f"Device: {device}")
```

### Zero-shot прогнозирование

```{code-cell} python
from lag_llama.gluon.estimator import LagLlamaEstimator

# Загрузка предобученной модели
ckpt_path = "lag-llama.ckpt"  # скачайте с HuggingFace
estimator = LagLlamaEstimator(
    ckpt_path=ckpt_path,
    prediction_length=prediction_length,
    context_length=context_length,
    input_size=1,
    n_layer=8,
    n_embd=32,
    n_head=4,
    device=device,
)

# Создание предиктора
predictor = estimator.create_lightning_predictor()

# Генерация прогнозов
forecast_it, ts_it = make_evaluation_predictions(
    dataset=dataset.test,
    predictor=predictor,
    num_samples=100,  # количество сэмплов для вероятностного прогноза
)

forecasts = list(forecast_it)
tss = list(ts_it)

# Оценка
evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(tss, forecasts)

print(f"Zero-shot CRPS: {agg_metrics['mean_wQuantileLoss']:.4f}")
print(f"Zero-shot MASE: {agg_metrics['MASE']:.4f}")
```

### Fine-tuning на целевом датасете

```{code-cell} python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

# Estimator для fine-tuning
estimator_ft = LagLlamaEstimator(
    ckpt_path=ckpt_path,
    prediction_length=prediction_length,
    context_length=context_length,
    input_size=1,
    n_layer=8,
    n_embd=32,
    n_head=4,
    # Fine-tuning параметры
    lr=1e-4,
    batch_size=32,
    num_batches_per_epoch=100,
    trainer_kwargs={
        "max_epochs": 50,
        "callbacks": [EarlyStopping(monitor="val_loss", patience=10)],
        "accelerator": "auto",
    },
)

# Fine-tuning
predictor_ft = estimator_ft.train(
    training_data=dataset.train,
    validation_data=dataset.test,  # в реальности используйте validation split
)

# Оценка после fine-tuning
forecast_it_ft, ts_it_ft = make_evaluation_predictions(
    dataset=dataset.test,
    predictor=predictor_ft,
    num_samples=100,
)

agg_metrics_ft, _ = evaluator(list(ts_it_ft), list(forecast_it_ft))

print(f"Fine-tuned CRPS: {agg_metrics_ft['mean_wQuantileLoss']:.4f}")
print(f"Fine-tuned MASE: {agg_metrics_ft['MASE']:.4f}")
```

### Визуализация вероятностного прогноза

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np

# Выбираем один ряд
idx = 0
forecast = forecasts[idx]
ts = tss[idx]

# Извлекаем данные
history = ts[:-prediction_length].values.flatten()
actual = ts[-prediction_length:].values.flatten()

# Квантили из сэмплов
samples = forecast.samples  # (num_samples, prediction_length)
median = np.median(samples, axis=0)
p10 = np.percentile(samples, 10, axis=0)
p90 = np.percentile(samples, 90, axis=0)

# Визуализация
fig, ax = plt.subplots(figsize=(12, 5))

# История
t_history = range(len(history))
ax.plot(t_history, history, 'k-', label='История', linewidth=1)

# Прогнозный период
t_forecast = range(len(history), len(history) + prediction_length)
ax.plot(t_forecast, actual, 'k--', label='Факт', linewidth=1)
ax.plot(t_forecast, median, 'b-', label='Медиана прогноза', linewidth=2)
ax.fill_between(t_forecast, p10, p90, alpha=0.3, color='blue', label='80% интервал')

ax.axvline(x=len(history), color='gray', linestyle=':', alpha=0.7)
ax.set_xlabel('Время')
ax.set_ylabel('Значение')
ax.set_title('Вероятностный прогноз Lag-Llama')
ax.legend()
plt.tight_layout()
plt.show()
```

## Резюме

Lag-Llama демонстрирует, что для создания сильной foundation-модели не обязательно изобретать радикально новые архитектуры. Классическая идея лаговых признаков, интегрированная с современным decoder-only трансформером и вероятностным distribution head, даёт модель, которая:

- Конкурирует с supervised-бейзлайнами в zero-shot режиме
- Достигает state-of-the-art после fine-tuning на малых данных
- Работает на CPU благодаря компактному размеру (~2.5M параметров)
- Полностью открыта — код, веса, данные доступны сообществу

Модель особенно привлекательна как отправная точка для практиков: попробуйте zero-shot, и если результат не устраивает — дообучите на своих данных за несколько минут.

## Ссылки

- **Оригинальная статья**: Rasul, K. et al. (2024). [Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting](https://arxiv.org/abs/2310.08278). arXiv.
- **GitHub**: https://github.com/time-series-foundation-models/lag-llama
- **HuggingFace**: https://huggingface.co/time-series-foundation-models/Lag-Llama
- **Colab Demo (zero-shot)**: [Demo 1](https://colab.research.google.com/github/time-series-foundation-models/lag-llama/blob/main/colab/LagLlama_demo.ipynb)
- **Colab Demo (fine-tuning)**: [Demo 2](https://colab.research.google.com/github/time-series-foundation-models/lag-llama/blob/main/colab/LagLlama_finetune_demo.ipynb)