# TiRex. xLSTM возвращается

## Ключевая идея

В 2025 году, когда всё внимание приковано к трансформерам и foundation models, команда NX-AI выпускает модель на основе xLSTM — модернизированной версии LSTM от Сеппа Хохрайтера, одного из создателей оригинальной архитектуры в 1997 году. И эта модель занимает топовые позиции на бенчмарках GIFT-Eval и Chronos-ZS, обходя многие трансформерные решения.

Название расшифровывается как Time Series Rex — «король временных рядов». Амбициозно, но результаты подтверждают: 35 миллионов параметров, zero-shot прогнозирование, вероятностный выход с квантилями, и при этом state-of-the-art качество как на коротких, так и на длинных горизонтах.

Главный вопрос, на который отвечает TiRex: можно ли взять идею LSTM, исправить её известные проблемы, и получить архитектуру, конкурентную с трансформерами? Ответ — да.

## Почему LSTM нужно было переизобретать

Классический LSTM имеет три фундаментальные проблемы, которые ограничивали его применение для современных задач.

**Затухание градиентов на длинных последовательностях.** Несмотря на механизм гейтов, информация всё равно «затухает» при прохождении через сотни и тысячи шагов. LSTM может «помнить» важное событие 50 шагов назад, но 500 шагов — уже сложно.

**Ограниченная ёмкость памяти.** Скрытое состояние LSTM — это вектор фиксированного размера. Вся информация о прошлом должна сжаться в этот вектор. Чем длиннее история, тем сильнее сжатие, тем больше потерь.

**Невозможность параллелизации.** Каждый шаг LSTM зависит от предыдущего: ht=f(ht−1,xt)h_t = f(h_{t-1}, x_t) ht​=f(ht−1​,xt​). Нельзя вычислить h100h_{100} h100​, не вычислив h1,h2,...,h99h_1, h_2, ..., h_{99} h1​,h2​,...,h99​. Это делает обучение медленным по сравнению с трансформерами, которые обрабатывают всю последовательность параллельно.

xLSTM — extended LSTM — решает первые две проблемы. Третья остаётся (это фундаментальное свойство рекуррентности), но современные GPU и оптимизированные CUDA-ядра делают её менее критичной.

## xLSTM: что изменилось

xLSTM вводит два новых типа ячеек: sLSTM (scalar LSTM) и mLSTM (matrix LSTM). TiRex использует mLSTM как основу.

### Экспоненциальные гейты

В классическом LSTM гейты используют сигмоиду, которая ограничена диапазоном (0, 1):

ft=σ(Wf⋅[ht−1,xt]+bf)f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)ft​=σ(Wf​⋅[ht−1​,xt​]+bf​)

Это создаёт проблему: даже если forget gate близок к 1 (почти ничего не забываем), после 100 шагов сигнал уменьшается до 0.99100≈0.370.99^{100} \approx 0.37 0.99100≈0.37. После 1000 шагов — практически до нуля.

xLSTM заменяет сигмоиду на экспоненту:

ft=exp⁡(wf⋅xt+bf)f_t = \exp(w_f \cdot x_t + b_f)ft​=exp(wf​⋅xt​+bf​)

Теперь гейт может быть больше 1, что позволяет сигналу не только сохраняться, но и усиливаться. Специальная нормализация предотвращает взрыв значений.

### Матричная память (mLSTM)

Классический LSTM хранит состояние в векторе ct∈Rdc_t \in \mathbb{R}^d ct​∈Rd. mLSTM хранит состояние в матрице Ct∈Rd×dC_t \in \mathbb{R}^{d \times d} Ct​∈Rd×d. Это квадратично увеличивает ёмкость памяти.

Обновление матричной памяти:

Ct=ft⊙Ct−1+it⊙(vt⊗kt)C_t = f_t \odot C_{t-1} + i_t \odot (v_t \otimes k_t)Ct​=ft​⊙Ct−1​+it​⊙(vt​⊗kt​)

где vtv_t vt​ — value vector, ktk_t kt​ — key vector, ⊗\otimes ⊗ — внешнее произведение. Это похоже на механизм key-value из трансформеров, но встроенный в рекуррентную структуру.

Чтение из памяти:

ht=ot⊙(Ct⋅qt)h_t = o_t \odot (C_t \cdot q_t)ht​=ot​⊙(Ct​⋅qt​)

где qtq_t qt​ — query vector. Снова аналогия с трансформерами: запрос извлекает релевантную информацию из памяти.

### Covariance update rule

mLSTM использует специальное правило обновления, которое можно интерпретировать как накопление ковариационной матрицы между ключами и значениями. Это позволяет модели выучивать ассоциации между паттернами во входных данных.

## Архитектура TiRex

TiRex строится на mLSTM, но добавляет компоненты, специфичные для временных рядов.

### Входной слой

Временной ряд разбивается на патчи (аналогично PatchTST), каждый патч проецируется в эмбеддинг:

```
Input: x ∈ ℝ^T (временной ряд)
    ↓
Patching: разбиение на патчи размера P
    ↓
Linear projection: каждый патч → эмбеддинг ∈ ℝ^D
    ↓
+ Positional encoding
```

Патчинг решает две задачи: снижает длину последовательности (меньше рекуррентных шагов) и создаёт более богатые токены (патч информативнее точки).

### mLSTM backbone

Последовательность эмбеддингов обрабатывается стеком mLSTM-слоёв:

```
Patch embeddings
    ↓
mLSTM Layer 1
    ↓
mLSTM Layer 2
    ↓
...
    ↓
mLSTM Layer L
    ↓
Hidden states для каждой позиции
```

Каждый mLSTM-слой включает:

- Матричную память с экспоненциальными гейтами
- Layer normalization
- Residual connection
- Feed-forward подсеть

### Выходной слой с квантилями

TiRex выдаёт не точечный прогноз, а набор квантилей:

```
Final hidden state
    ↓
Linear → 9 квантилей для каждого шага горизонта
    ↓
Output: (horizon, 9) — квантили [0.1, 0.2, ..., 0.9]
```

Это позволяет получить вероятностный прогноз без Monte Carlo sampling — быстрее, чем у DeepAR.

## Особенности обучения

### Предобучение на разнообразных данных

TiRex предобучен на большом корпусе временных рядов:

- Публичные датасеты (Monash, M-competitions)
- Синтетические ряды с контролируемыми паттернами
- Данные разных доменов (ритейл, энергетика, транспорт)

Разнообразие важно: модель должна видеть разные типы сезонности, разные масштабы, разные уровни шума, чтобы обобщать на новые ряды.

### Quantile loss

Вместо negative log-likelihood (как в DeepAR) TiRex оптимизирует quantile loss — сумму pinball losses для каждого квантиля:

Lq(y^,y)={q⋅(y−y^)если y≥y^(1−q)⋅(y^−y)если y<y^\mathcal{L}_q(\hat{y}, y) = \begin{cases} q \cdot (y - \hat{y}) & \text{если } y \geq \hat{y} \\ (1-q) \cdot (\hat{y} - y) & \text{если } y < \hat{y} \end{cases}Lq​(y^​,y)={q⋅(y−y^​)(1−q)⋅(y^​−y)​если y≥y^​если y<y^​​ L=∑q∈{0.1,...,0.9}Lq(y^q,y)\mathcal{L} = \sum_{q \in \{0.1, ..., 0.9\}} \mathcal{L}_q(\hat{y}_q, y)L=q∈{0.1,...,0.9}∑​Lq​(y^​q​,y)

Это прямой подход: модель учится предсказывать каждый квантиль напрямую, без предположений о форме распределения.

### CUDA-оптимизация

Авторы TiRex написали custom CUDA kernels для mLSTM, которые значительно ускоряют обучение и инференс. Это важно, потому что наивная реализация рекуррентных вычислений на Python/PyTorch была бы очень медленной.

Требования: GPU с compute capability 8.0+ (Ampere и новее). На более старых GPU модель работает, но медленнее.

## Сильные стороны

**State-of-the-art на бенчмарках.** TiRex занимает топовые позиции на GIFT-Eval и Chronos-ZS — двух основных бенчмарках для zero-shot прогнозирования. Это не просто «ещё одна модель», а реальный претендент на лидерство.

**Работает и на коротких, и на длинных горизонтах.** В отличие от многих моделей, которые хороши либо на коротких, либо на длинных горизонтах, TiRex показывает стабильные результаты в обоих режимах. Это следствие комбинации патчинга и матричной памяти.

**Эффективный вероятностный выход.** Квантили генерируются за один forward pass, без многократного сэмплирования. Это быстрее, чем Monte Carlo в DeepAR или Chronos.

**Относительно компактная модель.** 35M параметров — это меньше, чем у многих foundation models. Можно запускать на consumer-grade GPU.

**Полностью открытая.** Веса, код, данные для воспроизведения — всё доступно. В отличие от TimeGPT, здесь нет закрытого API.

## Ограничения

**Зависимость от CUDA kernels.** Для полной скорости нужны кастомные CUDA-ядра, которые требуют современного GPU и могут быть сложны в установке. Без них модель работает, но значительно медленнее.

**Последовательный инференс.** Несмотря на оптимизации, рекуррентная природа означает, что инференс всё равно последовательный. Для очень длинных входных последовательностей это может быть узким местом.

**Новизна архитектуры.** xLSTM появился в 2024 году, TiRex — в 2025. Практического опыта применения в продакшене пока мало. Могут быть неизвестные edge cases.

**Фиксированные квантили.** Модель выдаёт фиксированный набор квантилей (0.1, 0.2, ..., 0.9). Если нужен произвольный квантиль (например, 0.95), придётся интерполировать.

## Код: TiRex

python

```python
import torch
from tirex import load_model, ForecastModel

# Загрузка предобученной модели
model: ForecastModel = load_model("NX-AI/TiRex")
model.to('cuda')

# Подготовка данных
# TiRex ожидает тензор (batch, time)
data = torch.randn(5, 128).to('cuda')  # 5 рядов по 128 точек

# Прогнозирование
quantiles, mean = model.forecast(
    context=data,
    prediction_length=64
)

# quantiles: (batch, 9, horizon) — 9 квантилей
# mean: (batch, horizon) — среднее (медиана)

print(f"Прогноз shape: {mean.shape}")
print(f"Квантили shape: {quantiles.shape}")

# Доступ к конкретным квантилям
# Индексы: 0=0.1, 1=0.2, ..., 4=0.5 (медиана), ..., 8=0.9
p10 = quantiles[:, 0, :]  # 10-й перцентиль
p50 = quantiles[:, 4, :]  # медиана
p90 = quantiles[:, 8, :]  # 90-й перцентиль
```

### Работа с pandas DataFrame

python

```python
import pandas as pd
import numpy as np
from tirex import load_model

model = load_model("NX-AI/TiRex")
model.to('cuda')

# Загружаем данные в формате neuralforecast
df = pd.read_csv('train.csv')
df['ds'] = pd.to_datetime(df['ds'])

# Параметры
HORIZON = 16
CONTEXT_LENGTH = 128

# Подготовка: преобразуем в тензоры
def prepare_data(df, context_length):
    """Преобразует DataFrame в тензоры для TiRex."""
    series_list = []
    ids = df['unique_id'].unique()
    
    for uid in ids:
        series = df[df['unique_id'] == uid]['y'].values
        # Берём последние context_length точек
        if len(series) >= context_length:
            series_list.append(series[-context_length:])
    
    return torch.tensor(np.array(series_list), dtype=torch.float32)

context = prepare_data(df, CONTEXT_LENGTH).to('cuda')

# Прогноз
quantiles, mean = model.forecast(
    context=context,
    prediction_length=HORIZON
)

# Преобразуем обратно в DataFrame
forecast_df = pd.DataFrame({
    'unique_id': np.repeat(df['unique_id'].unique(), HORIZON),
    'horizon': np.tile(np.arange(1, HORIZON + 1), len(df['unique_id'].unique())),
    'forecast': mean.cpu().numpy().flatten(),
    'p10': quantiles[:, 0, :].cpu().numpy().flatten(),
    'p90': quantiles[:, 8, :].cpu().numpy().flatten()
})
```

### Сравнение с другими моделями

python

```python
from tirex import load_model
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, PatchTST
import time

# TiRex
tirex = load_model("NX-AI/TiRex").to('cuda')

# Prepare context
context = torch.randn(100, 256).to('cuda')  # 100 рядов

# Benchmark TiRex
start = time.time()
for _ in range(10):
    _, mean_tirex = tirex.forecast(context, prediction_length=32)
tirex_time = (time.time() - start) / 10
print(f"TiRex: {tirex_time:.3f}s per batch")

# Для сравнения с N-HiTS и PatchTST
# нужно использовать их через NeuralForecast API
# (разный интерфейс, разная подготовка данных)
```

### Визуализация вероятностного прогноза

python

```python
import matplotlib.pyplot as plt

def plot_tirex_forecast(history, quantiles, mean, title=''):
    """
    Визуализация прогноза TiRex с квантилями.
    
    Args:
        history: исторические значения (1D array)
        quantiles: квантили прогноза (9, horizon)
        mean: средний прогноз (horizon,)
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # История
    ax.plot(range(len(history)), history, 
            color='black', label='История', linewidth=1.5)
    
    # Прогноз
    forecast_start = len(history)
    forecast_range = range(forecast_start, forecast_start + len(mean))
    
    # Медиана
    ax.plot(forecast_range, mean, 
            color='blue', label='Медиана', linewidth=2)
    
    # 80% интервал (p10-p90)
    ax.fill_between(
        forecast_range,
        quantiles[0],  # p10
        quantiles[8],  # p90
        alpha=0.2, color='blue', label='80% интервал'
    )
    
    # 50% интервал (p25-p75)
    ax.fill_between(
        forecast_range,
        quantiles[2],  # p30 (ближайший к p25)
        quantiles[6],  # p70 (ближайший к p75)
        alpha=0.4, color='blue', label='50% интервал'
    )
    
    ax.axvline(x=forecast_start, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Время')
    ax.set_ylabel('Значение')
    
    plt.tight_layout()
    plt.show()

# Пример использования
sample_idx = 0
history = context[sample_idx].cpu().numpy()
sample_quantiles = quantiles[sample_idx].cpu().numpy()
sample_mean = mean[sample_idx].cpu().numpy()

plot_tirex_forecast(history, sample_quantiles, sample_mean, 
                    title='TiRex: прогноз с квантилями')
```

## TiRex vs другие модели

|Критерий|TiRex|DeepAR|PatchTST|Chronos|
|---|---|---|---|---|
|Архитектура|xLSTM|LSTM|Transformer|T5 (encoder-decoder)|
|Параметры|35M|~5M (зависит от config)|~10M|20M–700M|
|Zero-shot|✓|✗ (нужно обучать)|~ (с pretraining)|✓|
|Вероятностный выход|✓ Квантили|✓ Распределение|✗ Точечный|✓ Сэмплы|
|Ковариаты|✗|✓|✗|✗|
|Скорость инференса|Средняя|Низкая|Высокая|Средняя|
|GIFT-Eval ranking|Top-3|Не участвует|Top-10|Top-5|

**Когда использовать TiRex:**

- Нужен zero-shot прогноз без обучения на своих данных
- Важен баланс качества на коротких и длинных горизонтах
- Хочется вероятностный выход без сложностей Monte Carlo
- Есть современный GPU (Ampere+)

**Когда искать альтернативы:**

- Нужны ковариаты (динамические признаки) → DeepAR, TFT
- Критична скорость инференса → PatchTST, N-HiTS
- Нет доступа к GPU → Chronos на CPU, классические методы

## Что дальше

TiRex показывает, что улучшенные RNN (в форме xLSTM) могут конкурировать с трансформерами. Но это не единственный путь эволюции последовательных архитектур.

В следующем посте мы рассмотрим FlowState — модель от IBM, построенную на State Space Models (SSM). SSM — это другой подход к моделированию последовательностей, который сочетает преимущества RNN (эффективный инференс) и свёрточных сетей (параллельное обучение). FlowState добавляет к этому уникальную возможность — time-scale invariance, позволяющую одной модели работать с данными разной частоты дискретизации без переобучения.