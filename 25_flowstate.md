## Ключевая идея

FlowState задаёт вопрос, который другие модели даже не рассматривают: почему модель, обученная на часовых данных, не может прогнозировать минутные или дневные? Интуитивно, паттерн «рост утром, спад вечером» — это один и тот же паттерн, независимо от того, измеряем мы его каждую минуту или каждый час. Но большинство моделей привязаны к конкретной частоте дискретизации и требуют переобучения при её изменении.

FlowState решает эту проблему через комбинацию State Space Model (SSM) encoder и Functional Basis Decoder (FBD). SSM обрабатывает входную последовательность и строит скрытое представление. FBD декодирует это представление не в дискретные точки прогноза, а в непрерывную функцию времени. Эта функция может быть вычислена в любой момент — с любым шагом дискретизации.

Результат впечатляет: модель с менее чем 10 миллионами параметров занимает первое место на GIFT-Eval, обходя модели в 10+ раз больше. И при этом одна и та же модель работает на данных с частотой от 15 минут до месяца, просто меняя параметр scale_factor.

## State Space Models: третий путь

Прежде чем разбирать FlowState, нужно понять, что такое State Space Models и почему они интересны.

SSM — это класс моделей, который пришёл из теории управления и обработки сигналов. В контексте глубокого обучения SSM стали популярны благодаря работам S4 (2021) и Mamba (2023), которые показали, что SSM могут конкурировать с трансформерами на задачах моделирования последовательностей.

Ключевая идея SSM — моделировать скрытое состояние как непрерывный процесс, который затем дискретизируется для работы с дискретными данными.

**Непрерывная форма:**

dh(t)dt=Ah(t)+Bx(t)\frac{dh(t)}{dt} = Ah(t) + Bx(t)dtdh(t)​=Ah(t)+Bx(t) y(t)=Ch(t)+Dx(t)y(t) = Ch(t) + Dx(t)y(t)=Ch(t)+Dx(t)

где h(t)h(t) h(t) — скрытое состояние, x(t)x(t) x(t) — вход, y(t)y(t) y(t) — выход, а AA A, BB B, CC C, DD D — обучаемые матрицы.

**Дискретная форма** (после дискретизации с шагом Δ\Delta Δ):

ht=Aˉht−1+Bˉxth_t = \bar{A}h_{t-1} + \bar{B}x_tht​=Aˉht−1​+Bˉxt​ yt=Cht+Dxty_t = Ch_t + Dx_tyt​=Cht​+Dxt​

где Aˉ=exp⁡(ΔA)\bar{A} = \exp(\Delta A) Aˉ=exp(ΔA), Bˉ=(ΔA)−1(exp⁡(ΔA)−I)⋅ΔB\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B Bˉ=(ΔA)−1(exp(ΔA)−I)⋅ΔB.

Почему это важно? SSM имеют два режима вычислений:

**Рекуррентный режим** — как RNN, шаг за шагом. Эффективен для инференса: O(1)O(1) O(1) память на шаг, можно обрабатывать бесконечные потоки данных.

**Свёрточный режим** — вся последовательность обрабатывается параллельно через свёртку. Эффективен для обучения: полная параллелизация на GPU.

Трансформеры умеют только параллельный режим (и то с квадратичной сложностью). RNN умеют только рекуррентный режим. SSM умеют оба — и переключаются между ними в зависимости от задачи.

## Functional Basis Decoder: непрерывный прогноз

Вторая ключевая инновация FlowState — Functional Basis Decoder (FBD). Вместо того чтобы предсказывать дискретные значения y^1,y^2,...,y^H\hat{y}_1, \hat{y}_2, ..., \hat{y}_H y^​1​,y^​2​,...,y^​H​, модель предсказывает коэффициенты разложения по базисным функциям:

y^(t)=∑k=1Kck⋅ϕk(t)\hat{y}(t) = \sum_{k=1}^{K} c_k \cdot \phi_k(t)y^​(t)=k=1∑K​ck​⋅ϕk​(t)

где ϕk(t)\phi_k(t) ϕk​(t) — базисные функции (например, синусы и косинусы разных частот), ckc_k ck​ — обучаемые коэффициенты, которые выдаёт decoder.

Это превращает прогноз из набора точек в непрерывную функцию. Хотите прогноз на 24 часа с шагом 15 минут? Вычислите y^(t)\hat{y}(t) y^​(t) для t=0.25,0.5,0.75,...,24t = 0.25, 0.5, 0.75, ..., 24 t=0.25,0.5,0.75,...,24. Хотите тот же прогноз с шагом 1 час? Вычислите для t=1,2,3,...,24t = 1, 2, 3, ..., 24 t=1,2,3,...,24. Модель та же, коэффициенты те же, меняется только точка вычисления.

Идея похожа на basis expansion в N-BEATS, но там базис применялся к дискретному выходу. Здесь базис создаёт истинно непрерывное представление.

## Time-scale invariance: как это работает

Главная особенность FlowState — возможность адаптации к разным частотам дискретизации через единственный параметр `scale_factor`.

### Концепция сезонности

FlowState обучен с «базовой сезонностью» 24 — это соответствует часовым данным с суточным циклом (24 часа в сутках). Параметр `scale_factor` говорит модели, как соотносится сезонность ваших данных с базовой:

scale_factor=базовая сезонностьсезонность ваших данных=24N\text{scale\_factor} = \frac{\text{базовая сезонность}}{\text{сезонность ваших данных}} = \frac{24}{N}scale_factor=сезонность ваших данныхбазовая сезонность​=N24​

Примеры:

|Частота данных|Сезонность N|scale_factor|
|---|---|---|
|15 минут (суточный цикл)|96|24/96 = 0.25|
|30 минут|48|24/48 = 0.5|
|Часовая|24|24/24 = 1.0|
|Дневная (недельный цикл)|7|24/7 ≈ 3.43|
|Недельная|52|24/52 ≈ 0.46|
|Месячная (годовой цикл)|12|24/12 = 2.0|

### Что происходит внутри

Когда вы указываете `scale_factor`, FlowState масштабирует внутреннее представление времени. SSM-encoder «сжимает» или «растягивает» временную ось так, чтобы один период сезонности ваших данных соответствовал одному периоду в базовом пространстве модели.

FBD затем генерирует непрерывную функцию в этом масштабированном пространстве. При вычислении конкретных точек прогноза масштабирование применяется обратно.

Результат: модель, обученная на часовых данных, «понимает» суточную сезонность. Когда вы подаёте 15-минутные данные с `scale_factor=0.25`, модель распознаёт ту же суточную сезонность, просто выраженную в 96 точках вместо 24.

## Архитектура в деталях

```
Input: x ∈ ℝ^T (временной ряд)
    ↓
Input normalization (по среднему и std контекста)
    ↓
SSM Encoder:
  - Несколько SSM-блоков
  - Каждый блок: SSM layer + MLP + residual
  - Параллельное обучение (свёрточный режим)
    ↓
Latent representation ∈ ℝ^D
    ↓
Functional Basis Decoder:
  - Linear → коэффициенты базиса c_k
  - Базисные функции φ_k(t)
  - Сумма: ŷ(t) = Σ c_k · φ_k(t)
    ↓
Evaluate at desired time points
    ↓
Output: квантили прогноза
```

### SSM-блок

Каждый SSM-блок следует архитектуре, близкой к Mamba:

python

```python
def ssm_block(x, A, B, C, D, delta):
    # Дискретизация
    A_bar = discretize_A(A, delta)
    B_bar = discretize_B(B, delta)
    
    # SSM (свёрточный режим для обучения)
    h = ssm_conv(x, A_bar, B_bar)  # параллельно
    y = C @ h + D * x
    
    # Gating и MLP
    y = gate(y) * mlp(y)
    
    return y + x  # residual
```

### Базисные функции

FlowState использует комбинацию базисных функций для захвата разных типов паттернов:

**Полиномиальный базис** — для трендов:

ϕkpoly(t)=tk,k=0,1,2,...\phi_k^{\text{poly}}(t) = t^k, \quad k = 0, 1, 2, ...ϕkpoly​(t)=tk,k=0,1,2,...

**Гармонический базис** — для сезонности:

ϕksin(t)=sin⁡(2πkt/P)\phi_k^{\text{sin}}(t) = \sin(2\pi k t / P)ϕksin​(t)=sin(2πkt/P) ϕkcos(t)=cos⁡(2πkt/P)\phi_k^{\text{cos}}(t) = \cos(2\pi k t / P)ϕkcos​(t)=cos(2πkt/P)

где PP P — период базовой сезонности.

Комбинация позволяет моделировать и плавные тренды, и периодические колебания.

## Сильные стороны

**Time-scale invariance.** Уникальная способность работать с данными разной частоты без переобучения. Одна модель для 15-минутных, часовых, дневных и месячных данных.

**Компактность.** Менее 10M параметров — это на порядок меньше, чем у большинства foundation models. Быстрый инференс, низкие требования к памяти.

**State-of-the-art качество.** Первое место на GIFT-Eval на момент публикации, обгоняя модели с 100M+ параметрами.

**Эффективное обучение.** SSM в свёрточном режиме обучаются почти так же быстро, как трансформеры, избегая проблем последовательного обучения RNN.

**Вероятностный выход.** Модель выдаёт квантили, позволяя оценить неопределённость прогноза.

## Ограничения

**Необходимость знать сезонность.** Чтобы выбрать правильный `scale_factor`, нужно понимать сезонность своих данных. Для данных с неочевидной или множественной сезонностью выбор может быть нетривиальным.

**Ограничение горизонта.** Авторы рекомендуют прогнозировать не более 30 периодов сезонности. После этого качество падает. Для часовых данных с суточной сезонностью это 30 × 24 = 720 часов = 30 дней — обычно достаточно, но не для всех задач.

**Относительная новизна.** Модель вышла в 2025 году, практического опыта мало. SSM-архитектуры менее изучены, чем трансформеры или RNN.

**Нет поддержки ковариат.** Как и многие foundation models, FlowState работает только с историей целевой переменной. Если важны внешние факторы — нужны другие решения.

## Код: FlowState

python

```python
import torch
from tsfm_public import FlowStateForPrediction

# Загрузка модели
# Для исследовательских целей (non-commercial)
model = FlowStateForPrediction.from_pretrained("ibm-research/flowstate")
model.to('cuda')

# Подготовка данных
# FlowState ожидает формат (context_length, batch, channels)
# Обратите внимание: batch НЕ первое измерение по умолчанию!
context_length = 2048
batch_size = 32
n_channels = 1

time_series = torch.randn(context_length, batch_size, n_channels).to('cuda')

# Прогнозирование
forecast = model(
    time_series,
    scale_factor=0.25,      # для 15-минутных данных с суточной сезонностью
    prediction_length=960,   # 960 точек = 10 дней при 15-мин интервале
    batch_first=False        # важно: context первое измерение
)

# Результат
predictions = forecast.prediction_outputs
print(f"Shape: {predictions.shape}")
# (batch, quantiles, horizon, channels)
# quantiles: 9 штук [0.1, 0.2, ..., 0.9]
```

### Выбор scale_factor

python

```python
def calculate_scale_factor(frequency: str, seasonality_type: str = 'auto') -> float:
    """
    Вычисляет scale_factor для FlowState.
    
    Args:
        frequency: частота данных ('15min', '30min', 'H', 'D', 'W', 'M')
        seasonality_type: тип сезонности ('daily', 'weekly', 'yearly', 'auto')
    
    Returns:
        scale_factor для FlowState
    """
    BASE_SEASONALITY = 24  # базовая сезонность модели
    
    # Определяем сезонность данных
    seasonality_map = {
        # (frequency, seasonality_type): N
        ('15min', 'daily'): 96,    # 24*4 = 96 точек в сутках
        ('30min', 'daily'): 48,    # 24*2 = 48
        ('H', 'daily'): 24,        # 24 часа
        ('D', 'weekly'): 7,        # 7 дней в неделе
        ('D', 'yearly'): 365,      # 365 дней в году
        ('W', 'yearly'): 52,       # 52 недели
        ('M', 'yearly'): 12,       # 12 месяцев
    }
    
    if seasonality_type == 'auto':
        # Эвристика: для внутридневных — суточная, иначе — недельная/годовая
        if frequency in ['15min', '30min', 'H']:
            seasonality_type = 'daily'
        elif frequency == 'D':
            seasonality_type = 'weekly'
        else:
            seasonality_type = 'yearly'
    
    key = (frequency, seasonality_type)
    if key not in seasonality_map:
        raise ValueError(f"Unknown combination: {key}")
    
    N = seasonality_map[key]
    scale_factor = BASE_SEASONALITY / N
    
    return scale_factor

# Примеры
print(f"15-min data: {calculate_scale_factor('15min')}")      # 0.25
print(f"Hourly data: {calculate_scale_factor('H')}")          # 1.0
print(f"Daily data: {calculate_scale_factor('D')}")           # 3.43
print(f"Monthly data: {calculate_scale_factor('M')}")         # 2.0
```

### Работа с разными частотами — один и тот же ряд

python

```python
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_scale_invariance(model, base_series, title=''):
    """
    Демонстрация: одна модель, разные частоты дискретизации.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Исходный ряд — часовые данные, 7 дней
    hourly_data = base_series  # (168,) — 7*24 часов
    
    # Преобразуем в разные частоты
    # 15-минутные (интерполяция)
    minute_15_data = np.interp(
        np.linspace(0, len(hourly_data)-1, len(hourly_data)*4),
        np.arange(len(hourly_data)),
        hourly_data
    )
    
    # Дневные (агрегация)
    daily_data = hourly_data.reshape(-1, 24).mean(axis=1)
    
    datasets = [
        (minute_15_data, 0.25, '15-минутные', 96*2),  # прогноз на 2 дня
        (hourly_data, 1.0, 'Часовые', 24*2),
        (daily_data, 24/7, 'Дневные', 7),  # недельная сезонность
    ]
    
    for ax, (data, scale_factor, label, pred_len) in zip(axes, datasets):
        # Подготовка входа
        context = torch.tensor(data, dtype=torch.float32)
        context = context.unsqueeze(1).unsqueeze(2).to('cuda')  # (T, 1, 1)
        
        # Прогноз
        with torch.no_grad():
            forecast = model(
                context,
                scale_factor=scale_factor,
                prediction_length=pred_len,
                batch_first=False
            )
        
        # Медиана прогноза
        median = forecast.prediction_outputs[0, 4, :, 0].cpu().numpy()
        
        # Визуализация
        ax.plot(range(len(data)), data, 'b-', label='История')
        ax.plot(range(len(data), len(data) + len(median)), median, 
                'r-', label='Прогноз')
        ax.axvline(len(data), color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'{label} (scale_factor={scale_factor})')
        ax.legend()
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()

# Генерируем тестовый ряд с суточной сезонностью
t = np.arange(168)  # 7 дней почасовых данных
base_series = (
    100 +                           # базовый уровень
    20 * np.sin(2 * np.pi * t / 24) +  # суточная сезонность
    0.5 * t +                       # тренд
    np.random.normal(0, 5, len(t))  # шум
)

demonstrate_scale_invariance(model, base_series, 
                            'FlowState: одна модель, разные частоты')
```

### Сравнение с TiRex

python

```python
from tirex import load_model as load_tirex
from tsfm_public import FlowStateForPrediction

# Загружаем обе модели
tirex = load_tirex("NX-AI/TiRex").to('cuda')
flowstate = FlowStateForPrediction.from_pretrained("ibm-research/flowstate").to('cuda')

# Тестовые данные
context_length = 512
batch_size = 50
prediction_length = 96

# Подготовка данных (разный формат!)
data_np = np.random.randn(batch_size, context_length).astype(np.float32)

# TiRex: (batch, time)
tirex_input = torch.tensor(data_np).to('cuda')

# FlowState: (time, batch, channels)
flowstate_input = torch.tensor(data_np.T[:, :, np.newaxis]).to('cuda')

# Прогнозы
with torch.no_grad():
    # TiRex
    tirex_quantiles, tirex_mean = tirex.forecast(
        context=tirex_input,
        prediction_length=prediction_length
    )
    
    # FlowState
    flowstate_output = flowstate(
        flowstate_input,
        scale_factor=1.0,  # часовые данные
        prediction_length=prediction_length,
        batch_first=False
    )
    flowstate_median = flowstate_output.prediction_outputs[:, 4, :, 0]

print(f"TiRex output shape: {tirex_mean.shape}")           # (50, 96)
print(f"FlowState output shape: {flowstate_median.shape}") # (50, 96)

# Обе модели дают вероятностный выход с квантилями
# TiRex: 9 квантилей через tirex_quantiles
# FlowState: 9 квантилей в prediction_outputs[:, :, :, 0]
```

## FlowState vs TiRex vs другие модели

|Критерий|FlowState|TiRex|Chronos|DeepAR|
|---|---|---|---|---|
|Архитектура|SSM + FBD|xLSTM|T5|LSTM|
|Параметры|<10M|35M|20M–700M|~5M|
|Time-scale invariance|✓ Уникально|✗|✗|✗|
|Zero-shot|✓|✓|✓|✗|
|Вероятностный выход|✓ Квантили|✓ Квантили|✓ Сэмплы|✓ Распределение|
|GIFT-Eval ranking|#1|Top-3|Top-5|—|
|Открытость|✓ (research)|✓|✓|✓|

**Когда использовать FlowState:**

- Данные разной частоты в одном пайплайне (например, часовые и дневные метрики)
- Нужна компактная модель с минимальными ресурсами
- Важна адаптивность к изменению частоты сбора данных
- Zero-shot прогноз без дообучения

**Когда искать альтернативы:**

- Неизвестная или сложная сезонность → экспериментируйте или используйте TiRex
- Очень длинные горизонты (>30 периодов) → другие модели
- Нужны ковариаты → DeepAR, TFT
- Коммерческое использование → проверьте лицензию (research vs commercial)

## Что дальше

Мы завершили обзор RNN/SSM-based архитектур. DeepAR заложил основы вероятностного прогнозирования с глобальными моделями. TiRex показал, что модернизированные LSTM (xLSTM) конкурентоспособны с трансформерами. FlowState продемонстрировал уникальную возможность SSM — time-scale invariance.

В следующей части мы переходим к Foundation Models — моделям, предобученным на огромных коллекциях данных и способным работать zero-shot на новых задачах. Начнём с обзорного поста об идее foundation models для временных рядов, а затем разберём конкретные архитектуры: Chronos, TimeGPT, TimesFM, Moirai, Lag-Llama и Toto.