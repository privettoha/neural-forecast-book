# PatchTST. Патчи решают

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/privettoha/neural-forecast-book/blob/main/notebooks/21_patchtst.ipynb)

## Ключевая идея

PatchTST — это архитектура, которая делает две ставки, идущие вразрез с интуицией предыдущих работ. Первая ставка: вместо того чтобы применять attention к отдельным точкам временного ряда, нужно группировать точки в патчи и работать с патчами как с токенами. Вторая ставка: вместо того чтобы моделировать зависимости между каналами многомерного ряда, нужно обрабатывать каждый канал независимо.

Обе идеи звучат как упрощение, как отказ от выразительности. Но именно это упрощение оказывается ключом к эффективности. PatchTST показывает state-of-the-art результаты на стандартных бенчмарках, обходя и сложные трансформеры с cross-channel attention, и простые линейные модели вроде [DLinear](https://arxiv.org/abs/2205.13504).

Название расшифровывается как Patch Time Series Transformer — трансформер для временных рядов с патчами. Простое название для простой, но мощной идеи.

## Проблема, которую решает патчинг

Вспомним, почему point-wise attention плохо работает на временных рядах. Одна точка ряда — это просто число, которое само по себе не несёт почти никакой информации. Значение 150 может означать что угодно: высокие продажи, низкие продажи, аномалию, норму — всё зависит от контекста.

Когда трансформер вычисляет attention между точками, он сравнивает эти голые числа (точнее, их эмбеддинги). Точка со значением 150 будет похожа на другую точку со значением 148, даже если одна — пик сезона, а другая — случайный выброс. Механизм attention не видит локальный контекст вокруг точки.

Патчинг решает эту проблему радикально: вместо отдельных точек мы работаем с группами последовательных точек. Патч из 16 значений — это уже не голое число, а кусок временного ряда с видимым локальным паттерном. Трансформер сравнивает паттерны, а не числа.

## Механика патчинга

Допустим, у нас есть ряд длины $L = 96$ (например, 96 часов истории). Мы хотим разбить его на патчи размера $P = 16$ с шагом (stride) $S = 8$.

Первый патч: точки 0–15 Второй патч: точки 8–23 Третий патч: точки 16–31 И так далее...

Количество патчей:

$$N = \left\lfloor \frac{L - P}{S} \right\rfloor + 1$$

Для наших параметров: $N = \lfloor (96 - 16) / 8 \rfloor + 1 = 11$ патчей.

Каждый патч — это вектор размерности $P$. Дальше этот вектор проецируется в пространство размерности $D$ (размерность модели) через обычный линейный слой:

$$z_i = W \cdot \text{patch}_i + b$$

где $W \in \mathbb{R}^{D \times P}$ — матрица проекции.

После проекции добавляется позиционное кодирование — чтобы модель знала, какой патч идёт раньше, какой позже:

$$z_i' = z_i + \text{PE}_i$$

Теперь у нас есть последовательность из $N$ токенов (патчей) размерности $D$, и мы можем применить стандартный transformer encoder.

## Почему патчинг работает

**Богатая семантика токена.** Патч из 16 точек содержит информацию о локальном тренде (растёт или падает), локальной волатильности (стабильный или скачущий), форме паттерна (пик, провал, плато). Это осмысленная единица, которую можно сравнивать с другими.

**Снижение сложности.** Вместо $L$ токенов у нас $N \approx L/S$ токенов. Для $L = 512$ и $S = 8$ это снижение с 512 до ~64 токенов. Квадратичная сложность attention падает с $512^2 = 262144$ до $64^2 = 4096$ — в 64 раза.

**Более длинный контекст.** При фиксированном вычислительном бюджете патчинг позволяет смотреть на более длинную историю. Если мы можем обработать 64 токена, то без патчинга это 64 точки истории, а с патчингом ($P=16$, $S=8$) — это 512 + 16 = 528 точек при том же количестве токенов.

**Локальная инвариантность.** Небольшие сдвиги во времени не меняют патч радикально. Если паттерн «рост с понедельника по среду» сдвинулся на один час, патч всё равно будет похожим. Это своего рода встроенная аугментация.

## Channel independence: неожиданный выбор

Вторая ключевая идея PatchTST — обрабатывать каналы многомерного ряда независимо. Если у нас CC C каналов (например, 7 категорий товаров), мы не строим один большой трансформер, который видит все каналы сразу. Вместо этого мы применяем один и тот же трансформер к каждому каналу отдельно.

Формально: вход размера $(B, C, L)$ — батч × каналы × время — преобразуется в $(B \cdot C, L)$, обрабатывается как набор независимых одномерных рядов, и результат собирается обратно.

Это контринтуитивно. Кажется, что мы теряем информацию о зависимостях между каналами. Если продажи категории A влияют на продажи категории B — разве модель не должна это видеть?

На практике channel independence работает по нескольким причинам.

**Меньше параметров для обучения.** Cross-channel модель должна выучить $C^2$ потенциальных взаимодействий. Для 100 каналов это 10 000 взаимодействий, многие из которых слабые или отсутствуют. Channel-independent модель фокусируется на временных паттернах внутри каждого канала — более robust задача.

**Лучшая генерализация.** Модель, обученная на одном наборе каналов, может применяться к другому. Это особенно важно для transfer learning и foundation models.

**Достаточность локальной информации.** Для многих задач прогнозирования история самого ряда содержит достаточно информации для прогноза. Cross-channel зависимости дают marginal improvement, который не оправдывает увеличение сложности.

**Эмпирическое подтверждение.** Авторы PatchTST систематически сравнили channel-independent и channel-mixing варианты. На большинстве датасетов channel independence побеждает или показывает сравнимые результаты.

## Архитектура в деталях

Полная архитектура PatchTST для одного канала:

```
Input: x ∈ ℝ^L (временной ряд длины L)
    ↓
Instance Normalization (нормализация ряда)
    ↓
Patching: разбиение на N патчей размера P с шагом S
    ↓
Patch Embedding: линейная проекция каждого патча в ℝ^D
    ↓
+ Positional Encoding
    ↓
Transformer Encoder (несколько слоёв)
    ↓
Flatten + Linear: проекция в прогноз длины H
    ↓
Output: ŷ ∈ ℝ^H (прогноз на горизонт H)
```

**Instance Normalization.** Перед обработкой ряд нормализуется — вычитается среднее, делится на стандартное отклонение. Это делает модель инвариантной к масштабу и уровню ряда. После прогноза применяется обратное преобразование.

**Transformer Encoder.** Стандартный encoder из оригинальной статьи [«Attention Is All You Need»](https://arxiv.org/abs/1706.03762): multi-head self-attention, feed-forward network, residual connections, layer normalization. Никаких модификаций, специфичных для временных рядов — вся адаптация происходит через патчинг.

**Flatten + Linear.** После трансформера у нас $N$ токенов размерности $D$. Они конкатенируются в один вектор размерности $N \cdot D$ и проецируются линейным слоём в прогноз размерности $H$.

## Self-supervised pretraining

PatchTST предлагает дополнительную возможность — предобучение без разметки через masked patch prediction. Идея заимствована из [BERT](https://arxiv.org/abs/1810.04805): случайно маскируем часть патчей и обучаем модель их восстанавливать.

**Маскирование.** Выбираем случайные $r\%$ патчей (например, 40%) и заменяем их на обучаемый (learnable) mask-токен.

**Восстановление.** Модель должна предсказать значения замаскированных патчей на основе видимых.

**Loss.** Mean Squared Error между предсказанными и истинными значениями замаскированных патчей.

После предобучения модель дообучается на конкретную задачу прогнозирования. Предобучение особенно полезно, когда:

- Есть много исторических данных, но мало размеченных примеров для конкретной задачи
- Нужно перенести модель на новый домен с ограниченными данными
- Хочется построить универсальное представление для разных downstream задач

## Сильные стороны

**Эффективность на длинных горизонтах.** За счёт патчинга PatchTST может эффективно обрабатывать длинные входные последовательности и делать прогнозы на длинные горизонты. На бенчмарке ETTh1 с горизонтом 720 точек PatchTST значительно обходит конкурентов.

**Простота архитектуры.** Никаких специальных механизмов внимания, никаких сложных декомпозиций. Стандартный transformer encoder плюс патчинг — всё. Это упрощает реализацию, отладку и понимание модели.

**Возможность transfer learning.** Channel independence и self-supervised pretraining делают PatchTST хорошим кандидатом для переноса между задачами и доменами.

**Конкурентные результаты.** На момент публикации (2023) PatchTST показал лучшие результаты на большинстве стандартных бенчмарков для long-term forecasting, обходя и предыдущие трансформеры, и DLinear.

## Ограничения

**Фиксированные гиперпараметры патчинга.** Размер патча PP P и шаг SS S нужно выбирать под данные. Для часовых данных с суточной сезонностью патч в 24 точки имеет смысл. Для данных с другой структурой — нужен другой размер. Универсального рецепта нет.

**Потеря fine-grained информации.** Патчинг агрегирует локальную информацию. Если важны точные значения в конкретные моменты (например, пиковое потребление электроэнергии в 18:00), патчинг может размыть эту информацию.

**Channel independence как ограничение.** Для данных с сильными межканальными зависимостями (например, взаимодополняющие товары) отказ от cross-channel моделирования может стоить качества. В таких случаях [TSMixer](https://arxiv.org/abs/2303.06053) или [iTransformer](https://arxiv.org/abs/2310.06625) могут быть лучшим выбором.

**Вычислительная стоимость.** Хотя патчинг снижает сложность по сравнению с point-wise attention, трансформер всё равно тяжелее, чем MLP-based модели. Для inference в реальном времени или на edge-устройствах это может быть проблемой.

## Код: PatchTST на Store Sales

python

```python
import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from neuralforecast.losses.pytorch import MAE

# Загружаем данные
train = pd.read_csv('train.csv')
train['ds'] = pd.to_datetime(train['ds'])

# Параметры
HORIZON = 16
INPUT_SIZE = 96  # длина входного окна

# Конфигурация PatchTST
model = PatchTST(
    h=HORIZON,
    input_size=INPUT_SIZE,
    loss=MAE(),
    max_steps=1000,
    
    # Параметры патчинга
    patch_len=16,             # размер патча
    stride=8,                 # шаг между патчами
    
    # Архитектура трансформера
    hidden_size=128,          # размерность модели
    n_heads=4,                # количество голов внимания
    e_layers=3,               # количество слоёв encoder
    d_ff=256,                 # размерность feed-forward
    dropout=0.2,
    
    scaler_type='standard',
    random_seed=42
)

# Обучаем
nf = NeuralForecast(
    models=[model],
    freq='D'
)
nf.fit(df=train)

# Прогнозируем
forecasts = nf.predict()
```

### Выбор параметров патчинга

python

```python
def suggest_patch_params(input_size, season_length, horizon):
    """
    Эвристика для выбора параметров патчинга.
    
    Принципы:
    - Патч должен захватывать локальный паттерн (fraction сезона)
    - Stride обычно = patch_len / 2 для overlap
    - Количество патчей должно быть разумным (не слишком мало, не слишком много)
    """
    
    # Патч как доля сезона
    # Для недельной сезонности (7 дней) хороший патч — 2-3 дня
    patch_len = max(4, season_length // 3)
    
    # Округляем до степени двойки (удобно для GPU)
    patch_len = 2 ** int(np.log2(patch_len))
    
    # Stride — половина патча для 50% overlap
    stride = patch_len // 2
    
    # Проверяем, что получается разумное число патчей
    n_patches = (input_size - patch_len) // stride + 1
    
    if n_patches < 4:
        # Слишком мало патчей — уменьшаем patch_len
        patch_len = patch_len // 2
        stride = stride // 2
        n_patches = (input_size - patch_len) // stride + 1
    
    if n_patches > 64:
        # Слишком много патчей — увеличиваем stride
        stride = patch_len  # no overlap
        n_patches = (input_size - patch_len) // stride + 1
    
    return {
        'patch_len': patch_len,
        'stride': stride,
        'n_patches': n_patches
    }

# Пример для дневных данных с недельной сезонностью
params = suggest_patch_params(
    input_size=96,
    season_length=7,
    horizon=16
)
print(f"Suggested params: {params}")
# Output: Suggested params: {'patch_len': 4, 'stride': 2, 'n_patches': 47}
```

### Визуализация патчей

python

```python
import matplotlib.pyplot as plt

def visualize_patching(series, patch_len, stride, n_patches_to_show=5):
    """
    Визуализация того, как ряд разбивается на патчи.
    """
    fig, axes = plt.subplots(n_patches_to_show + 1, 1, 
                             figsize=(12, 2 * (n_patches_to_show + 1)))
    
    # Полный ряд
    axes[0].plot(series, color='black', linewidth=1)
    axes[0].set_title('Полный ряд')
    axes[0].set_xlim(0, len(series))
    
    # Подсвечиваем патчи разными цветами
    colors = plt.cm.tab10(np.linspace(0, 1, n_patches_to_show))
    
    for i in range(n_patches_to_show):
        start = i * stride
        end = start + patch_len
        
        if end > len(series):
            break
        
        # На полном ряде
        axes[0].axvspan(start, end, alpha=0.3, color=colors[i])
        
        # Отдельный патч
        patch = series[start:end]
        axes[i + 1].plot(patch, color=colors[i], linewidth=2)
        axes[i + 1].set_title(f'Патч {i + 1}: позиции {start}-{end}')
        axes[i + 1].set_xlim(0, patch_len)
    
    plt.tight_layout()
    plt.show()

# Пример
sample_series = train[train['unique_id'] == train['unique_id'].iloc[0]]['y'].values[-96:]
visualize_patching(sample_series, patch_len=16, stride=8)
```

### Сравнение с N-HiTS и TSMixer

python

```python
from neuralforecast.models import NHITS, TSMixer

models = [
    PatchTST(
        h=HORIZON,
        input_size=INPUT_SIZE,
        patch_len=16,
        stride=8,
        hidden_size=128,
        n_heads=4,
        e_layers=3,
        loss=MAE(),
        max_steps=1000,
        scaler_type='standard',
        random_seed=42
    ),
    NHITS(
        h=HORIZON,
        input_size=INPUT_SIZE,
        n_pool_kernel_size=[1, 2, 4],
        n_freq_downsample=[1, 2, 4],
        loss=MAE(),
        max_steps=1000,
        scaler_type='standard',
        random_seed=42
    ),
    TSMixer(
        h=HORIZON,
        input_size=INPUT_SIZE,
        n_block=4,
        ff_dim=64,
        loss=MAE(),
        max_steps=1000,
        scaler_type='standard',
        random_seed=42
    )
]

nf = NeuralForecast(models=models, freq='D')
nf.fit(df=train)
forecasts = nf.predict()

# Считаем метрики
results = {}
for model_name in ['PatchTST', 'NHITS', 'TSMixer']:
    y_true = test['y'].values
    y_pred = forecasts[model_name].values
    mae = np.mean(np.abs(y_true - y_pred))
    results[model_name] = mae

print("MAE по моделям:")
for name, mae in sorted(results.items(), key=lambda x: x[1]):
    print(f"  {name}: {mae:.2f}")
```

### Self-supervised pretraining (концептуально)

python

```python
# Примечание: реализация pretraining зависит от конкретной библиотеки
# Ниже — концептуальный пример логики

def create_masked_patches(patches, mask_ratio=0.4):
    """
    Создаёт маскированную версию патчей для pretraining.
    
    Args:
        patches: tensor размера (batch, n_patches, patch_len)
        mask_ratio: доля патчей для маскирования
    
    Returns:
        masked_patches: патчи с маской
        mask: булева маска (True = замаскировано)
        targets: оригинальные значения замаскированных патчей
    """
    batch_size, n_patches, patch_len = patches.shape
    n_mask = int(n_patches * mask_ratio)
    
    # Случайный выбор патчей для маскирования
    mask = torch.zeros(batch_size, n_patches, dtype=torch.bool)
    for i in range(batch_size):
        mask_idx = torch.randperm(n_patches)[:n_mask]
        mask[i, mask_idx] = True
    
    # Сохраняем targets
    targets = patches[mask].clone()
    
    # Заменяем на learnable mask token
    masked_patches = patches.clone()
    masked_patches[mask] = mask_token  # mask_token — обучаемый параметр
    
    return masked_patches, mask, targets

def pretrain_step(model, patches, mask_ratio=0.4):
    """
    Один шаг pretraining.
    """
    masked_patches, mask, targets = create_masked_patches(patches, mask_ratio)
    
    # Forward pass
    predictions = model(masked_patches)
    
    # Loss только на замаскированных патчах
    pred_masked = predictions[mask]
    loss = F.mse_loss(pred_masked, targets)
    
    return loss
```

## PatchTST vs другие модели: когда что выбирать

|Критерий|PatchTST|N-HiTS|TSMixer|
|---|---|---|---|
|Длинные входные последовательности|✓ Эффективен|✓ Многомасштабность|~ Линейная сложность|
|Длинные горизонты|✓ Хорошо|✓ Отлично|✓ Хорошо|
|Межканальные зависимости|✗ Channel independent|✗ Независимо|✓ Feature mixing|
|Transfer learning|✓ Pretraining|~ Ограниченно|~ Ограниченно|
|Вычислительные ресурсы|Высокие|Средние|Средние|
|Простота настройки|Средняя|Средняя|Высокая|

Практическое правило: выбирайте PatchTST, если у вас длинные входные последовательности (сотни точек), достаточно вычислительных ресурсов и важна возможность transfer learning. Для более простых задач N-HiTS или TSMixer могут быть эффективнее при меньших затратах.

## Что дальше

PatchTST решает проблему point-wise attention через группировку точек в патчи и обрабатывает каналы независимо. Но что если главные зависимости в данных — между каналами, а не во времени?

В следующем посте мы рассмотрим iTransformer — модель, которая переворачивает традиционный подход с ног на голову, применяя attention по каналам, а не по времени. Этот «инвертированный» взгляд оказывается удивительно эффективным для определённого класса многомерных задач.

:::{seealso}
**Источники и ссылки:**
- Nie, Y., et al. (2023). [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730). ICLR 2023.
- [Официальный код PatchTST](https://github.com/yuqinie98/PatchTST) — GitHub
- [Hugging Face: PatchTST Tutorial](https://huggingface.co/blog/patchtst)
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library) — Tsinghua GitHub
:::