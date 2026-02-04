# Foundation Models для временных рядов

## Ключевая идея

**Foundation models** — это pretrained модели, которые обучены на огромных объёмах данных и могут решать задачи **без дополнительного обучения** (zero-shot) или с минимальной адаптацией (few-shot, fine-tuning). В NLP это GPT, BERT, LLaMA. В computer vision — CLIP, SAM. Теперь эта парадигма пришла во временные ряды.

Идея проста: вместо обучения отдельной модели на каждом датасете — взять pretrained модель, которая уже «понимает» паттерны временных рядов из миллионов примеров, и применить её напрямую к новым данным.

**Почему это важно:** Foundation models решают главную проблему нейросетей во временных рядах — **необходимость большого количества данных**. Если у вас 100 точек истории, обучить N-BEATS или PatchTST сложно. Foundation model может прогнозировать сразу, без обучения.

## Парадигма pre-training

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ТРАДИЦИОННЫЙ ПОДХОД vs FOUNDATION MODELS             │
│                                                                         │
│  ТРАДИЦИОННЫЙ (task-specific):                                          │
│  ─────────────────────────────                                          │
│                                                                         │
│  Датасет A → Обучение модели A → Прогноз A                              │
│  Датасет B → Обучение модели B → Прогноз B                              │
│  Датасет C → Обучение модели C → Прогноз C                              │
│                                                                         │
│  • Отдельная модель для каждой задачи                                  │
│  • Требует достаточно данных для обучения                              │
│  • Не переносит знания между задачами                                  │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  FOUNDATION MODEL:                                                      │
│  ─────────────────                                                      │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  PRE-TRAINING (один раз)                                        │   │
│  │                                                                  │   │
│  │  Миллионы рядов → Большая модель → Pretrained weights          │   │
│  │  (энергетика, финансы, метео, ритейл, IoT, ...)                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  INFERENCE (многократно)                                        │   │
│  │                                                                  │   │
│  │  Датасет A → Foundation Model → Прогноз A  (zero-shot)         │   │
│  │  Датасет B → Foundation Model → Прогноз B  (zero-shot)         │   │
│  │  Датасет C → Foundation Model → Прогноз C  (zero-shot)         │   │
│  │                                                                  │   │
│  │  • Одна модель для всех задач                                  │   │
│  │  • Работает без дополнительного обучения                       │   │
│  │  • Переносит знания между доменами                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Три режима использования

Foundation models можно использовать по-разному в зависимости от количества данных и вычислительных ресурсов:

### 1. Zero-shot

Применяем pretrained модель напрямую, **без какого-либо обучения**.

```python
# Zero-shot с Chronos
from chronos import ChronosPipeline

model = ChronosPipeline.from_pretrained("amazon/chronos-t5-base")
forecast = model.predict(context=history, prediction_length=24)
```

**Когда использовать:**
- Мало исторических данных (<100 точек)
- Нет времени/ресурсов на обучение
- Нужен быстрый прототип
- Данные из нового домена

### 2. Few-shot / In-context learning

Модель получает несколько примеров в контексте и адаптируется на лету.

```
Контекст для модели:
"Вот примеры прогнозов для похожих рядов:
 Ряд 1: [история] → [прогноз]
 Ряд 2: [история] → [прогноз]
 Теперь прогнозируй для: [новая история] → ?"
```

**Когда использовать:**
- Есть несколько примеров похожих рядов
- Хотим направить модель без переобучения

### 3. Fine-tuning

Дообучаем веса модели на конкретном датасете.

```python
# Fine-tuning Chronos
model = ChronosPipeline.from_pretrained("amazon/chronos-t5-base")
model.fine_tune(
    train_data=my_dataset,
    epochs=10,
    learning_rate=1e-5
)
```

**Когда использовать:**
- Достаточно данных для fine-tuning (>1000 примеров)
- Домен сильно отличается от pre-training данных
- Нужна максимальная точность

## Подходы к pre-training

Разные foundation models используют разные стратегии pre-training:

### 1. Tokenization-based (Chronos)

Квантование значений временного ряда в дискретные токены. Ряд становится «текстом», и можно использовать language model архитектуры.

```
[100.5, 102.3, 105.1, 103.8] → [token_512, token_523, token_551, token_538]
```

**Преимущества:** Можно использовать готовые LLM архитектуры (T5, GPT)
**Недостатки:** Потеря точности при квантовании

### 2. Patching-based (TimesFM, Moirai)

Группировка соседних точек в патчи (как в PatchTST), но с pre-training на миллионах рядов.

```
[x₁, x₂, ..., x₁₆] → patch₁ → embedding
```

**Преимущества:** Сохраняет непрерывность значений
**Недостатки:** Требует кастомную архитектуру

### 3. Lag-based (Lag-Llama)

Использование лаговых признаков + decoder-only transformer.

```
Вход: [x_{t-1}, x_{t-2}, ..., x_{t-L}] + lagged_values
```

**Преимущества:** Probabilistic output из коробки
**Недостатки:** Фиксированный набор лагов

## Ландшафт моделей (2024-2025)

| Модель | Разработчик | Архитектура | Открытая? | Ключевая идея |
|--------|-------------|-------------|-----------|---------------|
| **Chronos**[^chronos] | Amazon | T5 (enc-dec) | Да | Токенизация значений |
| **TimeGPT**[^timegpt] | Nixtla | Transformer | Нет (API) | Первая коммерческая |
| **TimesFM**[^timesfm] | Google | Decoder-only | Да | Patching + масштаб |
| **Moirai**[^moirai] | Salesforce | Encoder | Да | Any-variate, masked |
| **Lag-Llama**[^lagllama] | CMU | LLaMA-style | Да | Probabilistic, lags |
| **Toto**[^toto] | Databricks | Transformer | Да | Multimodal + time series |

### Размеры моделей

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    РАЗМЕРЫ FOUNDATION MODELS                            │
│                                                                         │
│  Chronos-T5-Tiny      8M параметров    ████                            │
│  Chronos-T5-Mini     20M параметров    ████████                        │
│  Chronos-T5-Small    46M параметров    ████████████                    │
│  Chronos-T5-Base    200M параметров    ████████████████████            │
│  Chronos-T5-Large   710M параметров    ████████████████████████████    │
│                                                                         │
│  TimesFM-200M       200M параметров    ████████████████████            │
│                                                                         │
│  Moirai-Small        14M параметров    █████                           │
│  Moirai-Base        311M параметров    █████████████████████           │
│  Moirai-Large       385M параметров    ██████████████████████          │
│                                                                         │
│  Lag-Llama            7M параметров    ███                             │
│                                                                         │
│  Для сравнения:                                                        │
│  N-BEATS              ~5M параметров   ██                              │
│  PatchTST           ~300K параметров   █                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Когда использовать foundation models

### ✅ Хороший выбор

**Cold start / мало данных.** Если у вас 50-200 точек истории — обучить task-specific модель сложно. Foundation model справится.

```
Сценарий: Новый продукт в каталоге, 2 месяца истории продаж.
Task-specific: Недостаточно данных для обучения.
Foundation model: Zero-shot прогноз сразу.
```

**Много разнородных рядов.** Если у вас 10,000 рядов из разных доменов — foundation model адаптируется к каждому.

**Прототипирование.** Быстро получить baseline для оценки сложности задачи.

**Нет GPU для обучения.** Foundation models работают на inference даже на CPU (с меньшей скоростью).

### ❌ Плохой выбор

**Много однородных данных.** Если у вас 100,000 рядов из одного домена — task-specific модель (N-HiTS, PatchTST) часто будет лучше.

**Критичная точность.** Foundation models — generalists. На конкретной задаче специализированная модель может быть точнее.

**Latency-critical приложения.** Foundation models большие. Если нужен прогноз за <10ms — маленькая task-specific модель быстрее.

**Специфический домен.** Если данные сильно отличаются от pre-training (например, квантовые измерения) — foundation model может не помочь.

## Сравнение с task-specific моделями

| Критерий | Foundation Models | Task-Specific |
|----------|-------------------|---------------|
| **Данные для работы** | 0 (zero-shot) | Сотни-тысячи точек |
| **Время до первого прогноза** | Секунды | Часы (обучение) |
| **Точность на своём домене** | Хорошая | Отличная |
| **Точность на новом домене** | Хорошая | Плохая |
| **Размер модели** | 10M - 700M | 100K - 5M |
| **Inference latency** | 100ms - 1s | 1ms - 50ms |
| **Вычислительные ресурсы** | GPU желателен | CPU достаточно |

## Практические рекомендации

### Выбор модели

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ДЕРЕВО РЕШЕНИЙ: КАКУЮ МОДЕЛЬ ВЫБРАТЬ                │
│                                                                         │
│  Сколько данных?                                                        │
│       │                                                                 │
│       ├── < 100 точек ──→ Foundation Model (zero-shot)                 │
│       │                    • Chronos для point forecast                │
│       │                    • Lag-Llama для probabilistic               │
│       │                                                                 │
│       ├── 100 - 1000 точек ──→ Foundation Model (fine-tuning?)         │
│       │                        или Task-Specific с регуляризацией     │
│       │                                                                 │
│       └── > 1000 точек ──→ Task-Specific (N-HiTS, PatchTST)           │
│                            или Fine-tuned Foundation                  │
│                                                                         │
│  Нужен probabilistic output?                                           │
│       │                                                                 │
│       ├── Да ──→ Lag-Llama, Chronos, DeepAR                            │
│       │                                                                 │
│       └── Нет ──→ Chronos, TimesFM, Moirai                             │
│                                                                         │
│  Multivariate важен?                                                    │
│       │                                                                 │
│       ├── Да ──→ Moirai (any-variate)                                  │
│       │                                                                 │
│       └── Нет ──→ Chronos, TimesFM                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Workflow с foundation models

1. **Начните с zero-shot.** Получите baseline без обучения.

2. **Сравните с наивными бейзлайнами.** Если foundation model хуже Seasonal Naive — данные слишком специфичны.

3. **Попробуйте fine-tuning.** Если есть достаточно данных (>500 точек на ряд) и zero-shot не удовлетворяет.

4. **Сравните с task-specific.** Обучите N-HiTS или PatchTST на тех же данных. Выберите лучшее.

## Выводы

1. **Foundation models — новая парадигма.** Pre-training на миллионах рядов позволяет делать zero-shot прогнозы.

2. **Главное преимущество:** Работают без обучения. Решают проблему cold start.

3. **Главное ограничение:** Generalists. На конкретной задаче специализированная модель может быть лучше.

4. **Практика:** Начинайте с zero-shot, затем fine-tuning если нужно, затем сравнивайте с task-specific.

5. **Тренд:** Foundation models становятся лучше и доступнее. В 2025 году это уже не экзотика, а mainstream подход.

В следующих главах подробно рассмотрим конкретные модели: Chronos (токенизация), TimeGPT (коммерческий API), TimesFM (Google), Moirai (any-variate), Lag-Llama (probabilistic).

---

## Ссылки

[^chronos]: Ansari, A., et al. (2024). Chronos: Learning the Language of Time Series. *arXiv preprint*. https://arxiv.org/abs/2403.07815

[^timegpt]: Garza, A., & Mergenthaler-Canseco, M. (2023). TimeGPT-1. *arXiv preprint*. https://arxiv.org/abs/2310.03589

[^timesfm]: Das, A., et al. (2024). A decoder-only foundation model for time-series forecasting. *ICML 2024*. https://arxiv.org/abs/2310.10688

[^moirai]: Woo, G., et al. (2024). Unified Training of Universal Time Series Forecasting Transformers. *ICML 2024*. https://arxiv.org/abs/2402.02592

[^lagllama]: Rasul, K., et al. (2024). Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting. *arXiv preprint*. https://arxiv.org/abs/2310.08278

[^toto]: Boahen, E., et al. (2024). Toto: Time Series Optimized Transformer. *Databricks*. https://www.databricks.com/blog/introducing-toto-first-multimodal-foundation-model-time-series
