# TiRex. xLSTM возвращается

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/privettoha/neural-forecast-book/blob/main/notebooks/24_tirex.ipynb)

TiRex ([статья](https://arxiv.org/abs/2505.23719), NeurIPS 2025) — модель от NX-AI, которая в 2025 году занимает #1 позицию на бенчмарке [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval), обходя Chronos Bolt (Amazon), TimesFM (Google), Moirai (Salesforce) и TabPFN-TS (Prior Labs)[^tirex-leaderboard]. И всё это на основе xLSTM — модернизированной версии LSTM от [Сеппа Хохрайтера](https://en.wikipedia.org/wiki/Sepp_Hochreiter), одного из создателей оригинальной архитектуры в 1997 году

[^tirex-leaderboard]: Auer, A., et al. "TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning." NeurIPS 2025. Figure 10: "TiRex leads the GiftEval-ZS benchmark." https://arxiv.org/abs/2505.23719

## Идея

Название расшифровывается как **Time Series Rex** — «король временных рядов». Ключевой вопрос, на который отвечает TiRex: можно ли взять идею LSTM, исправить её известные проблемы, и получить архитектуру, конкурентную с трансформерами для zero-shot прогнозирования?

Главное преимущество LSTM над трансформерами и SSM — **state tracking**: способность отслеживать скрытые состояния процесса во времени[^xlstm-statetracking]. Трансформеры и SSM математически не способны решать задачи state tracking (Merrill et al., 2024), а для временных рядов это критично — нужно «помнить» текущее состояние системы, чтобы корректно прогнозировать будущее[^tirex-statetracking]

[^xlstm-statetracking]: Beck, M., et al. "xLSTM: Extended Long Short-Term Memory." NeurIPS 2024. Section 4.1: "Memory mixing enables to solve state tracking problems, and therefore LSTMs are more expressive than SSMs and Transformers." https://arxiv.org/abs/2405.04517

[^tirex-statetracking]: Auer, A., et al. "TiRex." NeurIPS 2025. Section 1: "Unlike transformers, state-space models, or parallelizable RNNs such as RWKV, TiRex retains state-tracking, a critical property for long-horizon forecasting." https://arxiv.org/abs/2505.23719

## Почему LSTM нужно было переизобретать

Классический LSTM имеет три фундаментальные проблемы[^xlstm-limitations]:

[^xlstm-limitations]: Beck, M., et al. "xLSTM." NeurIPS 2024. Section 1, Figure 2. https://arxiv.org/abs/2405.04517

➖ **Затухание градиентов**: несмотря на механизм гейтов, информация «затухает» при прохождении через сотни шагов. Сигмоидные гейты ограничены диапазоном (0, 1), после 100 шагов сигнал $0.99^{100} \approx 0.37$

➖ **Ограниченная ёмкость памяти**: скрытое состояние — вектор фиксированного размера. Вся информация о прошлом должна сжаться в этот вектор

➖ **Невозможность параллелизации**: каждый шаг зависит от предыдущего: $h_t = f(h_{t-1}, x_t)$. Нельзя вычислить $h_{100}$, не вычислив $h_1, ..., h_{99}$

## xLSTM: что изменилось

[xLSTM](https://arxiv.org/abs/2405.04517) (NeurIPS 2024) вводит два новых типа ячеек: **sLSTM** (scalar LSTM) и **mLSTM** (matrix LSTM)[^xlstm-variants]. TiRex использует **sLSTM** — именно он сохраняет способность к state tracking[^tirex-slstm]

[^xlstm-variants]: Beck, M., et al. "xLSTM." NeurIPS 2024. Section 3: "(i) sLSTM with a scalar memory, a scalar update, and new memory mixing, (ii) mLSTM that is fully parallelizable with a matrix memory and a covariance update rule." https://arxiv.org/abs/2405.04517

[^tirex-slstm]: Auer, A., et al. "TiRex." NeurIPS 2025. Figure 2: "TiRex adopts the block design proposed by Beck et al. (2025), but substitutes the mLSTM with a sLSTM module as the sequence mixing component. Only sLSTM allows for state-tracking." https://arxiv.org/abs/2505.23719

🔢 **Экспоненциальные гейты**
В классическом LSTM: $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ — сигмоида ограничена (0, 1)

В xLSTM: $f_t = \exp(w_f \cdot x_t + b_f)$ — гейт может быть >1, сигнал не только сохраняется, но и усиливается. Специальная нормализация предотвращает взрыв значений[^xlstm-expgating]

[^xlstm-expgating]: Beck, M., et al. "xLSTM." NeurIPS 2024. Section 3.1: "We introduce exponential gating with appropriate normalization and stabilization techniques." https://arxiv.org/abs/2405.04517

🔢 **Memory mixing (только sLSTM)**
sLSTM добавляет механизм «смешивания памяти» между ячейками — именно это позволяет решать задачи state tracking[^xlstm-memorymixing]. mLSTM не имеет memory mixing (зато полностью параллелизуем), поэтому TiRex выбирает sLSTM

[^xlstm-memorymixing]: Beck, M., et al. "xLSTM." NeurIPS 2024. Section 4.1: "In contrast to the new sLSTM, [other] approaches do not allow memory mixing. Memory mixing enables to solve state tracking problems." https://arxiv.org/abs/2405.04517

🔢 **mLSTM (для справки)**
mLSTM хранит состояние в матрице $C_t \in \mathbb{R}^{d \times d}$ вместо вектора — квадратично больше ёмкости. Использует covariance update rule, похожее на key-value из трансформеров[^xlstm-mlstm]. TiRex его не использует, но важно знать для понимания семейства xLSTM

[^xlstm-mlstm]: Beck, M., et al. "xLSTM." NeurIPS 2024. Section 3.2: "mLSTM that is fully parallelizable with a matrix memory and a covariance update rule." https://arxiv.org/abs/2405.04517

## Архитектура TiRex

TiRex строится на sLSTM с несколькими компонентами, специфичными для временных рядов[^tirex-arch]:

[^tirex-arch]: Auer, A., et al. "TiRex." NeurIPS 2025. Figure 2, Section 3. https://arxiv.org/abs/2505.23719

🔢 **Patching**
Временной ряд разбивается на патчи (аналогично [PatchTST](https://arxiv.org/abs/2211.14730)), каждый патч проецируется в эмбеддинг. Максимальная длина контекста — 2048[^tirex-context]

[^tirex-context]: AI Horizon Forecast. "TiRex: LSTMs Take The Lead Again." 2025: "The model supports a maximum context length of 2048." https://aihorizonforecast.substack.com/p/tirex-lstms-take-the-lead-again-in

```
Input: x ∈ ℝ^T
    ↓
Patching → последовательность патчей
    ↓
Linear projection + Instance Normalization
```

🔢 **sLSTM backbone**
Последовательность обрабатывается стеком sLSTM-блоков. Каждый блок включает:
➖ sLSTM с экспоненциальными гейтами
➖ RMSNorm (вместо LayerNorm)
➖ Feed-forward network
➖ Residual skip connections[^tirex-blocks]

[^tirex-blocks]: Auer, A., et al. "TiRex." NeurIPS 2025. Figure 2: "Each block comprises a sLSTM module followed by a feed-forward network, with both components preceded by RMSNorm. Additionally, all sLSTM and feedforward layers include residual skip connections." https://arxiv.org/abs/2505.23719

🔢 **CPM (Causal Patch Masking)**
Ключевая инновация TiRex — специальная стратегия маскирования при обучении[^tirex-cpm]. Вместо того чтобы подставлять предсказания обратно как входы (как в DeepAR), TiRex помечает будущие шаги как missing values. Скрытое состояние sLSTM переносит вперёд не точечную оценку, а полное predictive distribution

[^tirex-cpm]: Auer, A., et al. "TiRex." NeurIPS 2025. Section 3.2: "We propose a training-time masking strategy called CPM." https://arxiv.org/abs/2505.23719

Преимущества CPM:
➖ **Uncertainty propagation**: модель не коммитится к точечной оценке на границах патчей
➖ **Coherence**: квантильные прогнозы второго патча условны на всём posterior, а не на leaked estimate
➖ Избегает «variance collapse» при передаче медианы[^tirex-cpm-benefits]

[^tirex-cpm-benefits]: AI Horizon Forecast. "TiRex: LSTMs Take The Lead Again." 2025. https://aihorizonforecast.substack.com/p/tirex-lstms-take-the-lead-again-in

🔢 **Квантильный выход**
TiRex выдаёт 9 квантилей [0.1, 0.2, ..., 0.9] для каждого шага горизонта за один forward pass — без Monte Carlo sampling[^tirex-quantiles]

[^tirex-quantiles]: GitHub: NX-AI/tirex. "Quantile Predictions: TiRex provides both point estimates and quantile estimates." https://github.com/NX-AI/tirex

## Экспериментальные результаты

Авторы тестируют на двух основных бенчмарках[^tirex-benchmarks]:

[^tirex-benchmarks]: Auer, A., et al. "TiRex." NeurIPS 2025. Section 4. https://arxiv.org/abs/2505.23719

**GIFT-Eval** — 28 датасетов, >144 000 временных рядов, 177M точек данных. Покрывает short, medium, long-term горизонты[^tirex-gifteval]

[^tirex-gifteval]: GitHub: autogluon/autogluon. Issue #5146: "TiRex achieves state-of-the-art performance on the GiftEval benchmark, a large-scale collection of 28 datasets." https://github.com/autogluon/autogluon/issues/5146

**Chronos-ZS** — 27 датасетов для short-term forecasting

🔢 **Ключевые результаты** (Figure 10 в статье):
➖ TiRex — **#1 overall** на GiftEval-ZS
➖ TiRex — **первая zero-shot модель**, которая превосходит PatchTST и TFT в long-term задачах[^tirex-vs-patchtst]
➖ В отличие от других моделей, которые специализируются на short ИЛИ long-term, TiRex отлично работает на **обоих**[^tirex-both]

[^tirex-vs-patchtst]: AI Horizon Forecast. "TiRex: LSTMs Take The Lead Again." 2025: "TiRex becomes the first zero-shot model to beat PatchTST and TFT in long-term tasks." https://aihorizonforecast.substack.com/p/tirex-lstms-take-the-lead-again-in

[^tirex-both]: Auer, A., et al. "TiRex." NeurIPS 2025. Abstract: "TiRex sets a new state of the art in zero-shot time series forecasting... outperforming significantly larger models... across both short- and long-term forecasts." https://arxiv.org/abs/2505.23719

🔢 **Важно про data leakage**: авторы явно указывают, какие модели имеют overlap с тестовыми данными (отмечены как "Zero-shot Leak" в Figure 10). TiRex чист — версия 1.1-gifteval специально очищена от overlap[^tirex-leakage]

[^tirex-leakage]: HuggingFace: NX-AI/TiRex-1.1-gifteval. "This specific version includes the 1.1 improvements plus the pretraining dataset has been cleaned to remove overlaps with the GIFT-Eval test dataset." https://huggingface.co/NX-AI/TiRex-1.1-gifteval

## Что умеет

➖ **SOTA на zero-shot бенчмарках**: превосходит модели с 200M-500M параметров (Chronos Bolt, TimesFM-2.0, Toto), имея только 35M[^tirex-params]
➖ **State tracking**: уникальная способность среди foundation models, критичная для длинных горизонтов
➖ **Работает и на short, и на long horizons**: редкое сочетание
➖ **Эффективный вероятностный выход**: квантили за один forward pass
➖ **Компактность**: 35M параметров, можно запускать на edge devices[^tirex-edge]
➖ **Полностью открытая**: веса, код, данные доступны

[^tirex-params]: GitHub: NX-AI/tirex. "TiRex is a 35M parameter pre-trained time series forecasting model." https://github.com/NX-AI/tirex
[^tirex-edge]: NX-AI Solutions. "You can run TiRex on a PLC or an even smaller devices." https://www.nx-ai.com/en/solutions

## Когда использовать

👍 **Хорошо работает:**
➖ Zero-shot прогноз без обучения на своих данных
➖ Баланс качества на коротких и длинных горизонтах
➖ Вероятностный выход без сложностей Monte Carlo
➖ Nvidia GPU с compute capability ≥ 8.0 (Ampere+)[^tirex-gpu]
➖ Sparse/intermittent data (благодаря CPM)

[^tirex-gpu]: GitHub: NX-AI/tirex. "TiRex is currently only tested on Linux systems and Nvidia GPUs with compute capability >= 8.0." https://github.com/NX-AI/tirex

👎 **Проблемы:**
➖ **Зависимость от CUDA kernels**: без кастомных ядер — экспериментальный режим, «likely degrade forecasting results»[^tirex-cuda]
➖ **Не поддерживает ковариаты**: zero-shot режим не принимает static/dynamic features[^tirex-nocov]
➖ **Последовательный инференс**: рекуррентная природа sLSTM
➖ **Фиксированные квантили**: только [0.1, ..., 0.9], для 0.95 нужна интерполяция
➖ **Новизна**: мало практического опыта в продакшене

[^tirex-cuda]: GitHub: NX-AI/tirex. "However, this is at the moment EXPERIMENTAL, slows down TiRex considerably and likely degrade forecasting results!" https://github.com/NX-AI/tirex
[^tirex-nocov]: GitHub: autogluon/autogluon. "TiRex doesn't support covariates or static features in zero-shot mode." https://github.com/autogluon/autogluon/issues/5146

## TiRex vs другие модели

| Критерий | TiRex | DeepAR | PatchTST | Chronos |
|----------|-------|--------|----------|---------|
| Архитектура | sLSTM (xLSTM) | LSTM | Transformer | T5 |
| Параметры | 35M | ~5M | ~10M | 20M–700M |
| Zero-shot | ✓ | ✗ | ~ | ✓ |
| Вероятностный выход | ✓ Квантили | ✓ Распределение | ✗ | ✓ Сэмплы |
| Ковариаты | ✗ | ✓ | ✗ | ✗ |
| State tracking | ✓ | ✓ | ✗ | ✗ |
| GIFT-Eval rank | #1 | — | Top-10 | Top-5 |

## Реализации

| Ресурс | Ссылка |
|--------|--------|
| Официальный код | [NX-AI/tirex](https://github.com/NX-AI/tirex) |
| Модель на HuggingFace | [NX-AI/TiRex-1.1-gifteval](https://huggingface.co/NX-AI/TiRex-1.1-gifteval) |
| Демо | [HuggingFace Spaces](https://huggingface.co/spaces/NX-AI/TiRex-Demo) |
| xLSTM базовый код | [NX-AI/xlstm](https://github.com/NX-AI/xlstm) |

## Что дальше

TiRex показывает, что улучшенные RNN (в форме xLSTM) могут конкурировать с трансформерами и побеждать их благодаря способности к state tracking. Но это не единственный путь эволюции последовательных архитектур.

В следующем посте мы рассмотрим FlowState — модель от IBM, построенную на State Space Models (SSM). SSM — другой подход к моделированию последовательностей, который сочетает преимущества RNN (эффективный инференс) и свёрточных сетей (параллельное обучение). FlowState добавляет уникальную возможность — time-scale invariance.

:::{seealso}
**Источники:**
- Auer, A., Podest, P., Klotz, D., Böck, S., Klambauer, G., Hochreiter, S. (2025). [TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning](https://arxiv.org/abs/2505.23719). NeurIPS 2025
- Beck, M., et al. (2024). [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517). NeurIPS 2024 Spotlight
- [Официальный код TiRex](https://github.com/NX-AI/tirex) — NX-AI GitHub
- [GIFT-Eval Leaderboard](https://huggingface.co/spaces/Salesforce/GIFT-Eval) — Salesforce HuggingFace
:::