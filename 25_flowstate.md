# FlowState. SSM и инвариантность к частоте дискретизации

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/privettoha/neural-forecast-book/blob/main/notebooks/25_flowstate.ipynb)

FlowState ([статья](https://arxiv.org/abs/2508.05287), NeurIPS 2025 Workshop) — модель от IBM Research, которая задаёт вопрос, который другие модели даже не рассматривают: почему модель, обученная на часовых данных, не может прогнозировать минутные или дневные? С 9.1 миллионами параметров FlowState достигает #2 на GIFT-Eval среди zero-shot моделей, обходя конкурентов в 20+ раз больше[^flowstate-leaderboard]

[^flowstate-leaderboard]: IBM Research Blog. "IBM's time-series foundation model reaches #2 on GIFT-Eval." September 2025. https://research.ibm.com/blog/SSM-time-series-model

## Идея

Интуитивно, паттерн «рост утром, спад вечером» — один и тот же паттерн, независимо от того, измеряем мы его каждую минуту или каждый час. Но большинство моделей привязаны к конкретной частоте дискретизации (sampling rate) и требуют переобучения при её изменении[^flowstate-problem]

[^flowstate-problem]: Graf, L., et al. "FlowState: Sampling Rate Invariant Time Series Forecasting." arXiv:2508.05287. Section 1: "Existing TSFMs struggle with generalization across varying context and target lengths, lack adaptability to different sampling rates." https://arxiv.org/abs/2508.05287

FlowState решает эту проблему через комбинацию **S5-based encoder** и **Functional Basis Decoder (FBD)**. S5 encoder переводит временной ряд в timescale-invariant hidden state. FBD декодирует это представление не в дискретные точки, а в **непрерывную функцию времени**, которую можно вычислить с любым шагом дискретизации[^flowstate-core]

[^flowstate-core]: Graf, L., et al. "FlowState." arXiv:2508.05287. Abstract: "A state space model (SSM) based encoder and a functional basis decoder. This design enables continuous-time modeling and dynamic time-scale adjustment." https://arxiv.org/abs/2508.05287

Название отсылает к концепции «потока» (flow) — того состояния глубокого погружения в творческую деятельность, когда время кажется эластичным[^flowstate-name]

[^flowstate-name]: IBM Research Blog. "In choosing a name for the model, researchers turned to the concept of flow, that state of deep immersion during a creative endeavor when time can feel almost elastic." https://research.ibm.com/blog/SSM-time-series-model

## State Space Models: третий путь

SSM — класс моделей из теории управления и обработки сигналов. В глубоком обучении SSM стали популярны благодаря [S4](https://arxiv.org/abs/2111.00396) (ICLR 2022) и [Mamba](https://arxiv.org/abs/2312.00752) (2023)[^ssm-history]. FlowState использует **S5** — Simplified Structured State Space[^s5-paper]

[^ssm-history]: Gu, A., et al. "Efficiently Modeling Long Sequences with Structured State Spaces." ICLR 2022. https://arxiv.org/abs/2111.00396
[^s5-paper]: Smith, J.T.H., Warrington, A., Linderman, S. "Simplified State Space Layers for Sequence Modeling." ICLR 2023 (Notable-top-5% Oral). https://arxiv.org/abs/2208.04933

🔢 **Непрерывная форма SSM:**
$$\frac{dh(t)}{dt} = Ah(t) + Bx(t)$$
$$y(t) = Ch(t) + Dx(t)$$

где $h(t)$ — скрытое состояние, $x(t)$ — вход, $y(t)$ — выход, $A, B, C, D$ — обучаемые матрицы

🔢 **Дискретная форма** (после дискретизации с шагом $\Delta$):
$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$
$$y_t = Ch_t + Dx_t$$

где $\bar{A} = \exp(\Delta A)$, $\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$

🔢 **Почему SSM важны для time-scale invariance:**
SSM имеют два режима вычислений[^ssm-modes]:

[^ssm-modes]: Smith, J.T.H., et al. "Simplified State Space Layers." ICLR 2023. Section 1. https://arxiv.org/abs/2208.04933

➖ **Рекуррентный режим** — как RNN, шаг за шагом. $O(1)$ память на шаг
➖ **Свёрточный режим** — вся последовательность параллельно через свёртку. Эффективен для обучения

Главное преимущество: SSM нативно работают с непрерывным временем, поэтому могут адаптироваться к разным sampling rates без переобучения[^flowstate-ssm-advantage]

[^flowstate-ssm-advantage]: Graf, L., et al. "FlowState." arXiv:2508.05287. Section 2: "In contrast to the transformer- and MLP-based architectures, SSMs are stateful models... A main advantage of SSMs over RNNs is that the state update of SSMs block is linear." https://arxiv.org/abs/2508.05287

## S5: Simplified Structured State Space

FlowState использует **S5** — упрощённую версию S4[^s5-core]:

[^s5-core]: Smith, J.T.H., et al. "Simplified State Space Layers." ICLR 2023. Abstract: "S5 uses one multi-input, multi-output SSM... can leverage efficient and widely implemented parallel scans." https://arxiv.org/abs/2208.04933

➖ **S4** использует много независимых single-input, single-output (SISO) SSM
➖ **S5** использует один multi-input, multi-output (MIMO) SSM

Это упрощает архитектуру и позволяет использовать эффективные **parallel scans** вместо frequency-domain подхода S4[^s5-parallel]

[^s5-parallel]: Smith, J.T.H., et al. "Simplified State Space Layers." ICLR 2023. Section 4: "S5 uses an efficient and widely implemented parallel scan. This removes the need for the convolutional and frequency-domain approach used by S4." https://arxiv.org/abs/2208.04933

S5 достигает 87.4% на Long Range Arena benchmark и 98.5% на самой сложной задаче Path-X[^s5-results]

[^s5-results]: Smith, J.T.H., et al. "Simplified State Space Layers." ICLR 2023. Abstract. https://arxiv.org/abs/2208.04933

## Functional Basis Decoder (FBD)

Ключевая инновация FlowState — **Functional Basis Decoder**. Вместо предсказания дискретных значений $\hat{y}_1, \hat{y}_2, ..., \hat{y}_H$, модель предсказывает коэффициенты разложения по базисным функциям[^fbd-core]:

[^fbd-core]: Graf, L., et al. "FlowState." arXiv:2508.05287. Section 3: "FBD utilizes a set of continuous basis functions... creates a continuous output, which can be sampled at regular intervals Δ to produce the forecast." https://arxiv.org/abs/2508.05287

$$\hat{y}(t) = \sum_{k=1}^{K} c_k \cdot \phi_k(t)$$

где $\phi_k(t)$ — базисные функции, $c_k$ — коэффициенты из decoder

🔢 **Как это работает:**
➖ S5 encoder выдаёт hidden state
➖ FBD интерпретирует каждый элемент hidden state как коэффициент соответствующей базисной функции[^fbd-interpret]
➖ Получается непрерывная функция, которую можно вычислить в любой точке времени

[^fbd-interpret]: IBM Research Blog. "The decoder does this by interpreting each element in the hidden state as a coefficient to a corresponding basis function, leading to a continuous forecast that can be spliced at any interval." https://research.ibm.com/blog/SSM-time-series-model

Хотите прогноз на 24 часа с шагом 15 минут? Вычислите $\hat{y}(t)$ для $t = 0.25, 0.5, ..., 24$. Хотите с шагом 1 час? Вычислите для $t = 1, 2, ..., 24$. Модель та же, коэффициенты те же

## Архитектура

```
Input: x ∈ ℝ^T (временной ряд)
    ↓
Input normalization
    ↓
S5 Encoder:
  - N слоёв S5
  - Каждый слой: S5 block + MLP
  - Skip connections между слоями
    ↓
Timescale-invariant hidden state
    ↓
Functional Basis Decoder:
  - Интерпретирует hidden state как коэффициенты
  - Базисные функции φ_k(t)
  - Сумма: ŷ(t) = Σ c_k · φ_k(t)
    ↓
Sample at desired intervals Δ
    ↓
Output: прогноз
```

Рисунок из статьи (Figure 1) показывает: encoder состоит из N S5 слоёв, каждый из которых включает S5 block + MLP layer. Skip connection позволяет входам распространяться к более поздним слоям[^flowstate-arch]

[^flowstate-arch]: Graf, L., et al. "FlowState." arXiv:2508.05287. Figure 1b: "The SSM encoder consists of N S5 layers, each composed of an S5 block extended with an MLP layer." https://arxiv.org/abs/2508.05287

## Обучение с CPM

FlowState использует **Causal Patch Masking (CPM)** — технику, введённую в TiRex, которая позволяет Multi-Patch-Inference (MPI)[^flowstate-cpm]:

[^flowstate-cpm]: Graf, L., et al. "FlowState." arXiv:2508.05287. Section 2: "Auer et al. (2025) have improved this autoregressive technique with MPI. In particular, they adopt contiguous patch masking (CPM) during training." https://arxiv.org/abs/2508.05287

➖ Вместо классического autoregressive подхода (подставлять предсказания обратно как входы), CPM приучает модель делать предсказания после неизвестного количества timesteps
➖ Это позволяет прогнозировать несколько патчей параллельно

Данные для обучения: комбинация **TiRex pretraining corpus** (без synthetic data), GIFT-Eval-Pretrain, Chronos dataset, плюс синтетические ряды через Gaussian Processes (KernelSynth)[^flowstate-data]

[^flowstate-data]: Graf, L., et al. "FlowState." arXiv:2508.05287. Section 4.1. https://arxiv.org/abs/2508.05287

## Экспериментальные результаты

FlowState тестируется на двух основных бенчмарках[^flowstate-benchmarks]:

[^flowstate-benchmarks]: Graf, L., et al. "FlowState." arXiv:2508.05287. Section 4. https://arxiv.org/abs/2508.05287

🔢 **GIFT-Eval (GIFT-ZS)**
➖ FlowState — **лучшая zero-shot модель** на moment публикации
➖ #2 среди всех моделей (на сентябрь 2025)[^flowstate-gift]
➖ Самая маленькая модель в top-10: 9.1M параметров vs 200M+ у конкурентов
➖ Единственная SSM-based модель в лидерах

[^flowstate-gift]: HuggingFace: ibm-research/flowstate. "Despite being more than 10x smaller than the 3 next best models, FlowState is the best Zero-Shot model on the GIFT-Eval Leaderboard." https://huggingface.co/ibm-research/flowstate

🔢 **Chronos-ZS**
➖ State-of-the-art результаты

🔢 **Уникальная способность:**
FlowState может адаптироваться online к меняющимся sampling rates — ни одна другая модель этого не умеет[^flowstate-unique]

[^flowstate-unique]: Graf, L., et al. "FlowState." arXiv:2508.05287. Abstract: "We demonstrate its unique ability to adapt online to varying input sampling rates." https://arxiv.org/abs/2508.05287

## Что умеет

➖ **Sampling rate invariance** — уникальная способность работать с данными разной частоты без переобучения[^flowstate-invariance]
➖ **Компактность** — 9.1M параметров, можно запускать на edge devices
➖ **Динамический горизонт** — благодаря FBD можно менять prediction length без retraining
➖ **Параллельный инференс нескольких патчей** — благодаря CPM
➖ **Открытый код** — доступен на HuggingFace[^flowstate-hf]

[^flowstate-invariance]: Graf, L., et al. "FlowState." arXiv:2508.05287. Section 1: "FlowState inherently adapts its internal dynamics to the input scale, enabling smaller models, reduced data requirements, and improved efficiency." https://arxiv.org/abs/2508.05287
[^flowstate-hf]: HuggingFace: ibm-research/flowstate. https://huggingface.co/ibm-research/flowstate

## Когда использовать

👍 **Хорошо работает:**
➖ Данные разной частоты в одном пайплайне (часовые и дневные метрики)
➖ Нужна компактная модель с минимальными ресурсами
➖ Важна адаптивность к изменению частоты сбора данных
➖ Zero-shot прогноз без дообучения
➖ Single-variable forecasting (web clicks, traffic volumes)[^flowstate-use]

[^flowstate-use]: Joshua Berkowitz. "FlowState: IBM's Lightweight Powerhouse." 2025. https://joshuaberkowitz.us/blog/news-1/flowstate-ibms-lightweight-powerhouse-for-flexible-time-series-forecasting-1301

👎 **Проблемы:**
➖ **Только univariate** — multivariate пока не поддерживается, но команда работает над расширением[^flowstate-limit]
➖ **Нет ковариат** — как и многие foundation models
➖ **Research license** — для commercial use нужна Granite версия[^flowstate-license]
➖ **Новизна SSM** — меньше опыта применения, чем у трансформеров

[^flowstate-limit]: IBM Research Blog. "The researchers behind FlowState are working to extend the model to complex multi-variable problems." https://research.ibm.com/blog/SSM-time-series-model
[^flowstate-license]: HuggingFace: ibm-research/flowstate. "This model card contains the model-weights for research-use only... for commercial and enterprise use, please refer to our granite releases." https://huggingface.co/ibm-research/flowstate

## FlowState vs TiRex vs другие модели

| Критерий | FlowState | TiRex | Chronos | DeepAR |
|----------|-----------|-------|---------|--------|
| Архитектура | S5 (SSM) + FBD | sLSTM (xLSTM) | T5 | LSTM |
| Параметры | 9.1M | 35M | 20M–700M | ~5M |
| Sampling rate invariance | ✓ Уникально | ✗ | ✗ | ✗ |
| Zero-shot | ✓ | ✓ | ✓ | ✗ |
| Multivariate | ✗ | ✗ | ✗ | ✓ |
| GIFT-Eval rank | #2 | #1 | Top-5 | — |
| Ковариаты | ✗ | ✗ | ✗ | ✓ |

## Реализации

| Ресурс | Ссылка |
|--------|--------|
| Модель на HuggingFace | [ibm-research/flowstate](https://huggingface.co/ibm-research/flowstate) |
| Granite Time Series (commercial) | [IBM Granite](https://www.ibm.com/granite/docs/models/time-series) |
| S5 (базовый encoder) | [lindermanlab/S5](https://github.com/lindermanlab/S5) |

## Часть IBM Granite Time Series

FlowState — часть семейства IBM Granite Time Series вместе с TinyTimeMixers (TTM) и TSPulse[^granite-family]. Все три модели — ультра-компактные (<10M параметров), способные работать без GPU

[^granite-family]: IBM Granite Docs. "Flowstate, Tiny Time Mixer (TTM), and Time Series Pulse (TSPulse) are IBM's family of ultra-lightweight pre-trained models for time series data." https://www.ibm.com/granite/docs/models/time-series

## Что дальше

Мы завершили обзор RNN/SSM-based архитектур. DeepAR заложил основы вероятностного прогнозирования с глобальными моделями. TiRex показал, что модернизированные LSTM (xLSTM) конкурентоспособны с трансформерами благодаря state tracking. FlowState продемонстрировал уникальную возможность SSM — sampling rate invariance.

В следующей части мы переходим к Foundation Models — моделям, предобученным на огромных коллекциях данных и способным работать zero-shot. Начнём с [Chronos](https://arxiv.org/abs/2403.07815), затем разберём [TimeGPT](https://docs.nixtla.io/), [TimesFM](https://arxiv.org/abs/2310.10688), [Moirai](https://arxiv.org/abs/2402.02592), [Lag-Llama](https://arxiv.org/abs/2310.08278) и [Toto](https://www.datadoghq.com/blog/ai-forecasting/).

:::{seealso}
**Источники:**
- Graf, L., Ortner, T., Woźniak, S., Pantazi, A. (2025). [FlowState: Sampling Rate Invariant Time Series Forecasting](https://arxiv.org/abs/2508.05287). NeurIPS 2025 Workshop
- Smith, J.T.H., Warrington, A., Linderman, S. (2023). [Simplified State Space Layers for Sequence Modeling](https://arxiv.org/abs/2208.04933). ICLR 2023
- [IBM Research Blog: FlowState](https://research.ibm.com/blog/SSM-time-series-model)
- [HuggingFace: ibm-research/flowstate](https://huggingface.co/ibm-research/flowstate)
- Gu, A., et al. (2022). [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396). ICLR 2022
:::