# Moirai. Any-variate подход от Salesforce

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/privettoha/neural-forecast-book/blob/main/notebooks/34_moirai.ipynb)

Moirai ([статья](https://arxiv.org/abs/2402.02592), ICML 2024 Oral) — first any-variate foundation model для временных рядов от Salesforce. В отличие от TimesFM и Chronos, которые работают только с univariate рядами, Moirai способна обрабатывать multivariate данные произвольной размерности, включая dynamic covariates[^moirai-paper]

[^moirai-paper]: Woo, G., et al. "Unified Training of Universal Time Series Forecasting Transformers." ICML 2024 Oral. https://arxiv.org/abs/2402.02592

## Семейство Moirai

Salesforce выпустил несколько версий модели:

| Версия | Архитектура | Особенности |
|--------|-------------|-------------|
| **Moirai 1.0** | Masked encoder | Any-variate, multi-patch, mixture distribution[^moirai-10] |
| **Moirai 1.1** | Masked encoder | Improved version, Jun 2024[^moirai-11] |
| **Moirai-MoE** | Masked encoder + MoE | Sparse mixture of experts, Oct 2024[^moirai-moe] |
| **Moirai 2.0** | **Decoder-only** | Quantile loss, single patch, Nov 2025[^moirai-20] |

[^moirai-10]: Woo, G., et al. "Moirai." ICML 2024. https://arxiv.org/abs/2402.02592
[^moirai-11]: GitHub: SalesforceAIResearch/uni2ts. "Jun 2024: Released Moirai-1.1-R model weights." https://github.com/SalesforceAIResearch/uni2ts
[^moirai-moe]: Liu, X., et al. "Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts." arXiv:2410.10469. https://arxiv.org/abs/2410.10469
[^moirai-20]: Liu, C., et al. "Moirai 2.0: When Less Is More for Time Series Forecasting." arXiv:2511.11698. November 2025. https://arxiv.org/abs/2511.11698

**Moirai 2.0** — радикальный редизайн:
➖ **2x быстрее** и **30x меньше** чем Moirai 1.0-Large
➖ **#1 по MASE** на GIFT-Eval среди non-leaking моделей[^moirai-20-blog]

[^moirai-20-blog]: Salesforce Blog. "Introducing Moirai 2.0." November 2025. "Moirai 2.0 achieves the best MASE score among all non–test-data-leaking foundation models." https://www.salesforce.com/blog/moirai-2-0/

## Архитектура Moirai 1.x: Any-Variate

### Flattening

Центральная идея — «уплощение» multivariate ряда в единую последовательность[^moirai-flatten]:

[^moirai-flatten]: Woo, G., et al. "Moirai." ICML 2024. Section 3.2: "flatten multivariate time series, considering all variates as a single sequence." https://arxiv.org/abs/2402.02592

```
3-variate time series:
  Variate 0 (target): [p0, p1, p2] → patch tokens
  Variate 1 (target): [p0, p1, p2] → patch tokens
  Variate 2 (covariate): [p0, p1, p2, p3] → patch tokens (включая forecast horizon)
    ↓
Flattened sequence: [v0_p0, v0_p1, v0_p2, v1_p0, v1_p1, ...]
    + Time ID (позиция во времени)
    + Variate ID (какой ряд)
    ↓
Masked Encoder Transformer
    ↓
Mixture Distribution parameters
```

### Multi-Patch Size Projection

Moirai 1.x использует **5 размеров патчей** для разных частот[^moirai-patch]:

[^moirai-patch]: Woo, G., et al. "Moirai." ICML 2024. Section 3.1, Table 4. https://arxiv.org/abs/2402.02592

| Частота | Patch size |
|---------|------------|
| Seconds, minutes | 128 |
| Hours | 64 |
| Days | 32 |
| Weeks | 16 |
| Months, quarters, years | 8 |

Для каждого размера — отдельные input/output projection layers

### Any-Variate Attention с RoPE

Модифицированный attention с **Rotary Positional Embeddings (RoPE)**[^moirai-rope]:

[^moirai-rope]: Woo, G., et al. "Moirai." ICML 2024. Section 3.2: "RoPE is applied separately for temporal position and variate index." https://arxiv.org/abs/2402.02592

➖ **Permutation equivariance** — порядок подачи variates можно менять
➖ **Permutation invariance** — абсолютные индексы variates не важны

### Mixture Distribution

Moirai 1.x предсказывает параметры **смеси распределений**[^moirai-mixture]:

[^moirai-mixture]: Woo, G., et al. "Moirai." ICML 2024. Section 3.3, Appendix B.3. https://arxiv.org/abs/2402.02592

$$p(y) = \sum_{k=1}^{K} \pi_k \cdot f_k(y | \theta_k)$$

Включает Student-t (общий случай), Log-normal (положительные значения), Negative binomial (count data)

## Архитектура Moirai 2.0: Simpler is Better

Moirai 2.0 отказывается от ключевых решений 1.x[^moirai-20-changes]:

[^moirai-20-changes]: Liu, C., et al. "Moirai 2.0." arXiv:2511.11698. Section 4: "Moirai 2.0 replaces masked-encoder training, multi-patch inputs, and mixture-distribution outputs with a simpler decoder-only architecture, single patch, and quantile loss." https://arxiv.org/abs/2511.11698

| Компонент | Moirai 1.x | Moirai 2.0 |
|-----------|------------|------------|
| Архитектура | Masked encoder | **Decoder-only** |
| Patch sizes | Multi (5 sizes) | **Single** |
| Output | Mixture distribution | **Quantile forecasts** |
| Prediction | Single-token | **Multi-token** |

🔢 **Почему quantile loss лучше mixture?**
➖ Более robust к outliers
➖ Нет variance collapse/explosion
➖ Напрямую оптимизирует CRPS метрику
➖ Проще в обучении[^moirai-quantile]

[^moirai-quantile]: Liu, C., et al. "Moirai 2.0." arXiv:2511.11698. Section 4.1: "Compared to the distribution NLL loss, which may suffer from variance collapse, explosion, or unstable gradients under outliers, the quantile loss is more robust." https://arxiv.org/abs/2511.11698

## Moirai-MoE: Token-Level Specialization

Moirai-MoE (октябрь 2024) — альтернативный подход к heterogeneity[^moirai-moe-paper]:

[^moirai-moe-paper]: Liu, X., et al. "Moirai-MoE." arXiv:2410.10469. https://arxiv.org/abs/2410.10469

Вместо multi-patch (frequency-level specialization) → **Sparse MoE** (token-level specialization)

🔢 **Проблемы frequency-level:**
➖ Frequency — ненадёжный индикатор паттернов
➖ Даже в коротком окне ряды могут иметь разные distributions[^moirai-moe-problem]

[^moirai-moe-problem]: Liu, X., et al. "Moirai-MoE." arXiv:2410.10469. Abstract: "Frequency is not a reliable indicator for grouping pretraining data." https://arxiv.org/abs/2410.10469

🔢 **Результат:**
Moirai-MoE-Small (11M активных параметров) превосходит Moirai-Large (300M+) на многих бенчмарках

## LOTSA и GIFT-Eval

Salesforce создал важнейшую инфраструктуру для исследований:

🔢 **LOTSA (Large-scale Open Time Series Archive)**[^lotsa]
➖ **27B observations** across 9 domains
➖ Energy, transport, finance, nature/climate, web, retail, healthcare, cloud, banking
➖ Открытый датасет на HuggingFace

[^lotsa]: Woo, G., et al. "Moirai." ICML 2024. Section 4, Table 2-3. https://arxiv.org/abs/2402.02592

🔢 **GIFT-Eval**[^gifteval]
➖ Первый comprehensive benchmark для TSFM
➖ **37 foundation models** на leaderboard
➖ Explicit tracking of data leakage

[^gifteval]: Aksu, T., et al. "GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation." arXiv:2410.10393. https://arxiv.org/abs/2410.10393

## Размеры моделей

🔢 **Moirai 1.x:**

| Размер | Параметры |
|--------|-----------|
| Small | 14M |
| Base | 91M |
| Large | 311M |

🔢 **Moirai 2.0:**

| Размер | Характеристики |
|--------|----------------|
| Small | 30x меньше чем 1.0-Large, 2x быстрее[^moirai-20-size] |

[^moirai-20-size]: Salesforce Blog. "Moirai 2.0 is twice as fast and thirty times smaller than its prior best version, Moirai 1.0-Large, while also performing better." https://www.salesforce.com/blog/moirai-2-0/

## Данные обучения

🔢 **Moirai 1.x:** LOTSA (~27B observations)[^moirai-data-1]

[^moirai-data-1]: Woo, G., et al. "Moirai." ICML 2024. https://arxiv.org/abs/2402.02592

🔢 **Moirai 2.0:** Expanded corpus[^moirai-data-2]
➖ **36M series, ~295B observations**
➖ GIFT-Eval Pretrain + Train
➖ Chronos mixup data (non-leaking)
➖ KernelSynth synthetic data
➖ Salesforce internal operational data (~2.15M series)

[^moirai-data-2]: Liu, C., et al. "Moirai 2.0." arXiv:2511.11698. Section 2.4. https://arxiv.org/abs/2511.11698

## Что умеет

➖ **Any-variate** — произвольное количество переменных
➖ **Dynamic covariates** — ряды с известными future values
➖ **Zero-shot forecasting** — без обучения на ваших данных
➖ **Probabilistic forecasts** — quantiles или mixture distribution
➖ **Открытые веса** — research license

## Когда использовать

👍 **Хорошо работает:**
➖ Multivariate данные с корреляциями между рядами
➖ Dynamic covariates (известные future values)
➖ Нужен any-variate flexibility
➖ Zero-shot на разнородных данных

👎 **Проблемы:**
➖ **Вычислительная сложность** — any-variate attention квадратична[^moirai-complexity]
➖ **Masked encoder** (1.x) — фиксированный горизонт при обучении
➖ **Mixture distribution** (1.x) — сложнее в оптимизации[^moirai-mixture-problem]
➖ **Diminishing returns** — performance plateaus с увеличением параметров[^moirai-scaling]

[^moirai-complexity]: Woo, G., et al. "Moirai." ICML 2024. Section 3.2. https://arxiv.org/abs/2402.02592
[^moirai-mixture-problem]: Liu, C., et al. "Moirai 2.0." arXiv:2511.11698: "Mixture of distributions was an intuitive way to enhance probabilistic forecasting, it proved empirically less effective in practice." https://arxiv.org/abs/2511.11698
[^moirai-scaling]: Liu, C., et al. "Moirai 2.0." arXiv:2511.11698: "Model performance plateaus with increasing parameter count and declines at longer horizons." https://arxiv.org/abs/2511.11698

## Moirai vs другие модели

| Критерий | Moirai 2.0 | Moirai 1.x | Chronos-2 | TimesFM |
|----------|------------|------------|-----------|---------|
| Архитектура | Decoder-only | Masked enc | T5 encoder | Decoder-only |
| Multivariate | ✓ | ✓ | ✓ | ✗ |
| Covariates | ✓ | ✓ | ✓ | XReg |
| Output | Quantiles | Mixture | Quantiles | Quantiles |
| GIFT-Eval | #1 MASE* | Top-5 | SOTA | #1 open |

*среди non-leaking моделей

## Реализации

| Ресурс | Ссылка |
|--------|--------|
| Официальный код (Uni2TS) | [SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts) |
| Moirai 2.0 | [Salesforce/moirai-2.0-R-small](https://huggingface.co/Salesforce/moirai-2.0-R-small) |
| Moirai 1.1 | [Salesforce/moirai-1.1-R-large](https://huggingface.co/Salesforce/moirai-1.1-R-large) |
| Moirai-MoE | [Salesforce/moirai-moe-base](https://huggingface.co/Salesforce/moirai-moe-1.0-R-base) |
| LOTSA dataset | [Salesforce/lotsa_data](https://huggingface.co/datasets/Salesforce/lotsa_data) |
| GIFT-Eval Leaderboard | [Salesforce/GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) |

## Что дальше

Moirai показала, что any-variate forecasting возможен, а Moirai 2.0 продемонстрировала, что simpler is better: decoder-only + quantile loss + single patch превосходит сложную архитектуру 1.x.

GIFT-Eval стал стандартным benchmark для foundation models, а проблема data leakage получила explicit tracking.

:::{seealso}
**Источники:**
- Woo, G., et al. (2024). [Unified Training of Universal Time Series Forecasting Transformers](https://arxiv.org/abs/2402.02592). ICML 2024 Oral
- Liu, X., et al. (2024). [Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts](https://arxiv.org/abs/2410.10469). arXiv
- Liu, C., et al. (2025). [Moirai 2.0: When Less Is More for Time Series Forecasting](https://arxiv.org/abs/2511.11698). arXiv
- Aksu, T., et al. (2024). [GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation](https://arxiv.org/abs/2410.10393). arXiv
- [Uni2TS GitHub](https://github.com/SalesforceAIResearch/uni2ts)
- [LOTSA dataset](https://huggingface.co/datasets/Salesforce/lotsa_data)
- [Salesforce Blog: Moirai](https://www.salesforce.com/blog/moirai/)
- [Salesforce Blog: Moirai 2.0](https://www.salesforce.com/blog/moirai-2-0/)
:::