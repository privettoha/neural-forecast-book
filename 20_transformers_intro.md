## Трансформеры для временных рядов: проблемы

Когда трансформеры завоевали NLP (BERT[^bert], GPT[^gpt]) и computer vision (ViT[^vit]), казалось, что временные ряды следующие. Self-attention позволяет каждому элементу напрямую взаимодействовать с любым другим, что делает его потенциально идеальным инструментом для сложных временных зависимостей.

[^bert]: Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers." NAACL 2019. https://arxiv.org/abs/1810.04805
[^gpt]: Radford, A., et al. "Improving Language Understanding by Generative Pre-Training." 2018. https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
[^vit]: Dosovitskiy, A., et al. "An Image is Worth 16x16 Words." ICLR 2021. https://arxiv.org/abs/2010.11929

В 2020–2022 вышла серия работ: Informer[^informer], Autoformer[^autoformer], FEDformer[^fedformer]. Каждая обещала SOTA на длинных горизонтах.

[^informer]: Zhou, H., et al. "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting." AAAI 2021. https://arxiv.org/abs/2012.07436
[^autoformer]: Wu, H., et al. "Autoformer: Decomposition Transformers with Auto-Correlation." NeurIPS 2021. https://arxiv.org/abs/2106.13008
[^fedformer]: Zhou, T., et al. "FEDformer: Frequency Enhanced Decomposed Transformer." ICML 2022. https://arxiv.org/abs/2201.12740

А потом DLinear[^dlinear] — один линейный слой — побил их всех на стандартных бенчмарках. Как такое возможно?

[^dlinear]: Zeng, A., et al. "Are Transformers Effective for Time Series Forecasting?" AAAI 2023. https://arxiv.org/abs/2205.13504

### Пять проблем трансформеров на рядах

**🔢 Permutation invariance**

Self-attention вычисляет веса на основе сходства между элементами:

$$\alpha_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d})}{\sum_l \exp(q_i \cdot k_l / \sqrt{d})}$$

В формуле нет ничего, зависящего от позиций $i$ и $j$. Механизм инвариантен к перестановкам.

Для текста — positional encoding. Но для рядов позиция несёт богатую информацию: день недели, месяц, праздник. Синусоидальный encoding может быть недостаточно выразительным.

**🔢 Point-wise attention**

В NLP токен «университет» — концепт с семантикой, но во временных рядах точка 42.7 это просто число. Паттерны проявляются в последовательностях точек, не в отдельных значениях.

Можно привести аналогию, что сравнивать точки через attention это как понимать мелодию, сравнивая отдельные ноты вне контекста.

**🔢 Квадратичная сложность**

$O(L^2)$ по памяти и вычислениям. Для текста (сотни токенов) — терпимо. Для рядов (тысячи точек) — узкое место.

Informer предложил ProbSparse attention, Autoformer — auto-correlation, FEDformer — спектральные методы. Все снижают сложность, но вносят искажения.

DLinear имеет $O(L)$ без компромиссов — и работает не хуже.

**🔢 Сложность оптимизации**

Трансформеры требуют тщательной настройки: learning rate, warmup, dropout, количество слоёв и голов. Для NLP есть отработанные рецепты. Для рядов опыт только накапливается.

Простые модели (N-BEATS, DLinear) менее чувствительны к гиперпараметрам.

**🔢 Temporal vs channel attention**

Многомерные ряды: время ($T$) × каналы ($C$). Куда применять attention?

➖ **Temporal**: каждый момент смотрит на другие моменты — те же проблемы с point-wise и $O(L^2)$
➖ **Channel**: каждая переменная смотрит на другие — полезно для зависимостей между каналами, но игнорирует временную структуру

### Что работает

**Patching** — attention к группам точек, не к отдельным. Снижает сложность, патч несёт больше информации, чем точка. → PatchTST[^patchtst]

[^patchtst]: Nie, Y., et al. "A Time Series is Worth 64 Words." ICLR 2023. https://arxiv.org/abs/2211.14730

**Channel independence** — каждый канал обрабатывается отдельно. Упрощает задачу, часто работает не хуже cross-channel моделей.

**Inverted attention** — attention по каналам, не по времени. Каналов обычно меньше, чем точек → меньше сложность. → iTransformer[^itransformer]

[^itransformer]: Liu, Y., et al. "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting." ICLR 2024. https://arxiv.org/abs/2310.06625

**Pretrained representations** — трансформер для построения представления, прогноз — простым слоем. → Foundation models.

### Хронология

| Период | Что происходило |
|--------|-----------------|
| 2019–2020 | Первые попытки vanilla transformer на рядах, смешанные результаты |
| 2021 | Informer и последователи — фокус на эффективности (sparse attention, spectral methods) |
| 2022 | DLinear шок — линейная модель конкурентоспособна |
| 2023 | PatchTST, iTransformer — фундаментальные изменения архитектуры |
| 2023–2024 | Foundation models — фокус смещается к данным и предобучению |

### Что рассматриваем

**PatchTST** — решает point-wise attention через патчинг, channel independence как преимущество.

**iTransformer** — инвертирует подход, attention по каналам вместо времени.

Informer, Autoformer, FEDformer **не включаем** — исторически важны, но уступают поздним подходам и даже простым бейзлайнам.

### Когда трансформер оправдан

| Вопрос | Если «нет» — альтернатива |
|--------|---------------------------|
| Достаточно данных? | N-HiTS, N-BEATS |
| Сложная temporal structure? | DLinear, N-BEATS |
| Межканальные зависимости важны? | PatchTST (channel-independent) |
| Готовы к настройке гиперпараметров? | Начните с MLP-моделей |

### Ссылки

| Ресурс | URL |
|--------|-----|
| Attention Is All You Need | https://arxiv.org/abs/1706.03762 |
| Transformers in Time Series: A Survey | https://arxiv.org/abs/2202.07125 |
| Informer | https://arxiv.org/abs/2012.07436 |
| Autoformer | https://arxiv.org/abs/2106.13008 |
| FEDformer | https://arxiv.org/abs/2201.12740 |

