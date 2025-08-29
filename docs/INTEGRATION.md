
# Semantic Section Upgrade — Интеграция

Этот архив добавляет **смысловое разбиение по разделам**, **гибридный ретривал (BM25+вектора+MMR)**,
расширение запроса по синонимам и **человечные фолбэки** вместо «Ответ пуст».

## Что входит
- `app/services/chunking_sections.py` — разбиение по разделам и скользящие окна; метаданные секций; расширение запроса.
- `app/services/hybrid_retriever.py` — простой BM25, слияние скорингов, MMR, подмешивание релевантной секции.
- `app/services/generation_fallbacks.py` — дружелюбный фолбэк, если ответ не найден.
- `tools/apply_patch.py` — скрипт, который попытается автоматически внести минимальные правки в ваш проект.
- `examples/demo_split.py` — демо разбиения текста договора на секции.
- `examples/sample_contract.txt` — пример текста для проверки.

## Минимальные правки вручную (если не используете apply_patch.py)

### 1) Индексация: заменить старое разбиение на секционное
В `app/services/indexing.py` найдите место, где формируются чанки. Было что‑то вроде:

```python
from app.services.chunking import smart_split_text
chunks = smart_split_text(text)
```

Замените на:

```python
from app.services.chunking_sections import split_by_semantic_sections, serialize_section_chunks
section_chunks = split_by_semantic_sections(text)
chunks, metas = serialize_section_chunks(section_chunks, os.path.basename(file_path))

# Дальше — эмбеддинги на каждый чанк из `chunks`,
# а `metas` сохраните рядом с вашим индексом (Faiss/докфайл),
# чтобы в ретривере и генерации можно было использовать section_canonical.
```

### 2) Ретривал: расширение запроса, гибрид и MMR
Перед поиском по векторам:

```python
from app.services.chunking_sections import expand_query
q_expanded, target_canon = expand_query(user_query)
```

После получения косинусных скорингов от векторного индекса возьмите тексты `all_texts` и меты `all_metas`
(в том же порядке, что и скоринги) и примените BM25+fusion+MMR:

```python
from app.services.hybrid_retriever import BM25, fuse_scores, mmr_select, ensure_section_boost

bm = BM25(all_texts)
bm_scores = bm.score(q_expanded)
fused = fuse_scores(cosine_scores, bm_scores, alpha=0.65)
ranked = sorted(list(enumerate(fused)), key=lambda x: x[1], reverse=True)
selected = mmr_select(ranked, all_texts, top_k=5, lambda_=0.7)
selected = ensure_section_boost(selected, all_metas, all_texts, target_canon, limit_boost=2)
top_chunks = [all_texts[i] for i in selected]
top_metas  = [all_metas[i] for i in selected]
```

### 3) Генерация «пустого» ответа
Если `top_chunks` пуст или модель вернула пустую строку:

```python
from app.services.generation_fallbacks import empty_answer_message
return empty_answer_message(user_query, all_metas)
```

### 4) Размеры окон
Рекомендуемые параметры для юридических документов:
- `max_chars=1600`
- `overlap=250`

### 5) Логи (по желанию)
Пишите в лог top‑N секций: `section_canonical`, fused score, bm25 — это ускоряет диагностику.

## Автопатчером
Можно попробовать внести правки автоматически:

```bash
python tools/apply_patch.py /path/to/your/project/root
```

Скрипт:
- Скопирует новые файлы в `app/services/`.
- Попробует заменить `smart_split_text` на секционное разбиение в `indexing.py` (паттерн‑поиск).
- Добавит заготовку гибридного ретривера (если найдет участок с косинусными скорингами).

Если что‑то не нашёл — выведет подсказки и список мест для ручной правки.

## Проверка
```bash
python examples/demo_split.py
```

Ожидаемый вывод: список секций с каноническими названиями и чанками (1.6k символов с overlap).
