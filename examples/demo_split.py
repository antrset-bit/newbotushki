
# -*- coding: utf-8 -*-
import os, json, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.chunking_sections import split_by_semantic_sections, serialize_section_chunks

sample_path = os.path.join(os.path.dirname(__file__), "sample_contract.txt")
with open(sample_path, "r", encoding="utf-8") as f:
    txt = f.read()

chs = split_by_semantic_sections(txt)
texts, metas = serialize_section_chunks(chs, "sample_contract.txt")
print(json.dumps({"chunks": texts[:3], "metas": metas[:3], "total": len(texts)}, ensure_ascii=False, indent=2))
