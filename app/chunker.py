"""
Adaptive text chunker to reduce fragment explosion.

Key ideas:
- Respect headings and list blocks.
- Use sentence-aware chunks with adjustable size.
- Merge tiny pieces forward (min_size) and avoid micro-fragments.
- Optional sliding overlap (small, default=0).
- Deduplicate near-duplicate chunks via Jaccard.
- Collapse whitespace and hyphenation from OCR.
- Works without external tokenizers; sizes are character-based.
"""

import re
from typing import List, Tuple, Iterable, Optional
from dataclasses import dataclass

@dataclass
class ChunkingConfig:
    target_size: int = 1000         # target characters per chunk
    min_size: int = 400             # merge forward until at least this size
    max_size: int = 1400            # soft ceiling
    overlap: int = 0                # 0..200 characters of optional overlap
    dedupe_threshold: float = 0.92  # Jaccard similarity threshold for suppressing near-dupes
    keep_headings: bool = True      # don't merge across headings
    join_bullets: bool = True       # keep bullet lists together
    fix_hyphenation: bool = True    # join "hyphen-\nations" -> "hyphenations"
    strip_inline_cruft: bool = True # remove leftover artifacts

_heading = re.compile(r"^(#{1,6}\s+|[A-Z][^\n]{0,50}\n[-=]{3,}\s*$)", re.M)
_bullet  = re.compile(r"^[ \t]*([\-*•]|[0-9]+\.)\s+", re.M)

def _normalize(text: str, cfg: ChunkingConfig) -> str:
    t = text
    # Normalize newlines
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # Fix hyphenation from OCR line breaks: "micro-\n wave" -> "microwave"
    if cfg.fix_hyphenation:
        t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    # Collapse multiple spaces/newlines
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    if cfg.strip_inline_cruft:
        # Remove zero-width and control artifacts
        t = re.sub(r"[\u200B-\u200D\uFEFF]", "", t)
    return t.strip()

def _sentences(paragraph: str) -> List[str]:
    # Lightweight sentence splitter (no external deps)
    # Splits on period, question, exclamation, semicolon when followed by space+capital or newline.
    # Keeps punctuation.
    parts = re.split(r"(?<=[\.\?\!;])\s+(?=[A-ZА-ЯЁ0-9\"\'\(\[])|(?<=\n)", paragraph)
    # Fallback: if we produced too few, split on ". "
    if len(parts) <= 1 and ". " in paragraph:
        parts = paragraph.split(". ")
        parts = [p + ("" if p.endswith(".") else ".") for p in parts if p]
    return [p.strip() for p in parts if p and not p.isspace()]

def _paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

def _is_heading(block: str) -> bool:
    if _heading.search(block):
        return True
    # Heuristic: a single short line with Title Case
    lines = block.strip().split("\n")
    if len(lines) == 1 and 2 <= len(lines[0]) <= 80:
        line = lines[0]
        if sum(1 for c in line if c.isupper()) >= max(1, len(line)//5):
            return True
    return False

def _is_bullet_block(block: str) -> bool:
    return bool(_bullet.search(block))

def jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def dedupe(chunks: List[str], threshold: float) -> List[str]:
    out = []
    for c in chunks:
        if not any(jaccard(c, prev) >= threshold for prev in out):
            out.append(c)
    return out

def chunk_text(text: str, cfg: Optional[ChunkingConfig] = None) -> List[str]:
    if cfg is None:
        cfg = ChunkingConfig()
    t = _normalize(text, cfg)

    # Stage 1: split into logical blocks (paragraphs)
    paras = _paragraphs(t)

    # Stage 2: if keep_headings, keep headings as separate blocks
    blocks: List[Tuple[str, bool]] = []  # (text, is_heading)
    for p in paras:
        if _is_heading(p):
            blocks.append((p, True))
        else:
            blocks.append((p, False))

    # Stage 3: assemble chunks by walking blocks, respecting headings and bullets
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            chunks.append("\n\n".join(buf).strip())
            buf = []
            buf_len = 0

    i = 0
    while i < len(blocks):
        text, is_head = blocks[i]

        if is_head and cfg.keep_headings:
            # Flush current, then emit heading as its own tiny chunk or prefix to next
            flush()
            # Option: attach heading to the next content if it's small
            if i + 1 < len(blocks) and not blocks[i+1][1]:
                candidate = text + "\n\n" + blocks[i+1][0]
                if len(candidate) <= cfg.max_size:
                    chunks.append(candidate.strip())
                    i += 2
                    continue
            # Else keep as a standalone (rarely harms retrieval)
            chunks.append(text.strip())
            i += 1
            continue

        # Try to keep bullet blocks intact
        if cfg.join_bullets and _is_bullet_block(text):
            piece = text
        else:
            # Sentence-level packing to match target
            sents = _sentences(text)
            piece = ""
            cur = []
            for s in sents:
                tentative = (piece + " " + s).strip() if piece else s
                if len(tentative) <= cfg.target_size:
                    piece = tentative
                    cur.append(s)
                else:
                    # If adding would burst, finalize current piece into buffer
                    if piece:
                        # push old piece
                        if buf_len and buf_len + len(piece) > cfg.max_size:
                            flush()
                        buf.append(piece)
                        buf_len += len(piece)
                        piece = s
                    else:
                        piece = s
            # at end, piece holds tail
        # Append piece into buffer and merge forward
        if piece:
            if buf_len and buf_len + len(piece) > cfg.max_size:
                flush()
            buf.append(piece)
            buf_len += len(piece)

            # If buffer is still too small (< min_size), try merging with following non-heading blocks
            while buf_len < cfg.min_size and i + 1 < len(blocks) and not blocks[i+1][1]:
                nxt = blocks[i+1][0]
                if cfg.join_bullets and _is_bullet_block(nxt):
                    addition = "\n" + nxt
                else:
                    addition = "\n\n" + nxt
                if buf_len + len(addition) <= cfg.max_size:
                    buf[-1] = (buf[-1] + addition).strip()
                    buf_len += len(addition)
                    i += 1
                else:
                    break

        # If buffer over target or near max, flush
        if buf_len >= cfg.target_size or buf_len >= cfg.max_size:
            flush()

        i += 1

    flush()

    # Optional small overlap to help retrieval continuity
    if cfg.overlap and cfg.overlap > 0 and len(chunks) > 1:
        overlapped = []
        for idx, c in enumerate(chunks):
            if idx == 0:
                overlapped.append(c)
            else:
                prev_tail = chunks[idx-1][-cfg.overlap:]
                overlapped.append(prev_tail + c)
        chunks = overlapped

    # Deduplicate very similar chunks
    if cfg.dedupe_threshold < 0.999:
        chunks = dedupe(chunks, cfg.dedupe_threshold)

    # Final polish: strip extremes and drop trivial fragments
    final = [c.strip() for c in chunks if len(c.strip()) >= max(80, cfg.min_size // 3)]
    return final
