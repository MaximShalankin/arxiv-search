#!/usr/bin/env python3
"""Generate the draw.io source files for docs/reproduction.

Run from this directory:
    python3 build.py
Then export PNGs with the draw.io CLI (see export.sh).
"""
from __future__ import annotations

import html
from pathlib import Path

OUT = Path(__file__).resolve().parent

PALETTE = {
    "blue":   ("#dae8fc", "#6c8ebf"),
    "green":  ("#d5e8d4", "#82b366"),
    "yellow": ("#fff2cc", "#d6b656"),
    "purple": ("#e1d5e7", "#9673a6"),
    "red":    ("#f8cecc", "#b85450"),
    "orange": ("#ffe6cc", "#d79b00"),
    "gray":   ("#f5f5f5", "#666666"),
    "teal":   ("#b0e3e6", "#0e8088"),
    "white":  ("#ffffff", "#999999"),
}


def esc(value: str) -> str:
    return html.escape(value, quote=True).replace("\n", "&#10;")


class Diagram:
    def __init__(self, name: str, width: int, height: int, title: str):
        self.name = name
        self.width = width
        self.height = height
        self.cells: list[str] = []
        self.n = 0
        self._text("title", title, 40, 18, width - 80, 40,
                   size=24, bold=True, color="#1a1a1a")

    def _id(self, prefix: str = "c") -> str:
        self.n += 1
        return f"{prefix}{self.n}"

    def _text(self, cid: str, value: str, x, y, w, h,
              size=14, bold=False, color="#222222", align="center"):
        style = (
            "text;html=1;whiteSpace=wrap;align=%s;verticalAlign=middle;"
            "fontFamily=Helvetica;fontSize=%d;fontColor=%s;%s"
            % (align, size, color, "fontStyle=1;" if bold else "")
        )
        self.cells.append(
            f'<mxCell id="{cid}" value="{esc(value)}" style="{style}" '
            f'vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
            f'</mxCell>'
        )

    def panel(self, value, x, y, w, h, color="gray"):
        fill, stroke = PALETTE[color]
        cid = self._id("p")
        style = (
            "rounded=1;whiteSpace=wrap;html=1;container=0;"
            "fillColor=%s;strokeColor=%s;strokeWidth=2;dashed=1;"
            "verticalAlign=top;align=left;spacingLeft=14;spacingTop=10;"
            "fontFamily=Helvetica;fontSize=17;fontStyle=1;fontColor=#333333;"
            % (fill, stroke)
        )
        self.cells.append(
            f'<mxCell id="{cid}" value="{esc(value)}" style="{style}" '
            f'vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
            f'</mxCell>'
        )
        return cid

    def box(self, value, x, y, w, h, color="white", size=14, bold=False,
            dashed=False, align="center"):
        fill, stroke = PALETTE[color]
        cid = self._id("b")
        style = (
            "rounded=1;whiteSpace=wrap;html=1;fillColor=%s;strokeColor=%s;"
            "strokeWidth=2;fontFamily=Helvetica;fontSize=%d;fontColor=#222222;"
            "verticalAlign=middle;align=%s;spacingLeft=8;spacingRight=8;%s%s"
            % (fill, stroke, size, align,
               "fontStyle=1;" if bold else "",
               "dashed=1;" if dashed else "")
        )
        self.cells.append(
            f'<mxCell id="{cid}" value="{esc(value)}" style="{style}" '
            f'vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
            f'</mxCell>'
        )
        return cid

    def edge(self, src, dst, label="", dashed=False, color="#666666",
             width=2, points=None, exit_xy=None, entry_xy=None):
        cid = self._id("e")
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
            "jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=%d;"
            "strokeColor=%s;fontFamily=Helvetica;fontSize=12;"
            "labelBackgroundColor=#ffffff;%s"
            % (width, color, "dashed=1;" if dashed else "")
        )
        if exit_xy:
            style += f"exitX={exit_xy[0]};exitY={exit_xy[1]};exitDx=0;exitDy=0;"
        if entry_xy:
            style += f"entryX={entry_xy[0]};entryY={entry_xy[1]};entryDx=0;entryDy=0;"
        pts = ""
        if points:
            pts = '<Array as="points">' + "".join(
                f'<mxPoint x="{x}" y="{y}"/>' for x, y in points
            ) + "</Array>"
        self.cells.append(
            f'<mxCell id="{cid}" value="{esc(label)}" style="{style}" '
            f'edge="1" parent="1" source="{src}" target="{dst}">'
            f'<mxGeometry relative="1" as="geometry">{pts}</mxGeometry>'
            f'</mxCell>'
        )
        return cid

    def note(self, value, x, y, w, h, size=13):
        self._text(self._id("n"), value, x, y, w, h,
                   size=size, color="#666666", align="left")

    def save(self):
        body = "\n        ".join(self.cells)
        xml = f'''<mxfile host="app.diagrams.net" agent="arxiv_search reproduction" version="31.1.8">
  <diagram id="{self.name}" name="{self.name}">
    <mxGraphModel dx="1600" dy="950" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{self.width}" pageHeight="{self.height}" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        {body}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''
        (OUT / f"{self.name}.drawio").write_text(xml, encoding="utf-8")
        print("wrote", self.name + ".drawio")


# ─────────────────────────────────────────────────────────────────────────────
# 00 — overview
# ─────────────────────────────────────────────────────────────────────────────
d = Diagram("00-overview", 1600, 900,
            "arxiv_search — воспроизведение: общая архитектура")
p_src = d.panel("Публичные источники", 40, 80, 280, 470, "blue")
p_batch = d.panel("BATCH HOST — считает", 360, 80, 500, 470, "green")
p_serv = d.panel("SERVING HOST — отдаёт", 900, 80, 660, 470, "purple")
p_store = d.panel("Storage", 360, 590, 500, 250, "yellow")
p_ops = d.panel("Quality & operations", 900, 590, 660, 250, "red")

src = [
    ("Kaggle snapshot\nметаданные", 40, 140),
    ("OAI-PMH\nсвежий хвост", 40, 200),
    ("export API\nточечные ID", 40, 260),
    ("GCS mirror\nPDF", 40, 320),
    ("arxiv.org/pdf\nfallback", 40, 380),
]
src_ids = [d.box(t, x, y, 240, 46, "blue") for t, x, y in src]
d.box("LaTeX не используется", 40, 450, 240, 46, "gray", size=13)

batch = [
    ("1. delta metadata + dedup", 380, 140),
    ("2. policy selector abstract_tier", 380, 200),
    ("3. PDF download (GCS → arXiv)", 380, 260),
    ("4. Docling → markdown → chunks", 380, 320),
    ("5. BGE-M3 embeddings", 380, 380),
    ("6. write PostgreSQL + Qdrant", 380, 440),
]
batch_ids = [d.box(t, x, y, 460, 46, "green") for t, x, y in batch]
for a, b in zip(batch_ids, batch_ids[1:]):
    d.edge(a, b)

serv = [
    ("FastAPI :8080\n/search · /ask · /paper · /related", 920, 140, 620, 55),
    ("embed client\nBGE-M3 :8005", 920, 215, 300, 60),
    ("reranker\nin-process / HTTP :8006", 1240, 215, 300, 60),
    ("LLM generator\nQwen :8004", 920, 295, 300, 60),
    ("PostgreSQL reader :5432", 920, 375, 300, 60),
    ("Qdrant reader :6333", 1240, 375, 300, 60),
]
serv_ids = [d.box(t, x, y, w, h, "purple") for t, x, y, w, h in serv]
d.note("Один тяжёлый запрос за раз → 429", 920, 455, 620, 40)

pg = d.box("PostgreSQL\npapers · chunks\ningestion_progress\npromotion_queue · citations",
           380, 640, 220, 170, "yellow", size=13)
qd = d.box("Qdrant\ncollection\nnamed vectors:\ndense 1024-d\nsparse hashed-unigram\nбез text в payload",
           620, 640, 220, 170, "yellow", size=13)
ev = d.box("Gold set\nrecall@10 / nDCG@10\nRAG smoke", 920, 640, 300, 90, "red")
bk = d.box("Dumps\nPG + Qdrant\nпроверенный restore", 1240, 640, 300, 90, "red")
d.note("Гейты: counts · self-retrieval · latency · restore", 920, 750, 620, 40)

d.edge(p_src, p_batch, label="metadata + PDF",
       exit_xy=(1, 0.5), entry_xy=(0, 0.5))
d.edge(p_batch, p_store, label="write",
       exit_xy=(0.5, 1), entry_xy=(0.5, 0))
d.edge(p_store, p_serv, label="read", dashed=True,
       exit_xy=(1, 0.5), entry_xy=(0.5, 1),
       points=[(880, 715), (880, 560), (1230, 560)])
d.edge(p_serv, p_ops, label="logs / metrics", dashed=True,
       exit_xy=(0.5, 1), entry_xy=(0.5, 0))
d.edge(p_ops, p_store, label="restore", dashed=True,
       exit_xy=(0.5, 1), entry_xy=(0.5, 1),
       points=[(1230, 870), (610, 870)])

d.save()


# ─────────────────────────────────────────────────────────────────────────────
# 01 — data and corpus
# ─────────────────────────────────────────────────────────────────────────────
d = Diagram("01-data-and-corpus", 1600, 900,
            "01 — Данные и корпус: от публичных источников до fulltext / abstract")
p_meta = d.panel("Источники метаданных", 40, 90, 520, 170, "blue")
p_pdf = d.panel("Источники PDF", 940, 90, 600, 170, "blue")

d.box("Kaggle snapshot\n~5 ГБ JSON", 60, 130, 220, 50, "blue")
d.box("OAI-PMH\nсвежий хвост", 300, 130, 240, 50, "blue")
d.box("export API\nточечные ID", 60, 190, 220, 50, "blue")
d.box("GCS mirror\nPDF", 960, 140, 260, 60, "blue")
d.box("arxiv.org/pdf\nfallback", 1240, 140, 280, 60, "blue")

norm = d.box("нормализация ID\nсрезать версии", 40, 300, 280, 60, "teal")
filt = d.box("фильтр CS/STAT\nnew-style ID", 340, 300, 280, 60, "teal")
dedup = d.box("дедупликация\nпо arxiv_id", 640, 300, 280, 60, "teal")
delta = d.box("delta vs PostgreSQL\nуже проиндексировано −", 940, 300, 280, 60, "teal")
new = d.box("new IDs\nрабочий список", 1240, 300, 300, 60, "teal")

anchors = d.box("profile anchors\nверсионируемый список тем", 40, 420, 340, 90, "purple")
embed = d.box("embed title + abstract\nBGE-M3, 1024-d", 420, 420, 340, 90, "purple")
score = d.box("score = max cosine\nк anchors + core_threshold", 800, 420, 300, 90, "purple")
select = d.box("top-K по relevance\n+ category floor\n+ tie-break (-score, id)", 1240, 420, 300, 90, "purple")

p_out = d.panel("Outputs run", 40, 560, 1000, 170, "yellow")
ft = d.box("fulltext_ids.txt\nприоритетное подмножество", 60, 610, 300, 80, "green")
ab = d.box("abstract_ids.txt\nвсе остальные", 380, 610, 300, 80, "orange")
man = d.box("manifest.json\nверсии · хеши · counts", 700, 610, 320, 80, "yellow")
pdfnote = d.box("PDF archive\nGCS mirror → arxiv.org fallback\n<pdf_root>/<YYMM>/<id>.pdf",
                1140, 560, 420, 90, "blue", size=13)

ftc = d.box("full-text corpus\nPDF download → Docling → chunks", 40, 770, 340, 80, "green")
abc = d.box("abstract corpus\ntitle + abstract → 1 чанк", 420, 770, 340, 80, "orange")
promo = d.box("promotion queue\nabstract → full-text", 800, 770, 300, 80, "orange")

d.edge(p_meta, norm, exit_xy=(0.5, 1), entry_xy=(0.5, 0))
d.edge(norm, filt)
d.edge(filt, dedup)
d.edge(dedup, delta)
d.edge(delta, new)
d.edge(new, embed, exit_xy=(0.5, 1), entry_xy=(0.5, 0),
       points=[(1390, 390), (590, 390)])
d.edge(anchors, embed)
d.edge(embed, score)
d.edge(score, select)
d.edge(select, p_out, label="run outputs", exit_xy=(0.5, 1), entry_xy=(0.9, 0),
       points=[(1390, 540), (940, 540)])
d.edge(ft, ftc)
d.edge(ab, abc)
d.edge(ab, promo, label="по запросу", dashed=True,
       exit_xy=(0.5, 1), entry_xy=(0.5, 0),
       points=[(530, 740), (950, 740)])
d.note("Инвариант: fulltext ∪ abstract = delta; пересечение пусто; повторный запуск даёт те же списки.",
       40, 875, 1500, 30, size=14)

d.save()


# ─────────────────────────────────────────────────────────────────────────────
# 02 — ingestion and indexing
# ─────────────────────────────────────────────────────────────────────────────
d = Diagram("02-ingestion", 1700, 950,
            "02 — Ингест: PDF → Docling → чанки → BGE-M3 → PostgreSQL + Qdrant")
steps = [
    ("PDF\nиз архива", 40, 100, 240, 80, "blue"),
    ("Docling\nPDF → markdown", 320, 100, 260, 80, "green"),
    ("Chunker\n~300 токенов", 620, 100, 260, 80, "green"),
    ("BGE-M3\n1024-d embeddings", 920, 100, 260, 80, "purple"),
    ("write\nPG + Qdrant", 1220, 100, 300, 80, "yellow"),
    ("progress\nindexed/done", 1560, 100, 100, 80, "gray"),
]
ids = [d.box(t, x, y, w, h, c) for t, x, y, w, h, c in steps]
for a, b in zip(ids, ids[1:]):
    d.edge(a, b)

notes = [
    ("workers 6–10\nCPU/GPU\nSIGALRM 600 s", 40, 200, 240, 90, "gray"),
    ("target 300 tokens\nhard cap 9000 chars\nтаблицы атомарны", 320, 200, 260, 90, "gray"),
    ("L2 normalize\nbatch 256\nconcurrency 2", 620, 200, 260, 90, "gray"),
    ("upsert батчами 256\npayload без text\nidempotent", 920, 200, 260, 90, "gray"),
    ("resume: skip_done\nin_progress → retry", 1220, 200, 300, 90, "gray"),
]
for t, x, y, w, h, c in notes:
    d.box(t, x, y, w, h, c, size=12)

p_store = d.panel("Storage", 320, 350, 1200, 380, "yellow")
pg = d.box("PostgreSQL\npapers\nchunks (content_hash)\nchunk_aliases\ningestion_progress\npromotion_queue\npaper_tables",
           340, 400, 560, 190, "yellow", size=13)
qd = d.box("Qdrant\nnamed vectors:\ndense 1024-d INT8\nsparse hashed-unigram\npayload: ids, title, year,\ncategories, tier — без text",
           940, 400, 560, 190, "yellow", size=13)
mapping = d.box("hash → chunk_id\ndedup внутри статьи\nQdrant point id = chunk_id",
                340, 610, 560, 80, "teal", size=13)
payload = d.box("payload index\narxiv_id (keyword)\nфильтры без индекса = скан",
                940, 610, 560, 80, "teal", size=13)

d.edge(ids[4], p_store, label="PG + Qdrant",
       exit_xy=(1, 0.5), entry_xy=(1, 0.5),
       points=[(1540, 140), (1540, 540)])

ck = d.box("checkpoint / resume\nstage + status\nволновой runner", 40, 350, 220, 160, "gray")
rep = d.box("repair\nPG chunks есть,\nQdrant точек нет\n→ re-embed + upsert", 40, 580, 220, 160, "red")
d.edge(rep, p_store, label="rebuild points", dashed=True,
       exit_xy=(1, 0.5), entry_xy=(0, 0.5))

p_abs = d.panel("Abstract-only → promotion", 320, 760, 1200, 150, "orange")
ab1 = d.box("abstract-only ingest\ntitle + abstract", 340, 805, 280, 60, "orange")
ab2 = d.box("один чанк\nheading_path=Abstract", 660, 805, 280, 60, "orange")
ab3 = d.box("Qdrant\ntier=abstract", 980, 805, 220, 60, "orange")
promo = d.box("promotion queue\nabstract → full-text\n(explicit request)", 1240, 805, 260, 60, "orange")
d.edge(ab1, ab2)
d.edge(ab2, ab3)
d.edge(ab3, promo, label="по запросу", dashed=True)

d.note("Qdrant points = PG chunks для свежего корпуса; для инкремента проверяй Δ батча.",
       320, 925, 1200, 30, size=14)

d.save()


# ─────────────────────────────────────────────────────────────────────────────
# 03 — retrieval and RAG
# ─────────────────────────────────────────────────────────────────────────────
d = Diagram("03-retrieval-rag", 1700, 950,
            "03 — Retrieval и RAG: четыре режима, RRF, rerank, ответ с цитатами")
q = d.box("query", 40, 130, 200, 70, "blue")
dense = d.box("BGE-M3\nquery embedding", 300, 80, 260, 70, "purple")
qdd = d.box("Qdrant\ndense search", 620, 80, 260, 70, "yellow")
sparse = d.box("encode_sparse\nhashed-unigram", 300, 200, 260, 70, "purple")
qds = d.box("Qdrant\nsparse search", 620, 200, 260, 70, "yellow")
rrf = d.box("RRF\nk=60, weights 0.5/0.5", 940, 130, 220, 80, "teal")
drop = d.box("drop boilerplate\nReferences · Keywords · CCS", 1220, 130, 220, 80, "red")
rer = d.box("rerank\nBGE-reranker\n(/search: выход здесь)", 1480, 130, 180, 80, "purple")
hyd = d.box("hydrate text/title\nиз PostgreSQL", 1480, 270, 180, 80, "yellow")
llm = d.box("LLM RAG\nQwen / instruct", 1220, 420, 260, 90, "purple")
ans = d.box("answer + [N] citations\ngrounding + refusal", 940, 420, 260, 90, "green")
api = d.box("HTTP response", 660, 420, 260, 90, "green")

d.edge(q, dense)
d.edge(q, sparse)
d.edge(dense, qdd)
d.edge(sparse, qds)
d.edge(qdd, rrf)
d.edge(qds, rrf)
d.edge(rrf, drop)
d.edge(drop, rer)
d.edge(rer, hyd)
d.edge(hyd, llm)
d.edge(llm, ans)
d.edge(ans, api)

modes = d.box("Режимы /search\nbm25 = native sparse\ndense = BGE-M3\nhybrid = dense+sparse RRF\nrerank = hybrid + cross-encoder",
              40, 330, 380, 170, "green", size=13, align="left")
mt = d.box("multi-turn /ask\nsession_id → rewrite\n+ scope по arxiv_id\nTTL 1 h, 10 turns",
           40, 540, 380, 130, "blue", size=13, align="left")
paper = d.box("paper-scoped ask\n/paper/{id}/ask\nretrieval filtered by arxiv_id",
              40, 710, 380, 100, "blue", size=13, align="left")
heavy = d.box("heavy lane\nодин тяжёлый запрос\nвторой → 429\nQdrant down → 503 JSON",
              660, 560, 300, 130, "red", size=13)
extra = d.box("Дополнительные endpoints\n/paper · /outline · /section · /tables · /similar\n/related · /export · /claim-matrix\n/conflicting-evidence · /promote",
              1000, 560, 660, 130, "gray", size=13, align="left")
d.note("score: dense/RRF — нормированный; rerank — сырой логит; score_norm = sigmoid(score).",
       660, 730, 1000, 40, size=14)

d.save()


# ─────────────────────────────────────────────────────────────────────────────
# 04 — serving and operations
# ─────────────────────────────────────────────────────────────────────────────
d = Diagram("04-serving-ops", 1700, 950,
            "04 — Serving и эксплуатация: batch → storage → serving, запуск и restore")
p_b = d.panel("BATCH HOST — считает", 40, 80, 520, 310, "green")
p_s = d.panel("STORAGE", 620, 80, 400, 310, "yellow")
p_v = d.panel("SERVING HOST — отдаёт", 1080, 80, 580, 310, "purple")

d.box("Docling workers\nparse → chunks", 60, 140, 220, 70, "green")
d.box("embed client\n→ BGE-M3 :8005", 300, 140, 240, 70, "green")
batch_write = d.box("PG writer\nQdrant writer", 60, 240, 220, 70, "green")
d.box("promotion worker\nabstract → full-text", 300, 240, 240, 70, "green")

pg_box = d.box("PostgreSQL :5432\npapers · chunks · progress", 640, 140, 360, 80, "yellow")
qd_box = d.box("Qdrant :6333 / :6334\ndense + sparse\nnofile 65536", 640, 250, 360, 90, "yellow")
d.note("payload index arxiv_id\nno text in payload", 640, 350, 360, 35)

d.box("API :8080\n/search /ask /paper", 1100, 140, 540, 60, "purple")
d.box("embed :8005", 1100, 225, 170, 55, "purple")
d.box("LLM :8004", 1290, 225, 170, 55, "purple")
d.box("rerank :8006", 1480, 225, 160, 55, "purple")
serv_readers = d.box("PG reader · Qdrant reader\nheavy lane = 1", 1100, 300, 540, 55, "purple")

d.edge(batch_write, p_s, label="write",
       exit_xy=(0.5, 1), entry_xy=(0, 0.5),
       points=[(170, 335), (600, 335), (600, 235)])
d.edge(p_s, serv_readers, label="read", dashed=True,
       exit_xy=(1, 0.5), entry_xy=(0, 0.5),
       points=[(1040, 235), (1040, 327)])

# startup order
p_run = d.panel("Порядок запуска", 40, 450, 560, 390, "blue")
run = [
    ("1. Qdrant — первым,\nдождаться green", 60, 500),
    ("2. embeddings :8005", 60, 570),
    ("3. LLM :8004", 60, 630),
    ("4. reranker :8006", 60, 690),
    ("5. API :8080 → /health", 60, 750),
]
rids = [d.box(t, x, y, 500, 50, "blue") for t, x, y in run]
for a, b in zip(rids, rids[1:]):
    d.edge(a, b)
d.note("Qdrant первым и в одиночку: пик загрузки + модели = OOM", 60, 805, 500, 30, size=13)

# resources
p_res = d.panel("VRAM (reference, 32 ГБ)", 640, 450, 340, 390, "green")
d.box("BGE-M3 vLLM · ~2 ГБ", 660, 500, 300, 45, "green", size=13)
d.box("Qwen BF16+FP8 KV · ~13–17 ГБ", 660, 555, 300, 45, "green", size=13)
d.box("Reranker BF16 · ~1.1 ГБ", 660, 610, 300, 45, "green", size=13)
d.box("CUDA + прочее · ~2.7 ГБ", 660, 665, 300, 45, "green", size=13)
d.box("Запас · ~9–13 ГБ", 660, 720, 300, 45, "orange", size=13)
d.note("Меньше запаса — уменьшай LLM\nили разноси batch и serving", 660, 780, 300, 45, size=12)

# backup
p_bk = d.panel("Дамп и restore", 1020, 450, 640, 390, "red")
d.box("PG: pg_dump -Fc\n(или copy тома)", 1040, 500, 280, 70, "red", size=13)
d.box("Qdrant: snapshot\n(или stop + rsync)", 1340, 500, 300, 70, "red", size=13)
d.box("Проверка: counts,\nчисло файлов, checksums", 1040, 590, 280, 70, "red", size=13)
d.box("Носитель, который\nпереживёт машину", 1340, 590, 300, 70, "red", size=13)
d.box("Restore в пустые тома\n→ counts → smoke", 1040, 680, 600, 60, "red", size=13)
d.note("Правило: любая работа, меняющая данные,\nзаканчивается дампом на носителе.", 1040, 760, 600, 50, size=13)

d.note("API без аутентификации → только приватная сеть / firewall / VPN.", 40, 855, 1600, 35, size=14)

d.save()


# ─────────────────────────────────────────────────────────────────────────────
# 05 — evaluation
# ─────────────────────────────────────────────────────────────────────────────
d = Diagram("05-evaluation", 1600, 900,
            "05 — Оценка качества: gold set, метрики, приёмка, регрессия")
gold = d.box("Gold set\n100 реальных запросов\nA/B/C/D/E + unanswerable", 40, 100, 320, 110, "blue")
run = d.box("Прогон режимов\ndense · hybrid · rerank", 400, 100, 300, 110, "purple")
met = d.box("Метрики\nrecall@10 · nDCG@10\nbootstrap CI", 740, 100, 320, 110, "teal")
base = d.box("Сравнение с baseline\nпо категориям", 1100, 100, 320, 110, "yellow")
dec = d.box("Решение\nadopt / reject", 1100, 260, 320, 90, "red")

accept = d.box("Acceptance после ингеста\n• counts и partition\n• 0 missing\n• self-retrieval\n• Δ PG = Δ Qdrant\n• payload без text\n• promotion e2e",
               40, 300, 320, 260, "green", size=13, align="left")
smoke = d.box("RAG smoke 10 вопросов\n• 5 factual\n• 2 comparative\n• 2 exploratory\n• 1 без ответа\n≥8/10 с цитатой",
              400, 300, 300, 260, "green", size=13, align="left")
cit = d.box("Граф цитирования\n• refs-чанки → arXiv-ID\n• in-corpus edges\n• /related signals\n• покрытие и ограничения",
            740, 300, 320, 260, "green", size=13, align="left")
reg = d.box("Регрессия\n1 переменная за прогон\n• чанкер\n• эмбеддер\n• reranker\n• LLM / prompt\n• политика",
            1100, 420, 320, 260, "orange", size=13, align="left")

d.edge(gold, run)
d.edge(run, met)
d.edge(met, base)
d.edge(base, dec)
d.edge(accept, gold, label="после ингеста",
       exit_xy=(0.5, 0), entry_xy=(0.5, 1))
d.edge(smoke, run, label="перед релизом",
       exit_xy=(0.5, 0), entry_xy=(0.5, 1))
d.edge(cit, met, label="качество связей",
       exit_xy=(0.5, 0), entry_xy=(0.5, 1))
d.edge(reg, dec, label="изменение",
       exit_xy=(0.5, 0), entry_xy=(0.5, 1))
d.edge(reg, run, label="повторный eval", dashed=True,
       exit_xy=(0.5, 1), entry_xy=(1, 0.5),
       points=[(1260, 730), (720, 730), (720, 155)])
d.note("Правила: не менять gold set между прогонами; фиксировать версии;\nсмотреть по категориям, а не только overall; grounding — эвристика, не вероятность.",
       40, 760, 1000, 60, size=14)

d.save()
