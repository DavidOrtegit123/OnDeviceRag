# /// script
# requires-python = ">=3.11"
# dependencies = ["sentence-transformers==5.1.2", "transformers==4.57.3"]
# ///
# Writes the bundled tokenizer files and the Swift tests' reference token IDs + vectors.
# Usage (from Packages/OnDeviceRAG): uv run Tools/make_package_fixtures.py
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

NAME = "BAAI/bge-small-en-v1.5"
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "Sources/OnDeviceRAG/Resources/bge-small-en-v1.5"
FIX = ROOT / "Tests/OnDeviceRAGTests/Fixtures/bge_reference.json"
PREFIX = "Represent this sentence for searching relevant passages: "

model = SentenceTransformer(NAME)
tok = model.tokenizer
RES.mkdir(parents=True, exist_ok=True)
tok.save_pretrained(str(RES))
for extra in ("vocab.txt", "special_tokens_map.json"):  # tokenizer.json already contains these
    (RES / extra).unlink(missing_ok=True)
(RES / "LICENSE.txt").write_text(
    "bge-small-en-v1.5 tokenizer files (c) BAAI, MIT License. https://huggingface.co/BAAI/bge-small-en-v1.5\n"
)

texts = [
    ("passage", "Hello, my dog is cute"),
    ("passage", "Integers are encoded as i<n>e, e.g. encode_zorp(42) -> 'i42e'."),
    ("passage", "Café naïve résumé — accents and dashes… plus emoji 🙂 and 中文."),
    ("passage", "    def merge(a, b):\n        return sorted(a + b)"),
    ("passage", "x² + 2x − 3 = 0, so x = −3 or x = 1."),
    ("query", "How do I encode a list in ZORP-7?"),
    ("passage", " ".join(["token"] * 600)),  # longer than 512 tokens -> truncated
]
out = []
for kind, text in texts:
    full = PREFIX + text if kind == "query" else text
    ids = tok(full, truncation=True, max_length=512)["input_ids"]
    vec = model.encode(full, normalize_embeddings=True).tolist()
    out.append({"kind": kind, "text": text, "ids": ids, "vector": vec})
FIX.write_text(json.dumps(out))
print("wrote", RES, "and", FIX, [len(o["ids"]) for o in out])
