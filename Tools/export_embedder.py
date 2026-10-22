# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b2",
#     "coreai-torch==0.4.1",
#     "transformers==4.57.3",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
# Export BAAI/bge-small-en-v1.5 to Core AI: CLS pooling + L2 norm baked in,
# fixed-length functions embed_128/256/512 with input_ids + attention_mask (int32).
# Usage: uv run export_embedder.py float16   (from the 2026-10-22 iPhone spike)
import json, sys, time
from pathlib import Path
import torch, transformers
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table

NAME = "BAAI/bge-small-en-v1.5"
dtype = {"float32": torch.float32, "float16": torch.float16}[sys.argv[1] if len(sys.argv) > 1 else "float32"]

class Embedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(NAME, add_pooling_layer=False)
    def forward(self, input_ids, attention_mask):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(h, dim=-1)

model = Embedder().eval().to(dtype)
tok = transformers.AutoTokenizer.from_pretrained(NAME)
conv = TorchConverter(mode=TorchConverter.Mode.RELEASE)
for L in (128, 256, 512):
    ids = torch.zeros(1, L, dtype=torch.int32); mask = torch.ones(1, L, dtype=torch.int32)
    ep = torch.export.export(model, args=(), kwargs={"input_ids": ids, "attention_mask": mask})
    ep = ep.run_decompositions(get_decomp_table())
    conv.add_exported_program(exported_program=ep, input_names=["input_ids", "attention_mask"],
                              output_names=["embedding"], entrypoint_name=f"embed_{L}")
prog = conv.to_coreai()
prog.optimize()
md = AIModelAssetMetadata(); md.author = "BAAI"; md.license = "MIT"
md.model_description = "bge-small-en-v1.5 sentence embedding, CLS pooling, L2-normalized (spike)"
md.creation_date = int(time.time())
out = Path(f"bge_small_buckets_{str(dtype).split('.')[-1]}.aimodel")
prog.save_asset(out, md)
print("saved", out)
