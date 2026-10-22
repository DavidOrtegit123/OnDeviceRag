import json
from sentence_transformers import SentenceTransformer
import pathlib; R=str(pathlib.Path(__file__).resolve().parents[3] / 'evaluation' / 'rag') + '/'
P=json.load(open(R+'passages.json'))['passages']
C=json.load(open(R+'corpus.json'))['sections']
gold={s['task_id']:s['title'] for s in C}
Q=[q for q in json.load(open(R+'queries.json'))]
m=SentenceTransformer('BAAI/bge-small-en-v1.5'); tok=m.tokenizer
qp='Represent this sentence for searching relevant passages: '
ids=lambda t: tok(t, truncation=True, max_length=512)['input_ids']
par=["Hello, my dog is cute", P[1]['text']]
out={"parity":[{"text":t,"ids":ids(t),"vector":m.encode(t,normalize_embeddings=True).tolist()} for t in par],
     "passages":[{"section":p['section_title'],"ids":ids(p['text'])} for p in P],
     "queries":[{"task":q['task_id'],"variant":q['variant'],"gold":gold[q['task_id']],"ids":ids(qp+q['query'])} for q in Q]}
json.dump(out,open('EmbeddingSpikeReference.json','w'))
print(len(out['passages']),len(out['queries']),max(len(p['ids']) for p in out['passages']))
