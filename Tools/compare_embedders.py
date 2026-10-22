import json, sys, numpy as np
from sentence_transformers import SentenceTransformer
import pathlib; R=str(pathlib.Path(__file__).resolve().parents[3] / 'evaluation' / 'rag') + '/'
P=json.load(open(R+'passages.json'))['passages']
C=json.load(open(R+'corpus.json'))['sections']
gold={s['task_id']:s['title'] for s in C}
Q=[q for q in json.load(open(R+'queries.json')) if q['variant']==sys.argv[1]]
import re
def tok(t): return re.findall(r"[a-z0-9_]+", t.lower())
# BM25
from math import log
docs=[tok(p['text']) for p in P]; N=len(docs); avg=sum(map(len,docs))/N
df={}
for d in docs:
    for w in set(d): df[w]=df.get(w,0)+1
def bm25(q):
    qs=tok(q); out=[]
    for d in docs:
        s=0
        for w in qs:
            if w not in df: continue
            f=d.count(w); idf=log(1+(N-df[w]+.5)/(df[w]+.5))
            s+=idf*f*2.2/(f+1.2*(.25+.75*len(d)/avg))
        out.append(s)
    return np.array(out)
def evaluate(name, scores_fn):
    hit5=sec3=0; ranks=[]
    for q in Q:
        s=scores_fn(q['query']); order=np.argsort(-s)
        g=gold[q['task_id']]
        top5=[P[i]['section_title'] for i in order[:5]]
        hit5+= g in top5
        secs=[]
        for i in order:
            t=P[i]['section_title']
            if t not in secs: secs.append(t)
        sec3+= g in secs[:3]; ranks.append(secs.index(g)+1)
    print(f"{name:40s} hit@5 {hit5}/40  section-top3 {sec3}/40  median section rank {int(np.median(ranks))}  worst {max(ranks)}")
def rrf(*arrs):
    tot=np.zeros(N)
    for a in arrs:
        r=np.empty(N); r[np.argsort(-a)]=np.arange(1,N+1); tot+=1/(60+r)
    return tot
evaluate('BM25 only', bm25)
models={'bge-small-en-v1.5':('BAAI/bge-small-en-v1.5','Represent this sentence for searching relevant passages: '),
        'e5-small-v2':('intfloat/e5-small-v2','query: '),
        'all-MiniLM-L6-v2':('sentence-transformers/all-MiniLM-L6-v2',''),
}
for name,(mid,qp) in models.items():
    m=SentenceTransformer(mid)
    pp='passage: ' if 'e5' in mid else ''
    E=m.encode([pp+p['text'] for p in P], normalize_embeddings=True)
    cache={}
    def dense(q): return E@m.encode([qp+q],normalize_embeddings=True)[0]
    evaluate(name+' (dense only)', dense)
    evaluate(name+' + BM25 (RRF)', lambda q: rrf(dense(q), bm25(q)))
