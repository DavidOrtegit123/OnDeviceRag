# Changelog

## 0.2.2

- **History search no longer drowns a new topic.** With `history.useForSearch` on, the question is
  now searched alone *and* with the recent user turns, and the rankings are fused with weight 2 for
  the question alone and 1 for the history search. Before, the only query was question + history,
  so after two questions on other topics a new question's answer could drop out of the context.
  On a 63-page test manual this brings the right section back on every topic change (one was
  missing) and keeps or improves follow-ups. Without history nothing changes. Costs one extra
  query embedding when history is used.
- `rag-index query --history "<earlier question>"` (repeatable, oldest first) searches with chat
  history, to reproduce an app's search on a Mac.
- No API or file format change.

## 0.2.1

- `ChatTurn.recent(_:settings:counter:)`: the latest turns that fit a history budget, without an
  engine — the same rule as `RAGContext.history`, for apps that pin a document or run their own
  retrieval. Additive; no other change.

## 0.2.0

- **Conversation history, switchable** (`RAGConfiguration.history`, both off by default):
  `useForSearch` adds the last `searchTurns` (2) user turns to the search query, so follow-up
  questions find the right section; `includeInContext` returns the latest turns that fit
  `tokenBudget` (600 tokens, at most `maximumTurns` = 6, oldest dropped first) in
  `RAGContext.history` / `historyTokenCount`. `ChatTurn.format` writes them as labeled lines for a
  prompt. The package doesn't store conversations.
- **Breaking:** `history` is `[ChatTurn]` (role + text) instead of `[String]`;
  `RetrievalSettings.historyMessages` is removed (now `history.searchTurns`). Passing history no
  longer changes search unless `history.useForSearch` is on. Apps on "Up to Next Minor from 0.1.0"
  must switch to "from 0.2.0".
- `RAGEngine.renameDocument(id:to:)` — new title, re-chunked from the stored source; new
  `RAGError.unknownDocument`.
- Docs: the repository is public — downloading the model needs no login; `Tools/download_model.sh`
  defaults to this release.
- Stored documents and `.ragindex` files are unchanged (format version 1).

## 0.1.0

First release: PDF/text ingest (sections, code-aware chunks, Vision OCR for scanned pages),
bge-small embeddings via Core AI, hybrid dense + BM25 search, small-to-big packing within a token
budget, persistence, prebuilt read-only indexes, `rag-index` CLI.
