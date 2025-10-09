#!/usr/bin/env bash
set -euo pipefail

BASE="http://localhost:8000/api/v1"

echo "== Health =="
curl -fSs "$BASE/system/health" | uv run python - <<'PY'
import sys,json
d=json.load(sys.stdin)
print("health.success=",d.get("success"),"status=",d.get("status"))
print("components=",d.get("components"))
PY

echo "== Upload document =="
TMPFILE=$(mktemp)
printf "This is Lab05 manual test content.\nVector search filter DSL check.\n" > "$TMPFILE"
DOC_JSON_FILE=$(mktemp)
curl -fSs -X POST "$BASE/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@$TMPFILE;filename=lab05_manual_test.txt" \
  -F "title=Lab05 Manual Test" \
  -F "description=End-to-end test for Lab05" > "$DOC_JSON_FILE"

export DOC_JSON_FILE
DOC_ID=$(uv run python - <<'PY'
import os,json
path=os.environ['DOC_JSON_FILE']
with open(path) as f:
    j=json.load(f)
print(j.get("id") or j.get("document_id") or "")
PY)
FILENAME=$(uv run python - <<'PY'
import os,json
path=os.environ['DOC_JSON_FILE']
with open(path) as f:
    j=json.load(f)
print(j.get("filename") or j.get("document_filename") or "lab05_manual_test.txt")
PY)
echo "Uploaded doc_id=$DOC_ID filename=$FILENAME"

echo "== Chunks before vectorize =="
curl -fSs "$BASE/documents/$DOC_ID/chunks" | uv run python - <<'PY'
import sys,json
chunks=json.load(sys.stdin)
print("chunks_count=",len(chunks))
print("vectorized_flags=", [c.get("is_vectorized") for c in chunks])
PY

echo "== Vectorize document =="
curl -fSs -X POST "$BASE/vectors/vectorize" \
  -H "Content-Type: application/json" \
  -d "{\"document_ids\": [\"$DOC_ID\"], \"batch_size\": 16}" | python - <<'PY'
import sys,json
r=json.load(sys.stdin)
print("vectorize.success=",r.get("success"),"processed_count=",r.get("processed_count"),"failed_ids=",r.get("failed_ids"))
PY

echo "== Chunks after vectorize =="
curl -fSs "$BASE/documents/$DOC_ID/chunks" | uv run python - <<'PY'
import sys,json
chunks=json.load(sys.stdin)
print("vectorized_flags=", [c.get("is_vectorized") for c in chunks])
PY

echo "== Vector search with filters =="
curl -fSs -X POST "$BASE/vectors/search" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Lab05\",\"limit\":8,\"score_threshold\":0.0,\"filters\":[{\"op\":\"eq\",\"field\":\"document_filename\",\"value\":\"$FILENAME\"}]}" | python - <<'PY'
import sys,json
r=json.load(sys.stdin)
print("search.success=",r.get("success"),"total_found=",r.get("total_found"))
print("hits_filenames=", [res.get("document_filename") for res in r.get("results", [])])
PY

echo "== Reindex all =="
REINDEX_JSON=$(curl -sS -X POST "$BASE/vectors/reindex" \
  -H "Content-Type: application/json" \
  -d '{"force": true}' || echo '{}')
uv run python - <<'PY'
import os,json
r=json.loads(os.environ.get('REINDEX_JSON','{}'))
print("reindex.success=",r.get("success"),"message=",r.get("message"))
PY

echo "== Delete document =="
curl -fSs -X DELETE "$BASE/documents/$DOC_ID" | uv run python - <<'PY'
import sys,json
r=json.load(sys.stdin)
print("delete.success=",r.get("success"),"message=",r.get("message"))
PY

echo "== Verify cascade deletion via search =="
curl -fSs -X POST "$BASE/vectors/search" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Lab05\",\"limit\":8,\"filters\":[{\"op\":\"eq\",\"field\":\"document_filename\",\"value\":\"$FILENAME\"}]}" | python - <<'PY'
import sys,json
r=json.load(sys.stdin)
print("post_delete.total_found=",r.get("total_found"))
PY

echo "== System stats =="
curl -fSs "$BASE/system/stats" | uv run python - <<'PY'
import sys,json
r=json.load(sys.stdin)
print("stats=",r)
PY

echo "== Documents list redirect behavior =="
STATUS_NO_SLASH=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/documents")
STATUS_WITH_SLASH=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/documents/")
echo "status_no_slash=$STATUS_NO_SLASH status_with_slash=$STATUS_WITH_SLASH"

echo "== Done =="