#!/usr/bin/env bash
# The chapter's five steps on the cluster, plus three that show what the rules are for.
# Assumes `make up` has run (namespace, chroma, docs_v1 built, Service -> blue on docs_v1).
set -euo pipefail
NS=docbot
URL=http://localhost:8080     # the docbot Service is a LoadBalancer; Docker Desktop puts it on localhost
ask() {  # ask "question"  -> one line: which pod answered (the service reports its own pod name), on which index, with what
  # --retry: right after a cut-over, the Service's endpoints can lag the selector by a second or two
  curl -s --retry 8 --retry-delay 1 --retry-all-errors "$URL/ask?q=$(printf %s "$1" | sed 's/ /%20/g')" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('  ask -> pod', d['pod'], '[' + d['active_index'] + ']', d['hit'])"
}
active() { kubectl -n $NS get svc docbot -o jsonpath='{.spec.selector.color}'; }
idx()    { kubectl -n $NS get deployment docbot-$1 -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="ACTIVE_INDEX")].value}'; }
svc()    { echo "  Service docbot -> $(active)"; }
pods()   { kubectl -n $NS get pods -l app=docbot --no-headers | sort | awk '{printf "  pod %s %s %s\n", $1, $2, $3}'; }
rev()    { helm -n $NS history docbot -o json | python3 -c 'import sys,json; print(json.load(sys.stdin)[-1]["revision"])'; }
helmerr() { grep -iE 'error|failed' | head -1 | sed -E 's/.*error=\\?"([^"\\]*).*/\1/; s/\\n.*//' | sed 's/^/  helm: /'; }
podlog() { (kubectl -n $NS logs "$1" 2>/dev/null; kubectl -n $NS logs --previous "$1" 2>/dev/null) | grep -E "$2" | tail -1; }
crashing() { kubectl -n $NS get pods -l app=docbot,color=$1 --no-headers | awk '$2=="0/1"{print $1; exit}'; }

echo "========== 1) Shipped: Service -> $(active), $(active) on $(idx $(active)) =========="
echo "  image: $(kubectl -n $NS get deployment docbot-blue -o jsonpath='{.spec.template.spec.containers[0].image}')"
pods; ask "how to configure kafka batch consumption"

echo; echo "========== 2) Build v2 as its own release (blue keeps serving) =========="
make -s build-index V=2 | grep -E '^\[build\]|^ +make stage'
svc; ask "how to configure kafka batch consumption"

echo; echo "========== 3) Stage v2: make stage V=2 -> green on docs_v2, Ready; the Service has not moved =========="
make -s stage V=2 >/dev/null
echo "  green: $(idx green), $(kubectl -n $NS get deployment docbot-green -o jsonpath='{.status.readyReplicas}') pods Ready"
pods; svc; ask "how to configure kafka batch consumption"

echo; echo "========== 4) Cut over: make cutover -> the Service's selector moves; no pod restarts =========="
make -s cutover >/dev/null
echo "  release docbot revision $(rev)"; svc; pods
ask "how to configure kafka batch consumption"; ask "postgres vector search"

echo; echo "========== 5) Something wrong? make rollback -> the selector comes back; still no pod restarts =========="
make -s rollback >/dev/null
echo "  release docbot revision $(rev)"; svc; ask "how to configure kafka batch consumption"

echo; echo "========== 6) Forward for good, then retire docs_v1 =========="
make -s cutover >/dev/null
echo "  release docbot revision $(rev)"; svc
# capture, don't pipe: head/grep -m1 close the pipe early and, under pipefail,
# turn kubectl's harmless trailing "pod deleted" write into a failed step
out=$(make -s retire V=1 2>&1); grep -E '^  (blue|green) idled|^  deleted' <<<"$out" || { echo "  retire failed:"; echo "$out" | tail -3; exit 1; }
out=$(make -s list 2>&1);       grep '^  docs_'   <<<"$out" || { echo "  list failed:";   echo "$out" | tail -3; exit 1; }
pods; echo "  releases: $(helm -n $NS list -q | sort | tr '\n' ' ')"

echo; echo "========== 7) The deploy nobody should survive: make stage V=9 (no such collection) =========="
make -s stage V=9 2>&1 | helmerr || true
for i in $(seq 1 30); do kubectl -n $NS get pods -l app=docbot --no-headers | grep -q CrashLoopBackOff && break; sleep 2; done
pods
NEW=$(crashing blue)
echo "  new pod's log: $(podlog "$NEW" 'NotFoundError' | sed 's/.*NotFoundError/NotFoundError/')"
echo "  meanwhile, through the Service:"; svc; ask "how to configure kafka batch consumption"
make -s rollback >/dev/null
echo "  helm rollback -> revision $(rev): blue idle again"; pods

echo; echo "========== 8) Two refusals: a second build of a live version; a half-written collection =========="
out=$(make -s build-index V=2 2>&1 || true); helmerr <<<"$out"
out=$(make -s partial V=7 2>&1); grep '^  docs_v7' <<<"$out"
make -s stage V=7 2>&1 | helmerr || true
NEW=$(crashing blue)
echo "  new pod's log: $(podlog "$NEW" 'not marked complete|wrong number' | sed 's/.*SystemExit: //')"
svc; make -s rollback >/dev/null
echo "  helm rollback -> revision $(rev): blue idle again"
