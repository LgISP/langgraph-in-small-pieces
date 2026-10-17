# Chapter 27 — blue-green index deployment, on Kubernetes with Helm

The chapter's five steps as a real deployment: the vector store is a service,
each index build is its own Helm release, and the doc-bot is **two serving
environments** — Deployments `docbot-blue` and `docbot-green`, each with its own
`ACTIVE_INDEX` — behind **one Service whose selector names the colour that takes
traffic** (`active` in `values.yaml`). A new index is staged onto the idle colour
and made Ready while the other still answers every request; the cut-over is
`helm upgrade --set active=green`, one label, all traffic at once, no pod
restarted; a rollback is `helm rollback`, the same label back. The application
never writes any of these values.

```
deploy/ch27-bluegreen/
  app/            the serving process (FastAPI), the index-build Job, an admin CLI
  docs/           v1.json, v2.json -- the two document sets from the chapter
  chart/          the serving stack: Chroma (PVC+Deployment+Service), ConfigMap, doc-bot blue+green Deployments and the Service that picks one
  chart-build/    one index build = one release: a ConfigMap of source docs + a Job
  Dockerfile      python:3.12-slim + the pinned requirements
  Makefile        make -> helm, one target per step
  runbook.sh      the five steps end to end, plus three that show what the rules are for
```

## Prerequisites

- Docker Desktop with its Kubernetes enabled (the image is built locally and never
  pushed; `imagePullPolicy: IfNotPresent`). Any cluster that can see your local
  image store works; on kind, `kind load docker-image $(make -s image | sed 's/^image: //')`
  after each build. The tag is a hash of the app's content (`make image` prints
  it): a reused tag such as `:dev` leaves nodes serving a cached old image under
  the new name, which is exactly how this deploy went wrong once while being
  written.
- LM Studio serving the embedding model on `:1234`, reachable from pods as
  `host.docker.internal` (`chart/values.yaml`, `embedding.lmBase`; change it for
  another cluster).
- The `docbot` Service is a `LoadBalancer`; Docker Desktop puts it on
  `localhost:8080`, which is where `runbook.sh` asks its questions from. On a
  cluster without a load balancer, `kubectl -n docbot port-forward svc/docbot 8080:8080`
  gives the same address (note that a port-forward pins one pod, so the
  "which pod answered" lines will not spread across pods).
- `kubectl`, `helm` (3 or 4), `make`.

## The steps

```bash
make up                 # namespace; helm install docbot (store, config); build docs_v1; blue gets its pods on docs_v1, the Service points at blue
make build-index V=2    # helm install build-v2 ./chart-build: a Job writes docs_v2 -- and prints the next two commands, it does not run them
make stage V=2          # the idle colour (green) gets ACTIVE_INDEX=docs_v2 and its pods; helm --wait until they are Ready; the Service has not moved
make cutover            # helm upgrade --set active=green : the Service's selector moves -- every request now goes to green; no pod restarts
make rollback           # helm rollback docbot : the selector comes back to blue, instantly; still no pod restarts
make cutover            # forward for good
make retire V=1         # blue idled (0 pods, no index), docs_v1 deleted, build-v1 uninstalled -- only now
make deploy-image       # after a code change: build docbot-ch27:<hash>, helm upgrade --set image.tag (both colours roll onto it)
make status             # which colour the Service points at, each colour's index and pods
make history            # helm history docbot: every stage, cut-over and rollback, with its values
make down
```

`./runbook.sh` runs all of that and then three more: `make stage V=9`, a
collection that does not exist — the idle colour's new pod exits at import,
crash-loops and is never Ready, `helm upgrade --wait` reports the failure, the
Service never moved and the live colour kept answering, and `helm rollback` idles
the colour again; a second `build-index V=2` while `docs_v2` is live, which Helm
refuses because the release `build-v2` already exists (and the Job would refuse
the collection if it got that far); and a stage onto a collection a build left
half-written, which the startup checks refuse the same way.

## What each piece is, in the chapter's terms

| Chapter | Here |
|---|---|
| a "process" that read its config once | a pod; `ACTIVE_INDEX` is in its env, resolved at startup, never re-read |
| blue and green, both whole | Deployments `docbot-blue` and `docbot-green`, each on its own collection; `make stage V=N` gives the idle one an index and pods |
| `deploy("docs_v2")` | `helm upgrade docbot ./chart --set active=green` → the Service's selector changes → every request goes to green; the pods were already Ready |
| rollback = the old value again | `helm rollback docbot`: the previous revision's `active`; the blue pods never left, so nothing starts |
| "validate before you deploy" | staging: the process opens its collection at import and exits unless the build marked it complete and the count matches; `helm --wait` on the idle colour fails and `make cutover` refuses a colour that is not Ready |
| the build cannot move the pointer | the Job writes a collection and prints the next two commands; a person or a pipeline runs them |
| a published index is never written again | one release per build: `helm install build-vN` refuses a name in use; the Job refuses an existing collection as a second guard |

## Two things the manifests do that are easy to miss

- `enableServiceLinks: false` on every pod. Kubernetes injects
  `<SERVICE>_PORT=tcp://…` for every Service in the namespace; a Service named
  `chroma` hands the Chroma server `CHROMA_PORT=tcp://10.96.x.x:8000`, inside the
  `CHROMA_*` prefix its own config loader reads, and it panics at startup.
- Two kinds of bad target, both refused at startup. A name with nothing behind it
  (the raw chromadb `get_collection` raises; and `create_collection_if_not_exists=False`
  in `app/store.py` keeps the wrapper from inventing an empty one), and a collection a
  crashed build left half-written: the build records `expected_docs` at creation and
  sets `build=complete` only after the last write, and the service checks both.
