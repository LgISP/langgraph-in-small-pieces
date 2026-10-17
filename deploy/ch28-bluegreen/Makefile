# The chapter's five steps, as the commands you would actually run: make -> helm.
#
# The image tag is a hash of the app's content. A reused tag such as :dev is a
# trap on any cluster whose nodes cache images (imagePullPolicy: IfNotPresent):
# you rebuild on the host, the nodes keep serving the old bytes under the same
# name, and nothing tells you. A new tag per build is a changed pod template,
# which is a roll-out you can see.
NS  := docbot
TAG := $(shell tar cf - app requirements.txt Dockerfile 2>/dev/null | shasum | cut -c1-12)
IMG := docbot-ch27:$(TAG)
HELM := helm -n $(NS)

.PHONY: image up build-index stage cutover rollback endpoints-settled deploy-image status history list retire partial down

# The colour the Service points at, and the other one. Read from the live
# release, so every target below acts on the right Deployment.
ACTIVE = $$(helm -n $(NS) get values docbot -a -o json | python3 -c 'import sys,json; print(json.load(sys.stdin)["active"])')
IDLE   = $$([ "$(ACTIVE)" = blue ] && echo green || echo blue)

image:                      ## build the app image under its content hash
	docker build -t $(IMG) .
	@echo "image: $(IMG)"

up: image                   ## the serving stack, blue on docs_v1: build the index, then give blue its pods
	kubectl create namespace $(NS) --dry-run=client -o yaml | kubectl apply -f -
	$(HELM) upgrade --install docbot ./chart --set image.tag=$(TAG) --set active=blue --set blue.activeIndex=docs_v1 \
	  --set blue.replicas=0 --wait --timeout 180s     # store + config first, no doc-bot pods yet
	$(MAKE) build-index V=1
	$(HELM) upgrade docbot ./chart --reuse-values --set blue.replicas=2 --wait --timeout 180s

build-index:                ## build-index V=2  -> one release per version; a Job writes docs_v2 (does NOT deploy it)
	$(HELM) install build-v$(V) ./chart-build --set version=$(V) --set image.tag=$(TAG) --set-file docs=docs/v$(V).json
	@ok=0; for i in $$(seq 1 150); do \
	  st=$$(kubectl -n $(NS) get job/build-index-v$(V) -o jsonpath='{.status.succeeded}{"/"}{.status.failed}' 2>/dev/null); \
	  case "$$st" in 1/*) ok=1; break;; */1) kubectl -n $(NS) logs job/build-index-v$(V); echo "build FAILED"; exit 1;; esac; sleep 2; done; \
	  if [ "$$ok" != 1 ]; then echo "build-index-v$(V): no success after 300s (status '$$st')"; kubectl -n $(NS) get job/build-index-v$(V); exit 1; fi
	kubectl -n $(NS) logs job/build-index-v$(V)

stage:                      ## stage V=2  -> the idle colour gets docs_v2 and its pods; must be Ready before cutover; the Service does not move
	@idle=$(IDLE); echo "staging docs_v$(V) on $$idle (the Service stays on $(ACTIVE))"; \
	  $(HELM) upgrade docbot ./chart --reuse-values --set $$idle.activeIndex=docs_v$(V) --set $$idle.replicas=2 --wait --timeout 120s

cutover:                    ## the deploy: point the Service at the staged colour -- one label, all traffic at once
	@idle=$(IDLE); \
	  kubectl -n $(NS) wait --for=condition=available --timeout=10s deployment/docbot-$$idle >/dev/null || { echo "docbot-$$idle is not Ready: nothing to cut over to"; exit 1; }; \
	  echo "cutting over: Service docbot -> $$idle"; \
	  $(HELM) upgrade docbot ./chart --reuse-values --set active=$$idle --wait --timeout 60s
	@$(MAKE) -s endpoints-settled

rollback:                   ## the previous release back: the selector returns, no pod restarts
	$(HELM) rollback docbot --wait --timeout 120s
	@$(MAKE) -s endpoints-settled

# `helm --wait` returns when the Service object is updated; the endpoints behind
# it (and kube-proxy's rules) follow a moment later. A cut-over is done when every
# endpoint of the Service belongs to the active colour, so wait for that.
endpoints-settled:
	@active=$(ACTIVE); for i in $$(seq 1 30); do \
	  names=$$(kubectl -n $(NS) get endpointslices -l kubernetes.io/service-name=docbot -o jsonpath='{range .items[*].endpoints[*]}{.targetRef.name}{"\n"}{end}'); \
	  if [ -n "$$names" ] && ! grep -qv "^docbot-$$active-" <<<"$$names"; then sleep 1; exit 0; fi; sleep 1; done; \
	  echo "Service docbot: endpoints did not settle on $$active"; exit 1

deploy-image: image         ## after a code change: both colours roll onto the image just built
	$(HELM) upgrade docbot ./chart --reuse-values --set image.tag=$(TAG) --wait --timeout 180s

status:
	$(HELM) list
	@echo "Service docbot -> $(ACTIVE)"
	@for c in blue green; do printf "  %-6s " $$c; kubectl -n $(NS) get deployment docbot-$$c -o jsonpath='{.spec.replicas} pod(s) on {.spec.template.spec.containers[0].env[?(@.name=="ACTIVE_INDEX")].value}{"\n"}'; done
	kubectl -n $(NS) get pods -l app=docbot

history:
	$(HELM) history docbot

list:
	kubectl -n $(NS) run admin-$$RANDOM --rm -i --restart=Never --image=$(IMG) --image-pull-policy=IfNotPresent \
	  --env=CHROMA_HOST=chroma --env=CHROMA_PORT=8000 -- python -m app.admin list

retire:                     ## retire V=1  -> only after the cut-over is final: drop the collection and its build release, and idle the colour that served it
	@idle=$(IDLE); idx=$$(kubectl -n $(NS) get deployment docbot-$$idle -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="ACTIVE_INDEX")].value}'); \
	  if [ "$$idx" = docs_v$(V) ]; then $(HELM) upgrade docbot ./chart --reuse-values --set $$idle.replicas=0 --set $$idle.activeIndex="" --wait --timeout 60s >/dev/null; echo "  $$idle idled"; fi
	kubectl -n $(NS) run admin-$$RANDOM --rm -i --restart=Never --image=$(IMG) --image-pull-policy=IfNotPresent \
	  --env=CHROMA_HOST=chroma --env=CHROMA_PORT=8000 -- python -m app.admin delete docs_v$(V)
	-$(HELM) uninstall build-v$(V) >/dev/null 2>&1

partial:                    ## partial V=7  -> a half-written collection, for testing the startup checks
	kubectl -n $(NS) run admin-$$RANDOM --rm -i --restart=Never --image=$(IMG) --image-pull-policy=IfNotPresent \
	  --env=CHROMA_HOST=chroma --env=CHROMA_PORT=8000 -- python -m app.admin partial docs_v$(V) 3

down:
	-$(HELM) list -q | xargs -n1 $(HELM) uninstall >/dev/null 2>&1
	kubectl delete namespace $(NS) --ignore-not-found
