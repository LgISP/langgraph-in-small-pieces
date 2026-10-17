# Chapter 20 - client-side stream readers

Two ways for a front end to consume the streaming `POST /ask` endpoint from
`code/ch20_serving_async.py`. Both read the same SSE stream (`data: {"token": ...}`
per token, a closing `data: [DONE]`) and pull the run id from the `X-Run-Id`
response header.

- `stream_handwritten.mjs` - `fetch()` + a `ReadableStream` reader, no dependencies.
  This is `curl -N`'s read loop written out in JavaScript.
- `stream_fetch_event_source.mjs` - the same thing via `@microsoft/fetch-event-source`,
  which parses the events and adds reconnect/error handling.

## Run

```sh
# 1) start the server (needs LM Studio on :1234; from the book's venv)
../../.venv1x/bin/python ../ch20_serving_async.py    # serves on :8001

# 2a) hand-written reader - no install needed (Node 18+)
node stream_handwritten.mjs

# 2b) library version - install the one dependency first
npm install
node stream_fetch_event_source.mjs
```

Both print the answer token by token, then a `[runId=...] answer is N chars` line.
The token text is the model's and will drift run to run; what is stable is the
framing and the run id round-trip.
