// Chapter 21 - the same streaming /ask client, using @microsoft/fetch-event-source.
//
// The hand-written reader in stream_handwritten.mjs is worth seeing once. In
// production you reach for a library that keeps the POST + body + headers (which
// native EventSource cannot), parses the `data:` events for you, and adds reconnect
// and error handling. This is that version.
//
// Setup:  cd code/ch20_client && npm install
// Run  :  node code/ch20_client/stream_fetch_event_source.mjs
//         (with code/ch20_serving_async.py already serving on :8001)

// @microsoft/fetch-event-source targets the browser: it reaches for window.fetch,
// window.setTimeout/clearTimeout, and a document visibilitychange listener. In a
// browser you need none of the two lines below; here they let the same code run
// under Node so we can test it from the command line. Aliasing window to globalThis
// gives it the real fetch and timer functions; document only needs the two stubs.
globalThis.window ??= globalThis;
globalThis.document ??= { addEventListener() {}, removeEventListener() {} };

import { fetchEventSource } from "@microsoft/fetch-event-source";

const BASE = process.env.BASE_URL ?? "http://localhost:8001";

async function ask(question, threadId) {
  const ctrl = new AbortController();
  let runId = null;
  let answer = "";
  let runStatus = null;      // "paused" | "failed", as the server reported it
  let sawDone = false;       // did [DONE] arrive? that, not EOF, ends the protocol

  try {
    await fetchEventSource(`${BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, thread_id: threadId }),
      signal: ctrl.signal,
      fetch: globalThis.fetch,   // the lib defaults to window.fetch; Node has no window
      openWhenHidden: true,      // Node has no document; skip the visibility handling
      async onopen(res) {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        runId = res.headers.get("X-Run-Id");   // still in the header
      },
      onmessage(ev) {
        if (ev.data === "[DONE]") {
          sawDone = true;          // the server's last word
          ctrl.abort();            // our sentinel: stop, and don't let the lib reconnect
          return;
        }
        const evt = JSON.parse(ev.data);
        if (evt.paused) {   // parked at the gate: no answer, resume via the sync server
          process.stdout.write(`\n[paused] ${evt.paused.ask ?? "awaiting input"}\n`);
          runStatus = "paused";
          return;           // do NOT abort: [DONE] still has to arrive, or this was cut off
        }
        if (evt.failed) {   // a node raised mid-stream
          process.stdout.write(`\n[failed] ${evt.failed}; GET /run/${runId} for where it stopped\n`);
          runStatus = "failed";
          return;           // [DONE] still follows
        }
        answer += evt.token;
        process.stdout.write(evt.token);
      },
      onclose() {
        // The library's clean-close hook. Reaching it without [DONE] means the
        // response ended on its own, which is NOT proof the server finished.
      },
      onerror(err) {
        throw err;   // a demo bails on error instead of retrying forever
      },
    });
  } catch (err) {
    if (err.name !== "AbortError") throw err;   // AbortError = our clean [DONE] stop
  }

  if (!sawDone) {   // EOF with no [DONE]: the stream was cut off
    process.stdout.write(`\n[incomplete] stream ended without [DONE]; GET /run/${runId} for the run's state\n`);
    return { runId, answer, incomplete: true, runStatus };   // keep what the run reported
  }
  process.stdout.write("\n");
  return { runId, answer, ended: runStatus ?? "completed" };
}

const question = "How do I configure Kafka batch consumers?";
const { runId, answer, ended, incomplete, runStatus } = await ask(question, "demo-fes");
// runStatus is what the RUN reported; incomplete is whether the RESPONSE finished.
const state = incomplete ? `incomplete${runStatus ? ` (last reported: ${runStatus})` : ""}` : ended;
console.log(`\n[runId=${runId}] ${state}: answer is ${answer.length} chars`);
if (incomplete) process.exit(2);                 // partial text is not an answer
