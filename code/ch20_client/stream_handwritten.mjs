// Chapter 20 - reading the streaming /ask endpoint from a client, by hand.
//
// The browser's built-in EventSource cannot do this: it only does GET, sends no
// body, and sets no headers - and POST /ask needs all three. So we drive the
// stream with fetch() and a reader over the response body. Node 18+ ships global
// fetch / TextDecoder / ReadableStream, so this file has no dependencies.
//
// Run (with code/ch20_serving_async.py already serving on :8001):
//   node code/ch20_client/stream_handwritten.mjs

const BASE = process.env.BASE_URL ?? "http://localhost:8001";

async function ask(question, threadId) {
  const res = await fetch(`${BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, thread_id: threadId }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);

  const runId = res.headers.get("X-Run-Id");   // the id is in the header, not the body
  const reader = res.body.getReader();          // curl's read loop, in JS
  const decoder = new TextDecoder();
  let buffer = "";
  let answer = "";
  let runStatus = null;                           // "paused" | "failed", as the server reported it
  let sawDone = false;                            // did [DONE] arrive? that, not EOF, ends the protocol

  outer: while (true) {
    const { value, done } = await reader.read();
    if (done) break;                              // the socket closed; NOT proof the server finished
    buffer += decoder.decode(value, { stream: true });

    let i;
    while ((i = buffer.indexOf("\n\n")) !== -1) {   // one SSE event per blank line
      const event = buffer.slice(0, i);
      buffer = buffer.slice(i + 2);
      for (const line of event.split("\n")) {
        if (!line.startsWith("data:")) continue;    // the data: chunk; skip the rest
        const data = line.slice(5).trim();
        if (data === "[DONE]") { sawDone = true; break outer; }          // the server's last word
        const evt = JSON.parse(data);
        if (evt.paused) {                           // parked at the gate, no answer yet
          process.stdout.write(`\n[paused] ${evt.paused.ask ?? "awaiting input"}\n`);
          runStatus = "paused";
          continue;                                 // [DONE] still follows
        }
        if (evt.failed) {                           // a node raised mid-stream
          process.stdout.write(`\n[failed] ${evt.failed}; GET /run/${runId} for where it stopped\n`);
          runStatus = "failed";
          continue;                                 // [DONE] still follows
        }
        answer += evt.token;
        process.stdout.write(evt.token);            // render as each token lands
      }
    }
  }
  if (!sawDone) {                                 // EOF with no [DONE]: the stream was cut off
    process.stdout.write(`\n[incomplete] stream ended without [DONE]; GET /run/${runId} for the run's state\n`);
    return { runId, answer, incomplete: true, runStatus };   // keep what the run reported
  }
  process.stdout.write("\n");
  return { runId, answer, ended: runStatus ?? "completed" };
}

const question = "How do I configure Kafka batch consumers?";
const { runId, answer, ended, incomplete, runStatus } = await ask(question, "demo-hand");
// runStatus is what the RUN reported; incomplete is whether the RESPONSE finished.
const state = incomplete ? `incomplete${runStatus ? ` (last reported: ${runStatus})` : ""}` : ended;
console.log(`\n[runId=${runId}] ${state}: answer is ${answer.length} chars`);
if (incomplete) process.exit(2);                 // partial text is not an answer
