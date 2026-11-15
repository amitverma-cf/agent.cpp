// Self-test of src/engines/api/bridge.js against a local mock of an OpenAI-compatible server (no network).
// Run: bun tests/helpers/api_bridge_selftest.js src/engines/api/bridge.js   (ctest: api_bridge_js, when bun is installed)
const bridgePath = process.argv[2];
const KEY = "sk-test-secret-1234";
process.env.SELFTEST_KEY = KEY;

const sse = (...objs) => objs.map((o) => "data: " + (typeof o === "string" ? o : JSON.stringify(o)) + "\n\n").join("");
const chunk = (delta, extra) => ({ id: "x", object: "chat.completion.chunk", unknown_field: { a: 1 }, choices: [{ index: 0, delta, finish_reason: null }], ...extra });

let scenario = "";
let lastRequest = null;
const server = Bun.serve({
  port: 0,
  async fetch(req) {
    lastRequest = { headers: Object.fromEntries(req.headers), body: await req.json() };
    const sseHeaders = { "content-type": "text/event-stream" };
    switch (scenario) {
      case "stream":
        return new Response(
          sse(
            chunk({ role: "assistant", reasoning_content: "thinking..." }),
            chunk({ content: "Hel" }),
            chunk({ content: "lo" }),
            { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
            { choices: [], usage: { prompt_tokens: 11, completion_tokens: 2, prompt_tokens_details: { cached_tokens: 5 } } },
            "[DONE]",
          ),
          { headers: sseHeaders },
        );
      case "tools":
        return new Response(
          sse(
            chunk({ tool_calls: [{ index: 0, function: { name: "read_file", arguments: "" } }] }),
            chunk({ tool_calls: [{ index: 0, function: { arguments: '{"path":' } }] }),
            chunk({ tool_calls: [{ index: 0, function: { arguments: '"a.txt"}' } }, { index: 1, id: "call_b", function: { name: "todo", arguments: "{}" } }] }),
            { choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] },
            "[DONE]",
          ),
          { headers: sseHeaders },
        );
      case "plain-json":
        return Response.json({ id: "y", extra: [1, 2], choices: [{ index: 0, message: { role: "assistant", content: "whole reply", reasoning: "r" }, finish_reason: "stop" }], usage: { prompt_tokens: 3, completion_tokens: 2 } });
      case "503":
        return new Response("upstream busy", { status: 503 });
      case "429":
        return new Response(JSON.stringify({ error: { message: "slow down" } }), { status: 429 });
      case "401":
        return new Response(JSON.stringify({ error: { message: "invalid key " + KEY } }), { status: 401 });
      case "400":
        return new Response(JSON.stringify({ error: { message: "bad request" } }), { status: 400 });
      case "empty":
        return Response.json({ choices: [{ index: 0, message: { role: "assistant", content: "" }, finish_reason: "stop" }] });
      case "no-choices":
        return Response.json({ id: "z" });
      case "not-json":
        return new Response("<html>gateway</html>", { headers: { "content-type": "text/html" } });
      case "cut":
        return new Response(sse(chunk({ content: "partial" })), { headers: sseHeaders });
      case "bad-args":
        return new Response(sse(chunk({ tool_calls: [{ index: 0, id: "c", function: { name: "t", arguments: "{oops" } }] }), { choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] }, "[DONE]"), { headers: sseHeaders });
      case "length":
        return Response.json({ choices: [{ index: 0, message: { role: "assistant", content: "", reasoning_content: "long thought" }, finish_reason: "length" }] });
      case "hang": {
        const body = new ReadableStream({ start(c) { c.enqueue(new TextEncoder().encode(sse(chunk({ content: "a" })))); } });
        return new Response(body, { headers: sseHeaders });
      }
      case "stream-error":
        return new Response(sse(chunk({ content: "x" }), { error: { message: "model overloaded" } }), { headers: sseHeaders });
    }
    return new Response("?", { status: 500 });
  },
});

const child = Bun.spawn(["bun", bridgePath], { stdin: "pipe", stdout: "pipe", stderr: "inherit", env: process.env });
const lines = [];
let waiter = null;
(async () => {
  const dec = new TextDecoder();
  let buf = "";
  for await (const part of child.stdout) {
    buf += dec.decode(part, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const text = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      if (text) lines.push(JSON.parse(text));
      if (waiter) waiter();
    }
  }
})();
const send = (o) => { child.stdin.write(JSON.stringify(o) + "\n"); child.stdin.flush(); };
const waitFor = async (pred, ms = 8000) => {
  const end = Date.now() + ms;
  for (;;) {
    const hit = lines.find(pred);
    if (hit) return hit;
    if (Date.now() > end) throw new Error("timed out waiting for the bridge; saw " + JSON.stringify(lines));
    await new Promise((r) => { waiter = r; setTimeout(r, 50); });
  }
};

let nextId = 1;
async function ask(name, extra = {}) {
  scenario = name;
  const id = nextId++;
  send({ op: "chat", id, base_url: `http://127.0.0.1:${server.port}/v1/`, model: "m", key_env: "SELFTEST_KEY", timeout_ms: 3000, stream: true, messages: [{ role: "user", content: "hi" }], max_tokens: 50, temperature: 0.3, ...extra });
  const end = await waitFor((l) => l.id === id && (l.t === "done" || l.t === "error"));
  return { end, deltas: lines.filter((l) => l.id === id && l.t === "delta").map((l) => l.text).join(""), id };
}

let failed = 0;
function check(label, cond, detail) {
  if (!cond) { failed++; console.error("FAIL " + label + (detail ? ": " + JSON.stringify(detail) : "")); } else console.log("ok   " + label);
}

await waitFor((l) => l.t === "ready");

let r = await ask("stream");
check("stream: done with content", r.end.t === "done" && r.end.content === "Hello", r.end);
check("stream: deltas arrive and reasoning is not forwarded", r.deltas === "Hello", r.deltas);
check("stream: usage and cached tokens", r.end.usage.prompt_tokens === 11 && r.end.usage.completion_tokens === 2 && r.end.usage.cached_tokens === 5, r.end.usage);
check("stream: request has bearer key, model, stream flags, max_tokens", lastRequest.headers.authorization === "Bearer " + KEY && lastRequest.body.model === "m" && lastRequest.body.stream === true && lastRequest.body.stream_options.include_usage === true && lastRequest.body.max_tokens === 50, lastRequest.body);
check("the key is not in any output line", !JSON.stringify(lines).includes(KEY));

r = await ask("tools", { tools: [{ name: "read_file", description: "d", parameters: { type: "object", properties: {} } }] });
check("tools: sent in function form", lastRequest.body.tools[0].type === "function" && lastRequest.body.tools[0].function.name === "read_file", lastRequest.body.tools);
check("tools: arguments joined across chunks", r.end.t === "done" && r.end.tool_calls[0].name === "read_file" && r.end.tool_calls[0].arguments === '{"path":"a.txt"}', r.end);
check("tools: missing id is generated, given id kept", r.end.tool_calls[0].id.startsWith("call_") && r.end.tool_calls[1].id === "call_b", r.end.tool_calls);

r = await ask("plain-json", { stream: false });
check("non-stream reply with unknown fields", r.end.t === "done" && r.end.content === "whole reply" && r.end.usage.prompt_tokens === 3 && !lastRequest.body.stream, r.end);
r = await ask("plain-json");
check("provider ignoring stream:true still works", r.end.t === "done" && r.end.content === "whole reply", r.end);

for (const [name, status, retry] of [["503", 503, true], ["429", 429, true], ["400", 400, false], ["401", 401, false]]) {
  r = await ask(name);
  check(`${name}: error, retryable=${retry}`, r.end.t === "error" && r.end.status === status && r.end.retryable === retry, r.end);
}
check("401: the key is scrubbed from the provider's text", !JSON.stringify(lines).includes(KEY) && r.end.message.includes("***"), r.end.message);

for (const name of ["empty", "no-choices", "not-json", "cut", "bad-args", "stream-error"]) {
  r = await ask(name);
  check(`${name}: retryable error`, r.end.t === "error" && r.end.retryable === true, r.end);
}
r = await ask("length");
check("length with only reasoning: not retryable", r.end.t === "error" && r.end.retryable === false, r.end);

r = await ask("hang", { timeout_ms: 1000 });
check("silence for timeout_ms: retryable time-out", r.end.t === "error" && r.end.retryable === true && /timed out/.test(r.end.message), r.end);

scenario = "hang";
const hid = nextId++;
send({ op: "chat", id: hid, base_url: `http://127.0.0.1:${server.port}/v1`, model: "m", key_env: "SELFTEST_KEY", timeout_ms: 20000, stream: true, messages: [] });
await waitFor((l) => l.id === hid && l.t === "delta");
send({ op: "cancel", id: hid });
const cancelled = await waitFor((l) => l.id === hid && l.t === "error");
check("cancel aborts the request", cancelled.cancelled === true && cancelled.retryable === false, cancelled);

r = await ask("stream", { extra_body: { top_k: 7, model: "ignored" } });
check("extra_body merged, core fields win", lastRequest.body.top_k === 7 && lastRequest.body.model === "m", lastRequest.body);

send({ op: "quit" });
await child.exited;
server.stop(true);
if (failed) {
  console.error(failed + " check(s) failed");
  process.exit(1);
}
console.log("all bridge checks passed");
process.exit(0);
