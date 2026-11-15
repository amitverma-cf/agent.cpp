// agent.cpp "api" engine bridge: one attempt of an OpenAI-compatible chat completion per request, over line-delimited JSON.
//
// stdin  (one JSON object per line)
//   {"op":"chat","id":1,"base_url":"https://host/v1","model":"id","key_env":"NAME","timeout_ms":90000,"stream":true,
//    "max_tokens":512,"temperature":0.7,"messages":[...],"tools":[{"name","description","parameters"}],"extra_body":{...}}
//   {"op":"cancel","id":1}
//   {"op":"quit"}
// stdout (one JSON object per line)
//   {"t":"ready"}
//   {"t":"delta","id":1,"text":"..."}                              streamed answer text (reasoning is not forwarded)
//   {"t":"done","id":1,"content":"...","tool_calls":[{"id","name","arguments"}],"usage":{...},"finish":"stop","reasoning_chars":0}
//   {"t":"error","id":1,"message":"...","retryable":true,"status":503,"cancelled":false}
//
// Retrying, backoff and the conversation live in the C++ engine; this script makes exactly one HTTP attempt per request.
// Plain fetch, no packages. The API key is read from the environment variable named in key_env, goes only into the
// Authorization header and is removed from every message this script prints.

const active = new Map();
const send = (o) => process.stdout.write(JSON.stringify(o) + "\n");

function keyOf(req) {
  const name = typeof req.key_env === "string" ? req.key_env : "";
  return name ? process.env[name] || "" : "";
}

function scrub(text, key) {
  let s = String(text);
  if (key && key.length >= 4) s = s.split(key).join("***");
  return s.length > 600 ? s.slice(0, 600) + "..." : s;
}

class Failure extends Error {
  constructor(message, retryable, status) {
    super(message);
    this.retryable = retryable;
    this.status = status || 0;
  }
}

function toolsOf(list) {
  if (!Array.isArray(list) || !list.length) return undefined;
  return list.map((t) =>
    t && t.type === "function" && t.function
      ? t
      : { type: "function", function: { name: t.name, description: t.description || "", parameters: t.parameters || { type: "object", properties: {} } } },
  );
}

function buildBody(req, withUsage) {
  const body = { model: req.model, messages: req.messages || [] };
  if (Number.isFinite(req.max_tokens) && req.max_tokens > 0) body.max_tokens = req.max_tokens;
  if (Number.isFinite(req.temperature) && req.temperature >= 0) body.temperature = req.temperature;
  const tools = toolsOf(req.tools);
  if (tools) body.tools = tools;
  if (req.stream) {
    body.stream = true;
    if (withUsage) body.stream_options = { include_usage: true };
  }
  if (req.extra_body && typeof req.extra_body === "object") for (const [k, v] of Object.entries(req.extra_body)) if (!(k in body)) body[k] = v;
  return body;
}

// Collects the pieces of one answer, whichever shape the provider sends them in. Unknown fields are ignored.
class Answer {
  constructor(emit) {
    this.emit = emit;
    this.content = "";
    this.reasoning = 0;
    this.calls = new Map();
    this.finish = "";
    this.usage = null;
    this.chunks = 0;
    this.done = false;
  }
  text(s) {
    this.content += s;
    if (this.emit) this.emit(s);
  }
  add(obj) {
    if (!obj || typeof obj !== "object") return;
    if (obj.error) {
      const e = obj.error;
      throw new Failure("provider error: " + (typeof e === "string" ? e : e.message || JSON.stringify(e)), true, 0);
    }
    this.chunks++;
    if (obj.usage && typeof obj.usage === "object") this.usage = obj.usage;
    const choice = Array.isArray(obj.choices) ? obj.choices[0] : null;
    if (!choice) return;
    const d = choice.delta || choice.message || {};
    if (typeof d.content === "string") {
      if (d.content) this.text(d.content);
    } else if (Array.isArray(d.content)) {
      for (const p of d.content) if (p && typeof p.text === "string" && p.text) this.text(p.text);
    }
    for (const r of [d.reasoning_content, d.reasoning]) if (typeof r === "string") this.reasoning += r.length;
    if (Array.isArray(d.tool_calls)) {
      for (const tc of d.tool_calls) {
        if (!tc || typeof tc !== "object") continue;
        const idx = Number.isInteger(tc.index) ? tc.index : this.calls.size;
        const cur = this.calls.get(idx) || { id: "", name: "", arguments: "" };
        if (typeof tc.id === "string" && tc.id) cur.id = tc.id;
        const f = tc.function || {};
        if (typeof f.name === "string") cur.name += f.name;
        if (typeof f.arguments === "string") cur.arguments += f.arguments;
        else if (f.arguments && typeof f.arguments === "object") cur.arguments += JSON.stringify(f.arguments);
        this.calls.set(idx, cur);
      }
    }
    if (typeof choice.finish_reason === "string" && choice.finish_reason) this.finish = choice.finish_reason;
  }
  result() {
    const calls = [...this.calls.entries()].sort((a, b) => a[0] - b[0]).map((e) => e[1]);
    for (let i = 0; i < calls.length; i++) {
      const c = calls[i];
      if (!c.name) throw new Failure("malformed reply: a tool call without a name", true, 0);
      if (!c.id) c.id = "call_" + i + "_" + Date.now().toString(36);
      const a = c.arguments.trim();
      if (!a) c.arguments = "{}";
      else {
        let v;
        try {
          v = JSON.parse(a);
        } catch {
          throw new Failure("malformed reply: tool call arguments are not valid JSON", true, 0);
        }
        if (!v || typeof v !== "object" || Array.isArray(v)) throw new Failure("malformed reply: tool call arguments are not a JSON object", true, 0);
      }
    }
    if (!this.content && !calls.length) {
      if (this.finish === "length")
        throw new Failure("the model used its whole token limit before it wrote an answer (finish_reason length); raise the maximum answer length", false, 0);
      throw new Failure(this.chunks ? "empty reply" : "malformed reply: no choices in the response", true, 0);
    }
    if (!this.finish && !this.done) throw new Failure("the reply ended early (the stream was cut off)", true, 0);
    const u = this.usage || {};
    const details = u.prompt_tokens_details || {};
    return {
      content: this.content,
      tool_calls: calls,
      finish: this.finish,
      reasoning_chars: this.reasoning,
      usage: { prompt_tokens: u.prompt_tokens | 0, completion_tokens: u.completion_tokens | 0, cached_tokens: details.cached_tokens | 0 },
    };
  }
}

async function readStream(res, answer, touch) {
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  const line = (raw) => {
    const l = raw.replace(/\r$/, "");
    if (!l.startsWith("data:")) return;
    const data = l.slice(5).trim();
    if (!data) return;
    if (data === "[DONE]") {
      answer.done = true;
      return;
    }
    let obj;
    try {
      obj = JSON.parse(data);
    } catch {
      return; // a broken line is skipped; an answer made of nothing else fails as empty
    }
    answer.add(obj);
  };
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    touch();
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      line(buf.slice(0, nl));
      buf = buf.slice(nl + 1);
    }
  }
  buf += dec.decode();
  if (buf) line(buf);
}

async function chat(req) {
  const id = req.id;
  const key = keyOf(req);
  const ac = new AbortController();
  const st = { cancelled: false, timedOut: false };
  active.set(id, { ac, st });
  let timer;
  // The per-try limit is the time without any data: it restarts whenever bytes arrive, so a long answer that keeps streaming is fine.
  const touch = () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      st.timedOut = true;
      ac.abort();
    }, Math.max(1000, req.timeout_ms || 90000));
  };
  try {
    touch();
    const url = String(req.base_url || "").replace(/\/+$/, "") + "/chat/completions";
    const headers = { "content-type": "application/json", accept: req.stream ? "text/event-stream, application/json" : "application/json" };
    if (key) headers.authorization = "Bearer " + key;
    let res;
    for (let withUsage = true; ; withUsage = false) {
      res = await fetch(url, { method: "POST", headers, body: JSON.stringify(buildBody(req, withUsage)), signal: ac.signal });
      if (res.ok) break;
      const text = await res.text().catch(() => "");
      if (withUsage && req.stream && res.status === 400 && /stream_options|include_usage/.test(text)) continue; // a provider that rejects the option
      let detail = text;
      try {
        const j = JSON.parse(text);
        const e = j.error || j;
        detail = typeof e === "string" ? e : e.message || text;
      } catch {}
      const hint = res.status === 401 || res.status === 403 ? " (check the key in the environment variable " + (req.key_env || "api_key_env") + ")" : "";
      const retryable = res.status === 408 || res.status === 425 || res.status === 429 || res.status >= 500;
      throw new Failure("HTTP " + res.status + ": " + scrub(detail || res.statusText, key) + hint, retryable, res.status);
    }
    touch();
    const answer = new Answer(req.stream ? (text) => send({ t: "delta", id, text }) : null);
    const type = (res.headers.get("content-type") || "").toLowerCase();
    if (type.includes("text/event-stream") && res.body) {
      await readStream(res, answer, touch);
    } else {
      const text = await res.text();
      let obj;
      try {
        obj = JSON.parse(text);
      } catch {
        throw new Failure("malformed reply: the body is not JSON (" + scrub(text.slice(0, 80), key) + ")", true, 0);
      }
      // A provider that ignored stream:true sends one message; it counts as complete.
      answer.add(obj);
      answer.done = true;
    }
    send({ t: "done", id, ...answer.result() });
  } catch (e) {
    if (st.cancelled) send({ t: "error", id, message: "cancelled", retryable: false, cancelled: true });
    else if (st.timedOut) send({ t: "error", id, message: "timed out after " + Math.round((req.timeout_ms || 90000) / 1000) + " s without data", retryable: true });
    else if (e instanceof Failure) send({ t: "error", id, message: scrub(e.message, key), retryable: e.retryable, status: e.status });
    else send({ t: "error", id, message: "connection failed: " + scrub(e && e.message ? e.message : e, key), retryable: true });
  } finally {
    clearTimeout(timer);
    active.delete(id);
  }
}

let pending = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  pending += chunk;
  let nl;
  while ((nl = pending.indexOf("\n")) >= 0) {
    const text = pending.slice(0, nl).trim();
    pending = pending.slice(nl + 1);
    if (!text) continue;
    let req;
    try {
      req = JSON.parse(text);
    } catch {
      continue;
    }
    if (req.op === "chat") chat(req);
    else if (req.op === "cancel") {
      const a = active.get(req.id);
      if (a) {
        a.st.cancelled = true;
        a.ac.abort();
      }
    } else if (req.op === "quit") process.exit(0);
  }
});
process.stdin.on("end", () => process.exit(0));
send({ t: "ready" });
