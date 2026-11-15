// free-model-aggregator-02: one server, two client wires, one live free model
// behind every answer.
//
// The server reads the brief at $MAKEFUN_PROMPT_FILE at runtime (the free
// model list with ratings/capabilities, plus the six-prompt manifest), picks
// a free port, registers it with the env, serves GET /v1/models and both
// completion wires, dials every prompt on the gateway's Chat Completions
// route, and exits 0 when GET $MAKEFUN_ENV_URL/work says done.
//
// Routing: eligible models ranked by rating_milli, best first; a gateway
// refusal fails over to the next-best eligible model and is reported in the
// x-makefun-upstream-failures header. A model that just refused is skipped
// for the next prompt too (a deterministic, count-based cooldown: the
// cooldown is a function of prompts seen, never of the wall clock).
// Nothing is ever answered with locally invented text.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import {
  buildCatalog,
  freeModels,
  isRecord,
  rankedEligible,
  readBrief,
  requirementsFromBody,
  type Brief,
  type Catalog,
  type Json,
  type ModelRow,
  type Requirement,
} from "./lib/brief.ts";
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  openAIUpstreamBody,
  type ServedModel,
} from "./lib/translate.ts";

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const MAX_BODY_BYTES = 8 * 1024 * 1024;
const GATEWAY_TIMEOUT_MS = 30000;
const MAX_ATTEMPTS_PER_PROMPT = 4;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/+$/, "");
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? "";
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? "").replace(/\/+$/, "");
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? "";

const brief: Brief = readBrief(PROMPT_FILE);
const catalog: Catalog = buildCatalog(brief);

/** Prompts served so far, in arrival order (the env sends them in manifest order). */
let promptIndex = 0;
/** model id -> the prompt index until which it is cooling down (exclusive). */
const cooldownUntil = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** The gateway's Chat Completions route. */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === "") return "";
  return GATEWAY_URL.endsWith("/v1")
    ? `${GATEWAY_URL}/chat/completions`
    : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The models this server lists, in the brief's order. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((m) => ({
    id: m.id,
    provider: m.provider,
    display_name: m.display_name,
  }));
}

/** Requirements for the prompt now arriving: the manifest row, else the body. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[promptIndex];
  promptIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/**
 * Eligible models for a prompt, best rating first, skipping models that are
 * cooling down after a refusal on an earlier prompt. Within one prompt's
 * failover chain the caller additionally skips models already tried.
 */
function candidatesFor(req: Requirement, tried: Set<string>): ModelRow[] {
  return rankedEligible(brief, req).filter((m) => {
    if (tried.has(m.id)) return false;
    const until = cooldownUntil.get(m.id);
    return until === undefined || promptIndex > until;
  });
}

interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** The error code string of a refused gateway answer, from its error body. */
function codeOf(status: number, text: string): string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      if (typeof parsed.error.code === "string" && parsed.error.code !== "") {
        return parsed.error.code;
      }
      if (typeof parsed.error.type === "string" && parsed.error.type !== "") {
        return parsed.error.type;
      }
    }
    if (isRecord(parsed) && typeof parsed.code === "string" && parsed.code !== "") {
      return parsed.code;
    }
  } catch {
    // Not JSON: fall through to the status-based code.
  }
  return `http_${status}`;
}

interface DialResult {
  ok: boolean;
  response: Response | null;
  failure: FailureRow | null;
}

/** One gateway call for `model`, with the 30s per-prompt client timeout. */
async function dial(model: string, body: Json): Promise<DialResult> {
  let response: Response;
  try {
    response = await fetch(chatCompletionsUrl(), {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${GATEWAY_KEY}`,
        ...(body.stream === true ? { accept: "text/event-stream" } : {}),
      },
      body: JSON.stringify({ ...body, model }),
      signal: AbortSignal.timeout(GATEWAY_TIMEOUT_MS),
    });
  } catch (err) {
    const name = (err as Error).name ?? "";
    return {
      ok: false,
      response: null,
      failure: {
        status: 504,
        code: name === "TimeoutError" ? "timeout" : "upstream_unreachable",
        model,
      },
    };
  }
  if (response.ok) return { ok: true, response, failure: null };
  const text = await response.text().catch(() => "");
  return {
    ok: false,
    response: null,
    failure: { status: response.status, code: codeOf(response.status, text), model },
  };
}

/**
 * Dial a prompt's upstream body, failing over down the rating-ranked list.
 * Returns the first successful response with the model that served it and
 * the trail of refusals behind it. When every attempt refused, `response`
 * is null and `failures` holds the trail (empty only when no model in the
 * brief met the requirements at all).
 * A refused model cools down for the next prompt (count-based, deterministic).
 */
async function dialWithFailover(
  upstreamBody: Json,
  req: Requirement,
): Promise<{ response: Response | null; model: string | null; failures: FailureRow[] }> {
  const failures: FailureRow[] = [];
  const tried = new Set<string>();
  for (let attempt = 0; attempt < MAX_ATTEMPTS_PER_PROMPT; attempt++) {
    const candidates = candidatesFor(req, tried);
    if (candidates.length === 0) break;
    const model = candidates[0];
    tried.add(model.id);
    const result = await dial(model.id, upstreamBody);
    if (result.ok && result.response !== null) {
      return { response: result.response, model: model.id, failures };
    }
    if (result.failure !== null) {
      failures.push(result.failure);
      // Cool this model down for the next prompt: it is skipped while
      // promptIndex is still at or below the index it failed on + 1.
      // promptIndex was already advanced past this prompt, so +1 covers
      // exactly the next prompt and no more (count-based, deterministic).
      cooldownUntil.set(model.id, promptIndex + 1);
      log(`gateway refused ${model.id}: ${result.failure.status} ${result.failure.code}`);
    }
  }
  return { response: null, model: null, failures };
}

/** The failure trail header value: at most 8 rows, as the rules spell it. */
function failuresHeader(failures: FailureRow[]): string | null {
  if (failures.length === 0) return null;
  return JSON.stringify(failures.slice(0, 8));
}

/** One client request body, or null when it is not one JSON object or too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(
  res: ServerResponse,
  status: number,
  body: unknown,
  extraHeaders?: Record<string, string>,
): void {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(text),
    ...extraHeaders,
  });
  res.end(text);
}

function sendError(
  res: ServerResponse,
  wire: "openai" | "anthropic",
  status: number,
  type: string,
  message: string,
): void {
  if (wire === "anthropic") sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** The response headers for a served prompt, carrying any failure trail. */
function servedHeaders(failures: FailureRow[]): Record<string, string> {
  const header = failuresHeader(failures);
  return header === null ? {} : { "x-makefun-upstream-failures": header };
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers["x-api-key"] === "string") sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** A prompt every eligible model refused: a 502 in the client's wire. */
function allRefused(
  res: ServerResponse,
  wire: "openai" | "anthropic",
  failures: FailureRow[],
): void {
  const last = failures[failures.length - 1];
  const detail = last === undefined ? "no free model answered" : `gateway refused ${last.model ?? "unknown"}: HTTP ${last.status} ${last.code}`;
  sendError(res, wire, 502, wire === "anthropic" ? "api_error" : "upstream_error", detail);
}

/** POST /v1/chat/completions: OpenAI wire, answered by a real free model. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, "openai", 400, "invalid_request_error", "the request body is not one JSON object");
  }
  const reqs = requirementsFor(body);
  const upstream = openAIUpstreamBody(body, "auto");
  const dialed = await dialWithFailover(upstream, reqs);
  if (dialed.response === null || dialed.model === null) {
    if (dialed.failures.length === 0) {
      return sendError(res, "openai", 503, "no_free_model", "no free model in the brief meets this request");
    }
    return allRefused(res, "openai", dialed.failures);
  }
  if (dialed.response.body === null) {
    return allRefused(res, "openai", dialed.failures);
  }
  const headers = servedHeaders(dialed.failures);
  if (body.stream === true) {
    res.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
      ...headers,
    });
    try {
      for await (const frame of openAIStreamWithModel(dialed.response.body, dialed.model)) {
        res.write(frame);
      }
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await dialed.response.json().catch(() => null);
  if (!isRecord(answer)) {
    return allRefused(res, "openai", [
      ...dialed.failures,
      { status: 502, code: "upstream_unreachable", model: dialed.model },
    ]);
  }
  sendJson(res, 200, openAIJsonWithModel(answer, dialed.model), headers);
}

/** POST /v1/messages: Anthropic wire, translated to Chat Completions and back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, "anthropic", 400, "invalid_request_error", "the request body is not one JSON object");
  }
  const reqs = requirementsFor(body);
  const upstream = anthropicToOpenAI(body, "auto");
  const dialed = await dialWithFailover(upstream, reqs);
  if (dialed.response === null || dialed.model === null) {
    if (dialed.failures.length === 0) {
      return sendError(res, "anthropic", 503, "overloaded_error", "no free model in the brief meets this request");
    }
    return allRefused(res, "anthropic", dialed.failures);
  }
  if (dialed.response.body === null) {
    return allRefused(res, "anthropic", dialed.failures);
  }
  const headers = servedHeaders(dialed.failures);
  if (body.stream === true) {
    res.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
      ...headers,
    });
    try {
      for await (const event of anthropicEventsFromOpenAI(dialed.response.body, dialed.model)) {
        res.write(event);
      }
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await dialed.response.json().catch(() => null);
  if (!isRecord(answer)) {
    return allRefused(res, "anthropic", [
      ...dialed.failures,
      { status: 502, code: "upstream_unreachable", model: dialed.model },
    ]);
  }
  sendJson(res, 200, openAIToAnthropic(answer, dialed.model), headers);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? "/", "http://entry");
  if (req.method === "GET" && url.pathname === "/v1/models") {
    models(req, res);
    return;
  }
  if (req.method === "POST" && url.pathname === "/v1/chat/completions") {
    await chat(req, res);
    return;
  }
  if (req.method === "POST" && url.pathname === "/v1/messages") {
    await messages(req, res);
    return;
  }
  sendError(
    res,
    url.pathname === "/v1/messages" ? "anthropic" : "openai",
    404,
    "not_found",
    `no route ${req.method ?? ""} ${url.pathname}`,
  );
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => resolve());
  });
}

/** POST /register with the port and the one catalog. */
async function register(port: number, report: Catalog): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) {
    throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  }
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === "number") {
    log(`registered: the trial holds ${reply.requests} prompts`);
  }
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes("--list")) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === "") throw new Error("MAKEFUN_ENV_URL is not set");
  if (chatCompletionsUrl() === "") throw new Error("OPENAI_BASE_URL is not set");
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(
          res,
          req.url?.startsWith("/v1/messages") === true ? "anthropic" : "openai",
          500,
          "internal_error",
          "the server failed to answer",
        );
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("the server has no port");
  const port = address.port;
  if (port === 8080) throw new Error("the OS picked 8080, which the env owns on one layout: restart the trial");
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on("SIGTERM", () => process.exit(0));
  process.on("SIGINT", () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}

