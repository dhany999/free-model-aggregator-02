// Wire translation between the two client APIs and the gateway's Chat
// Completions wire, plus the SSE relays. Written fresh for
// free-model-aggregator-02 from RULES.txt / prompt.txt.
// Every function is a pure function of its arguments (except the stream
// relays, which only forward bytes); no clock, no randomness, so a replay
// of the same gateway answers produces the same bytes.

import { isRecord, type Json } from "./brief.ts";

export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

// Fixed timestamps in model listings: nothing in a reply reads the clock.
const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = "2026-01-01T00:00:00Z";

export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: "list",
    data: models.map((m) => ({
      id: m.id,
      object: "model",
      created: FIXED_CREATED,
      owned_by: m.provider,
    })),
  };
}

export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({
      type: "model",
      id: m.id,
      display_name: m.display_name,
      created_at: FIXED_CREATED_AT,
    })),
    has_more: false,
  };
}

export function anthropicError(type: string, message: string): Json {
  return { type: "error", error: { type, message } };
}

export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

function textOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .flatMap((b) =>
      isRecord(b) && b.type === "text" && typeof b.text === "string" ? [b.text] : [],
    )
    .join("\n");
}

function toolResultText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== "" ? text : JSON.stringify(content);
  }
  return content === undefined ? "" : JSON.stringify(content);
}

function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: empty input.
    }
  }
  return {};
}

export function stopReasonOf(finish: unknown): string {
  if (finish === "length") return "max_tokens";
  if (finish === "tool_calls" || finish === "function_call") return "tool_use";
  if (finish === "content_filter") return "refusal";
  return "end_turn";
}

function numOrZero(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

/**
 * Anthropic Messages request -> OpenAI Chat Completions request naming `model`.
 * System prompt becomes a system message; image blocks become image_url parts;
 * tool_use blocks become tool_calls; tool_result blocks become tool messages;
 * tools / tool_choice are rewritten; max_tokens, temperature, top_p, stop
 * sequences and streaming pass through. The gateway requires an output cap,
 * so a missing max_tokens gets a safe default (never above the 8192 cap).
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== "") messages.push({ role: "system", content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === "assistant";
    const content = raw.content;
    if (typeof content === "string") {
      messages.push({ role: assistant ? "assistant" : "user", content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== "tool_use") continue;
        toolCalls.push({
          id: typeof block.id === "string" ? block.id : "",
          type: "function",
          function: {
            name: typeof block.name === "string" ? block.name : "",
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: "assistant",
        content: text !== "" ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === "tool_result") {
        messages.push({
          role: "tool",
          tool_call_id: typeof block.tool_use_id === "string" ? block.tool_use_id : "",
          content: toolResultText(block.content),
        });
        continue;
      }
      if (block.type === "text" && typeof block.text === "string") {
        parts.push({ type: "text", text: block.text });
      } else if (block.type === "image" && isRecord(block.source)) {
        const src = block.source;
        if (src.type === "base64" && typeof src.data === "string") {
          const media = typeof src.media_type === "string" ? src.media_type : "image/png";
          parts.push({ type: "image_url", image_url: { url: `data:${media};base64,${src.data}` } });
        } else if (src.type === "url" && typeof src.url === "string") {
          parts.push({ type: "image_url", image_url: { url: src.url } });
        }
      }
    }
    if (parts.length > 0) messages.push({ role: "user", content: parts });
  }

  const out: Json = { model, messages };
  out.max_tokens =
    typeof body.max_tokens === "number" && body.max_tokens > 0
      ? Math.min(body.max_tokens, 8192)
      : 1024;
  if (typeof body.temperature === "number") out.temperature = body.temperature;
  if (typeof body.top_p === "number") out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) {
    out.stop = body.stop_sequences;
  }
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== "string") return [];
      return [
        {
          type: "function",
          function: {
            name: raw.name,
            description: typeof raw.description === "string" ? raw.description : "",
            parameters: isRecord(raw.input_schema)
              ? raw.input_schema
              : { type: "object", properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === "auto") out.tool_choice = "auto";
    else if (choice.type === "any") out.tool_choice = "required";
    else if (choice.type === "tool" && typeof choice.name === "string") {
      out.tool_choice = { type: "function", function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** OpenAI Chat Completions body for a client body: model replaced, cap enforced. */
export function openAIUpstreamBody(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  const cap =
    typeof body.max_tokens === "number"
      ? body.max_tokens
      : typeof body.max_completion_tokens === "number"
        ? body.max_completion_tokens
        : 0;
  if (cap > 0) out.max_tokens = Math.min(cap, 8192);
  else out.max_tokens = 1024;
  if (body.stream === true) out.stream_options = { include_usage: true };
  return out;
}

/** Usage with whole counts; output is at least 1 when the answer carries text. */
function wholeUsage(usage: unknown, hasText: boolean): Json {
  const u = isRecord(usage) ? usage : {};
  let input = numOrZero(u.prompt_tokens);
  let output = numOrZero(u.completion_tokens);
  if (output === 0 && hasText) output = 1;
  return { prompt_tokens: input, completion_tokens: output, total_tokens: input + output };
}

/** Chat Completions answer -> Anthropic Messages object, `model` = id dialled. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === "string" ? message.content : textOf(message.content);
  if (text !== "") content.push({ type: "text", text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: "tool_use",
      id: typeof call.id === "string" ? call.id : "toolu_unknown",
      name: typeof call.function.name === "string" ? call.function.name : "",
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: "text", text: "" });
  const usage = wholeUsage(body.usage, text !== "" || calls.length > 0);
  return {
    id: `msg_${typeof body.id === "string" ? body.id : "aggregate"}`,
    type: "message",
    role: "assistant",
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: usage.prompt_tokens,
      output_tokens: usage.completion_tokens,
    },
  };
}

/** Chat Completions answer with `model` set to the id dialled, usage kept whole. */
export function openAIJsonWithModel(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === "string" ? message.content : textOf(message.content);
  const hasText = text !== "" || (Array.isArray(message.tool_calls) && message.tool_calls.length > 0);
  return { ...body, model, usage: wholeUsage(body.usage, hasText) };
}

export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? "" : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Rewrite every JSON data payload's `model` in one SSE frame to `model`. */
function frameWithModel(frame: string, model: string): string {
  const lines = frame.split("\n").map((line) => {
    const trimmed = line.trim();
    if (!trimmed.startsWith("data:")) return line;
    const payload = trimmed.slice(5).trim();
    if (payload === "" || payload === "[DONE]") return line;
    try {
      const chunk: unknown = JSON.parse(payload);
      if (isRecord(chunk)) return `data: ${JSON.stringify({ ...chunk, model })}`;
    } catch {
      // Not JSON: relay as it came.
    }
    return line;
  });
  return `${lines.join("\n")}\n\n`;
}

/** Split an SSE byte stream into frames on blank lines. */
async function* framesOf(upstream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let at = buffer.indexOf("\n\n");
      while (at !== -1) {
        yield buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf("\n\n");
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== "") yield buffer;
  } finally {
    reader.releaseLock();
  }
}

/** Chat Completions SSE relayed as-is, every chunk's `model` = id dialled. */
export async function* openAIStreamWithModel(
  upstream: ReadableStream<Uint8Array>,
  model: string,
): AsyncGenerator<string> {
  for await (const frame of framesOf(upstream)) yield frameWithModel(frame, model);
}

/**
 * Chat Completions SSE -> Anthropic event sequence: message_start on the first
 * chunk, content_block_start/delta/stop per text block or tool call, then
 * message_delta with the stop reason and usage, then message_stop.
 */
export async function* anthropicEventsFromOpenAI(
  upstream: ReadableStream<Uint8Array>,
  model: string,
): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent("message_start", {
        type: "message_start",
        message: {
          id: "msg_stream",
          type: "message",
          role: "assistant",
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent("content_block_stop", { type: "content_block_stop", index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) {
      out.push(sseEvent("content_block_stop", { type: "content_block_stop", index }));
    }
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent("message_delta", {
        type: "message_delta",
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { output_tokens: outputTokens > 0 ? outputTokens : 1 },
      }),
      sseEvent("message_stop", { type: "message_stop" }),
    ];
  };

  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === "[DONE]") {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? "end_turn"));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numOrZero(chunk.usage.prompt_tokens);
        outputTokens = numOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === "string" && delta.content !== "") {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(
            sseEvent("content_block_start", {
              type: "content_block_start",
              index: textIndex,
              content_block: { type: "text", text: "" },
            }),
          );
        }
        out.push(
          sseEvent("content_block_delta", {
            type: "content_block_delta",
            index: textIndex,
            delta: { type: "text_delta", text: delta.content },
          }),
        );
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === "number" ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent("content_block_stop", { type: "content_block_stop", index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent("content_block_start", {
              type: "content_block_start",
              index,
              content_block: {
                type: "tool_use",
                id: typeof call.id === "string" && call.id !== "" ? call.id : `toolu_${index}`,
                name: typeof fn.name === "string" ? fn.name : "",
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === "string" && fn.arguments !== "") {
          out.push(
            sseEvent("content_block_delta", {
              type: "content_block_delta",
              index,
              delta: { type: "input_json_delta", partial_json: fn.arguments },
            }),
          );
        }
      }
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
        pendingStop = stopReasonOf(choice.finish_reason);
      }
    }
    return out;
  };

  for await (const frame of framesOf(upstream)) {
    for (const event of eventsOfFrame(frame)) yield event;
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? "end_turn")) yield event;
  }
}
