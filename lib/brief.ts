// Brief parsing, catalog, and rating-ordered model selection.
// Written fresh for free-model-aggregator-02 from RULES.txt / prompt.txt.
// The brief is prose with two JSON documents inside it; each document is found
// by its `schema` member, never by guessing at the prose around it.

import { readFileSync } from "node:fs";

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json =>
  v !== null && typeof v === "object" && !Array.isArray(v);

export interface ModelRow {
  id: string;
  provider: string;
  display_name: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface PromptRow {
  id: string;
  api: "openai" | "anthropic";
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

export interface Catalog {
  providers: Array<{
    id: string;
    base_url: string;
    key_env: string;
    wire: string;
    models: Array<{ id: string; free: boolean; context_length: number | null }>;
  }>;
}

const SCHEMA_PREFIX = "free-model-aggregator-02/";

/** Index of the `}` closing the object opened at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === "\\") i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` carrying a string `schema`, in order of appearance. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf("{"); start !== -1; start = text.indexOf("{", start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === "string") out.push(value);
    } catch {
      // Not a standalone JSON object: keep scanning.
    }
  }
  return out;
}

function str(v: unknown): string {
  return typeof v === "string" ? v : "";
}
function numOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}
function boolOrNull(v: unknown): boolean | null {
  return typeof v === "boolean" ? v : null;
}

function modelOf(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const raw = str(v.id) !== "" ? str(v.id) : str(v.model);
  if (raw === "") return null;
  const named = str(v.provider) !== "" ? str(v.provider) : str(v.wire);
  const id =
    named === "openrouter" && !raw.toLowerCase().startsWith("openrouter/")
      ? `openrouter/${raw}`
      : raw;
  return {
    id,
    provider: named === "" ? "openrouter" : named,
    display_name: str(v.display_name) !== "" ? str(v.display_name) : raw,
    rating_milli: numOrNull(v.rating_milli) ?? 0,
    context_length: numOrNull(v.context_length),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
    free: v.free !== false,
  };
}

function promptOf(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = str(v.id);
  if (id === "") return null;
  const req = isRecord(v.requires) ? v.requires : {};
  const requires: Requirement = {};
  if (req.tools === true) requires.tools = true;
  if (req.vision === true) requires.vision = true;
  if (req.json === true) requires.json = true;
  if (typeof req.min_context === "number" && Number.isFinite(req.min_context)) {
    requires.min_context = req.min_context;
  }
  return {
    id,
    api: v.api === "anthropic" ? "anthropic" : "openai",
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires,
  };
}

/** Read the brief at `path`; empty models/prompts plus `error` when unreadable. */
export function readBrief(path: string): Brief {
  if (path === "") return { models: [], prompts: [], error: "MAKEFUN_PROMPT_FILE is not set" };
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  let models: ModelRow[] = [];
  let prompts: PromptRow[] = [];
  for (const doc of docs) {
    if (!str(doc.schema).includes(SCHEMA_PREFIX)) continue;
    for (const key of ["models", "ratings"]) {
      if (Array.isArray(doc[key]) && models.length === 0) {
        models = (doc[key] as unknown[]).flatMap((v) => {
          const m = modelOf(v);
          return m === null ? [] : [m];
        });
      }
    }
    for (const key of ["prompts", "manifest"]) {
      if (Array.isArray(doc[key]) && prompts.length === 0) {
        prompts = (doc[key] as unknown[]).flatMap((v) => {
          const p = promptOf(v);
          return p === null ? [] : [p];
        });
      }
    }
  }
  if (models.length === 0) {
    return { models: [], prompts, error: `the brief at ${path} holds no free-model list` };
  }
  return { models, prompts, error: null };
}

/** The free models in the brief, in the brief's order. */
export function freeModels(brief: Brief): ModelRow[] {
  return brief.models.filter((m) => m.free);
}

/** Whether a model meets a prompt's requirements (unknown capability never meets one). */
export function meets(model: ModelRow, req: Requirement): boolean {
  if (req.tools === true && model.tools !== true) return false;
  if (req.vision === true && model.vision !== true) return false;
  if (req.json === true && model.json !== true) return false;
  if (
    req.min_context !== undefined &&
    (model.context_length === null || model.context_length < req.min_context)
  ) {
    return false;
  }
  return true;
}

/**
 * Eligible models for a prompt, best rating first (ties: brief order, which is
 * deterministic). Quality is 60% of the score, so the best eligible model is
 * always tried first; failover walks down this same list.
 */
export function rankedEligible(brief: Brief, req: Requirement): ModelRow[] {
  return freeModels(brief)
    .map((m, i) => ({ m, i }))
    .filter(({ m }) => meets(m, req))
    .sort((a, b) => b.m.rating_milli - a.m.rating_milli || a.i - b.i)
    .map(({ m }) => m);
}

/** The catalog document: exactly the brief's free models, once each. */
export function buildCatalog(brief: Brief): Catalog {
  const seen = new Set<string>();
  const models: Catalog["providers"][number]["models"] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.context_length });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: "openrouter",
        base_url: "https://openrouter.ai/api/v1",
        key_env: "OPENROUTER_API_KEY",
        wire: "openai",
        models,
      },
    ],
  };
}

/** Requirements read off a request body, used only when the brief has no row for it. */
export function requirementsFromBody(body: Json): Requirement {
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === "image" || block.type === "image_url")) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0 ? true : undefined,
    vision: vision ? true : undefined,
    json: format === "json_object" || format === "json_schema" ? true : undefined,
  };
}
