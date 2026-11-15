# free-model-aggregator-02

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API (JSON and streaming) and answers every prompt with a real
completion from a live free OpenRouter model, dialled through the grader's
gateway.

## Run

```sh
node main.ts --list   # print the catalog document and exit 0
node main.ts          # serve: register with $MAKEFUN_ENV_URL, answer prompts,
                      # exit 0 when GET $MAKEFUN_ENV_URL/work says done
```

Node 22 standard library only; nothing is installed. The server reads the
brief at `$MAKEFUN_PROMPT_FILE` at runtime: the free-model list (with the
house's ratings and capabilities) and the six-prompt manifest are parsed
from it, never hardcoded.

## How routing works

- **Model selection:** for each prompt, eligible free models (JSON-mode
  prompts need a JSON-capable model, tools prompts a tools-capable one)
  are ranked by the brief's `rating_milli`, best first, and the best is
  dialled. In trial-0 that is Inkling 2 Small (1475) for plain/tools
  prompts and Nemotron 3.5 Super VL (1303) for JSON-mode prompts.
- **Failover:** a gateway 4xx/5xx (including 429) is recorded in the
  `x-makefun-upstream-failures` response header and the prompt is retried
  on the next-best eligible model (up to 4 attempts per prompt). A model
  that just refused is skipped for the next prompt as well (deterministic,
  count-based cooldown — no wall clock). Nothing is ever answered with
  locally invented text: every completion comes from a real gateway call.
- **Wires:** both client wires are translated to the gateway's Chat
  Completions route (`POST $OPENAI_BASE_URL/chat/completions`); the
  gateway's Messages route is never used for these ids. The `model` field
  on every answer and streamed chunk is the id actually dialled.
- **Determinism:** every choice is a function of the brief, the requests,
  and the gateway answers seen so far. No wall clock, no unseeded
  randomness.
