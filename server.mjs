#!/usr/bin/env node
// ============================================================================
// THE_BRAIN — custom MCP server
//
// Exposes one tool, `search_business_records`, to Samir's Claude. On each call
// it embeds the natural-language question with text-embedding-3-small (the SAME
// model the records were stored with), then runs a semantic (vector) search
// against the `thg_business_records` Turbopuffer namespace and returns the
// most relevant records.
//
// This is what lets Samir ask plain-English questions and get real answers —
// the off-the-shelf Turbopuffer MCP does NOT embed the query for you.
//
// IMPORTANT: stdio MCP servers must never write to stdout (it carries the
// JSON-RPC protocol). All logging goes to stderr via console.error.
// ============================================================================
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import OpenAI from "openai";
import Turbopuffer from "@turbopuffer/turbopuffer";

const {
  OPENAI_API_KEY,
  OPENAI_EMBEDDING_MODEL = "text-embedding-3-small",
  TURBOPUFFER_API_KEY,
  TURBOPUFFER_REGION,
  TURBOPUFFER_BASE_URL,
  TURBOPUFFER_NAMESPACE = "thg_business_records",
} = process.env;

if (!OPENAI_API_KEY) {
  console.error("[thg-brain] FATAL: OPENAI_API_KEY is not set");
  process.exit(1);
}
if (!TURBOPUFFER_API_KEY) {
  console.error("[thg-brain] FATAL: TURBOPUFFER_API_KEY is not set");
  process.exit(1);
}

// The Turbopuffer client needs EXACTLY ONE of: region (e.g. "aws-us-east-1")
// OR a fully-resolved baseURL (e.g. "https://aws-us-east-1.turbopuffer.com").
// Passing both — or neither — throws. Prefer region; fall back to baseURL.
const tpufLocation = TURBOPUFFER_REGION
  ? { region: TURBOPUFFER_REGION }
  : TURBOPUFFER_BASE_URL
    ? { baseURL: TURBOPUFFER_BASE_URL }
    : null;

if (!tpufLocation) {
  console.error(
    "[thg-brain] FATAL: set TURBOPUFFER_REGION (e.g. aws-us-east-1) " +
      "or TURBOPUFFER_BASE_URL (e.g. https://aws-us-east-1.turbopuffer.com)",
  );
  process.exit(1);
}

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
const tpuf = new Turbopuffer({
  apiKey: TURBOPUFFER_API_KEY,
  ...tpufLocation,
  logLevel: "error",
});
const ns = tpuf.namespace(TURBOPUFFER_NAMESPACE);

async function embedQuery(text) {
  const res = await openai.embeddings.create({
    model: OPENAI_EMBEDDING_MODEL,
    input: text,
  });
  return res.data[0].embedding;
}

const server = new McpServer({ name: "thg-brain", version: "1.0.0" });

server.registerTool(
  "search_business_records",
  {
    title: "Search THG business records",
    description:
      "Semantic search across The Hoop Gang's business data — Shopify (orders, products, variants, inventory, refunds), Gmail, Klaviyo (email/SMS campaigns, flows, profiles), and Meta Ads. Use this for ANY question about the business: sales performance, what's selling, customer issues, marketing results, ad spend, etc. The question is embedded and matched by meaning, so plain-language queries work well.",
    inputSchema: {
      query: z
        .string()
        .describe(
          "Natural-language search query, e.g. 'best selling products last month', 'customer emails about late orders', 'which Klaviyo campaign drove the most revenue'",
        ),
      limit: z
        .number()
        .int()
        .min(1)
        .max(25)
        .optional()
        .describe("Max records to return (default 8)"),
    },
    annotations: { readOnlyHint: true, openWorldHint: true },
  },
  async ({ query, limit }) => {
    try {
      const vector = await embedQuery(query);
      const result = await ns.query({
        rank_by: ["vector", "ANN", vector],
        top_k: limit ?? 8,
        include_attributes: true,
      });

      const rows = result.rows ?? [];
      if (rows.length === 0) {
        return {
          content: [{ type: "text", text: `No records found for: "${query}"` }],
        };
      }

      const text = rows
        .map((row, i) => {
          // Attributes are spread at the TOP LEVEL of each row, and the
          // distance is `$dist`. Drop `vector` — it's the full 1536-dim
          // embedding and must never reach the model's context.
          const { $dist, id, vector, content, record_type, ...rest } = row;
          const meta = Object.entries(rest)
            .filter(([, v]) => v != null && v !== "")
            .map(
              ([k, v]) =>
                `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`,
            )
            .join(" · ");
          const dist =
            typeof $dist === "number" ? ` · dist ${$dist.toFixed(3)}` : "";
          const body =
            content != null
              ? String(content).slice(0, 1500)
              : "(no content field)";
          return `#${i + 1} [${record_type ?? "record"}] ${id ?? ""}${dist}\n${body}${
            meta ? `\n— ${meta}` : ""
          }`;
        })
        .join("\n\n———\n\n");

      return { content: [{ type: "text", text }] };
    } catch (err) {
      const msg = err?.message ?? String(err);
      console.error("[thg-brain] search error:", msg);
      return {
        isError: true,
        content: [
          {
            type: "text",
            text:
              `Search failed: ${msg}\n` +
              `Check OPENAI_API_KEY, TURBOPUFFER_API_KEY, ` +
              `TURBOPUFFER_REGION/TURBOPUFFER_BASE_URL ` +
              `(${TURBOPUFFER_REGION ?? TURBOPUFFER_BASE_URL ?? "unset"}), ` +
              `and namespace "${TURBOPUFFER_NAMESPACE}".`,
          },
        ],
      };
    }
  },
);

const transport = new StdioServerTransport();
await server.connect(transport);
console.error("[thg-brain] MCP server connected over stdio");
