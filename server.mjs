#!/usr/bin/env node
// ============================================================================
// THE_BRAIN — custom MCP server (v2)
//
// Semantic + structured access to The Hoop Gang's business data stored in the
// Turbopuffer namespace `thg_business_records` (Shopify, Klaviyo, Gmail, ...).
//
// Tools
//   1. search_business_records — semantic vector search (+ optional filters)
//   2. list_records            — exact filter-only query, sorted, no embedding
//   3. get_records_by_id       — fetch specific record(s) by id
//   4. aggregate_records       — server-side Count / Sum, optional group_by
//   5. upsert_records          — embed + write new/updated records (ingest)
//   6. describe_namespace      — inspect the stored schema (fields + types)
//
// Every record carries: id, content, record_type, source, source_created_at,
// synced_at, plus type-specific attributes (e.g. total_refunded, price,
// inventory_quantity, channel, status, from, subject, ...).
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
// OR a fully-resolved baseURL. Passing both — or neither — throws.
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

// Canonical per-record event timestamp present on every source. Stored as an
// ISO-8601 string, so lexicographic Gte/Lte comparisons sort chronologically.
const DATE_FIELD = "source_created_at";

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------
async function embedOne(text) {
  const res = await openai.embeddings.create({
    model: OPENAI_EMBEDDING_MODEL,
    input: text,
  });
  return res.data[0].embedding;
}

async function embedMany(texts) {
  const res = await openai.embeddings.create({
    model: OPENAI_EMBEDDING_MODEL,
    input: texts,
  });
  // OpenAI preserves input order in res.data.
  return res.data.map((d) => d.embedding);
}

// Turbopuffer filter grammar: [field, Op, value]; combine with ["And", [...]].
// Ops include Eq, NotEq, In, Gt, Gte, Lt, Lte, Contains, Glob, etc.
function buildFilters({ record_type, source, since, until, equals } = {}) {
  const clauses = [];

  if (record_type != null) {
    clauses.push(
      Array.isArray(record_type)
        ? ["record_type", "In", record_type]
        : ["record_type", "Eq", record_type],
    );
  }
  if (source != null) {
    clauses.push(
      Array.isArray(source) ? ["source", "In", source] : ["source", "Eq", source],
    );
  }
  if (since != null) clauses.push([DATE_FIELD, "Gte", since]);
  if (until != null) clauses.push([DATE_FIELD, "Lte", until]);

  if (equals && typeof equals === "object") {
    for (const [k, v] of Object.entries(equals)) {
      if (v == null) continue;
      clauses.push(Array.isArray(v) ? [k, "In", v] : [k, "Eq", v]);
    }
  }

  if (clauses.length === 0) return undefined;
  if (clauses.length === 1) return clauses[0];
  return ["And", clauses];
}

function formatRows(rows, query) {
  if (!rows || rows.length === 0) {
    return query ? `No records found for: "${query}".` : "No records found.";
  }
  const blocks = rows.map((row, i) => {
    // Attributes are spread at the TOP LEVEL of each row; distance is `$dist`.
    // Drop `vector` — it's the full 1536-dim embedding and must never reach
    // the model's context.
    const { $dist, id, vector, content, record_type, ...rest } = row;
    const meta = Object.entries(rest)
      .filter(([, v]) => v != null && v !== "")
      .map(([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`)
      .join(" · ");
    const dist = typeof $dist === "number" ? ` · dist ${$dist.toFixed(3)}` : "";
    const body =
      content != null ? String(content).slice(0, 1500) : "(no content field)";
    const head = `#${i + 1} [${record_type ?? "record"}] ${id ?? ""}${dist}`;
    return meta ? `${head}\n${body}\n— ${meta}` : `${head}\n${body}`;
  });
  return blocks.join("\n\n———\n\n");
}

function ok(text) {
  return { content: [{ type: "text", text }] };
}

function fail(err, hint) {
  const msg = err?.message ?? String(err);
  const loc = TURBOPUFFER_REGION ?? TURBOPUFFER_BASE_URL ?? "unset";
  console.error("[thg-brain] error:", msg);
  let text = `Request failed: ${msg}`;
  if (hint) text += `\n${hint}`;
  text += `\n(namespace "${TURBOPUFFER_NAMESPACE}", region/url ${loc})`;
  return { isError: true, content: [{ type: "text", text }] };
}

// Reusable zod fields for the structured filters shared by several tools.
const filterShape = {
  record_type: z
    .union([z.string(), z.array(z.string())])
    .optional()
    .describe(
      "Restrict to one or more record types, e.g. 'shopify_refund', " +
        "'shopify_variant', 'klaviyo_campaign', 'gmail_message'.",
    ),
  source: z
    .union([z.string(), z.array(z.string())])
    .optional()
    .describe("Restrict to data source(s): 'shopify', 'klaviyo', 'gmail'."),
  since: z
    .string()
    .optional()
    .describe(
      "Inclusive lower bound on source_created_at (ISO-8601), e.g. '2026-04-01'.",
    ),
  until: z
    .string()
    .optional()
    .describe("Inclusive upper bound on source_created_at (ISO-8601)."),
  equals: z
    .record(z.any())
    .optional()
    .describe(
      "Extra exact-match attribute filters as an object, e.g. " +
        '{"channel":"sms","status":"Sent"}. Arrays become IN filters.',
    ),
};

const server = new McpServer({ name: "thg-brain", version: "2.0.0" });

// ----------------------------------------------------------------------------
// 1. Semantic search (+ optional structured filters)
// ----------------------------------------------------------------------------
server.registerTool(
  "search_business_records",
  {
    title: "Search THG business records (semantic)",
    description:
      "Semantic vector search across The Hoop Gang's business data — Shopify " +
      "(orders, products, variants, inventory, refunds), Gmail, and Klaviyo " +
      "(email/SMS campaigns, flows, profiles). The query is embedded and " +
      "matched by meaning. Optionally narrow with record_type / source / date " +
      "filters. Use this for fuzzy, plain-language questions.",
    inputSchema: {
      query: z
        .string()
        .describe(
          "Natural-language search, e.g. 'customer emails about late orders'.",
        ),
      limit: z
        .number()
        .int()
        .min(1)
        .max(50)
        .optional()
        .describe("Max records (default 8)"),
      ...filterShape,
    },
    annotations: { readOnlyHint: true, openWorldHint: true },
  },
  async ({ query, limit, ...filterArgs }) => {
    try {
      const vector = await embedOne(query);
      const params = {
        rank_by: ["vector", "ANN", vector],
        top_k: limit ?? 8,
        include_attributes: true,
      };
      const filters = buildFilters(filterArgs);
      if (filters) params.filters = filters;
      const result = await ns.query(params);
      return ok(formatRows(result.rows, query));
    } catch (err) {
      return fail(err, "Check OPENAI_API_KEY / TURBOPUFFER credentials.");
    }
  },
);

// ----------------------------------------------------------------------------
// 2. List records — exact filter-only query, no embedding, sorted
// ----------------------------------------------------------------------------
server.registerTool(
  "list_records",
  {
    title: "List THG records (exact filter)",
    description:
      "Return records matching exact filters (record_type / source / date " +
      "range / attribute equality), sorted by an attribute. No semantic " +
      "matching — use when you know precisely what you want, e.g. 'all " +
      "shopify_refund records since 2026-04-01' or 'klaviyo_campaign where " +
      "channel = sms'. Pass source='gmail' to act like an email-only search.",
    inputSchema: {
      ...filterShape,
      order_by: z
        .string()
        .optional()
        .describe("Attribute to sort by (default source_created_at)."),
      direction: z
        .enum(["asc", "desc"])
        .optional()
        .describe("Sort direction (default desc / newest first)."),
      limit: z
        .number()
        .int()
        .min(1)
        .max(50)
        .optional()
        .describe("Max records (default 20)"),
    },
    annotations: { readOnlyHint: true, openWorldHint: true },
  },
  async ({ order_by, direction, limit, ...filterArgs }) => {
    const params = { top_k: limit ?? 20, include_attributes: true };
    const filters = buildFilters(filterArgs);
    if (filters) params.filters = filters;
    const sortField = order_by ?? DATE_FIELD;
    const sortDir = direction ?? "desc";
    try {
      const result = await ns.query({ ...params, rank_by: [sortField, sortDir] });
      return ok(formatRows(result.rows));
    } catch (err) {
      // Some attributes aren't sortable; fall back to an unranked filter query.
      try {
        const result = await ns.query(params);
        return ok(formatRows(result.rows));
      } catch (err2) {
        return fail(err2, `Could not sort by '${sortField}'.`);
      }
    }
  },
);

// ----------------------------------------------------------------------------
// 3. Get records by id — exact lookup
// ----------------------------------------------------------------------------
server.registerTool(
  "get_records_by_id",
  {
    title: "Get THG records by id",
    description:
      "Fetch one or more specific records by their exact id, e.g. " +
      "'shopify:refund:994175385790' or 'gmail:message:19e1d17ab27458ec'. " +
      "Use after a search/list to pull full records for known ids.",
    inputSchema: {
      ids: z
        .array(z.string())
        .min(1)
        .max(50)
        .describe("Exact record ids to fetch."),
    },
    annotations: { readOnlyHint: true, openWorldHint: true },
  },
  async ({ ids }) => {
    try {
      const result = await ns.query({
        filters: ["id", "In", ids],
        rank_by: ["id", "asc"],
        top_k: ids.length,
        include_attributes: true,
      });
      return ok(formatRows(result.rows));
    } catch (err) {
      return fail(err);
    }
  },
);

// ----------------------------------------------------------------------------
// 4. Aggregate — server-side Count / Sum, optional group_by
// ----------------------------------------------------------------------------
server.registerTool(
  "aggregate_records",
  {
    title: "Aggregate THG records (count / sum)",
    description:
      "Compute server-side aggregates over matching records: a Count, and an " +
      "optional Sum of a numeric attribute (e.g. sum_field='total_refunded' " +
      "or 'price'). Optionally group_by an attribute (e.g. 'record_type', " +
      "'source', 'channel', 'item_title') for a breakdown. When grouped, " +
      "results are returned as a leaderboard ranked descending by the Sum " +
      "(or Count if no sum_field). Use for 'best sellers by units' " +
      "(record_type='shopify_order_item', sum_field='quantity', " +
      "group_by='item_title'), 'total refunded in May', 'campaigns by " +
      "channel'. Sum only works on numeric attributes.",
    inputSchema: {
      sum_field: z
        .string()
        .optional()
        .describe("Numeric attribute to Sum, e.g. 'total_refunded', 'price', 'quantity'."),
      group_by: z
        .string()
        .optional()
        .describe("Attribute to group results by, e.g. 'record_type', 'channel', 'item_title'."),
      max_groups: z
        .number()
        .int()
        .min(1)
        .max(1000)
        .optional()
        .describe("Max groups to fetch when group_by is set (default 100)."),
      normalize_keys: z
        .boolean()
        .optional()
        .describe(
          "Merge groups whose key matches case-insensitively (default true), " +
            "so 'Black and Cream' and 'BLACK AND CREAM' count as one. Set " +
            "false to keep exact-case groups separate.",
        ),
      ...filterShape,
    },
    annotations: { readOnlyHint: true, openWorldHint: true },
  },
  async ({ sum_field, group_by, max_groups, normalize_keys, ...filterArgs }) => {
    try {
      const metricKey = sum_field ? `sum_${sum_field}` : "count";
      const aggregate_by = { count: ["Count"] };
      if (sum_field) aggregate_by[metricKey] = ["Sum", sum_field];

      // NOTE: aggregate queries must NOT send include_attributes, and group_by
      // must be a SEQUENCE of attribute names, not a bare string.
      const params = { aggregate_by, top_k: group_by ? max_groups ?? 100 : 1 };
      const filters = buildFilters(filterArgs);
      if (filters) params.filters = filters;
      if (group_by) params.group_by = Array.isArray(group_by) ? group_by : [group_by];

      const result = await ns.query(params);

      // ---- Ungrouped: single aggregate row ----
      if (!group_by) {
        const agg = result.aggregations ?? {};
        const text = Object.entries(agg)
          .map(([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`)
          .join(" · ");
        return ok(text || "No matching records.");
      }

      // ---- Grouped: normalize → merge → rank into a leaderboard ----
      let rows = (result.aggregation_groups ?? []).map((g) => {
        const label =
          g[group_by] ?? Object.values(g).find((v) => typeof v === "string");
        const count = Number(g.count ?? 0);
        const metric = sum_field ? Number(g[metricKey] ?? 0) : count;
        return { label: label == null ? "(unknown)" : String(label), count, metric };
      });

      if (rows.length === 0) return ok("No matching records.");

      const merge = normalize_keys !== false; // default ON
      if (merge) {
        const byKey = new Map();
        for (const r of rows) {
          const key = r.label.trim().toLowerCase();
          const cur = byKey.get(key);
          if (cur) {
            cur.count += r.count;
            cur.metric += r.metric;
          } else {
            byKey.set(key, { ...r }); // keep first-seen original casing as label
          }
        }
        rows = [...byKey.values()];
      }

      rows.sort((a, b) => b.metric - a.metric);

      const lines = rows.map((r, i) => {
        const extra = sum_field ? ` · count ${r.count}` : "";
        return `${i + 1}. ${r.label} — ${metricKey} ${r.metric}${extra}`;
      });
      const header =
        `Ranked by ${metricKey}, group_by '${group_by}'` +
        (merge ? ", case-merged" : "") +
        `:`;
      return ok(`${header}\n${lines.join("\n")}`);
    } catch (err) {
      return fail(
        err,
        sum_field
          ? `Sum requires '${sum_field}' to be a numeric attribute.`
          : undefined,
      );
    }
  },
);

// ----------------------------------------------------------------------------
// 5. Upsert — embed text and write new/updated records (ingest)
// ----------------------------------------------------------------------------
server.registerTool(
  "upsert_records",
  {
    title: "Upsert THG records (write)",
    description:
      "Insert or update records in the namespace. Each record's `content` is " +
      "embedded with the same model used for search, so new records become " +
      "semantically searchable immediately. Provide a stable `id` to update " +
      "in place. Use sparingly — this writes to the live brain.",
    inputSchema: {
      records: z
        .array(
          z.object({
            id: z
              .string()
              .describe("Stable unique id, e.g. 'note:2026-06-16:supplier-x'."),
            content: z
              .string()
              .describe("Text to embed + store (the searchable body)."),
            record_type: z
              .string()
              .optional()
              .describe("Logical type, e.g. 'note', 'manual_record'."),
            source: z.string().optional().describe("Origin tag, e.g. 'manual'."),
            attributes: z
              .record(z.any())
              .optional()
              .describe("Any extra structured fields to store on the row."),
          }),
        )
        .min(1)
        .max(100)
        .describe("Records to insert/update."),
    },
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
  },
  async ({ records }) => {
    try {
      const vectors = await embedMany(records.map((r) => r.content));
      const now = new Date().toISOString();
      const upsert_rows = records.map((r, i) => ({
        id: r.id,
        vector: vectors[i],
        content: r.content,
        record_type: r.record_type ?? "manual_record",
        source: r.source ?? "manual",
        source_created_at: now,
        synced_at: now,
        ...(r.attributes ?? {}),
      }));
      const res = await ns.write({ upsert_rows });
      const affected = res?.rows_affected ?? upsert_rows.length;
      return ok(
        `Upserted ${affected} record(s). ids: ${records
          .map((r) => r.id)
          .join(", ")}`,
      );
    } catch (err) {
      return fail(err, "Write failed — check API keys and row shapes.");
    }
  },
);

// ----------------------------------------------------------------------------
// 6. Describe namespace — schema introspection
// ----------------------------------------------------------------------------
server.registerTool(
  "describe_namespace",
  {
    title: "Describe THG namespace schema",
    description:
      "List the attributes stored in the namespace with their types and " +
      "whether they are filterable. Use this to discover what fields exist " +
      "before building filters or aggregations.",
    inputSchema: {},
    annotations: { readOnlyHint: true, openWorldHint: true },
  },
  async () => {
    try {
      const schema = await ns.schema();
      const lines = Object.entries(schema)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([field, cfg]) => {
          const type = cfg?.type ?? "?";
          const filterable =
            cfg?.filterable === false ? " (not filterable)" : "";
          const fts = cfg?.full_text_search ? " (full-text)" : "";
          return `${field}: ${type}${filterable}${fts}`;
        });
      return ok(
        `Namespace "${TURBOPUFFER_NAMESPACE}" — ${lines.length} attributes:\n` +
          lines.join("\n"),
      );
    } catch (err) {
      return fail(err);
    }
  },
);

const transport = new StdioServerTransport();
await server.connect(transport);
console.error("[thg-brain] MCP server v2 connected over stdio (6 tools).");
