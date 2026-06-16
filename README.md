# thg-brain-mcp

Custom MCP server giving Claude semantic **and** structured access to The Hoop
Gang's business data stored in a Turbopuffer namespace (`thg_business_records`):
Shopify (orders, products, variants, inventory, refunds), Klaviyo (email/SMS
campaigns, flows, profiles), and Gmail.

## Tools (v2)

| Tool | What it does |
|------|--------------|
| `search_business_records` | Semantic vector search. Embeds the query and matches by meaning. Now also accepts optional `record_type` / `source` / `since` / `until` / `equals` filters. |
| `list_records` | Exact, filter-only query (no embedding), sorted by an attribute. Use when you know exactly what you want, e.g. "all `shopify_refund` since 2026-04-01". Pass `source:"gmail"` for an email-only view. |
| `get_records_by_id` | Fetch specific records by exact id, e.g. `shopify:refund:994175385790`. |
| `aggregate_records` | Server-side `Count` and optional `Sum` of a numeric field (`total_refunded`, `price`, `quantity`, ...), with optional `group_by` (`record_type`, `source`, `channel`, `item_title`, ...). Grouped results come back as a **leaderboard ranked by the Sum** (or Count). By default it **case-merges** duplicate group keys (e.g. "Black and Cream" + "BLACK AND CREAM"); pass `normalize_keys:false` to keep them separate. |
| `upsert_records` | Embed `content` and write new/updated records into the namespace so they're immediately searchable. |
| `describe_namespace` | List stored attributes with their types and whether they're filterable. Run this first to discover field names. |

## Environment variables

| Var | Required | Default |
|-----|----------|---------|
| `OPENAI_API_KEY` | yes | — |
| `OPENAI_EMBEDDING_MODEL` | no | `text-embedding-3-small` |
| `TURBOPUFFER_API_KEY` | yes | — |
| `TURBOPUFFER_REGION` *or* `TURBOPUFFER_BASE_URL` | one of them | — |
| `TURBOPUFFER_NAMESPACE` | no | `thg_business_records` |

The embedding model **must** match the model the records were stored with
(`text-embedding-3-small`, 1536-dim), or semantic search quality degrades.

## Run

```bash
npm install
node server.mjs
```

It speaks MCP over stdio, so it's launched by the host (Claude), not run
interactively. All logging goes to **stderr** — stdout is reserved for the
JSON-RPC protocol.

## Notes / limits

- **Filters** use Turbopuffer's grammar: `[field, "Eq", value]`, combined with
  `["And", [...]]`. The `equals` arg lets Claude pass arbitrary attribute
  matches; arrays become `IN` filters.
- **Date filters** operate on `source_created_at` as ISO-8601 string
  comparisons (which sort chronologically).
- **Sum** only works on genuinely numeric attributes; non-numeric fields error
  (the tool returns a helpful message).
- Data is a **synced snapshot**, not live — freshness depends on the external
  sync pipeline that populates the namespace (that pipeline lives outside this
  repo).
