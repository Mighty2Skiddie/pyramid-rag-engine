const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface IngestResponse {
  session_id: string;
  chunk_count: number;
  chunks: ChunkSummary[];
}

export interface ChunkSummary {
  chunk_id: string;
  raw_text: string;
  summary: string;
  category: string;
  category_confidence: number;
  keywords: string[];
}

export interface QueryResponse {
  query: string;
  results: QueryResultItem[];
  result_count: number;
}

export interface QueryResultItem {
  chunk_id: string;
  score: number;
  best_level: string;
  level_scores: Record<string, number>;
  raw_text: string;
  summary: string;
  category: string;
  keywords: string[];
}

export interface SolveResponse {
  query: string;
  query_type: string;
  handler_name: string;
  answer: string;
  confidence: number;
  reasoning_trace: string[];
  metadata: Record<string, unknown>;
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "API error");
  }
  return res.json();
}

export const api = {
  ingestText: (text: string) =>
    apiFetch<IngestResponse>("/api/pyramid/ingest", {
      method: "POST",
      body: JSON.stringify({ text }),
    }),

  queryPyramid: (session_id: string, query: string) =>
    apiFetch<QueryResponse>("/api/pyramid/query", {
      method: "POST",
      body: JSON.stringify({ session_id, query }),
    }),

  exploreLevel: (session_id: string, level: number) =>
    apiFetch<{ chunks: unknown[] }>(`/api/pyramid/explore/${session_id}/${level}`),

  solveReasoning: (query: string) =>
    apiFetch<SolveResponse>("/api/reasoning/solve", {
      method: "POST",
      body: JSON.stringify({ query }),
    }),

  health: () => apiFetch<{ status: string }>("/api/health"),
};
