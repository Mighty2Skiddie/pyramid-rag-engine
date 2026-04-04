"use client";

import { useState, useRef } from "react";
import { api, type IngestResponse, type QueryResultItem } from "@/lib/api";

const SAMPLE_TEXT = `Nexagen Technologies — 2024 Annual Report (AI & Infrastructure Division)

EXECUTIVE SUMMARY
Nexagen Technologies achieved record revenue of $2.4B in FY2024, representing 34% year-over-year growth driven primarily by enterprise AI adoption. Our AI-powered solutions now serve 1,200+ enterprise clients across healthcare, finance, and legal sectors.

FINANCIAL HIGHLIGHTS
Total revenue: $2.4B (↑34% YoY). Gross margin: 68.2%. Operating income: $487M. R&D investment: $340M (14% of revenue). Cash reserves: $890M.

AI PLATFORM PERFORMANCE
Our machine learning models achieved 94.7% diagnostic accuracy in clinical AI deployments. Healthcare AI system processed 45M+ patient records with HIPAA-compliant infrastructure. Legal AI platform reviewed 2.3M contracts with 99.1% accuracy.

INFRASTRUCTURE & TECHNOLOGY
Distributed microservices architecture across 3 cloud providers. Real-time streaming with Apache Kafka and Apache Spark processing 15PB of active data. Zero-trust security framework with 247 SOC 2 Type II compliance certification.

MARKET EXPANSION
Expanded to 28 new markets. Europe and APAC now represent 41% of total revenue. Strategic partnerships with 15 Fortune 500 companies signed in Q4. Acquired DataVault Analytics for $340M.`;

const LEVEL_COLORS: Record<number, { bg: string; border: string; text: string; label: string }> = {
  1: { bg: "rgba(59,130,246,0.08)", border: "rgba(59,130,246,0.3)", text: "#60a5fa", label: "L1 — Raw Text" },
  2: { bg: "rgba(139,92,246,0.08)", border: "rgba(139,92,246,0.3)", text: "#a78bfa", label: "L2 — Summary" },
  3: { bg: "rgba(6,182,212,0.08)", border: "rgba(6,182,212,0.3)", text: "#22d3ee", label: "L3 — Category" },
  4: { bg: "rgba(16,185,129,0.08)", border: "rgba(16,185,129,0.3)", text: "#34d399", label: "L4 — Distilled" },
};

function PyramidViz({ highlighted }: { highlighted?: string }) {
  const layers = [
    { level: 4, label: "L4 — Distilled Keywords + Embeddings", width: "30%" },
    { level: 3, label: "L3 — Category Labels", width: "50%" },
    { level: 2, label: "L2 — Extractive Summaries", width: "70%" },
    { level: 1, label: "L1 — Raw Text Chunks", width: "90%" },
  ];
  return (
    <div className="flex flex-col items-center gap-2 py-4">
      {layers.map(({ level, label, width }) => {
        const c = LEVEL_COLORS[level];
        const isHL = highlighted === `L${level}`;
        return (
          <div key={level}
            className="rounded-lg px-4 py-2 text-center text-xs font-medium transition-all duration-300"
            style={{
              width,
              background: isHL ? c.border : c.bg,
              border: `1px solid ${isHL ? c.text : c.border}`,
              color: c.text,
              boxShadow: isHL ? `0 0 20px ${c.border}` : "none",
              transform: isHL ? "scale(1.03)" : "scale(1)",
            }}>
            {label}
          </div>
        );
      })}
    </div>
  );
}

function ScoreBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-3 text-xs">
      <span className="w-6 text-gray-500 shrink-0">{label}</span>
      <div className="flex-1 h-1.5 rounded-full" style={{ background: "rgba(255,255,255,0.06)" }}>
        <div className="h-full rounded-full transition-all duration-700"
          style={{ width: `${Math.min(value * 100 * 2, 100)}%`, background: color }} />
      </div>
      <span className="text-gray-400 w-10 text-right">{(value * 100).toFixed(0)}%</span>
    </div>
  );
}

export default function DocumentLabPage() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState<"ingesting" | "querying" | null>(null);
  const [session, setSession] = useState<IngestResponse | null>(null);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<QueryResultItem[]>([]);
  const [error, setError] = useState("");
  const [highlightedLevel, setHighlightedLevel] = useState<string | undefined>();
  const fileRef = useRef<HTMLInputElement>(null);

  const handleIngest = async (inputText: string) => {
    if (!inputText.trim()) return;
    setLoading("ingesting");
    setError("");
    setSession(null);
    setResults([]);
    try {
      const res = await api.ingestText(inputText);
      setSession(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to ingest document.");
    } finally {
      setLoading(null);
    }
  };

  const handleQuery = async () => {
    if (!session || !query.trim()) return;
    setLoading("querying");
    setError("");
    try {
      const res = await api.queryPyramid(session.session_id, query);
      setResults(res.results);
      if (res.results[0]) setHighlightedLevel(res.results[0].best_level);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Query failed.");
    } finally {
      setLoading(null);
    }
  };

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const content = await file.text();
    setText(content);
    await handleIngest(content);
  };

  const levelBarColors: Record<string, string> = {
    L1: "#3b82f6", L2: "#8b5cf6", L3: "#06b6d4", L4: "#10b981",
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-12">
      {/* Header */}
      <div className="mb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs mb-4"
          style={{ background: "rgba(59,130,246,0.1)", border: "1px solid rgba(59,130,246,0.25)", color: "#60a5fa" }}>
          Part 1 — Document Pipeline
        </div>
        <h1 className="text-4xl font-bold text-white mb-3">Document Intelligence Lab</h1>
        <p className="text-gray-400">Upload a document → build the Knowledge Pyramid → query it with natural language</p>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Left: Upload + Pyramid */}
        <div className="space-y-6">
          {/* Upload Card */}
          <div className="glass-card rounded-2xl p-6">
            <h2 className="text-white font-semibold mb-4 flex items-center gap-2">
              <span className="w-6 h-6 rounded-md text-xs flex items-center justify-center font-bold"
                style={{ background: "rgba(59,130,246,0.2)", color: "#60a5fa" }}>1</span>
              Upload Document
            </h2>
            <textarea
              className="w-full h-40 rounded-xl p-4 text-sm text-gray-300 resize-none outline-none focus:ring-2"
              style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", focusRingColor: "#3b82f6" }}
              placeholder="Paste document text here..."
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
            <div className="flex gap-3 mt-3">
              <button className="btn-primary text-sm flex-1" disabled={!text.trim() || !!loading}
                onClick={() => handleIngest(text)}>
                {loading === "ingesting" ? "Building Pyramid..." : "Build Pyramid →"}
              </button>
              <button className="btn-secondary text-sm" onClick={() => fileRef.current?.click()}>
                📁 Upload File
              </button>
              <button className="btn-secondary text-sm"
                onClick={() => { setText(SAMPLE_TEXT); handleIngest(SAMPLE_TEXT); }}>
                Sample
              </button>
              <input ref={fileRef} type="file" accept=".txt,.md,.pdf" hidden onChange={handleFile} />
            </div>
          </div>

          {/* Pyramid Visualization */}
          <div className="glass-card rounded-2xl p-6">
            <h2 className="text-white font-semibold mb-1 flex items-center gap-2">
              <span className="w-6 h-6 rounded-md text-xs flex items-center justify-center font-bold"
                style={{ background: "rgba(139,92,246,0.2)", color: "#a78bfa" }}>2</span>
              Knowledge Pyramid
              {session && <span className="ml-auto text-xs text-gray-500">{session.chunk_count} chunks</span>}
            </h2>
            <p className="text-xs text-gray-500 mb-4">
              {session ? "Pyramid built — query on the right to highlight the best matching layer" : "Build pyramid to see the visualization"}
            </p>
            <PyramidViz highlighted={session ? highlightedLevel : undefined} />
            {session && (
              <div className="mt-4 grid grid-cols-2 gap-3">
                {session.chunks.slice(0, 2).map((c) => (
                  <div key={c.chunk_id} className="rounded-lg p-3 text-xs"
                    style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)" }}>
                    <div className="text-gray-500 mb-1">Category</div>
                    <div className="text-cyan-400 font-medium capitalize">{c.category}</div>
                    <div className="text-gray-500 mt-2 mb-1">Top Keywords</div>
                    <div className="text-gray-300 truncate">{c.keywords.slice(0, 4).join(", ")}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right: Query + Results */}
        <div className="space-y-6">
          {/* Query */}
          <div className="glass-card rounded-2xl p-6">
            <h2 className="text-white font-semibold mb-4 flex items-center gap-2">
              <span className="w-6 h-6 rounded-md text-xs flex items-center justify-center font-bold"
                style={{ background: "rgba(6,182,212,0.2)", color: "#22d3ee" }}>3</span>
              Query the Pyramid
            </h2>
            <div className="flex gap-3">
              <input
                className="flex-1 rounded-xl px-4 py-3 text-sm text-gray-300 outline-none focus:ring-2"
                style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)" }}
                placeholder={session ? "Ask anything about the document..." : "Build pyramid first"}
                value={query}
                disabled={!session}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleQuery()}
              />
              <button className="btn-primary text-sm px-5" disabled={!session || !query.trim() || !!loading}
                onClick={handleQuery}>
                {loading === "querying" ? "..." : "Ask"}
              </button>
            </div>
            {/* Suggested queries */}
            {session && (
              <div className="mt-3 flex flex-wrap gap-2">
                {["What are the revenue figures?", "Describe the AI platform", "What is the gross margin?"].map(q => (
                  <button key={q} className="text-xs px-3 py-1 rounded-full text-gray-400 hover:text-blue-400 transition-colors"
                    style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.06)" }}
                    onClick={() => { setQuery(q); }}>
                    {q}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Error */}
          {error && (
            <div className="rounded-xl p-4 text-sm text-red-400"
              style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)" }}>
              ⚠ {error}
            </div>
          )}

          {/* Results */}
          {results.length > 0 && (
            <div className="space-y-4">
              <h2 className="text-white font-semibold flex items-center gap-2">
                <span className="w-6 h-6 rounded-md text-xs flex items-center justify-center font-bold"
                  style={{ background: "rgba(16,185,129,0.2)", color: "#34d399" }}>4</span>
                Results
                <span className="ml-auto text-xs text-gray-500">{results.length} match{results.length > 1 ? "es" : ""}</span>
              </h2>
              {results.map((r, i) => {
                const c = LEVEL_COLORS[parseInt(r.best_level.replace(/\D/g, "")) || 4];
                return (
                  <div key={r.chunk_id} className={`glass-card rounded-xl p-5 ${i === 0 ? "animate-fadeInUp" : ""}`}
                    style={{ borderColor: i === 0 ? c.border : undefined }}>
                    <div className="flex items-start justify-between mb-3">
                      <span className="text-xs font-medium px-2.5 py-1 rounded-full"
                        style={{ background: c.bg, color: c.text, border: `1px solid ${c.border}` }}>
                        Best match at {r.best_level}
                      </span>
                      <span className="text-xs font-bold" style={{ color: c.text }}>
                        {(r.score * 100).toFixed(0)}%
                      </span>
                    </div>
                    <p className="text-gray-300 text-sm leading-relaxed mb-3 line-clamp-3">{r.raw_text}</p>
                    <div className="space-y-1.5 mb-3">
                      {Object.entries(r.level_scores).map(([lvl, score]) => (
                        <ScoreBar key={lvl} label={lvl} value={score} color={levelBarColors[lvl] ?? "#6b7280"} />
                      ))}
                    </div>
                    <div className="flex gap-2 flex-wrap">
                      {r.keywords.slice(0, 5).map(kw => (
                        <span key={kw} className="text-xs px-2 py-0.5 rounded-full text-gray-400"
                          style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.06)" }}>
                          {kw}
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Empty state */}
          {!session && !loading && (
            <div className="glass-card rounded-2xl p-12 text-center">
              <div className="text-4xl mb-4">🔺</div>
              <p className="text-gray-500 text-sm">Upload a document to get started. <br />
                Click <strong className="text-gray-400">Sample</strong> for an instant demo.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
