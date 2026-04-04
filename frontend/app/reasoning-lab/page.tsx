"use client";

import { useState } from "react";
import { api, type SolveResponse } from "@/lib/api";

const SAMPLE_PROBLEMS = {
  Easy: "John has 15 apples. He gives 7 to his sister and buys 4 more from the store. How many apples does John have now?",
  Medium: "A store sells notebooks for $3.50 each and pens for $1.25 each. Sarah buys 4 notebooks and 6 pens. If she pays with a $25 bill, how much change does she receive?",
  Hard: "A train travels from City A to City B at 60 mph. The return journey is at 40 mph due to weather. If the distance between cities is 120 miles, what is the average speed for the entire round trip?",
};

const DOMAIN_COLORS: Record<string, { bg: string; border: string; text: string; icon: string }> = {
  math:    { bg: "rgba(59,130,246,0.08)", border: "rgba(59,130,246,0.3)", text: "#60a5fa", icon: "🧮" },
  legal:   { bg: "rgba(139,92,246,0.08)", border: "rgba(139,92,246,0.3)", text: "#a78bfa", icon: "⚖️" },
  factual: { bg: "rgba(6,182,212,0.08)",  border: "rgba(6,182,212,0.3)",  text: "#22d3ee", icon: "📚" },
  code:    { bg: "rgba(16,185,129,0.08)", border: "rgba(16,185,129,0.3)", text: "#34d399", icon: "💻" },
  general: { bg: "rgba(107,114,128,0.08)",border: "rgba(107,114,128,0.3)",text: "#9ca3af", icon: "💬" },
};

function TypewriterText({ text, speed = 12 }: { text: string; speed?: number }) {
  const [displayed, setDisplayed] = useState("");
  const [done, setDone] = useState(false);
  const ref = { current: 0 };

  if (!done && text && displayed.length < text.length) {
    setTimeout(() => {
      setDisplayed(text.slice(0, displayed.length + 1));
      if (displayed.length + 1 >= text.length) setDone(true);
    }, speed);
  }

  return (
    <span>
      {displayed}
      {!done && <span className="cursor-blink text-blue-400">|</span>}
    </span>
  );
}

export default function ReasoningLabPage() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SolveResponse | null>(null);
  const [error, setError] = useState("");
  const [animKey, setAnimKey] = useState(0);

  const handleSolve = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await api.solveReasoning(query);
      setResult(res);
      setAnimKey(k => k + 1);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Request failed. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  const dc = result ? (DOMAIN_COLORS[result.query_type] ?? DOMAIN_COLORS.general) : null;

  return (
    <div className="max-w-5xl mx-auto px-6 py-12">
      {/* Header */}
      <div className="mb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs mb-4"
          style={{ background: "rgba(139,92,246,0.1)", border: "1px solid rgba(139,92,246,0.25)", color: "#a78bfa" }}>
          Bonus — Reasoning Adapter
        </div>
        <h1 className="text-4xl font-bold text-white mb-3">Reasoning Lab</h1>
        <p className="text-gray-400">
          Queries are classified into domains (Math, Legal, Factual, Code) and dispatched to domain-specific handlers via the Strategy Pattern adapter.
        </p>
      </div>

      {/* Model Info */}
      <div className="glass-card rounded-2xl p-6 mb-8">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-6">
          {[
            { label: "Architecture", value: "Strategy Pattern" },
            { label: "Domains", value: "5 (Math, Legal, Factual, Code, General)" },
            { label: "Classifier", value: "Keyword-based routing" },
            { label: "Upgrade Path", value: "DistilBERT classifier" },
          ].map(({ label, value }) => (
            <div key={label}>
              <div className="text-xs text-gray-500 mb-1">{label}</div>
              <div className="text-sm text-gray-200 font-medium">{value}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Input Panel */}
        <div className="space-y-4">
          <div className="glass-card rounded-2xl p-6">
            <h2 className="text-white font-semibold mb-4">Enter a Query</h2>
            <textarea
              className="w-full h-36 rounded-xl p-4 text-sm text-gray-300 resize-none outline-none"
              style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)" }}
              placeholder="Type any question — math, legal, factual, code, or general..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && e.ctrlKey && handleSolve()}
            />
            <button className="btn-primary w-full mt-3 text-sm" disabled={!query.trim() || loading}
              onClick={handleSolve}>
              {loading ? "Routing..." : "Solve → (Ctrl+Enter)"}
            </button>

            {/* Sample buttons */}
            <div className="mt-4">
              <p className="text-xs text-gray-500 mb-2">Sample Problems:</p>
              <div className="flex gap-2 flex-wrap">
                {Object.entries(SAMPLE_PROBLEMS).map(([label, text]) => (
                  <button key={label}
                    className="text-xs px-3 py-1 rounded-full text-gray-400 hover:text-blue-400 transition-colors"
                    style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.06)" }}
                    onClick={() => setQuery(text)}>
                    {label}
                  </button>
                ))}
              </div>
            </div>

            {/* Quick domain tests */}
            <div className="mt-3">
              <p className="text-xs text-gray-500 mb-2">Test Other Domains:</p>
              <div className="flex gap-2 flex-wrap">
                {[
                  { label: "Legal", q: "What are the legal implications of breach of contract?" },
                  { label: "Code", q: "How do I implement a binary search in Python?" },
                  { label: "Factual", q: "Who invented the World Wide Web?" },
                ].map(({ label, q }) => (
                  <button key={label}
                    className="text-xs px-3 py-1 rounded-full text-gray-400 hover:text-purple-400 transition-colors"
                    style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.06)" }}
                    onClick={() => setQuery(q)}>
                    {label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="rounded-xl p-4 text-sm text-red-400"
              style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)" }}>
              ⚠ {error} <br />
              <span className="text-xs text-red-500">Make sure the backend is running: <code>uvicorn backend.main:app --reload</code></span>
            </div>
          )}
        </div>

        {/* Output Panel */}
        <div>
          {result && dc ? (
            <div className="space-y-4 animate-fadeInUp" key={animKey}>
              {/* Domain Badge */}
              <div className="glass-card rounded-2xl p-5"
                style={{ borderColor: dc.border }}>
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">{dc.icon}</span>
                    <div>
                      <div className="text-xs text-gray-500 uppercase tracking-widest">Detected Domain</div>
                      <div className="font-bold capitalize text-lg" style={{ color: dc.text }}>
                        {result.query_type}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-xs text-gray-500 mb-1">Confidence</div>
                    <div className="text-xl font-bold" style={{ color: dc.text }}>
                      {(result.confidence * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
                {/* Confidence bar */}
                <div className="h-1.5 rounded-full" style={{ background: "rgba(255,255,255,0.06)" }}>
                  <div className="h-full rounded-full transition-all duration-1000"
                    style={{ width: `${result.confidence * 100}%`, background: dc.text }} />
                </div>
              </div>

              {/* Handler */}
              <div className="glass-card rounded-xl p-4 text-sm">
                <div className="text-xs text-gray-500 mb-1">Handler Dispatched To</div>
                <div className="text-gray-200 font-mono">{result.handler_name}</div>
              </div>

              {/* Answer */}
              <div className="glass-card rounded-2xl p-5"
                style={{ borderColor: dc.border }}>
                <div className="text-xs text-gray-500 mb-3 uppercase tracking-widest">Answer</div>
                <div className="text-gray-200 text-sm leading-relaxed font-mono whitespace-pre-wrap">
                  <TypewriterText key={animKey} text={result.answer} speed={8} />
                </div>
              </div>

              {/* Reasoning trace */}
              {result.reasoning_trace.length > 0 && (
                <div className="glass-card rounded-xl p-5">
                  <div className="text-xs text-gray-500 mb-3 uppercase tracking-widest">Reasoning Trace</div>
                  <div className="space-y-2">
                    {result.reasoning_trace.map((step, i) => (
                      <div key={i} className="flex gap-3 text-xs text-gray-400">
                        <span className="text-gray-600 shrink-0 font-mono">{String(i + 1).padStart(2, "0")}</span>
                        <span>{step}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="glass-card rounded-2xl p-12 text-center h-full flex flex-col items-center justify-center">
              <div className="text-5xl mb-4">🔀</div>
              <p className="text-gray-500 text-sm">
                Ask a question to see the router in action.<br />
                <strong className="text-gray-400">Math</strong> queries compute answers.{" "}
                <strong className="text-gray-400">Legal</strong> adds disclaimers.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
