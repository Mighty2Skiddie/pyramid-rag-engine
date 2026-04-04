const pyramidLayers = [
  { level: "L4", title: "Distilled Knowledge", desc: "Top TF-IDF keywords + 128-d hash-seeded embedding vector. Highest retrieval weight (45%).", color: "#10b981", weight: "45%" },
  { level: "L3", title: "Category Labels",     desc: "Domain classification: Finance, Legal, Technical, Medical, General. Keyword dict matching.", color: "#06b6d4", weight: "10%" },
  { level: "L2", title: "Extractive Summaries", desc: "Top-2 sentences by word-frequency scoring (simplified TextRank). Second-highest weight (30%).", color: "#8b5cf6", weight: "30%" },
  { level: "L1", title: "Raw Text Chunks",    desc: "Original 2000-char sliding window chunks with 15% overlap. TF-IDF cosine matching (15%).", color: "#3b82f6", weight: "15%" },
];

const trainingSteps = [
  { step: "GSM8K", desc: "3,000 train / 1,000 test, seed=42" },
  { step: "Tokenizer", desc: "max_length=512, causal LM labels" },
  { step: "4-bit NF4", desc: "QLoRA, fits on 16GB T4 VRAM" },
  { step: "LoRA r=8", desc: "q_proj + v_proj, α=16, ~16M params" },
  { step: "Training", desc: "3 epochs, batch=32 via grad_accum" },
  { step: "Evaluation", desc: "Exact match → ~25–35% accuracy" },
];

const adapterDomains = [
  { type: "MATH",    icon: "🧮", handler: "MathReasoningHandler",    desc: "Chain-of-thought arithmetic" },
  { type: "LEGAL",   icon: "⚖️", handler: "LegalReasoningHandler",   desc: "Clause-aware + disclaimer" },
  { type: "FACTUAL", icon: "📚", handler: "FactualReasoningHandler", desc: "Precision-first retrieval" },
  { type: "CODE",    icon: "💻", handler: "CodeReasoningHandler",    desc: "Technical context parsing" },
  { type: "GENERAL", icon: "💬", handler: "DefaultHandler",          desc: "Standard fallback" },
];

const techTable = [
  { layer: "Frontend",  tech: "Next.js 14",    purpose: "App Router, SSR, fast initial loads" },
  { layer: "Frontend",  tech: "Tailwind CSS",  purpose: "Utility-first styling" },
  { layer: "Backend",   tech: "FastAPI",       purpose: "Python API, auto OpenAPI docs" },
  { layer: "AI — Part 1", tech: "scikit-learn TF-IDF", purpose: "Sparse matrix similarity, deterministic" },
  { layer: "AI — Part 1", tech: "rapidfuzz",  purpose: "10–100× faster than fuzzywuzzy" },
  { layer: "AI — Part 2", tech: "peft + trl", purpose: "LoRA fine-tuning, standard HF ecosystem" },
  { layer: "AI — Part 2", tech: "bitsandbytes", purpose: "4-bit NF4 quantization (QLoRA)" },
];

export default function ArchitecturePage() {
  return (
    <div className="max-w-6xl mx-auto px-6 py-12 space-y-20">
      <div>
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs mb-4"
          style={{ background: "rgba(6,182,212,0.1)", border: "1px solid rgba(6,182,212,0.25)", color: "#22d3ee" }}>
          System Design
        </div>
        <h1 className="text-4xl font-bold text-white mb-3">Architecture</h1>
        <p className="text-gray-400">How the system is designed — not just what it does, but why each decision was made.</p>
      </div>

      {/* System Overview */}
      <section>
        <h2 className="text-2xl font-bold text-white mb-6">System Overview</h2>
        <div className="glass-card rounded-2xl p-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center text-sm">
            {[
              { label: "Browser (Next.js)", sub: "5 pages, dark theme, live demos", color: "#3b82f6" },
              { label: "FastAPI Backend", sub: "/api/pyramid + /api/reasoning", color: "#8b5cf6" },
              { label: "AI Engines", sub: "Part 1 pipeline + Adapter", color: "#06b6d4" },
              { label: "Session Store", sub: "In-memory, TTL 30 minutes", color: "#10b981" },
            ].map(({ label, sub, color }) => (
              <div key={label} className="rounded-xl p-4" style={{ background: `${color}15`, border: `1px solid ${color}30` }}>
                <div className="font-semibold text-white mb-1 text-sm">{label}</div>
                <div className="text-xs text-gray-400">{sub}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Pyramid */}
      <section>
        <h2 className="text-2xl font-bold text-white mb-2">Knowledge Pyramid — Layer by Layer</h2>
        <p className="text-gray-400 text-sm mb-6">Each chunk enriched into 4 representations. Retrieval scores all 4 and combines with fixed weights.</p>
        <div className="space-y-3">
          {pyramidLayers.map(({ level, title, desc, color, weight }) => (
            <div key={level} className="glass-card rounded-xl p-5 flex items-start gap-5">
              <div className="shrink-0 w-14 h-14 rounded-xl flex flex-col items-center justify-center text-sm font-bold"
                style={{ background: `${color}15`, border: `1px solid ${color}30`, color }}>
                <div className="text-lg">{level}</div>
                <div className="text-xs">{weight}</div>
              </div>
              <div className="flex-1">
                <div className="font-semibold text-white mb-1">{title}</div>
                <div className="text-sm text-gray-400">{desc}</div>
              </div>
              <div className="hidden sm:block shrink-0">
                <div className="h-2 w-24 rounded-full" style={{ background: "rgba(255,255,255,0.06)" }}>
                  <div className="h-full rounded-full" style={{ width: weight, background: color }} />
                </div>
                <div className="text-xs text-gray-500 mt-1 text-right">weight</div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Training Pipeline */}
      <section>
        <h2 className="text-2xl font-bold text-white mb-2">GSM8K Training Pipeline</h2>
        <p className="text-gray-400 text-sm mb-6">LLaMA 3.2 1B fine-tuned with QLoRA on Google Colab T4 GPU (free tier).</p>
        <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
          {trainingSteps.map(({ step, desc }, i) => (
            <div key={step} className="glass-card rounded-xl p-3 text-center">
              <div className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold mx-auto mb-2"
                style={{ background: "linear-gradient(135deg,#7c3aed,#4f46e5)", color: "#c4b5fd" }}>
                {i + 1}
              </div>
              <div className="text-xs font-semibold text-white mb-1 font-mono">{step}</div>
              <div className="text-xs text-gray-500 leading-snug">{desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Reasoning Adapter */}
      <section>
        <h2 className="text-2xl font-bold text-white mb-2">Reasoning Adapter — Strategy Pattern</h2>
        <p className="text-gray-400 text-sm mb-6">Adding a new domain requires only registration — zero changes to routing or existing handlers.</p>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="glass-card rounded-2xl p-6 space-y-3 text-sm">
            {["User Query", "KeywordQueryClassifier → (QueryType, confidence)", "Handler Registry lookup", "ReasoningResponse (answer + trace + confidence)"].map((label, i) => (
              <div key={i}>
                <div className="p-3 rounded-lg text-center font-mono"
                  style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", color: ["#60a5fa","#22d3ee","#a78bfa","#34d399"][i] }}>
                  {label}
                </div>
                {i < 3 && <div className="text-center text-gray-600 text-xs py-1">↓</div>}
              </div>
            ))}
          </div>
          <div className="space-y-2">
            {adapterDomains.map(({ type, icon, handler, desc }) => (
              <div key={type} className="glass-card rounded-xl p-3 flex items-center gap-4">
                <span className="text-xl">{icon}</span>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-semibold text-white">{type}</div>
                  <div className="text-xs text-gray-500 font-mono truncate">{handler}</div>
                </div>
                <div className="text-xs text-gray-400 hidden sm:block">{desc}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Tech Table */}
      <section>
        <h2 className="text-2xl font-bold text-white mb-6">Technology Choices</h2>
        <div className="glass-card rounded-2xl overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/5">
                <th className="px-5 py-3 text-left text-xs text-gray-500 uppercase tracking-widest">Layer</th>
                <th className="px-5 py-3 text-left text-xs text-gray-500 uppercase tracking-widest">Technology</th>
                <th className="px-5 py-3 text-left text-xs text-gray-500 uppercase tracking-widest">Reason</th>
              </tr>
            </thead>
            <tbody>
              {techTable.map(({ layer, tech, purpose }, i) => (
                <tr key={i} className="border-b border-white/5">
                  <td className="px-5 py-3 text-gray-500">{layer}</td>
                  <td className="px-5 py-3 font-mono text-blue-400">{tech}</td>
                  <td className="px-5 py-3 text-gray-400">{purpose}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
