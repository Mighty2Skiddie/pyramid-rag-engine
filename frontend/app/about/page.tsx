import Link from "next/link";

const checklist = [
  { item: "Sliding Window Chunker (2000 chars, 15% overlap)", done: true },
  { item: "4-Layer Knowledge Pyramid (L1→L4)", done: true },
  { item: "Multi-level weighted retrieval (TF-IDF, fuzzy, Jaccard, cosine)", done: true },
  { item: "GSM8K Fine-tuning with LoRA (r=8) on LLaMA 3.2 1B", done: true },
  { item: "QLoRA (4-bit NF4) for free-tier Colab T4 GPU", done: true },
  { item: "Exact match evaluation with regex answer extraction", done: true },
  { item: "Bonus: Plug-and-play Reasoning Adapter (Strategy Pattern)", done: true },
  { item: "Full test suite (39 tests, all passing)", done: true },
  { item: "FastAPI backend wrapping all Python modules", done: true },
  { item: "Next.js 14 frontend with 5 interactive pages", done: true },
];

const keyDecisions = [
  { q: "Why TF-IDF over real embeddings?", a: "Zero external dependencies, fully deterministic, reproducible. Clear swap point at generate_mock_embedding()." },
  { q: "Why multi-level retrieval?", a: "Single-score retrieval forces a choice between lexical and semantic. Multi-level captures complementary signals." },
  { q: "Why LoRA r=8?", a: "~16M trainable params (1.6% of 1B). Prevents overfitting on 3K samples while having capacity to shift math reasoning." },
  { q: "Why Strategy Pattern for the adapter?", a: "Classifier, handler logic, and routing are three concerns that change independently. OCP in practice." },
];

export default function AboutPage() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-12 space-y-16">
      <div>
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs mb-4"
          style={{ background: "rgba(59,130,246,0.1)", border: "1px solid rgba(59,130,246,0.25)", color: "#60a5fa" }}>
          About This Project
        </div>
        <h1 className="text-4xl font-bold text-white mb-3">About & Docs</h1>
        <p className="text-gray-400">Vexoo Labs AI Engineer Assignment — built by Pranav Sharma.</p>
      </div>

      {/* Bio */}
      <section className="glass-card rounded-2xl p-8">
        <h2 className="text-xl font-bold text-white mb-4">About This Submission</h2>
        <p className="text-gray-400 leading-relaxed mb-4">
          This project implements all three assignment requirements — document ingestion with a hierarchical Knowledge Pyramid,
          GSM8K fine-tuning of LLaMA 3.2 1B, and a plug-and-play reasoning adapter — and then deploys them as a live web platform.
        </p>
        <p className="text-gray-400 leading-relaxed">
          The system is designed for clarity and upgradeability: every major component has a clearly marked swap point for production upgrades
          (real embeddings, LLM-based summarization, ML-based classifier). The architecture documentation explains the <em>why</em> behind each decision,
          not just what was built.
        </p>
        <div className="flex gap-4 mt-6">
          <a href="https://github.com/Mighty2Skiddie/pyramid-rag-engine" target="_blank" rel="noreferrer"
            className="btn-primary text-sm">GitHub Repository →</a>
          <Link href="/architecture" className="btn-secondary text-sm">View Architecture</Link>
        </div>
      </section>

      {/* Assignment Checklist */}
      <section>
        <h2 className="text-2xl font-bold text-white mb-6">Assignment Coverage</h2>
        <div className="space-y-2">
          {checklist.map(({ item, done }) => (
            <div key={item} className="glass-card rounded-xl px-5 py-3 flex items-center gap-4">
              <div className="w-5 h-5 rounded-full flex items-center justify-center shrink-0 text-xs"
                style={{ background: done ? "rgba(16,185,129,0.15)" : "rgba(107,114,128,0.15)", border: `1px solid ${done ? "#10b981" : "#6b7280"}`, color: done ? "#34d399" : "#9ca3af" }}>
                {done ? "✓" : "−"}
              </div>
              <span className={`text-sm ${done ? "text-gray-300" : "text-gray-500"}`}>{item}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Key Decisions Q&A */}
      <section>
        <h2 className="text-2xl font-bold text-white mb-6">Key Design Decisions</h2>
        <div className="space-y-4">
          {keyDecisions.map(({ q, a }) => (
            <div key={q} className="glass-card rounded-xl p-5">
              <div className="text-white font-medium mb-2">{q}</div>
              <div className="text-gray-400 text-sm leading-relaxed">{a}</div>
            </div>
          ))}
        </div>
        <div className="mt-4 text-sm">
          <Link href="/architecture" className="text-blue-400 hover:text-blue-300 transition-colors">
            Full architecture analysis → docs/engineering_decisions.md
          </Link>
        </div>
      </section>

      {/* Quick Links */}
      <section>
        <h2 className="text-2xl font-bold text-white mb-6">Documentation</h2>
        <div className="grid sm:grid-cols-2 gap-4">
          {[
            { title: "System Design", desc: "Architecture, data flow, scalability", href: "/architecture", icon: "🏗️" },
            { title: "Try Document Lab", desc: "Upload a document and query it live", href: "/document-lab", icon: "🔺" },
            { title: "Try Reasoning Lab", desc: "See the adapter route queries in real time", href: "/reasoning-lab", icon: "🔀" },
            { title: "GitHub Repo", desc: "All source code, tests, and documentation", href: "https://github.com/Mighty2Skiddie/pyramid-rag-engine", icon: "💻", external: true },
          ].map(({ title, desc, href, icon, external }) => (
            <a key={title} href={href} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}
              className="glass-card rounded-xl p-5 flex items-start gap-4 hover:border-blue-500/20 transition-all group">
              <span className="text-2xl">{icon}</span>
              <div>
                <div className="text-white font-medium group-hover:text-blue-400 transition-colors">{title}</div>
                <div className="text-gray-400 text-sm mt-1">{desc}</div>
              </div>
            </a>
          ))}
        </div>
      </section>
    </div>
  );
}
