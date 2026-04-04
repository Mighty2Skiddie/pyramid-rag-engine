import Link from "next/link";

const features = [
  {
    icon: "🔺",
    title: "Knowledge Pyramid RAG",
    desc: "Documents ingested into a 4-layer pyramid (Raw → Summary → Category → Keywords). Multi-level weighted retrieval for precise answers.",
    color: "#3b82f6",
  },
  {
    icon: "🧮",
    title: "GSM8K Fine-Tuned Reasoning",
    desc: "LLaMA 3.2 1B fine-tuned with QLoRA on 3,000 math problems. Step-by-step chain-of-thought reasoning with ~30% exact match accuracy.",
    color: "#8b5cf6",
  },
  {
    icon: "🔀",
    title: "Plug-and-Play Adapter",
    desc: "Strategy Pattern router classifies queries into domains (Math, Legal, Factual, Code) and dispatches to the right handler. Zero changes to add new domains.",
    color: "#06b6d4",
  },
];

const steps = [
  { step: "01", title: "Upload Document", desc: "Drop a .txt or .pdf file, or paste text directly" },
  { step: "02", title: "Build Pyramid", desc: "4-layer Knowledge Pyramid constructed automatically" },
  { step: "03", title: "Query & Retrieve", desc: "Ask anything — multi-level retrieval finds the best answer" },
];

const techStack = ["Python", "FastAPI", "Next.js", "LLaMA 3.2", "LoRA / QLoRA", "scikit-learn", "rapidfuzz", "PyMuPDF"];

export default function LandingPage() {
  return (
    <div className="relative overflow-hidden">

      {/* Hero background blobs */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-[-20%] left-[-10%] w-[600px] h-[600px] rounded-full"
          style={{ background: "radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%)" }} />
        <div className="absolute top-[10%] right-[-10%] w-[500px] h-[500px] rounded-full"
          style={{ background: "radial-gradient(circle, rgba(139,92,246,0.10) 0%, transparent 70%)" }} />
      </div>

      {/* ── Hero ─────────────────────────────────────── */}
      <section className="max-w-7xl mx-auto px-6 pt-24 pb-20 text-center relative">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium mb-8"
          style={{ background: "rgba(59,130,246,0.1)", border: "1px solid rgba(59,130,246,0.25)", color: "#60a5fa" }}>
          <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse inline-block" />
          Vexoo Labs AI Engineer Assignment — Live Demo
        </div>

        <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold leading-tight mb-6 text-white">
          AI Document Intelligence
          <br />
          <span className="gradient-text">+ Math Reasoning</span>
        </h1>

        <p className="text-lg text-gray-400 max-w-2xl mx-auto mb-10">
          Upload any document and query it through a 4-layer Knowledge Pyramid.
          Ask math problems solved step-by-step by a fine-tuned LLM. All running live.
        </p>

        <div className="flex flex-wrap gap-4 justify-center">
          <Link href="/document-lab" className="btn-primary text-sm">
            Try Document Lab →
          </Link>
          <Link href="/reasoning-lab"
            className="px-6 py-2.5 rounded-lg border border-purple-500/40 text-purple-400 bg-purple-500/10 hover:bg-purple-500/20 text-sm font-semibold transition-all">
            Try Reasoning Lab →
          </Link>
        </div>
      </section>

      {/* ── Feature Cards ─────────────────────────────── */}
      <section className="max-w-7xl mx-auto px-6 py-16">
        <div className="grid md:grid-cols-3 gap-6">
          {features.map((f) => (
            <div key={f.title} className="glass-card rounded-2xl p-6 group hover:border-blue-500/20 transition-all duration-300">
              <div className="w-12 h-12 rounded-xl flex items-center justify-center text-2xl mb-4"
                style={{ background: `${f.color}15`, border: `1px solid ${f.color}30` }}>
                {f.icon}
              </div>
              <h3 className="text-white font-semibold text-lg mb-2">{f.title}</h3>
              <p className="text-gray-400 text-sm leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── How It Works ──────────────────────────────── */}
      <section className="max-w-7xl mx-auto px-6 py-16">
        <h2 className="text-3xl font-bold text-white text-center mb-12">
          How It Works
        </h2>
        <div className="grid md:grid-cols-3 gap-8 relative">
          {/* connector line */}
          <div className="hidden md:block absolute top-8 left-1/3 right-1/3 h-px"
            style={{ background: "linear-gradient(90deg, transparent, rgba(59,130,246,0.4), transparent)" }} />

          {steps.map((s) => (
            <div key={s.step} className="flex flex-col items-center text-center">
              <div className="w-16 h-16 rounded-2xl flex items-center justify-center text-xl font-bold mb-4 animate-pulseGlow"
                style={{ background: "linear-gradient(135deg,#1d4ed8,#4c1d95)", color: "#93c5fd", border: "1px solid rgba(59,130,246,0.3)" }}>
                {s.step}
              </div>
              <h3 className="text-white font-semibold mb-2">{s.title}</h3>
              <p className="text-gray-400 text-sm">{s.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Tech Stack Badges ─────────────────────────── */}
      <section className="max-w-7xl mx-auto px-6 py-12 text-center">
        <p className="text-gray-500 text-sm uppercase tracking-widest font-medium mb-6">Built With</p>
        <div className="flex flex-wrap justify-center gap-3">
          {techStack.map((t) => (
            <span key={t}
              className="px-4 py-1.5 rounded-full text-sm font-medium text-gray-300 transition-colors hover:text-blue-400"
              style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)" }}>
              {t}
            </span>
          ))}
        </div>
      </section>

      {/* ── CTA Banner ────────────────────────────────── */}
      <section className="max-w-5xl mx-auto px-6 py-16">
        <div className="glass-card rounded-3xl p-10 text-center"
          style={{ background: "linear-gradient(135deg, rgba(59,130,246,0.08), rgba(139,92,246,0.08))", border: "1px solid rgba(59,130,246,0.2)" }}>
          <h2 className="text-3xl font-bold text-white mb-4">See It In Action</h2>
          <p className="text-gray-400 mb-8 max-w-xl mx-auto">
            Upload your own document or use the pre-loaded Nexagen Technologies annual report — then ask anything.
          </p>
          <Link href="/document-lab" className="btn-primary text-sm">
            Open Document Lab →
          </Link>
        </div>
      </section>

    </div>
  );
}
