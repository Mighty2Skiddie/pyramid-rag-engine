export default function Footer() {
  return (
    <footer className="border-t border-white/5 mt-20">
      <div className="max-w-7xl mx-auto px-6 py-10 flex flex-col sm:flex-row justify-between items-center gap-4">
        <div className="text-sm text-gray-500">
          Built by <span className="text-gray-300 font-medium">Pranav Sharma</span> · Vexoo Labs AI Engineer Assignment
        </div>
        <div className="flex items-center gap-6 text-sm text-gray-500">
          <a href="https://github.com/Mighty2Skiddie/pyramid-rag-engine" target="_blank" rel="noreferrer" className="hover:text-blue-400 transition-colors">GitHub</a>
          <a href="https://linkedin.com" target="_blank" rel="noreferrer" className="hover:text-blue-400 transition-colors">LinkedIn</a>
          <a href="/about" className="hover:text-blue-400 transition-colors">Docs & Report</a>
        </div>
      </div>
    </footer>
  );
}
