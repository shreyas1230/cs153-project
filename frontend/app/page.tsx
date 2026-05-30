"use client";
import { useState, useRef, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  Upload, FileText, X, Loader2, Layers, Search,
  ArrowRight, BookOpen, Zap, AlertTriangle, Clock, ExternalLink,
  CheckCircle2, Library, Trash2,
} from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";

type RecentEntry = { id: string; question: string; papers: string[]; ts: number };
type SavedPaper = { name: string; uploaded_at: number };

export default function HomePage() {
  const router = useRouter();
  const [files, setFiles] = useState<File[]>([]);
  const [question, setQuestion] = useState("");
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [recent, setRecent] = useState<RecentEntry[]>([]);
  const [savedPapers, setSavedPapers] = useState<SavedPaper[]>([]);
  const [selectedPapers, setSelectedPapers] = useState<string[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const stored = localStorage.getItem("recent_analyses");
    if (stored) setRecent(JSON.parse(stored));
    fetch(`${API_BASE}/api/papers`)
      .then((r) => r.json())
      .then(setSavedPapers)
      .catch(() => {});
  }, []);

  const togglePaper = (name: string) =>
    setSelectedPapers((prev) =>
      prev.includes(name) ? prev.filter((p) => p !== name) : [...prev, name]
    );

  const deletePaper = async (name: string, e: React.MouseEvent) => {
    e.stopPropagation();
    await fetch(`${API_BASE}/api/papers/${encodeURIComponent(name)}`, { method: "DELETE" });
    setSavedPapers((prev) => prev.filter((p) => p.name !== name));
    setSelectedPapers((prev) => prev.filter((p) => p !== name));
  };

  const addFiles = (incoming: FileList | File[]) => {
    const pdfs = Array.from(incoming).filter((f) => f.type === "application/pdf");
    setFiles((prev) => {
      const names = new Set(prev.map((f) => f.name));
      return [...prev, ...pdfs.filter((f) => !names.has(f.name))].slice(0, 10);
    });
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    addFiles(e.dataTransfer.files);
  }, []);

  const removeFile = (name: string) => setFiles((prev) => prev.filter((f) => f.name !== name));

  const totalPapers = files.length + selectedPapers.length;

  const submit = async () => {
    if (totalPapers < 2) { setError("Select or upload at least 2 papers to compare."); return; }
    if (!question.trim()) { setError("Enter a research question."); return; }
    setError("");
    setLoading(true);
    try {
      const form = new FormData();
      form.append("question", question.trim());
      files.forEach((f) => form.append("files", f));
      selectedPapers.forEach((name) => form.append("paper_names", name));
      const res = await fetch(`${API_BASE}/api/analyze`, { method: "POST", body: form });
      if (!res.ok) throw new Error((await res.json()).detail ?? "Server error");
      const { session_id } = await res.json();
      const allPaperNames = [...selectedPapers, ...files.map((f) => f.name)];
      const entry: RecentEntry = { id: session_id, question: question.trim(), papers: allPaperNames, ts: Date.now() };
      const updated = [entry, ...recent].slice(0, 5);
      localStorage.setItem("recent_analyses", JSON.stringify(updated));
      setRecent(updated);
      // Refresh saved papers list (new uploads are now cached)
      fetch(`${API_BASE}/api/papers`).then((r) => r.json()).then(setSavedPapers).catch(() => {});
      router.push(`/results/${session_id}`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-white">
      {/* ── Navbar ── */}
      <nav className="fixed top-0 z-50 w-full border-b border-slate-200 bg-white/90 backdrop-blur-sm">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-2">
            <Layers className="h-5 w-5 text-blue-600" />
            <span className="font-bold text-slate-900">ClaimLens</span>
          </div>
          <div className="flex items-center gap-6 text-sm text-slate-500">
            <a href="#how-it-works" className="hover:text-slate-900 transition-colors">How it works</a>
            <a href="#analyze" className="hover:text-slate-900 transition-colors">Try it</a>
            <a
              href="https://github.com/shreyas1230/cs153-project"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 rounded-full border border-slate-200 px-3 py-1 hover:bg-slate-50 transition-colors"
            >
              GitHub <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section
        className="relative overflow-hidden bg-slate-900 pb-24 pt-32 px-4"
        style={{ backgroundImage: "radial-gradient(circle, #334155 1px, transparent 1px)", backgroundSize: "28px 28px" }}
      >
        <div className="mx-auto max-w-3xl text-center">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-800/80 px-4 py-1.5 text-xs text-slate-300">
            <Zap className="h-3 w-3 text-blue-400" />
            CS 153 · Stanford · Spring 2026
          </div>
          <h1 className="mt-4 text-4xl font-extrabold tracking-tight text-white sm:text-5xl leading-tight">
            Discover what papers<br />
            <span className="text-blue-400">actually disagree</span> about
          </h1>
          <p className="mt-6 text-lg leading-relaxed text-slate-300 max-w-2xl mx-auto">
            Upload scientific PDFs and ask a question. Surface real contradictions, hidden support,
            and apparent conflicts caused by terminology drift — not synthesized away.
          </p>
          <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
            <a
              href="#analyze"
              className="flex items-center gap-2 rounded-lg bg-blue-600 px-6 py-3 font-semibold text-white hover:bg-blue-500 transition-colors"
            >
              Start Analyzing <ArrowRight className="h-4 w-4" />
            </a>
            <a
              href="#how-it-works"
              className="flex items-center gap-2 rounded-lg border border-slate-600 px-6 py-3 font-semibold text-slate-300 hover:bg-slate-800 transition-colors"
            >
              How it works
            </a>
          </div>
        </div>

        {/* Floating stats */}
        <div className="mx-auto mt-14 max-w-2xl grid grid-cols-3 gap-4 text-center">
          {[
            { value: "3", label: "Pipeline stages" },
            { value: "F1 = 0.68", label: "Aggregate accuracy" },
            { value: "$0.003", label: "Avg. cost / analysis" },
          ].map(({ value, label }) => (
            <div key={label} className="rounded-xl border border-slate-700 bg-slate-800/60 px-4 py-4">
              <p className="text-xl font-bold text-white">{value}</p>
              <p className="mt-0.5 text-xs text-slate-400">{label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Problem callout ── */}
      <section className="border-b border-amber-200 bg-amber-50 px-4 py-8">
        <div className="mx-auto max-w-3xl">
          <p className="text-base leading-relaxed text-slate-700">
            <span className="font-semibold text-amber-700">The problem: </span>
            Tools like Perplexity and NotebookLM synthesize papers into confident summaries.
            For cross-domain research, this{" "}
            <em>hides exactly the disagreements you need to see</em> — especially when two papers
            use the same word to mean completely different things.
            ClaimLens makes those disagreements first-class outputs.
          </p>
        </div>
      </section>

      {/* ── How it works ── */}
      <section id="how-it-works" className="bg-white px-4 py-20">
        <div className="mx-auto max-w-4xl">
          <div className="mb-12 text-center">
            <h2 className="text-2xl font-bold text-slate-900">How it works</h2>
            <p className="mt-2 text-slate-500">A three-stage LLM pipeline runs on your papers</p>
          </div>
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-3">
            <PipelineStep
              n={1} color="blue" icon={<FileText className="h-5 w-5 text-blue-600" />}
              title="Extract Claims"
              desc="Each paper is broken into atomic, falsifiable claims tagged with verbatim source passages, key terms, and domain signals."
            />
            <PipelineStep
              n={2} color="violet" icon={<Layers className="h-5 w-5 text-violet-600" />}
              title="Normalize Terms"
              desc="A cross-paper glossary flags where the same word means different things across fields — the root cause of most apparent conflicts."
            />
            <PipelineStep
              n={3} color="emerald" icon={<Search className="h-5 w-5 text-emerald-600" />}
              title="Detect Conflicts"
              desc="Claim pairs are classified with a plain-language explanation and, for incommensurable pairs, a terminology note."
            />
          </div>

          {/* Relationship legend */}
          <div className="mt-10 flex flex-wrap justify-center gap-3 text-xs">
            <RelBadge color="green" label="SUPPORT" desc="Claims reinforce each other" />
            <RelBadge color="red" label="CONTRADICT" desc="Genuine empirical or logical conflict" />
            <RelBadge color="yellow" label="INCOMMENSURABLE" desc="Apparent conflict due to terminology drift" />
          </div>
        </div>
      </section>

      {/* ── Upload section ── */}
      <section id="analyze" className="bg-slate-50 px-4 py-20">
        <div className="mx-auto max-w-2xl space-y-8">
          <div className="text-center">
            <h2 className="text-2xl font-bold text-slate-900">Try it yourself</h2>
            <p className="mt-2 text-slate-500">Upload 2–10 scientific PDFs and ask a question</p>
          </div>

          <div className="rounded-2xl border border-slate-200 bg-white p-8 shadow-sm space-y-6">
            {/* My Papers library */}
            {savedPapers.length > 0 && (
              <div>
                <div className="mb-2 flex items-center gap-2">
                  <Library className="h-4 w-4 text-violet-500" />
                  <span className="text-sm font-medium text-slate-700">My Papers</span>
                  <span className="text-xs text-slate-400">— select from library, upload new ones below, or both</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {savedPapers.map((p) => {
                    const selected = selectedPapers.includes(p.name);
                    return (
                      <button
                        key={p.name}
                        type="button"
                        onClick={() => togglePaper(p.name)}
                        className={`group flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs transition-all ${
                          selected
                            ? "border-violet-400 bg-violet-50 text-violet-700"
                            : "border-slate-200 bg-white text-slate-600 hover:border-violet-300 hover:bg-violet-50"
                        }`}
                      >
                        {selected
                          ? <CheckCircle2 className="h-3.5 w-3.5 text-violet-500" />
                          : <FileText className="h-3.5 w-3.5 text-slate-400" />
                        }
                        <span className="max-w-[160px] truncate">{p.name}</span>
                        <span
                          onClick={(e) => deletePaper(p.name, e)}
                          className="ml-0.5 hidden rounded-full p-0.5 hover:bg-red-100 group-hover:inline-flex"
                        >
                          <Trash2 className="h-3 w-3 text-red-400" />
                        </span>
                      </button>
                    );
                  })}
                </div>
                {selectedPapers.length > 0 && (
                  <p className="mt-2 text-xs text-violet-600">
                    {selectedPapers.length} paper{selectedPapers.length > 1 ? "s" : ""} selected from library
                    {files.length > 0 ? ` + ${files.length} new upload${files.length > 1 ? "s" : ""}` : ""}
                  </p>
                )}
              </div>
            )}

            {/* Divider shown only when library is present */}
            {savedPapers.length > 0 && (
              <div className="flex items-center gap-3">
                <div className="h-px flex-1 bg-slate-200" />
                <span className="text-xs text-slate-400">+ add new papers</span>
                <div className="h-px flex-1 bg-slate-200" />
              </div>
            )}

            {/* Drop zone */}
            <div
              onDrop={onDrop}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onClick={() => inputRef.current?.click()}
              className={`cursor-pointer rounded-xl border-2 border-dashed p-10 text-center transition-colors ${
                dragging ? "border-blue-500 bg-blue-50" : "border-slate-200 bg-slate-50 hover:border-blue-300 hover:bg-blue-50/30"
              }`}
            >
              <Upload className="mx-auto mb-3 h-8 w-8 text-slate-400" />
              <p className="font-medium text-slate-700">Drop PDFs here or click to upload</p>
              <p className="mt-1 text-sm text-slate-400">2–10 papers · PDF only</p>
              <input
                ref={inputRef} type="file" accept="application/pdf" multiple className="hidden"
                onChange={(e) => e.target.files && addFiles(e.target.files)}
              />
            </div>

            {/* File list */}
            {files.length > 0 && (
              <ul className="space-y-2">
                {files.map((f) => (
                  <li key={f.name} className="flex items-center gap-3 rounded-lg border border-slate-100 bg-slate-50 px-4 py-2.5">
                    <FileText className="h-4 w-4 shrink-0 text-blue-400" />
                    <span className="flex-1 truncate text-sm text-slate-700">{f.name}</span>
                    <span className="text-xs text-slate-400">{(f.size / 1024).toFixed(0)} KB</span>
                    <button onClick={() => removeFile(f.name)} className="text-slate-300 hover:text-red-500 transition-colors">
                      <X className="h-4 w-4" />
                    </button>
                  </li>
                ))}
              </ul>
            )}

            {/* Question */}
            <div>
              <label className="mb-1.5 block text-sm font-medium text-slate-700">Research question</label>
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                rows={3}
                placeholder='e.g. "What is attention and how does it work?" or "Does regularization prevent overfitting?"'
                className="w-full rounded-lg border border-slate-300 px-4 py-3 text-sm placeholder-slate-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 resize-none"
              />
              <p className="mt-1 text-xs text-slate-400">
                Each analysis is question-specific. Start a new analysis to explore a different angle.
              </p>
            </div>

            {error && (
              <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                <AlertTriangle className="h-4 w-4 shrink-0" /> {error}
              </div>
            )}

            <button
              onClick={submit}
              disabled={loading}
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-6 py-3 font-semibold text-white transition-colors hover:bg-blue-700 disabled:opacity-50"
            >
              {loading
                ? <><Loader2 className="h-4 w-4 animate-spin" /> Submitting…</>
                : <><ArrowRight className="h-4 w-4" /> Analyze {totalPapers > 0 ? `${totalPapers} ` : ""}Papers</>}
            </button>
          </div>

          {/* Recent analyses */}
          {recent.length > 0 && (
            <div>
              <div className="mb-3 flex items-center justify-between">
                <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-500">
                  <Clock className="h-4 w-4" /> Recent analyses
                </h3>
                <button
                  onClick={() => { localStorage.removeItem("recent_analyses"); setRecent([]); }}
                  className="text-xs text-slate-400 hover:text-red-500 transition-colors"
                >
                  Clear
                </button>
              </div>
              <div className="space-y-2">
                {recent.map((r) => (
                  <a
                    key={r.id}
                    href={`/results/${r.id}`}
                    className="flex items-start justify-between gap-4 rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm hover:border-blue-300 hover:shadow-sm transition-all"
                  >
                    <div className="min-w-0">
                      <p className="font-medium text-slate-800 truncate">{r.question}</p>
                      <p className="mt-0.5 text-xs text-slate-400 truncate">
                        {r.papers.map((p) => p.replace(".pdf", "")).join(" · ")}
                      </p>
                    </div>
                    <div className="flex shrink-0 items-center gap-1 text-xs text-slate-400">
                      <BookOpen className="h-3 w-3" />
                      View
                    </div>
                  </a>
                ))}
              </div>
            </div>
          )}
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="border-t border-slate-200 bg-white px-4 py-8">
        <div className="mx-auto flex max-w-5xl items-center justify-between text-sm text-slate-400">
          <div className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-blue-500" />
            <span>ClaimLens · CS 153 Final Project · Stanford 2026</span>
          </div>
          <a
            href="https://github.com/shreyas1230/cs153-project"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 hover:text-slate-700 transition-colors"
          >
            GitHub <ExternalLink className="h-3 w-3" />
          </a>
        </div>
      </footer>
    </div>
  );
}

function PipelineStep({ n, color, icon, title, desc }: {
  n: number; color: "blue" | "violet" | "emerald";
  icon: React.ReactNode; title: string; desc: string;
}) {
  const numBg = { blue: "bg-blue-600", violet: "bg-violet-600", emerald: "bg-emerald-600" }[color];
  const iconBg = { blue: "bg-blue-50", violet: "bg-violet-50", emerald: "bg-emerald-50" }[color];
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="mb-4 flex items-center gap-3">
        <span className={`flex h-6 w-6 items-center justify-center rounded-full ${numBg} text-xs font-bold text-white`}>{n}</span>
        <div className={`rounded-lg p-2 ${iconBg}`}>{icon}</div>
      </div>
      <h3 className="font-semibold text-slate-900">{title}</h3>
      <p className="mt-2 text-sm leading-relaxed text-slate-500">{desc}</p>
    </div>
  );
}

function RelBadge({ color, label, desc }: { color: "green" | "red" | "yellow"; label: string; desc: string }) {
  const cls = {
    green: "border-green-200 bg-green-50 text-green-800",
    red: "border-red-200 bg-red-50 text-red-800",
    yellow: "border-yellow-200 bg-yellow-50 text-yellow-800",
  }[color];
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 ${cls}`}>
      <span className="font-bold">{label}</span>
      <span className="opacity-60">—</span>
      <span>{desc}</span>
    </span>
  );
}
