"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import ConflictCard from "./ConflictCard";
import SourceDrawer from "./SourceDrawer";
import {
  FileText, AlertTriangle, CheckCheck, BookOpen,
  Zap, Download, Link2, ArrowLeft, Layers, Check,
} from "lucide-react";

type Claim = {
  id: string;
  paper_title: string;
  paper_index: number;
  claim_text: string;
  source_passage: string;
  page_number?: number;
  key_terms: string[];
  domain_signals: string[];
};

type ClaimPair = {
  claim_a_id: string;
  claim_b_id: string;
  claim_a_text: string;
  claim_b_text: string;
  paper_a_title: string;
  paper_b_title: string;
  relationship: "SUPPORT" | "CONTRADICT" | "INCOMMENSURABLE";
  explanation: string;
  terminology_note?: string;
};

type TermDefinition = { term: string; paper_index: number; paper_title: string; definition: string };
type TermConflict = {
  term: string;
  conflict_type: "CONSISTENT" | "RELATED" | "DIVERGENT";
  explanation: string;
  definitions: TermDefinition[];
};

type UsageSummary = { total_tokens: number; total_cost_usd: number; llm_calls: number };

type AnalysisResult = {
  session_id: string;
  question: string;
  papers: string[];
  claims: Claim[];
  term_conflicts: TermConflict[];
  claim_pairs: ClaimPair[];
  usage?: UsageSummary;
};

type Filter = "ALL" | "CONTRADICT" | "SUPPORT" | "INCOMMENSURABLE";

export default function ClaimMap({ result }: { result: Record<string, unknown> }) {
  const data = result as AnalysisResult;
  const router = useRouter();
  const [selectedPair, setSelectedPair] = useState<ClaimPair | null>(null);
  const [filter, setFilter] = useState<Filter>("ALL");
  const [copied, setCopied] = useState(false);

  const contradicts = data.claim_pairs.filter((p) => p.relationship === "CONTRADICT");
  const supports = data.claim_pairs.filter((p) => p.relationship === "SUPPORT");
  const incomm = data.claim_pairs.filter((p) => p.relationship === "INCOMMENSURABLE");
  const divergentTerms = data.term_conflicts.filter((t) => t.conflict_type === "DIVERGENT");

  const visiblePairs =
    filter === "ALL" ? data.claim_pairs : data.claim_pairs.filter((p) => p.relationship === filter);

  const claimsPerPaper = data.papers.map((_, idx) => data.claims.filter((c) => c.paper_index === idx).length);

  const handleExport = () => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `claimlens-${data.session_id.slice(0, 8)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleCopyLink = async () => {
    await navigator.clipboard.writeText(window.location.href);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex h-screen flex-col bg-white">
      {/* ── Header ── */}
      <header className="shrink-0 border-b border-slate-200 bg-white px-5 py-3">
        {/* Top row: logo + actions */}
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <button
              onClick={() => router.push("/")}
              className="flex items-center gap-1.5 text-sm text-slate-400 hover:text-slate-700 transition-colors"
            >
              <ArrowLeft className="h-4 w-4" />
            </button>
            <div className="flex items-center gap-1.5">
              <Layers className="h-4 w-4 text-blue-600" />
              <span className="font-bold text-slate-800 text-sm">ClaimLens</span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {data.usage && (
              <div className="flex items-center gap-1.5 rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-500">
                <Zap className="h-3 w-3 text-blue-400" />
                <span>{data.usage.total_tokens.toLocaleString()} tokens</span>
                <span className="text-slate-300">·</span>
                <span className="font-semibold text-slate-700">${data.usage.total_cost_usd.toFixed(4)}</span>
              </div>
            )}
            <button
              onClick={handleCopyLink}
              className="flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-xs text-slate-600 hover:bg-slate-50 transition-colors"
            >
              {copied ? <><Check className="h-3 w-3 text-emerald-500" /> Copied</> : <><Link2 className="h-3 w-3" /> Share</>}
            </button>
            <button
              onClick={handleExport}
              className="flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-xs text-slate-600 hover:bg-slate-50 transition-colors"
            >
              <Download className="h-3 w-3" /> Export
            </button>
          </div>
        </div>

        {/* Question */}
        <p className="mt-2 text-sm font-medium text-slate-700 line-clamp-1">
          Q: {data.question}
        </p>

        {/* Paper chips */}
        <div className="mt-2 flex flex-wrap gap-2">
          {data.papers.map((paper, idx) => (
            <span key={paper} className="flex items-center gap-1.5 rounded-full border border-blue-100 bg-blue-50 px-3 py-0.5 text-xs text-blue-700">
              <FileText className="h-3 w-3" />
              {paper}
              <span className="rounded-full bg-blue-200 px-1.5 py-0.5 text-xs font-semibold text-blue-800">
                {claimsPerPaper[idx]} claims
              </span>
            </span>
          ))}
        </div>

        {/* Summary stats */}
        <div className="mt-3 flex flex-wrap gap-3 text-xs">
          <StatChip icon={<BookOpen className="h-3.5 w-3.5" />} label="Claims" value={data.claims.length} color="slate" />
          <StatChip icon={<AlertTriangle className="h-3.5 w-3.5 text-red-400" />} label="Contradictions" value={contradicts.length} color="red" />
          <StatChip icon={<CheckCheck className="h-3.5 w-3.5 text-emerald-500" />} label="Support" value={supports.length} color="green" />
          <StatChip icon={<AlertTriangle className="h-3.5 w-3.5 text-yellow-400" />} label="Incommensurable" value={incomm.length} color="yellow" />
          <StatChip icon={<Zap className="h-3.5 w-3.5 text-violet-400" />} label="Term conflicts" value={divergentTerms.length} color="violet" />
        </div>
      </header>

      {/* ── Body ── */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left panel */}
        <div className={`flex flex-col overflow-hidden border-r border-slate-200 bg-white transition-all ${selectedPair ? "w-2/5" : "w-full"}`}>
          {/* Filter tabs */}
          <div className="flex gap-2 border-b border-slate-100 px-4 py-2.5">
            {(["ALL", "CONTRADICT", "SUPPORT", "INCOMMENSURABLE"] as Filter[]).map((f) => {
              const count = f === "ALL" ? data.claim_pairs.length
                : f === "CONTRADICT" ? contradicts.length
                : f === "SUPPORT" ? supports.length
                : incomm.length;
              return (
                <button
                  key={f}
                  onClick={() => setFilter(f)}
                  className={`rounded-full px-3 py-1 text-xs font-semibold transition-colors ${
                    filter === f ? "bg-blue-600 text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                  }`}
                >
                  {f === "ALL" ? "All" : f.charAt(0) + f.slice(1).toLowerCase()} ({count})
                </button>
              );
            })}
          </div>

          <div className="flex-1 overflow-y-auto">
            {/* Terminology divergences */}
            {divergentTerms.length > 0 && filter === "ALL" && (
              <section className="px-4 pt-4 pb-2">
                <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-yellow-700">
                  Terminology Divergences ({divergentTerms.length})
                </h2>
                <div className="space-y-2">
                  {divergentTerms.map((tc) => (
                    <div key={tc.term} className="rounded-lg border border-yellow-200 bg-yellow-50 p-3">
                      <p className="font-semibold text-yellow-900">"{tc.term}"</p>
                      <p className="mt-0.5 text-xs text-yellow-700">{tc.explanation}</p>
                      <div className="mt-2 space-y-1">
                        {tc.definitions.map((d) => (
                          <p key={d.paper_index} className="text-xs text-yellow-800">
                            <span className="font-medium">{d.paper_title}:</span> {d.definition}
                          </p>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* Claim pairs */}
            <section className="space-y-2 px-4 pb-6 pt-4">
              <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
                Claim Relationships ({visiblePairs.length})
              </h2>
              {visiblePairs.length === 0 ? (
                <p className="text-sm text-slate-400">No pairs in this category.</p>
              ) : (
                visiblePairs.map((pair) => (
                  <ConflictCard
                    key={`${pair.claim_a_id}-${pair.claim_b_id}`}
                    pair={pair}
                    onSelect={setSelectedPair}
                    selected={
                      selectedPair?.claim_a_id === pair.claim_a_id &&
                      selectedPair?.claim_b_id === pair.claim_b_id
                    }
                  />
                ))
              )}
            </section>
          </div>
        </div>

        {/* Right panel */}
        {selectedPair && (
          <div className="w-3/5 overflow-hidden bg-white">
            <SourceDrawer pair={selectedPair} claims={data.claims} onClose={() => setSelectedPair(null)} />
          </div>
        )}
      </div>
    </div>
  );
}

function StatChip({ icon, label, value, color }: {
  icon: React.ReactNode; label: string; value: number;
  color: "slate" | "red" | "green" | "yellow" | "violet";
}) {
  const cls = {
    slate: "bg-slate-100 text-slate-600",
    red: "bg-red-50 text-red-700",
    green: "bg-emerald-50 text-emerald-700",
    yellow: "bg-yellow-50 text-yellow-700",
    violet: "bg-violet-50 text-violet-700",
  }[color];
  return (
    <span className={`flex items-center gap-1 rounded-full px-2.5 py-1 font-medium ${cls}`}>
      {icon} {value} {label}
    </span>
  );
}
