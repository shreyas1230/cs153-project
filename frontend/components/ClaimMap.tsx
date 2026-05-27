"use client";
import { useState } from "react";
import ConflictCard from "./ConflictCard";
import SourceDrawer from "./SourceDrawer";
import { FileText, AlertTriangle, CheckCheck, BookOpen } from "lucide-react";

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

type AnalysisResult = {
  session_id: string;
  question: string;
  papers: string[];
  claims: Claim[];
  term_conflicts: TermConflict[];
  claim_pairs: ClaimPair[];
};

type Filter = "ALL" | "CONTRADICT" | "SUPPORT" | "INCOMMENSURABLE";

export default function ClaimMap({ result }: { result: Record<string, unknown> }) {
  const data = result as AnalysisResult;
  const [selectedPair, setSelectedPair] = useState<ClaimPair | null>(null);
  const [filter, setFilter] = useState<Filter>("ALL");

  const contradicts = data.claim_pairs.filter((p) => p.relationship === "CONTRADICT");
  const supports = data.claim_pairs.filter((p) => p.relationship === "SUPPORT");
  const incomm = data.claim_pairs.filter((p) => p.relationship === "INCOMMENSURABLE");
  const divergentTerms = data.term_conflicts.filter((t) => t.conflict_type === "DIVERGENT");

  const visiblePairs =
    filter === "ALL" ? data.claim_pairs : data.claim_pairs.filter((p) => p.relationship === filter);

  return (
    <div className="flex h-screen flex-col">
      {/* Top bar */}
      <header className="border-b border-gray-200 bg-white px-6 py-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="font-bold text-gray-900">Claim Conflict Detector</h1>
            <p className="mt-0.5 text-sm text-gray-500 max-w-xl line-clamp-2">
              Q: {data.question}
            </p>
          </div>
          <div className="flex shrink-0 gap-4 text-sm">
            <Stat icon={<FileText className="h-4 w-4" />} label="Papers" value={data.papers.length} />
            <Stat icon={<BookOpen className="h-4 w-4" />} label="Claims" value={data.claims.length} />
            <Stat icon={<AlertTriangle className="h-4 w-4 text-red-500" />} label="Contradictions" value={contradicts.length} />
            <Stat icon={<CheckCheck className="h-4 w-4 text-yellow-500" />} label="Term conflicts" value={divergentTerms.length} />
          </div>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Left panel */}
        <div className={`flex flex-col overflow-hidden border-r border-gray-200 bg-white transition-all ${selectedPair ? "w-2/5" : "w-full"}`}>
          {/* Filter tabs */}
          <div className="flex gap-2 border-b border-gray-200 px-4 py-3">
            {(["ALL", "CONTRADICT", "SUPPORT", "INCOMMENSURABLE"] as Filter[]).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`rounded-full px-3 py-1 text-xs font-semibold transition-colors ${
                  filter === f
                    ? "bg-blue-600 text-white"
                    : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                }`}
              >
                {f === "ALL" ? `All (${data.claim_pairs.length})` : f}
              </button>
            ))}
          </div>

          <div className="flex-1 overflow-y-auto">
            {/* Terminology divergences */}
            {divergentTerms.length > 0 && (filter === "ALL") && (
              <section className="px-4 pt-4 pb-2">
                <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-yellow-700">
                  Terminology Divergences ({divergentTerms.length})
                </h2>
                <div className="space-y-2">
                  {divergentTerms.map((tc) => (
                    <div
                      key={tc.term}
                      className="rounded-lg border border-yellow-200 bg-yellow-50 p-3"
                    >
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
            <section className="px-4 pt-4 pb-6 space-y-2">
              <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-gray-500">
                Claim Relationships ({visiblePairs.length})
              </h2>
              {visiblePairs.length === 0 ? (
                <p className="text-sm text-gray-400">No pairs in this category.</p>
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

        {/* Right panel — source drawer */}
        {selectedPair && (
          <div className="w-3/5 overflow-hidden bg-white">
            <SourceDrawer
              pair={selectedPair}
              claims={data.claims}
              onClose={() => setSelectedPair(null)}
            />
          </div>
        )}
      </div>
    </div>
  );
}

function Stat({ icon, label, value }: { icon: React.ReactNode; label: string; value: number }) {
  return (
    <div className="flex items-center gap-1.5">
      {icon}
      <span className="font-semibold text-gray-900">{value}</span>
      <span className="text-gray-500">{label}</span>
    </div>
  );
}
