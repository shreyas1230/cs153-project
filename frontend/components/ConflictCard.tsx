"use client";
import { ChevronRight } from "lucide-react";

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

const STYLES: Record<ClaimPair["relationship"], { border: string; badge: string; label: string; hover: string; selectedRing: string }> = {
  SUPPORT: {
    border: "border-l-4 border-l-green-500",
    badge: "bg-green-100 text-green-800",
    label: "Supports",
    hover: "hover:border-green-300 hover:bg-green-50",
    selectedRing: "ring-2 ring-green-400 bg-green-50",
  },
  CONTRADICT: {
    border: "border-l-4 border-l-red-500",
    badge: "bg-red-100 text-red-800",
    label: "Contradicts",
    hover: "hover:border-red-300 hover:bg-red-50",
    selectedRing: "ring-2 ring-red-400 bg-red-50",
  },
  INCOMMENSURABLE: {
    border: "border-l-4 border-l-yellow-500",
    badge: "bg-yellow-100 text-yellow-800",
    label: "Incommensurable",
    hover: "hover:border-yellow-300 hover:bg-yellow-50",
    selectedRing: "ring-2 ring-yellow-400 bg-yellow-50",
  },
};

export default function ConflictCard({
  pair,
  onSelect,
  selected,
}: {
  pair: ClaimPair;
  onSelect: (pair: ClaimPair) => void;
  selected: boolean;
}) {
  const style = STYLES[pair.relationship];

  return (
    <button
      onClick={() => onSelect(pair)}
      className={`group w-full rounded-lg border bg-white p-4 text-left shadow-sm transition-all duration-150 hover:shadow-md hover:-translate-y-px cursor-pointer ${style.border} ${style.hover} ${selected ? style.selectedRing : ""}`}
    >
      <div className="mb-2 flex items-center gap-2">
        <span className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${style.badge}`}>
          {style.label}
        </span>
        <ChevronRight className="ml-auto h-3.5 w-3.5 text-gray-400 transition-transform group-hover:translate-x-0.5 group-hover:text-gray-600" />
      </div>

      <div className="space-y-1.5 text-sm">
        <div>
          <span className="font-medium text-gray-500">{pair.paper_a_title}: </span>
          <span className="text-gray-800 line-clamp-2">{pair.claim_a_text}</span>
        </div>
        <div>
          <span className="font-medium text-gray-500">{pair.paper_b_title}: </span>
          <span className="text-gray-800 line-clamp-2">{pair.claim_b_text}</span>
        </div>
      </div>

      <p className="mt-2 text-xs text-gray-500 line-clamp-2">{pair.explanation}</p>
    </button>
  );
}
