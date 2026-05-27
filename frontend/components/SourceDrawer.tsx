"use client";
import { X } from "lucide-react";

type Claim = {
  id: string;
  paper_title: string;
  paper_index: number;
  claim_text: string;
  source_passage: string;
  page_number?: number;
  key_terms: string[];
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

function PassageBlock({ claim }: { claim: Claim }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-gray-50 p-4">
      <div className="mb-2 flex items-center gap-2">
        <span className="font-semibold text-gray-800">{claim.paper_title}</span>
        {claim.page_number != null && (
          <span className="text-xs text-gray-400">p. {claim.page_number}</span>
        )}
      </div>
      <p className="mb-3 text-sm leading-relaxed text-gray-700">{claim.claim_text}</p>
      <blockquote className="border-l-2 border-gray-300 pl-3 text-xs italic text-gray-500">
        {claim.source_passage}
      </blockquote>
      {claim.key_terms.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1">
          {claim.key_terms.map((t) => (
            <span
              key={t}
              className="rounded-full bg-blue-50 px-2 py-0.5 text-xs text-blue-700"
            >
              {t}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export default function SourceDrawer({
  pair,
  claims,
  onClose,
}: {
  pair: ClaimPair;
  claims: Claim[];
  onClose: () => void;
}) {
  const claimMap = Object.fromEntries(claims.map((c) => [c.id, c]));
  const claimA = claimMap[pair.claim_a_id];
  const claimB = claimMap[pair.claim_b_id];

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <div className="flex items-center justify-between border-b border-gray-200 px-6 py-4">
        <h2 className="font-semibold text-gray-900">Source Passages</h2>
        <button onClick={onClose} className="text-gray-400 hover:text-gray-700">
          <X className="h-5 w-5" />
        </button>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto p-6">
        {/* Relationship summary */}
        <div className="rounded-lg bg-white p-4 shadow-sm">
          <p className="text-sm font-medium text-gray-700">
            Relationship:{" "}
            <span
              className={
                pair.relationship === "SUPPORT"
                  ? "text-green-700"
                  : pair.relationship === "CONTRADICT"
                  ? "text-red-700"
                  : "text-yellow-700"
              }
            >
              {pair.relationship}
            </span>
          </p>
          <p className="mt-1 text-sm text-gray-600">{pair.explanation}</p>
          {pair.terminology_note && (
            <p className="mt-2 rounded bg-yellow-50 p-2 text-xs text-yellow-800">
              Terminology note: {pair.terminology_note}
            </p>
          )}
        </div>

        {claimA && <PassageBlock claim={claimA} />}
        {claimB && <PassageBlock claim={claimB} />}
      </div>
    </div>
  );
}
