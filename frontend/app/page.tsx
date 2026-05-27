"use client";
import { useState, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import { Upload, FileText, X, Loader2 } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function HomePage() {
  const router = useRouter();
  const [files, setFiles] = useState<File[]>([]);
  const [question, setQuestion] = useState("");
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const addFiles = (incoming: FileList | File[]) => {
    const pdfs = Array.from(incoming).filter((f) => f.type === "application/pdf");
    setFiles((prev) => {
      const names = new Set(prev.map((f) => f.name));
      return [...prev, ...pdfs.filter((f) => !names.has(f.name))].slice(0, 10);
    });
  };

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      addFiles(e.dataTransfer.files);
    },
    [],
  );

  const removeFile = (name: string) =>
    setFiles((prev) => prev.filter((f) => f.name !== name));

  const submit = async () => {
    if (files.length < 2) {
      setError("Upload at least 2 PDFs to compare.");
      return;
    }
    if (!question.trim()) {
      setError("Enter a research question.");
      return;
    }
    setError("");
    setLoading(true);

    try {
      const form = new FormData();
      form.append("question", question.trim());
      files.forEach((f) => form.append("files", f));

      const res = await fetch(`${API_BASE}/api/analyze`, { method: "POST", body: form });
      if (!res.ok) {
        const detail = (await res.json()).detail ?? "Server error";
        throw new Error(detail);
      }
      const { session_id } = await res.json();
      router.push(`/results/${session_id}`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen flex flex-col items-center justify-center px-4 py-16">
      <div className="w-full max-w-2xl space-y-8">
        {/* Header */}
        <div className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight">Claim Conflict Detector</h1>
          <p className="text-gray-500 text-lg">
            Upload scientific papers and ask a question. Discover where they disagree —
            and whether those disagreements are real or just terminology drift.
          </p>
        </div>

        {/* Drop zone */}
        <div
          onDrop={onDrop}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onClick={() => inputRef.current?.click()}
          className={`cursor-pointer rounded-xl border-2 border-dashed p-10 text-center transition-colors ${
            dragging ? "border-blue-500 bg-blue-50" : "border-gray-300 bg-white hover:border-gray-400"
          }`}
        >
          <Upload className="mx-auto mb-3 h-8 w-8 text-gray-400" />
          <p className="font-medium text-gray-700">Drop PDFs here or click to upload</p>
          <p className="mt-1 text-sm text-gray-400">2–10 papers · PDF only</p>
          <input
            ref={inputRef}
            type="file"
            accept="application/pdf"
            multiple
            className="hidden"
            onChange={(e) => e.target.files && addFiles(e.target.files)}
          />
        </div>

        {/* File list */}
        {files.length > 0 && (
          <ul className="space-y-2">
            {files.map((f) => (
              <li
                key={f.name}
                className="flex items-center gap-3 rounded-lg border border-gray-200 bg-white px-4 py-3"
              >
                <FileText className="h-4 w-4 shrink-0 text-gray-400" />
                <span className="flex-1 truncate text-sm text-gray-700">{f.name}</span>
                <button onClick={() => removeFile(f.name)} className="text-gray-400 hover:text-red-500">
                  <X className="h-4 w-4" />
                </button>
              </li>
            ))}
          </ul>
        )}

        {/* Question input */}
        <div>
          <label className="mb-1 block text-sm font-medium text-gray-700">
            Research question
          </label>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            rows={3}
            placeholder="What question are you trying to answer across these papers? e.g. 'Does attention improve neural network performance?'"
            className="w-full rounded-lg border border-gray-300 px-4 py-3 text-sm placeholder-gray-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>

        {error && <p className="text-sm text-red-600">{error}</p>}

        <button
          onClick={submit}
          disabled={loading}
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-6 py-3 font-semibold text-white transition-colors hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Submitting…
            </>
          ) : (
            "Analyze Papers"
          )}
        </button>
      </div>
    </main>
  );
}
