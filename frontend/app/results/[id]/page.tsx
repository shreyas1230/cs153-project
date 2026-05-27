"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { Loader2 } from "lucide-react";
import ClaimMap from "@/components/ClaimMap";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Status = { status: string; progress: number; error?: string };

export default function ResultsPage() {
  const { id } = useParams<{ id: string }>();
  const [status, setStatus] = useState<Status | null>(null);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [pollError, setPollError] = useState("");

  useEffect(() => {
    if (!id) return;
    const poll = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/status/${id}`);
        const data: Status = await res.json();
        setStatus(data);

        if (data.status === "done") {
          clearInterval(poll);
          const r = await fetch(`${API_BASE}/api/results/${id}`);
          setResult(await r.json());
        } else if (data.status === "error") {
          clearInterval(poll);
          setPollError(data.error ?? "Unknown pipeline error");
        }
      } catch {
        setPollError("Could not reach the server.");
        clearInterval(poll);
      }
    }, 2500);
    return () => clearInterval(poll);
  }, [id]);

  if (pollError) {
    return (
      <main className="flex min-h-screen items-center justify-center">
        <p className="text-red-600">{pollError}</p>
      </main>
    );
  }

  if (!result) {
    const stage = status?.status ?? "starting";
    const progress = status?.progress ?? 0;
    return (
      <main className="flex min-h-screen flex-col items-center justify-center gap-6">
        <Loader2 className="h-10 w-10 animate-spin text-blue-500" />
        <div className="w-64 space-y-2 text-center">
          <p className="font-medium capitalize text-gray-700">{stage.replace("_", " ")}…</p>
          <div className="h-2 w-full rounded-full bg-gray-200">
            <div
              className="h-2 rounded-full bg-blue-500 transition-all duration-700"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      </main>
    );
  }

  return <ClaimMap result={result} />;
}
