"use client";
import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { CheckCircle2, Loader2, Layers, FileText, Search, ArrowLeft, AlertCircle } from "lucide-react";
import ClaimMap from "@/components/ClaimMap";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";

type Status = { status: string; progress: number; error?: string };

const STEPS = [
  {
    key: "extracting",
    icon: FileText,
    title: "Extracting Claims",
    desc: "Reading each paper and identifying atomic scientific assertions with verbatim source passages…",
  },
  {
    key: "normalizing",
    icon: Layers,
    title: "Building Glossary",
    desc: "Comparing how each paper defines key terms, flagging divergent usage across fields…",
  },
  {
    key: "detecting",
    icon: Search,
    title: "Detecting Conflicts",
    desc: "Classifying claim pairs as SUPPORT, CONTRADICT, or INCOMMENSURABLE with explanations…",
  },
];

export default function ResultsPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const [status, setStatus] = useState<Status | null>(null);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [pollError, setPollError] = useState("");

  useEffect(() => {
    if (!id) return;
    let intervalId: ReturnType<typeof setInterval> | null = null;

    const checkStatus = async (): Promise<boolean> => {
      try {
        const res = await fetch(`${API_BASE}/api/status/${id}`);
        const data: Status = await res.json();
        setStatus(data);
        if (data.status === "done") {
          const r = await fetch(`${API_BASE}/api/results/${id}`);
          const resultData = await r.json();
          setResult(resultData);
          // Save to recent analyses so shared links also appear on home page
          const recent = JSON.parse(localStorage.getItem("recent_analyses") ?? "[]");
          if (!recent.find((e: { id: string }) => e.id === id)) {
            const entry = {
              id,
              question: resultData.question ?? "",
              papers: resultData.papers ?? [],
              ts: Date.now(),
            };
            localStorage.setItem("recent_analyses", JSON.stringify([entry, ...recent].slice(0, 5)));
          }
          return true;
        } else if (data.status === "error") {
          setPollError(data.error ?? "Unknown pipeline error");
          return true;
        }
        return false;
      } catch {
        setPollError("Could not reach the server.");
        return true;
      }
    };

    // Check immediately so shared links don't flash the loading screen
    checkStatus().then((done) => {
      if (!done) {
        intervalId = setInterval(async () => {
          const finished = await checkStatus();
          if (finished && intervalId) clearInterval(intervalId);
        }, 2500);
      }
    });

    return () => { if (intervalId) clearInterval(intervalId); };
  }, [id]);

  if (pollError) {
    return (
      <main className="flex min-h-screen flex-col items-center justify-center gap-4 bg-slate-50 px-4">
        <div className="flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 px-6 py-4 text-red-700 max-w-md">
          <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" />
          <div>
            <p className="font-semibold">Analysis failed</p>
            <p className="mt-0.5 text-sm opacity-80">{pollError}</p>
          </div>
        </div>
        <button
          onClick={() => router.push("/")}
          className="flex items-center gap-2 text-sm text-slate-500 hover:text-slate-800 transition-colors"
        >
          <ArrowLeft className="h-4 w-4" /> Start a new analysis
        </button>
      </main>
    );
  }

  if (!result) {
    const currentStage = status?.status ?? "extracting";
    const progress = status?.progress ?? 0;
    const currentIdx = STEPS.findIndex((s) => s.key === currentStage);

    return (
      <main className="flex min-h-screen flex-col items-center justify-center bg-slate-50 px-4 py-12">
        {/* Logo */}
        <div className="mb-10 flex items-center gap-2 text-slate-600">
          <Layers className="h-5 w-5 text-blue-600" />
          <span className="font-bold">ClaimLens</span>
        </div>

        <div className="w-full max-w-md space-y-6">
          <div className="text-center">
            <h1 className="text-xl font-semibold text-slate-900">Analyzing your papers…</h1>
            <p className="mt-1 text-sm text-slate-500">Usually takes 20–60 seconds</p>
          </div>

          {/* Animated step cards */}
          <div className="space-y-3">
            {STEPS.map((step, i) => {
              const Icon = step.icon;
              const done = i < currentIdx;
              const active = step.key === currentStage;
              return (
                <div
                  key={step.key}
                  className={`flex items-start gap-4 rounded-xl border p-4 transition-all duration-500 ${
                    active
                      ? "border-blue-200 bg-blue-50 shadow-sm"
                      : done
                      ? "border-emerald-200 bg-emerald-50"
                      : "border-slate-100 bg-white opacity-40"
                  }`}
                >
                  <div
                    className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full transition-colors ${
                      done ? "bg-emerald-500" : active ? "bg-blue-600" : "bg-slate-200"
                    }`}
                  >
                    {done ? (
                      <CheckCircle2 className="h-5 w-5 text-white" />
                    ) : active ? (
                      <Loader2 className="h-4 w-4 animate-spin text-white" />
                    ) : (
                      <Icon className="h-4 w-4 text-slate-400" />
                    )}
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className={`text-sm font-medium ${done ? "text-emerald-700" : active ? "text-blue-700" : "text-slate-400"}`}>
                      {step.title}
                    </p>
                    {active && <p className="mt-0.5 text-xs leading-relaxed text-blue-500">{step.desc}</p>}
                    {done && <p className="mt-0.5 text-xs text-emerald-600">Complete</p>}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Progress bar */}
          <div className="space-y-1.5">
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-200">
              <div
                className="h-full rounded-full bg-blue-500 transition-all duration-700"
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="text-right text-xs text-slate-400">{progress}%</p>
          </div>
        </div>
      </main>
    );
  }

  return <ClaimMap result={result} />;
}
