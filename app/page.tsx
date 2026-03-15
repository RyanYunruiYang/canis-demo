"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import data from "./data.json";

type SystemKey = keyof typeof data.systems;

const ATTEMPTS = [
  {
    name: "v1: Single-GPU cuDF",
    result: "6% slower than Sirius",
    status: "baseline" as const,
    desc: "Direct cuDF operations on one H200. Simple but limited by single-GPU memory bandwidth.",
  },
  {
    name: "v2: Dask Distributed",
    result: "Abandoned — 2-3s scheduler overhead",
    status: "failed" as const,
    desc: "Dask DataFrame API across 4 GPUs. Task graph compilation and scheduler round-trips killed sub-ms queries.",
  },
  {
    name: "v3: Multiprocessing + Pipes",
    result: "#1 on ClickBench",
    status: "winner" as const,
    desc: "One process per GPU, cuDF inside each, tiny results merged on CPU. Zero Dask overhead, consistent sub-ms latency.",
  },
  {
    name: "v4: Custom C++ CUDA (single GPU)",
    result: "8/43 queries, scaffolding only",
    status: "partial" as const,
    desc: "Hand-written fused scan/reduce CUDA kernels with CUB warp reductions. Fast but too narrow.",
  },
  {
    name: "v5: Custom C++ CUDA (multi-GPU)",
    result: "8/43 queries, not production-ready",
    status: "partial" as const,
    desc: "Extended v4 to multiple GPUs with cudaMemcpyPeer. Only trivial scan queries implemented.",
  },
];

const STACK = [
  { label: "GPUs", value: "4x NVIDIA H200 141GB (NVLink)" },
  { label: "Framework", value: "RAPIDS cuDF 26.2 (Python)" },
  { label: "Parallelism", value: "multiprocessing.spawn, one process per GPU" },
  { label: "IPC", value: "pickle over Pipe (tiny reduced results only)" },
  { label: "Data format", value: "Apache Parquet (read directly, no conversion)" },
  { label: "CUDA", value: "12.5 (via cuDF, no custom kernels in final version)" },
];

// --- Race ---

const TIME_SCALE = 8;
const MIN_QUERY_MS = 30;

function formatMs(ms: number): string {
  if (ms < 1) return "<1ms";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function Race({ leftKey, rightKey }: { leftKey: SystemKey; rightKey: SystemKey }) {
  const left = data.systems[leftKey];
  const right = data.systems[rightKey];

  const [leftProgress, setLeftProgress] = useState(0);
  const [rightProgress, setRightProgress] = useState(0);
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(false);
  const animRef = useRef<number | null>(null);
  const startRef = useRef(0);
  const leftCum = useRef<number[]>([]);
  const rightCum = useRef<number[]>([]);

  useEffect(() => {
    let s = 0;
    leftCum.current = left.times.map((t) => { s += Math.max(t / TIME_SCALE, MIN_QUERY_MS); return s; });
    s = 0;
    rightCum.current = right.times.map((t) => { s += Math.max(t / TIME_SCALE, MIN_QUERY_MS); return s; });
  }, [left, right]);

  const start = useCallback(() => {
    setLeftProgress(0);
    setRightProgress(0);
    setRunning(true);
    setDone(false);
    startRef.current = performance.now();

    const tick = (now: number) => {
      const elapsed = now - startRef.current;
      let l = 0, r = 0;
      for (let i = 0; i < 43; i++) { if (elapsed >= leftCum.current[i]) l = i + 1; else break; }
      for (let i = 0; i < 43; i++) { if (elapsed >= rightCum.current[i]) r = i + 1; else break; }
      setLeftProgress(l);
      setRightProgress(r);
      if (l >= 43 && r >= 43) { setRunning(false); setDone(true); }
      else animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);
  }, [left, right]);

  useEffect(() => () => { if (animRef.current) cancelAnimationFrame(animRef.current); }, []);

  const leftTotal = left.times.reduce((a, b) => a + b, 0);
  const rightTotal = right.times.reduce((a, b) => a + b, 0);
  const leftWins = leftTotal <= rightTotal;
  const speedup = leftWins ? rightTotal / leftTotal : leftTotal / rightTotal;

  return (
    <div className="w-full max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="text-sm font-semibold text-green-400">{left.name}</div>
          <div className="text-xs text-zinc-500">{left.hw}</div>
        </div>
        <button
          onClick={start}
          disabled={running}
          className="px-6 py-2 bg-white text-black font-bold rounded-lg hover:bg-zinc-200 disabled:opacity-30 disabled:cursor-not-allowed text-sm"
        >
          {done ? "Race Again" : running ? "Racing..." : "Start Race"}
        </button>
        <div className="text-right">
          <div className="text-sm font-semibold text-blue-400">{right.name}</div>
          <div className="text-xs text-zinc-500">{right.hw}</div>
        </div>
      </div>

      {/* Progress bars */}
      <div className="flex items-center gap-3 mb-3 px-1">
        <div className="flex-1">
          <div className="flex items-center justify-between mb-1">
            <span className="text-green-400 font-mono text-sm">{leftProgress}/43</span>
            {leftProgress > 0 && (
              <span className="text-zinc-500 text-xs font-mono">
                {formatMs(left.times.slice(0, leftProgress).reduce((a, b) => a + b, 0))}
              </span>
            )}
          </div>
          <div className="h-3 bg-zinc-900 rounded-full overflow-hidden border border-zinc-800">
            <div
              className="h-full bg-green-500 rounded-full transition-all duration-75"
              style={{ width: `${(leftProgress / 43) * 100}%` }}
            />
          </div>
        </div>
        <div className="flex-1">
          <div className="flex items-center justify-between mb-1">
            <span className="text-blue-400 font-mono text-sm">{rightProgress}/43</span>
            {rightProgress > 0 && (
              <span className="text-zinc-500 text-xs font-mono">
                {formatMs(right.times.slice(0, rightProgress).reduce((a, b) => a + b, 0))}
              </span>
            )}
          </div>
          <div className="h-3 bg-zinc-900 rounded-full overflow-hidden border border-zinc-800">
            <div
              className="h-full bg-blue-500 rounded-full transition-all duration-75"
              style={{ width: `${(rightProgress / 43) * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Per-query bars */}
      <div className="bg-zinc-900/50 rounded-xl p-3 border border-zinc-800 max-h-[520px] overflow-y-auto">
        {data.queries.map((q, i) => {
          const lDone = i < leftProgress;
          const rDone = i < rightProgress;
          const maxMs = Math.max(left.times[i], right.times[i], 1);
          const lPct = (left.times[i] / maxMs) * 100;
          const rPct = (right.times[i] / maxMs) * 100;
          const bothDone = lDone && rDone;
          const lWon = bothDone && left.times[i] <= right.times[i];
          const rWon = bothDone && right.times[i] <= left.times[i];
          const isCurrent = i === Math.max(leftProgress, rightProgress);

          return (
            <div key={q.id} className={`flex items-center gap-2 py-[3px] ${isCurrent && running ? "bg-white/5 rounded" : ""}`}>
              <span className="w-8 text-[10px] text-zinc-600 text-right font-mono shrink-0">{q.id}</span>
              <div className="flex-1 flex justify-end items-center gap-1">
                {lDone && <span className="text-[9px] text-zinc-600 font-mono">{formatMs(left.times[i])}</span>}
                <div
                  className={`h-2.5 rounded-sm transition-all duration-75 ${lWon ? "bg-green-500" : lDone ? "bg-red-400/60" : "bg-zinc-800"}`}
                  style={{ width: lDone ? `${Math.max(lPct, 4)}%` : "0%", maxWidth: "75%" }}
                />
              </div>
              <div className="w-px h-3 bg-zinc-800 shrink-0" />
              <div className="flex-1 flex items-center gap-1">
                <div
                  className={`h-2.5 rounded-sm transition-all duration-75 ${rWon ? "bg-green-500" : rDone ? "bg-red-400/60" : "bg-zinc-800"}`}
                  style={{ width: rDone ? `${Math.max(rPct, 4)}%` : "0%", maxWidth: "75%" }}
                />
                {rDone && <span className="text-[9px] text-zinc-600 font-mono">{formatMs(right.times[i])}</span>}
              </div>
              <span className="w-28 text-[9px] text-zinc-700 truncate shrink-0">{q.desc}</span>
            </div>
          );
        })}
      </div>

      {done && (
        <div className="mt-4 text-center">
          <span className={`text-xl font-bold ${leftWins ? "text-green-400" : "text-blue-400"}`}>
            {leftWins ? left.name : right.name} wins!
          </span>
          <span className="text-zinc-500 ml-2 text-sm">
            {speedup.toFixed(1)}x faster total query time
          </span>
        </div>
      )}
    </div>
  );
}

// --- Components ---

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-zinc-900 rounded-xl p-5 border border-zinc-800 text-center">
      <div className="text-zinc-500 text-[10px] uppercase tracking-wider mb-1">{label}</div>
      <div className="text-3xl font-bold font-mono">{value}</div>
      {sub && <div className="text-zinc-500 text-[10px] mt-1">{sub}</div>}
    </div>
  );
}

function Leaderboard() {
  const entries = [
    { name: "RAPIDS-Bench", hw: "4x NVIDIA H200", hot: 1.32, cold: 0.77, load: "6.5s", size: "14.8 GB", combined: 1.43, ours: true },
    { name: "Sirius", hw: "1x NVIDIA GH200", hot: 1.45, cold: 52.8, load: "26.3s", size: "26.9 GB", combined: 2.86, ours: false },
    { name: "ClickHouse (web)", hw: "c8g.metal-48xl", hot: 3.27, cold: 8.47, load: "0s", size: "14.6 GB", combined: 3.06, ours: false },
    { name: "ClickHouse Cloud", hw: "2x 236GiB (GCP)", hot: 3.9, cold: 5.82, load: "11.5s", size: "10.2 GB", combined: 3.06, ours: false },
    { name: "MotherDuck", hw: "mega", hot: 3.53, cold: 2.56, load: "60.7s", size: "23.6 GB", combined: 3.16, ours: false },
  ];
  const maxC = Math.max(...entries.map((e) => e.combined));

  return (
    <div className="w-full max-w-3xl mx-auto space-y-2">
      {entries.map((e, i) => (
        <div key={i} className="flex items-center gap-3">
          <span className="text-zinc-500 text-sm w-5 text-right font-mono">{i + 1}</span>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className={`text-sm font-medium ${e.ours ? "text-green-400" : "text-zinc-300"}`}>{e.name}</span>
              <span className="text-[10px] text-zinc-600">{e.hw}</span>
              <span className="ml-auto text-[10px] text-zinc-600 font-mono hidden md:inline">
                hot={e.hot} cold={e.cold} load={e.load} size={e.size}
              </span>
            </div>
            <div className="h-5 bg-zinc-900 rounded overflow-hidden border border-zinc-800">
              <div
                className={`h-full rounded ${e.ours ? "bg-green-500" : "bg-zinc-700"}`}
                style={{ width: `${(e.combined / maxC) * 100}%` }}
              />
            </div>
          </div>
          <span className={`font-mono text-sm w-14 text-right ${e.ours ? "text-green-400" : "text-zinc-400"}`}>
            x{e.combined.toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  );
}

export default function Home() {
  const [raceRight, setRaceRight] = useState<SystemKey>("clickhouse");
  const [showAllQueries, setShowAllQueries] = useState(false);

  const displayedQueries = showAllQueries ? data.queries : data.queries.slice(0, 10);

  return (
    <main className="min-h-screen bg-black">
      {/* Hero */}
      <section className="pt-20 pb-16 px-6 text-center">
        <div className="inline-block bg-green-500/10 border border-green-500/30 rounded-full px-4 py-1 mb-6">
          <span className="text-green-400 text-sm font-medium">#1 out of 676 systems</span>
        </div>
        <h1 className="text-6xl md:text-8xl font-bold tracking-tight mb-4">
          <span className="text-green-400">#1</span> on ClickBench
        </h1>
        <p className="text-zinc-400 text-lg md:text-xl max-w-2xl mx-auto mb-4">
          Multi-GPU analytics with RAPIDS cuDF on 4x NVIDIA H200.<br />
          43 queries on 100M rows. Faster than every database on the leaderboard.
        </p>
        <p className="text-zinc-600 text-sm">March 2026</p>
      </section>

      {/* Stats */}
      <section className="pb-16 px-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 max-w-3xl mx-auto">
          <StatCard label="Combined Score" value="x1.43" sub="weighted geo mean (lower is better)" />
          <StatCard label="Hot Run" value="x1.32" sub="vs per-query best across all systems" />
          <StatCard label="Load Time" value="6.5s" sub="14.8 GB parquet into 4 GPUs" />
          <StatCard label="Total Queries" value="43/43" sub="100M rows, 105 columns" />
        </div>
      </section>

      {/* What is ClickBench */}
      <section className="pb-16 px-6">
        <div className="max-w-3xl mx-auto">
          <h2 className="text-2xl font-bold mb-4">What is ClickBench?</h2>
          <div className="text-zinc-400 text-sm space-y-3">
            <p>
              <a href="https://benchmark.clickhouse.com" className="text-zinc-300 underline" target="_blank" rel="noopener">ClickBench</a> is
              the standard benchmark for analytical databases, maintained by ClickHouse.
              It runs <strong className="text-zinc-200">43 SQL queries</strong> on a real-world web analytics dataset
              of <strong className="text-zinc-200">100 million rows</strong> with 105 columns (~14 GB as Parquet).
            </p>
            <p>
              The queries range from simple scans (<code className="text-zinc-300 bg-zinc-800 px-1 rounded text-xs">COUNT(*)</code>) to
              complex multi-column GROUP BYs with string filtering, DISTINCT counts, and REGEXP operations.
              Every major database has an entry: ClickHouse, DuckDB, PostgreSQL, Snowflake, BigQuery, and 670+ more.
            </p>
            <p>
              Scoring uses a <strong className="text-zinc-200">geometric mean of ratios</strong> to the per-query best across all systems,
              with a 10ms constant shift to prevent log(0). The combined score weights:
              60% hot run, 20% cold run, 10% load time, 10% data size. Lower is better.
            </p>
          </div>
        </div>
      </section>

      {/* Queries */}
      <section className="pb-16 px-6">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-2xl font-bold mb-2">The 43 Queries</h2>
          <p className="text-zinc-500 text-sm mb-4">
            Real SQL from production web analytics. Categories: scans, GROUP BY, string search, sorting, filtered analytics.
          </p>
          <div className="bg-zinc-900/50 rounded-xl border border-zinc-800 overflow-hidden">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-zinc-800 text-zinc-500">
                  <th className="px-3 py-2 text-left w-12">#</th>
                  <th className="px-3 py-2 text-left">SQL</th>
                  <th className="px-3 py-2 text-right w-20">Our time</th>
                  <th className="px-3 py-2 text-left w-16">Type</th>
                </tr>
              </thead>
              <tbody>
                {displayedQueries.map((q, i) => {
                  const idx = data.queries.indexOf(q);
                  const ms = data.systems.rapids.times[idx];
                  return (
                    <tr key={q.id} className="border-b border-zinc-800/50 hover:bg-white/[0.02]">
                      <td className="px-3 py-1.5 font-mono text-zinc-500">{q.id}</td>
                      <td className="px-3 py-1.5">
                        <code className="text-zinc-400 break-all">{q.sqlShort}</code>
                      </td>
                      <td className="px-3 py-1.5 text-right font-mono text-green-400">{formatMs(ms)}</td>
                      <td className="px-3 py-1.5">
                        <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                          q.category === "scan" ? "bg-blue-500/10 text-blue-400" :
                          q.category === "groupby" ? "bg-purple-500/10 text-purple-400" :
                          q.category === "string" ? "bg-yellow-500/10 text-yellow-400" :
                          q.category === "sort" ? "bg-cyan-500/10 text-cyan-400" :
                          q.category === "filtered" ? "bg-orange-500/10 text-orange-400" :
                          q.category === "distinct" ? "bg-pink-500/10 text-pink-400" :
                          q.category === "compute" ? "bg-red-500/10 text-red-400" :
                          "bg-zinc-500/10 text-zinc-400"
                        }`}>
                          {q.category}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {!showAllQueries && (
              <button
                onClick={() => setShowAllQueries(true)}
                className="w-full py-2 text-xs text-zinc-500 hover:text-zinc-300 border-t border-zinc-800"
              >
                Show all 43 queries...
              </button>
            )}
          </div>
        </div>
      </section>

      {/* Leaderboard */}
      <section className="pb-20 px-6">
        <h2 className="text-2xl font-bold text-center mb-2">Leaderboard (Combined Score)</h2>
        <p className="text-zinc-500 text-sm text-center mb-8">
          Lower is better. Weighted geometric mean across hot run, cold run, load time, and data size.
        </p>
        <Leaderboard />
      </section>

      {/* Race */}
      <section className="pb-20 px-6">
        <h2 className="text-2xl font-bold text-center mb-2">Head-to-Head Race</h2>
        <p className="text-zinc-500 text-sm text-center mb-6">
          All 43 ClickBench queries, simulated at real measured hot-run speeds.<br />
          Green bar = winner on that query. Timings are real benchmark data.
        </p>
        <div className="flex justify-center gap-2 mb-6">
          <span className="text-zinc-500 text-sm self-center">Race against:</span>
          {(["sirius", "clickhouse", "duckdb", "motherduck"] as SystemKey[]).map((k) => (
            <button
              key={k}
              onClick={() => setRaceRight(k)}
              className={`px-3 py-1.5 rounded-lg text-sm transition-all ${
                raceRight === k
                  ? "bg-blue-500/20 text-blue-400 border border-blue-500/40"
                  : "bg-zinc-900 text-zinc-500 border border-zinc-800 hover:border-zinc-600"
              }`}
            >
              {data.systems[k].name}
            </button>
          ))}
        </div>
        <Race key={raceRight} leftKey="rapids" rightKey={raceRight} />
      </section>

      {/* Architecture: What We Tried */}
      <section className="pb-16 px-6">
        <h2 className="text-2xl font-bold text-center mb-2">What We Tried</h2>
        <p className="text-zinc-500 text-sm text-center mb-8">
          Five approaches to multi-GPU analytics. Only one made #1.
        </p>
        <div className="max-w-3xl mx-auto space-y-3">
          {ATTEMPTS.map((a, i) => (
            <div
              key={i}
              className={`rounded-xl p-4 border ${
                a.status === "winner"
                  ? "bg-green-500/5 border-green-500/30"
                  : a.status === "failed"
                  ? "bg-zinc-900/50 border-zinc-800 opacity-60"
                  : "bg-zinc-900/50 border-zinc-800"
              }`}
            >
              <div className="flex items-center gap-3 mb-1">
                <span className={`text-sm font-medium ${a.status === "winner" ? "text-green-400" : "text-zinc-300"}`}>
                  {a.name}
                </span>
                <span className={`ml-auto text-xs font-mono ${
                  a.status === "winner" ? "text-green-400" :
                  a.status === "failed" ? "text-red-400/70" :
                  a.status === "baseline" ? "text-zinc-400" : "text-zinc-500"
                }`}>
                  {a.result}
                </span>
              </div>
              <div className="text-xs text-zinc-500">{a.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Architecture: Winning Stack */}
      <section className="pb-16 px-6">
        <h2 className="text-2xl font-bold text-center mb-2">Winning Architecture</h2>
        <p className="text-zinc-500 text-sm text-center mb-8">
          The key insight: reduce locally on each GPU, only send tiny results over IPC.
        </p>

        {/* Diagram */}
        <div className="w-full max-w-3xl mx-auto bg-zinc-900/50 rounded-xl border border-zinc-800 p-6 mb-6">
          <div className="grid grid-cols-4 gap-3 mb-4">
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="bg-zinc-800 rounded-lg p-3 text-center border border-zinc-700">
                <div className="text-green-400 font-mono text-xs mb-1">GPU {i}</div>
                <div className="text-[10px] text-zinc-500">H200 141GB</div>
                <div className="mt-2 space-y-1">
                  <div className="bg-zinc-900 rounded px-2 py-0.5 text-[10px] text-zinc-400">25M rows</div>
                  <div className="bg-zinc-900 rounded px-2 py-0.5 text-[10px] text-zinc-400">cuDF query</div>
                  <div className="bg-green-500/10 rounded px-2 py-0.5 text-[10px] text-green-400">local reduce</div>
                </div>
              </div>
            ))}
          </div>
          <div className="flex items-center justify-center gap-2 text-zinc-600">
            <div className="h-px flex-1 bg-zinc-700" />
            <span className="text-[10px]">scalars / top-K rows via pickle Pipe</span>
            <div className="h-px flex-1 bg-zinc-700" />
          </div>
          <div className="mt-3 bg-zinc-800 rounded-lg p-3 text-center border border-zinc-700">
            <div className="text-zinc-300 font-mono text-xs">Coordinator (CPU)</div>
            <div className="text-[10px] text-zinc-500 mt-1">merge 4 tiny partials &rarr; final answer</div>
          </div>
        </div>

        {/* Stack table */}
        <div className="max-w-2xl mx-auto">
          <div className="bg-zinc-900/50 rounded-xl border border-zinc-800 overflow-hidden">
            {STACK.map((s, i) => (
              <div key={i} className={`flex items-center px-4 py-2.5 text-sm ${i > 0 ? "border-t border-zinc-800/50" : ""}`}>
                <span className="text-zinc-500 w-28 shrink-0 text-xs">{s.label}</span>
                <span className="text-zinc-300 font-mono text-xs">{s.value}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Key Insights */}
      <section className="pb-20 px-6">
        <h2 className="text-2xl font-bold text-center mb-8">Key Insights</h2>
        <div className="max-w-2xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            { title: "Python + cuDF > custom C++ CUDA", desc: "High-level GPU libraries, when parallelized correctly, outperform hand-written kernels with 10x less code." },
            { title: "Reduce locally, never shuffle", desc: "Each GPU returns scalars or top-K rows over IPC. The coordinator merges kilobytes, not gigabytes. Zero NVLink needed." },
            { title: "One line change = 88x speedup", desc: ".nlargest(10) on GPU before .to_pandas(). Avoids copying 56M rows to CPU for a 10-row result." },
            { title: "Eager load > lazy load", desc: "Load all columns once at startup (6.5s). Sirius lazy-loads per query, paying 0.3-70s cold start each time." },
          ].map((item, i) => (
            <div key={i} className="bg-zinc-900/50 rounded-xl p-4 border border-zinc-800">
              <div className="text-sm font-medium text-zinc-200 mb-1">{item.title}</div>
              <div className="text-xs text-zinc-500 leading-relaxed">{item.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="pb-12 text-center text-zinc-600 text-xs space-y-1">
        <p>
          Built with RAPIDS cuDF on 4x NVIDIA H200. Benchmark data from{" "}
          <a href="https://benchmark.clickhouse.com" className="text-zinc-500 underline" target="_blank" rel="noopener">ClickBench</a>.
        </p>
        <p>All timings are real measured values from hot runs (min of 2nd and 3rd execution).</p>
      </footer>
    </main>
  );
}
