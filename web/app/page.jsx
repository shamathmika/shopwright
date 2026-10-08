"use client";

import { useState } from "react";

function parseEvents(buffer) {
  const events = [];
  const blocks = buffer.split("\n\n");
  const rest = blocks.pop();
  for (const block of blocks) {
    const event = block.match(/^event: (.*)$/m)?.[1];
    const data = block.match(/^data: (.*)$/m)?.[1];
    if (event && data) events.push({ event, data: JSON.parse(data) });
  }
  return { events, rest };
}

export default function Home() {
  const [question, setQuestion] = useState("Sony headphones under $100 with good reviews for comfort");
  const [steps, setSteps] = useState([]);
  const [answer, setAnswer] = useState("");
  const [stats, setStats] = useState(null);
  const [busy, setBusy] = useState(false);

  async function ask(e) {
    e.preventDefault();
    setSteps([]);
    setAnswer("");
    setStats(null);
    setBusy(true);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const { events, rest } = parseEvents(buffer);
        buffer = rest;
        for (const { event, data } of events) {
          if (event === "tool_call") setSteps((s) => [...s, `Calling ${data.tool} ${JSON.stringify(data.args)}`]);
          if (event === "tool_result") setSteps((s) => [...s, `${data.tool} ${data.error ? "failed" : "done"} in ${data.ms} ms`]);
          if (event === "token") setAnswer((a) => a + data.text);
          if (event === "done") setStats(data);
          if (event === "error") setAnswer((a) => `${a}\n[error] ${data.message}`);
        }
      }
    } catch (err) {
      setAnswer(`[error] ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <h1>ShopWright</h1>
      <form onSubmit={ask} style={{ display: "flex", gap: 8 }}>
        <input value={question} onChange={(e) => setQuestion(e.target.value)} style={{ flex: 1, padding: 8 }} />
        <button disabled={busy || !question.trim()} style={{ padding: "8px 16px" }}>{busy ? "Thinking…" : "Ask"}</button>
      </form>
      {steps.length > 0 && (
        <ol style={{ color: "#666", fontSize: 14 }}>
          {steps.map((s, i) => <li key={i}>{s}</li>)}
        </ol>
      )}
      <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.5 }}>{answer}</div>
      {stats && <p style={{ color: "#666", fontSize: 13 }}>{stats.steps} steps · {(stats.latency_ms / 1000).toFixed(1)} s</p>}
    </main>
  );
}
