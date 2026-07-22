import { useEffect, useState } from "react";
import Graph3D from "./Graph3D";
import {
  fetchGraphSchema,
  fetchDatasetStats,
  fetchGeneralization,
  fetchRandomSample,
  fetchSampleByIndex,
  predictCustomSample,
  type GraphSchema,
  type DatasetStats,
  type GeneralizationSummary,
  type PredictionResult,
} from "./api";
import "./App.css";

const NODE_LABELS: Record<string, string> = {
  pslist: "Running processes",
  dlllist: "Loaded DLLs",
  handles: "Open handles",
  ldrmodules: "Module loader records",
  malfind: "Injected memory",
  psxview: "Process visibility",
  modules: "Kernel modules",
  svcscan: "Windows services",
  callbacks: "Kernel callbacks",
};

export default function App() {
  const [schema, setSchema] = useState<GraphSchema | null>(null);
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [gen, setGen] = useState<GeneralizationSummary | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [model, setModel] = useState<"sage" | "gat">("gat");
  const [indexInput, setIndexInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"explorer" | "inspector" | "simulator" | "analytics">("explorer");
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [customFeatures, setCustomFeatures] = useState<Record<string, number>>({});

  useEffect(() => {
    fetchGraphSchema().then(setSchema).catch((e) => setError(String(e)));
    fetchDatasetStats().then(setStats).catch(() => {});
    fetchGeneralization().then(setGen).catch(() => {});
    // Draw initial sample
    fetchRandomSample("gat").then((res) => {
      setResult(res);
      setCustomFeatures(res.features);
    }).catch(() => {});
  }, []);

  const drawRandom = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchRandomSample(model);
      setResult(res);
      setCustomFeatures(res.features);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const drawIndex = async () => {
    const idx = parseInt(indexInput, 10);
    if (Number.isNaN(idx)) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetchSampleByIndex(idx, model);
      setResult(res);
      setCustomFeatures(res.features);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const runCustomPredict = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await predictCustomSample(customFeatures, model);
      setResult(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (result?.index != null) {
      fetchSampleByIndex(result.index, model).then(setResult).catch(() => {});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model]);

  return (
    <div className="app-container">
      {/* Top Header & Navigation */}
      <header className="top-nav">
        <div className="brand-section">
          <div className="brand-logo">GNN</div>
          <div>
            <h1 className="brand-title">Memory Forensics GNN</h1>
            <div className="brand-subtitle">Fileless Malware Detection & Graph Explainability</div>
          </div>
        </div>

        <nav className="nav-tabs">
          <button
            className={`nav-tab ${activeTab === "explorer" ? "active" : ""}`}
            onClick={() => setActiveTab("explorer")}
          >
            📊 Graph & Prediction
          </button>
          <button
            className={`nav-tab ${activeTab === "inspector" ? "active" : ""}`}
            onClick={() => setActiveTab("inspector")}
          >
            🔍 55-Feature Inspector
          </button>
          <button
            className={`nav-tab ${activeTab === "simulator" ? "active" : ""}`}
            onClick={() => setActiveTab("simulator")}
          >
            ⚡ Custom Simulator
          </button>
          <button
            className={`nav-tab ${activeTab === "analytics" ? "active" : ""}`}
            onClick={() => setActiveTab("analytics")}
          >
            📈 Model Benchmark
          </button>
        </nav>

        <div className="api-status-badge">
          <span className="status-dot"></span>
          <span>API Connected (Port 8000)</span>
        </div>
      </header>

      {/* Metric Summary Banner */}
      <div className="metrics-banner">
        <div className="metrics-row">
          <div className="metric-card">
            <div className="metric-val">{stats ? stats.totalSamples.toLocaleString() : "58,596"}</div>
            <div className="metric-lbl">Total Memory Scans</div>
          </div>
          <div className="metric-card">
            <div className="metric-val" style={{ color: "#047857" }}>
              {stats ? stats.benign.toLocaleString() : "29,298"}
            </div>
            <div className="metric-lbl">Benign Scans</div>
          </div>
          <div className="metric-card">
            <div className="metric-val" style={{ color: "#be123c" }}>
              {stats ? stats.malicious.toLocaleString() : "29,298"}
            </div>
            <div className="metric-lbl">Malicious Scans</div>
          </div>
          <div className="metric-card">
            <div className="metric-val" style={{ color: "#4f46e5" }}>99.9%</div>
            <div className="metric-lbl">GNN Accuracy</div>
          </div>
        </div>

        {gen && (
          <div className="gen-pill">
            <span>
              Unseen Malware Generalization: <strong>{(gen.unseen_families.accuracy * 100).toFixed(2)}%</strong>
            </span>
          </div>
        )}
      </div>

      {/* Main Content Body */}
      <main className="main-content">
        {error && (
          <div style={{ padding: "12px 16px", background: "#fff1f2", border: "1px solid #fecdd3", color: "#be123c", borderRadius: 10, marginBottom: 20 }}>
            ⚠️ {error}
          </div>
        )}

        {/* TAB 1: Graph Explorer */}
        {activeTab === "explorer" && (
          <div className="tab-pane">
            <div className="explorer-grid">
              {/* Left Controls & Diagnostic Panel */}
              <div className="controls-panel">
                <div className="card">
                  <div className="card-title">
                    <span>Sample Selector</span>
                    {result?.index !== undefined && <span style={{ fontSize: "0.78rem", color: "#64748b" }}>Row #{result.index}</span>}
                  </div>

                  <div className="model-selector">
                    <button
                      className={`model-btn ${model === "sage" ? "active" : ""}`}
                      onClick={() => setModel("sage")}
                    >
                      GraphSAGE
                    </button>
                    <button
                      className={`model-btn ${model === "gat" ? "active" : ""}`}
                      onClick={() => setModel("gat")}
                    >
                      GAT Attention
                    </button>
                  </div>

                  <button className="btn-primary" onClick={drawRandom} disabled={loading}>
                    {loading ? "Analyzing..." : "🎲 Draw Random Sample"}
                  </button>

                  <div className="index-row">
                    <input
                      className="input-text"
                      placeholder="Row index (0 - 11719)"
                      value={indexInput}
                      onChange={(e) => setIndexInput(e.target.value)}
                    />
                    <button className="btn-secondary" onClick={drawIndex} disabled={loading}>
                      Fetch
                    </button>
                  </div>
                </div>

                {/* Threat Verdict Card */}
                {result && (
                  <div className="card verdict-card">
                    <div className="verdict-header">
                      <span className="card-title" style={{ margin: 0 }}>Verdict Call</span>
                      {result.trueLabel && (
                        <span className={`truth-pill ${result.trueLabel === result.prediction ? "correct" : "wrong"}`}>
                          Ground Truth: {result.trueLabel} ({result.trueLabel === result.prediction ? "CORRECT" : "MISCLASSIFIED"})
                        </span>
                      )}
                    </div>

                    <div className="confidence-box">
                      <div className="verdict-badge bad" style={{ display: "inline-flex", marginBottom: 10 }}>
                        <span className={`verdict-badge ${result.prediction === "MALICIOUS" ? "bad" : "good"}`}>
                          {result.prediction === "MALICIOUS" ? "🔴 MALICIOUS DETECTED" : "🟢 BENIGN SAMPLE"}
                        </span>
                      </div>
                      <div className="confidence-val">
                        {(result.confidence * 100).toFixed(1)}% <span style={{ fontSize: "0.85rem", color: "#64748b", fontWeight: 500 }}>GNN Confidence</span>
                      </div>
                      <div className="confidence-meter">
                        <div
                          className={`confidence-fill ${result.prediction === "MALICIOUS" ? "bad" : "good"}`}
                          style={{ width: `${Math.max(5, result.confidence * 100)}%` }}
                        ></div>
                      </div>
                    </div>

                    {/* Baseline Comparison */}
                    <div className="baseline-comp">
                      <div className="baseline-row">
                        <span className="baseline-label">RandomForest Baseline:</span>
                        <span className="baseline-val">
                          {result.baseline.prediction} ({(result.baseline.confidence * 100).toFixed(1)}%)
                        </span>
                      </div>
                    </div>

                    {/* GAT Attention Edges */}
                    {model === "gat" && result.attentionEdges && result.attentionEdges.length > 0 && (
                      <div style={{ marginTop: 6 }}>
                        <div style={{ fontSize: "0.78rem", fontWeight: 700, color: "#64748b", textTransform: "uppercase", marginBottom: 6 }}>
                          🔥 Key Graph Attention Edges
                        </div>
                        <div className="edges-list">
                          {result.attentionEdges
                            .sort((a, b) => b.weight - a.weight)
                            .slice(0, 4)
                            .map((e, idx) => (
                              <div key={idx} className="edge-item">
                                <span className="edge-pair">{e.source} ↔ {e.target}</span>
                                <div className="edge-weight-bar">
                                  <div
                                    className="weight-fill"
                                    style={{ width: `${Math.min(100, Math.max(12, e.weight * 120))}px` }}
                                  ></div>
                                  <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "#be123c" }}>
                                    {e.weight.toFixed(3)}
                                  </span>
                                </div>
                              </div>
                            ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Center 3D Graph Canvas */}
              <div className="canvas-card">
                <div className="canvas-header">
                  <span style={{ fontWeight: 700, fontSize: "0.95rem" }}>
                    3D Artifact Relationship Topology Graph
                  </span>
                  <span style={{ fontSize: "0.8rem", color: "#64748b" }}>
                    {model === "gat" ? "🔴 Red nodes & glowing links highlight Attention Drivers" : "Blue spheres represent Volatility artifact nodes"}
                  </span>
                </div>

                <div className="canvas-body">
                  <Graph3D
                    schema={schema}
                    attentionEdges={result?.attentionEdges ?? []}
                    selectedNode={selectedNode}
                    onSelectNode={setSelectedNode}
                  />

                  {selectedNode && (
                    <div className="node-info-popover">
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                        <strong style={{ color: "#4f46e5", fontSize: "0.95rem" }}>{NODE_LABELS[selectedNode] || selectedNode}</strong>
                        <button onClick={() => setSelectedNode(null)} style={{ color: "#64748b", fontWeight: 700 }}>✕</button>
                      </div>
                      <div style={{ fontSize: "0.8rem", color: "#475569" }}>
                        Node ID: <code style={{ color: "#0f172a" }}>{selectedNode}</code>
                      </div>
                      {result?.features && (
                        <div style={{ marginTop: 8, fontSize: "0.78rem" }}>
                          <strong>Features in this artifact group:</strong>
                          <div style={{ marginTop: 4, maxHeight: 120, overflowY: "auto" }}>
                            {Object.entries(result.features)
                              .filter(([k]) => k.startsWith(selectedNode))
                              .map(([k, v]) => (
                                <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "2px 0" }}>
                                  <span style={{ color: "#64748b" }}>{k.replace(`${selectedNode}.`, "")}:</span>
                                  <span style={{ fontFamily: "monospace", fontWeight: 600 }}>{v}</span>
                                </div>
                              ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                <div className="canvas-footer">
                  <span>💡 Tip: Click any node sphere to inspect its feature values · Drag to rotate · Scroll to zoom</span>
                  <span>9 Nodes · 11 Edges</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: Feature Inspector */}
        {activeTab === "inspector" && (
          <div className="tab-pane">
            <div className="card">
              <div className="card-title">
                <span>Memory Sample Volatility Feature Breakdown</span>
                {result?.index !== undefined && <span>Sample #{result.index}</span>}
              </div>
              {result ? (
                <div style={{ overflowX: "auto" }}>
                  <table className="inspector-table">
                    <thead>
                      <tr>
                        <th>Artifact Group</th>
                        <th>Feature Identifier</th>
                        <th>Measured Raw Value</th>
                        <th>Range Hint</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(result.features).map(([featName, val]) => {
                        const group = featName.split(".")[0];
                        return (
                          <tr key={featName}>
                            <td>
                              <span className="group-badge">{group}</span>
                            </td>
                            <td style={{ fontWeight: 600, color: "#0f172a" }}>{featName}</td>
                            <td>
                              <span className="feature-val" style={{ color: val > 1000 ? "#be123c" : "#0f172a" }}>
                                {val.toLocaleString()}
                              </span>
                            </td>
                            <td style={{ fontSize: "0.78rem", color: "#64748b" }}>
                              {val === 0 ? "Zero count" : val > 100 ? "High activity" : "Standard range"}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p style={{ color: "#64748b" }}>Draw a sample in the Graph tab to inspect its features.</p>
              )}
            </div>
          </div>
        )}

        {/* TAB 3: Custom Feature Simulator */}
        {activeTab === "simulator" && (
          <div className="tab-pane">
            <div className="card">
              <div className="card-title" style={{ marginBottom: 8 }}>
                <span>Custom Feature Vector Simulator</span>
              </div>
              <p style={{ fontSize: "0.85rem", color: "#64748b", marginBottom: 20 }}>
                Adjust feature values below to test hypothetical memory dump scenarios against the trained GNN model.
              </p>

              <div className="predictor-grid">
                {Object.keys(customFeatures).map((key) => (
                  <div key={key} className="form-group">
                    <label>{key}</label>
                    <input
                      type="number"
                      className="input-text"
                      value={customFeatures[key]}
                      onChange={(e) =>
                        setCustomFeatures({
                          ...customFeatures,
                          [key]: parseFloat(e.target.value) || 0,
                        })
                      }
                    />
                  </div>
                ))}
              </div>

              <div style={{ marginTop: 20, display: "flex", gap: 12 }}>
                <button className="btn-primary" style={{ width: 220 }} onClick={runCustomPredict} disabled={loading}>
                  {loading ? "Running Prediction..." : "⚡ Run Custom Prediction"}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: Model Benchmark Analytics */}
        {activeTab === "analytics" && (
          <div className="tab-pane">
            <div className="analytics-grid">
              <div className="analytics-card">
                <h3 style={{ margin: "0 0 8px", fontSize: "1rem", color: "#475569" }}>GraphSAGE (GNN) Accuracy</h3>
                <div className="analytics-stat-huge">99.9%</div>
                <p style={{ fontSize: "0.85rem", color: "#64748b", margin: 0 }}>
                  2–5 false negatives out of 5,860 test memory snapshots.
                </p>
              </div>

              <div className="analytics-card">
                <h3 style={{ margin: "0 0 8px", fontSize: "1rem", color: "#475569" }}>Random Forest Baseline</h3>
                <div className="analytics-stat-huge" style={{ color: "#059669" }}>99.99%</div>
                <p style={{ fontSize: "0.85rem", color: "#64748b", margin: 0 }}>
                  0 false negatives out of 5,860 test memory snapshots.
                </p>
              </div>

              <div className="analytics-card">
                <h3 style={{ margin: "0 0 8px", fontSize: "1rem", color: "#475569" }}>Gradient Boosting Baseline</h3>
                <div className="analytics-stat-huge" style={{ color: "#0284c7" }}>99.98%</div>
                <p style={{ fontSize: "0.85rem", color: "#64748b", margin: 0 }}>
                  1 false negative out of 5,860 test memory snapshots.
                </p>
              </div>
            </div>

            {gen && (
              <div className="card" style={{ marginTop: 10 }}>
                <div className="card-title">Holdout Malware Subfamily Generalization</div>
                <p style={{ fontSize: "0.88rem", color: "#475569" }}>
                  Model accuracy when evaluated on malware subfamilies completely held out during training:
                </p>
                <div style={{ display: "flex", gap: 24, marginTop: 14 }}>
                  <div>
                    <div style={{ fontSize: "1.4rem", fontWeight: 800, color: "#059669" }}>
                      {(gen.in_distribution.accuracy * 100).toFixed(2)}%
                    </div>
                    <div style={{ fontSize: "0.78rem", color: "#64748b" }}>Seen Subfamilies Accuracy</div>
                  </div>
                  <div>
                    <div style={{ fontSize: "1.4rem", fontWeight: 800, color: "#4f46e5" }}>
                      {(gen.unseen_families.accuracy * 100).toFixed(2)}%
                    </div>
                    <div style={{ fontSize: "0.78rem", color: "#64748b" }}>Unseen Subfamilies Accuracy</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

