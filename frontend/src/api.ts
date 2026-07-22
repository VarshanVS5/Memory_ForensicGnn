const API_BASE = "http://localhost:8000";

export interface GraphSchema {
  nodes: { id: string; label: string }[];
  edges: { source: string; target: string }[];
}

export interface AttentionEdge {
  source: string;
  target: string;
  weight: number;
}

export interface PredictionResult {
  prediction: "BENIGN" | "MALICIOUS";
  confidence: number;
  baseline: { prediction: "BENIGN" | "MALICIOUS"; confidence: number };
  attentionEdges: AttentionEdge[];
  features: Record<string, number>;
  index?: number;
  trueLabel?: "BENIGN" | "MALICIOUS";
}

export interface DatasetStats {
  totalSamples: number;
  benign: number;
  malicious: number;
  families: number;
}

export interface GeneralizationSummary {
  held_out_subfamilies: string[];
  in_distribution: { accuracy: number; fnr: number };
  unseen_families: { accuracy: number; fnr: number };
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json();
}

export const fetchGraphSchema = () => get<GraphSchema>("/api/graph-schema");
export const fetchDatasetStats = () => get<DatasetStats>("/api/dataset-stats");
export const fetchGeneralization = () => get<GeneralizationSummary>("/api/generalization").catch(() => null);
export const fetchRandomSample = (model: string) => get<PredictionResult>(`/api/sample/random?model=${model}`);
export const fetchSampleByIndex = (index: number, model: string) =>
  get<PredictionResult>(`/api/sample/${index}?model=${model}`);

export const predictCustomSample = (features: Record<string, number>, model: string = "gat") =>
  post<PredictionResult>("/api/predict", { model, features });

