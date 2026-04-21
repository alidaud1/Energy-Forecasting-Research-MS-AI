export function apiBase(): string {
  return (import.meta as any).env?.VITE_API_BASE || "http://127.0.0.1:8000";
}

export function wsBase(): string {
  const base = apiBase();
  if (base.startsWith("https://")) return "wss://" + base.slice("https://".length);
  if (base.startsWith("http://")) return "ws://" + base.slice("http://".length);
  return base;
}

export async function getJson<T>(path: string): Promise<T> {
  const r = await fetch(`${apiBase()}${path}`);
  if (!r.ok) throw new Error(`GET ${path} failed: ${r.status}`);
  return (await r.json()) as T;
}

export async function postJson<T>(path: string, body: any): Promise<T> {
  const r = await fetch(`${apiBase()}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!r.ok) throw new Error(`POST ${path} failed: ${r.status}`);
  return (await r.json()) as T;
}

export function staticUrl(path: string): string {
  const p = (path || "").replaceAll("\\", "/");
  if (p.startsWith("logs/")) return `${apiBase()}/static/${p.slice("logs/".length)}`;
  return `${apiBase()}/static/${p}`;
}

