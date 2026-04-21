export type PoolMessage = {
  seq: number;
  topic: string;
  kind: string;
  level: string;
  sender: string;
  sender_role: string;
  run_id: string;
  correlation_id?: string | null;
  payload?: any;
};

export type WsEnvelope =
  | { type: "message"; data: PoolMessage }
  | { type: "ack"; ok: boolean }
  | { type: string; [k: string]: any };

export type PredictionRow = {
  timestamp: string;
  y_true: number;
  y_pred: number;
  ZONEID: number;
  horizon: number;
};

export type PredictionsResponse = {
  run_id: string;
  track: "load" | "solar";
  variant: string;
  model: string;
  model_family: string;
  rows: PredictionRow[];
};

