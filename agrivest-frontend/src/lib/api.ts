/**
 * AgriVest – TypeScript API client  (v2 — Agro-Climatic Advisory Engine)
 * All fetch calls to the FastAPI backend live here.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

// ---------------------------------------------------------------------------
// Types (mirror the Pydantic models in app/main.py)
// ---------------------------------------------------------------------------

export interface PortfolioRequest {
  state: string;
  district: string;
  season: string;
  land_area_ha: number;
  survival_budget_inr: number;
  irrigation_type: string;
  risk_aversion?: number;
  // Optional weather overrides — omit to let the backend auto-detect
  rainfall_mm?: number | null;
  temperature_c?: number | null;
  climate_regime?: string | null;
  available_crops: string[];
}

export interface PriceRiskInfo {
  price_cv: number;
  rho_py: number;
  policy: string;
  natural_hedge: string;
  revenue_vol_multiplier: number;
}

export interface CropDetail {
  expected_yield: number;
  raw_ml_yield: number;
  agronomic_multiplier: number;
  agronomic_reasons: string[];
  worst_case_yield: number;
  market_price: number;
  expected_revenue: number;
  worst_case_revenue: number;
  pmfby_payout: number;
  premium_cost: number;
  net_worst_case: number;
  price_cv?: number;
  price_yield_corr?: number;
  price_policy?: string;
  price_risk_mult?: number;
}

export interface ClimateInfo {
  climate_regime: string;
  oni_value: number;
  rainfall_mm: number;
  temperature_c: number;
  baseline_source: string;
  baseline_rain?: number;
  baseline_temp?: number;
  enso_rain_adj_pct?: number;
  enso_temp_adj_c?: number;
}

export interface PortfolioResponse {
  success: boolean;
  budget_met: boolean;
  message: string;
  weights: Record<string, number>;
  area_ha: Record<string, number>;
  expected_income: number;
  worst_case_income: number;
  total_premium: number;
  net_expected_profit: number;
  portfolio_volatility_inr: number;
  sharpe_ratio: number;
  risk_aversion: number;
  proxied_crops?: Record<string, string>;
  crop_details: Record<string, CropDetail>;
  optimizer_status: string;
  climate_info: ClimateInfo;
  agronomic_diagnostics: Record<string, string[]>;
  agmarknet_price_risk?: Record<string, PriceRiskInfo>;
}

export interface ClimatePreview {
  district: string;
  season: string;
  climate_regime: string;
  oni_value: number;
  rainfall_mm: number;
  temperature_c: number;
  baseline_source: string;
  baseline_rain: number;
  baseline_temp: number;
  enso_rain_adj_pct: number;
  enso_temp_adj_c: number;
  note: string;
}

export interface ApiError {
  detail: string | { loc: string[]; msg: string; type: string }[];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatApiError(body: ApiError): string {
  if (typeof body.detail === "string") return body.detail;
  return body.detail
    .map((e) => `[${e.loc.slice(1).join(".")}] ${e.msg}`)
    .join("\n");
}

// ---------------------------------------------------------------------------
// API calls
// ---------------------------------------------------------------------------

export async function optimizePortfolio(
  payload: PortfolioRequest
): Promise<PortfolioResponse> {
  // Remove null/undefined optional weather fields so the backend auto-detects
  const cleaned: Record<string, unknown> = { ...payload };
  if (cleaned.rainfall_mm   == null) delete cleaned.rainfall_mm;
  if (cleaned.temperature_c == null) delete cleaned.temperature_c;
  if (cleaned.climate_regime == null) delete cleaned.climate_regime;

  const res = await fetch(`${API_BASE}/api/optimize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cleaned),
  });

  if (!res.ok) {
    const errBody: ApiError = await res.json();
    throw new Error(formatApiError(errBody));
  }

  return res.json() as Promise<PortfolioResponse>;
}

export async function fetchClimatePreview(
  district: string,
  season: string,
  state?: string
): Promise<ClimatePreview> {
  const params = new URLSearchParams({ district, season });
  if (state) params.append("state", state);

  const res = await fetch(`${API_BASE}/api/climate-preview?${params}`);
  if (!res.ok) throw new Error("Failed to fetch climate preview.");
  return res.json() as Promise<ClimatePreview>;
}

export interface CropsCatalogue {
  crops: string[];
  proxied: Record<string, string>;
}

export async function fetchCrops(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/api/crops`);
  if (!res.ok) throw new Error("Failed to load crop list from API.");
  const data = await res.json();
  return data.crops as string[];
}

export async function fetchCropsCatalogue(): Promise<CropsCatalogue> {
  const res = await fetch(`${API_BASE}/api/crops`);
  if (!res.ok) throw new Error("Failed to load crop list from API.");
  return res.json() as Promise<CropsCatalogue>;
}

export async function fetchSeasons(): Promise<{
  seasons: string[];
  regimes: string[];
}> {
  const res = await fetch(`${API_BASE}/api/seasons`);
  if (!res.ok) throw new Error("Failed to load season list from API.");
  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/`, { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Multi-season rotation planning
// ---------------------------------------------------------------------------
export interface RotationSeason {
  season_index: number;
  weights: Record<string, number>;
  yield_multiplier: Record<string, number>;
  plot_transitions: { from: string; to: string; share: number }[];
  expected_profit: number;
  worst_case_income: number;
  volatility: number;
}

export interface RotationResponse {
  n_seasons: number;
  feasible: boolean;
  risk_aversion: number;
  rotation_plan: RotationSeason[];
  greedy_plan: RotationSeason[];
  static_plan: RotationSeason[];
  interaction_matrix: Record<string, Record<string, number>>;
  crop_family: Record<string, string>;
  summary: {
    rotation_discounted_profit: number;
    greedy_discounted_profit: number;
    static_discounted_profit: number;
    gain_vs_static_inr: number;
    gain_vs_static_pct: number;
    gain_vs_greedy_inr: number;
    gain_vs_greedy_pct: number;
    legume_share_rotation: number[];
    legume_share_greedy: number[];
    legume_share_static: number[];
  };
  assumption_note: string;
}

export async function optimizeRotation(
  payload: PortfolioRequest & { n_seasons?: number; previous_weights?: Record<string, number> | null }
): Promise<RotationResponse> {
  const cleaned: Record<string, unknown> = { ...payload };
  if (cleaned.rainfall_mm == null) delete cleaned.rainfall_mm;
  if (cleaned.temperature_c == null) delete cleaned.temperature_c;
  if (cleaned.climate_regime == null) delete cleaned.climate_regime;
  if (cleaned.previous_weights == null) delete cleaned.previous_weights;
  const res = await fetch(`${API_BASE}/api/rotation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cleaned),
  });
  if (!res.ok) {
    const errBody: ApiError = await res.json();
    throw new Error(formatApiError(errBody));
  }
  return res.json() as Promise<RotationResponse>;
}
