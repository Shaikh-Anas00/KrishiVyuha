/**
 * AgriVest Dashboard – Main Page  (v2: Agro-Climatic Advisory Engine)
 *
 * Layout:
 *   [Sidebar — Inputs]     [Main Area — Results]
 *
 * What's new in v2:
 *  - Rainfall, Temperature, ENSO removed from primary form
 *  - Irrigation Access dropdown added (Rainfed / Borewell / Canal)
 *  - "Live Climate Sync" card auto-populates on district+season change
 *  - "Advanced Manual Overrides" collapsible section for environmental controls
 *  - agronomic_diagnostics displayed in ExplainabilityPanel
 */
"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  optimizePortfolio,
  optimizeRotation,
  fetchClimatePreview,
  fetchCrops,
  fetchCropsCatalogue,
  fetchSeasons,
  checkHealth,
  type PortfolioRequest,
  type PortfolioResponse,
  type ClimatePreview,
  type RotationResponse,
} from "@/lib/api";
import MetricCard            from "@/components/MetricCard";
import AllocationChart       from "@/components/AllocationChart";
import ExplainabilityPanel   from "@/components/ExplainabilityPanel";
import CropTable             from "@/components/CropTable";
import RotationPanel          from "@/components/RotationPanel";

// ---------------------------------------------------------------------------
// Irrigation options
// ---------------------------------------------------------------------------
const IRRIGATION_OPTIONS = [
  { value: "Rainfed",       label: "Rainfed (Dryland)",                    badge: "+0 mm"   },
  { value: "Well_Borewell", label: "Borewell / Open Well",                 badge: "+350 mm" },
  { value: "Canal_River",   label: "Canal / River / Assured Supply",       badge: "+700 mm" },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const INR = (n: number) =>
  new Intl.NumberFormat("en-IN", {
    style: "currency", currency: "INR", maximumFractionDigits: 0,
  }).format(n);

const REGIME_EMOJI: Record<string, string> = {
  El_Nino: "🔴",
  La_Nina: "🔵",
  Normal:  "🟢",
};

// Continuous power-curve mapping: smooth, strictly monotonic, C^0 continuous at origin
function sliderPosToLambda(s: number): number {
  const t = Math.max(0, Math.min(100, s)) / 100.0;
  return Number((3.50 * Math.pow(t, 0.75)).toFixed(2));
}

function lambdaToSliderPos(lam: number): number {
  if (lam <= 0.0) return 0;
  const t = Math.pow(Math.min(3.50, lam) / 3.50, 4.0 / 3.0);
  return Math.min(100, Math.max(0, Math.round(t * 100.0)));
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function DashboardPage() {
  // ── API health ──────────────────────────────────────────────────────────
  const [apiOnline, setApiOnline] = useState(false);

  // ── Primary form state ──────────────────────────────────────────────────
  const [state,           setState]          = useState("Maharashtra");
  const [district,        setDistrict]       = useState("Pune");
  const [season,          setSeason]         = useState("Kharif");
  const [irrigationType,  setIrrigationType] = useState("Rainfed");
  const [landAreaHa,      setLandAreaHa]     = useState(5);
  const [survivalBudget,  setSurvivalBudget] = useState(100000);
  const [riskAversion,    setRiskAversion]   = useState(1.5);

  // Crop selection & proxy tracking
  const [availableCrops, setAvailableCrops] = useState<string[]>([]);
  const [proxiedCrops,   setProxiedCrops]   = useState<Record<string, string>>({
    Tomato: "Estimated via horticultural proxy (Onion baseline)",
  });
  const [selectedCrops,  setSelectedCrops]  = useState<string[]>(["Rice", "Cotton", "Soyabean", "Jowar"]);
  const [cropInput,      setCropInput]      = useState("");

  // ── Season list from API ────────────────────────────────────────────────
  const [seasonList, setSeasonList] = useState([
    "Kharif","Rabi","Summer","Zaid","Whole Year","Winter","Autumn",
  ]);

  // ── Live Climate Sync card ──────────────────────────────────────────────
  const [climatePreview,        setClimatePreview]       = useState<ClimatePreview | null>(null);
  const [climatePreviewLoading, setClimatePreviewLoading] = useState(false);

  // ── Advanced overrides (collapsible) ───────────────────────────────────
  const [showOverrides, setShowOverrides]   = useState(false);
  const [rainfallOverride, setRainfallOverride]     = useState<string>("");
  const [tempOverride,     setTempOverride]          = useState<string>("");
  const [regimeOverride,   setRegimeOverride]        = useState<string>("");

  // ── Results ─────────────────────────────────────────────────────────────
  const [result,    setResult]   = useState<PortfolioResponse | null>(null);
  const [loading,   setLoading]  = useState(false);
  const [error,     setError]    = useState<string | null>(null);
  const [rotation,  setRotation]  = useState<RotationResponse | null>(null);
  const [rotLoading, setRotLoading] = useState(false);

  // ── Voice interaction state ──────────────────────────────────────────────
  const [isListening,   setIsListening]   = useState(false);
  const [voiceFeedback, setVoiceFeedback] = useState<string | null>(null);
  const [voiceError,    setVoiceError]    = useState<string | null>(null);

  // ── Initialise API data ─────────────────────────────────────────────────
  useEffect(() => {
    checkHealth().then(setApiOnline);
    fetchCropsCatalogue()
      .then((cat) => {
        setAvailableCrops(cat.crops);
        if (cat.proxied) setProxiedCrops(cat.proxied);
      })
      .catch(() => {
        fetchCrops().then(setAvailableCrops).catch(() => {});
      });
    fetchSeasons().then((d) => setSeasonList(d.seasons)).catch(() => {});
  }, []);

  // ── Live climate preview (debounced) ────────────────────────────────────
  const triggerClimatePreview = useCallback(() => {
    if (!district.trim() || !season) return;
    setClimatePreviewLoading(true);
    fetchClimatePreview(district, season, state || undefined)
      .then(setClimatePreview)
      .catch(() => setClimatePreview(null))
      .finally(() => setClimatePreviewLoading(false));
  }, [district, season, state]);

  // Trigger on district or season change
  useEffect(() => {
    const timer = setTimeout(triggerClimatePreview, 600);
    return () => clearTimeout(timer);
  }, [triggerClimatePreview]);

  // ── Crop selection ───────────────────────────────────────────────────────
  const toggleCrop = (crop: string) => {
    setSelectedCrops((prev) =>
      prev.includes(crop) ? prev.filter((c) => c !== crop) : [...prev, crop]
    );
  };

  const addCustomCrop = () => {
    const c = cropInput.trim();
    if (c && !selectedCrops.includes(c)) {
      setSelectedCrops((prev) => [...prev, c]);
      setCropInput("");
    }
  };

  // ── Voice interaction handler ──────────────────────────────────────────
  const handleVoiceInput = () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const SpeechRecognition = typeof window !== "undefined" && ((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition);
    if (!SpeechRecognition) {
      setVoiceError("Voice input is supported in Chrome, Edge & Safari. Tap the cards below.");
      setTimeout(() => setVoiceError(null), 5000);
      return;
    }

    try {
      const recognition = new SpeechRecognition();
      recognition.lang = "en-IN";
      recognition.continuous = false;
      recognition.interimResults = false;

      recognition.onstart = () => {
        setIsListening(true);
        setVoiceError(null);
        setVoiceFeedback("Listening… Speak area, district, season, or strategy.");
      };

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        const text = transcript.toLowerCase();
        const updates: string[] = [];

        // 1. Strategy / Risk
        if (text.includes("defensive") || text.includes("safe") || text.includes("safety") || text.includes("low risk")) {
          setRiskAversion(2.5);
          updates.push("Strategy: Safety-First");
        } else if (text.includes("balanced") || text.includes("moderate") || text.includes("normal") || text.includes("medium")) {
          setRiskAversion(1.5);
          updates.push("Strategy: Balanced");
        } else if (text.includes("aggressive") || text.includes("profit") || text.includes("high return") || text.includes("max")) {
          setRiskAversion(0.0);
          updates.push("Strategy: High Return");
        }

        // 2. Land area (e.g. 5 acres, 10 hectares, 4 ha)
        const areaMatch = text.match(/(\d+(\.\d+)?)\s*(acre|acres|hectare|hectares|ha)\b/i);
        if (areaMatch) {
          const val = parseFloat(areaMatch[1]);
          const unit = areaMatch[3].toLowerCase();
          if (unit.startsWith("acre")) {
            const ha = Number((val * 0.404686).toFixed(1));
            setLandAreaHa(ha);
            updates.push(`${val} acres (${ha} ha)`);
          } else {
            setLandAreaHa(val);
            updates.push(`${val} ha`);
          }
        }

        // 3. Season
        if (text.includes("kharif")) { setSeason("Kharif"); updates.push("Kharif season"); }
        else if (text.includes("rabi")) { setSeason("Rabi"); updates.push("Rabi season"); }
        else if (text.includes("summer")) { setSeason("Summer"); updates.push("Summer season"); }
        else if (text.includes("zaid")) { setSeason("Zaid"); updates.push("Zaid season"); }

        // 4. Irrigation
        if (text.includes("borewell") || text.includes("tube well") || text.includes("well")) {
          setIrrigationType("Well_Borewell");
          updates.push("Borewell irrigation");
        } else if (text.includes("canal") || text.includes("river")) {
          setIrrigationType("Canal_River");
          updates.push("Canal irrigation");
        } else if (text.includes("rainfed") || text.includes("dryland")) {
          setIrrigationType("Rainfed");
          updates.push("Rainfed");
        }

        // 5. Districts
        const knownDistricts = ["Pune", "Nashik", "Nagpur", "Ahmednagar", "Satara", "Solapur", "Jalgaon", "Kolhapur", "Aurangabad", "Amravati", "Nanded", "Wardha", "Yavatmal", "Dhule", "Sangli", "Thane"];
        for (const d of knownDistricts) {
          if (text.includes(d.toLowerCase())) {
            setDistrict(d);
            updates.push(`District: ${d}`);
            break;
          }
        }

        // 6. Budget
        const budgetMatch = text.match(/(?:budget|survival|revenue)\s*(?:of|is|at)?\s*(\d+)/i);
        if (budgetMatch) {
          const bud = parseFloat(budgetMatch[1]);
          setSurvivalBudget(bud);
          updates.push(`Budget: ₹${bud.toLocaleString("en-IN")}`);
        }

        setVoiceFeedback(
          updates.length > 0
            ? `Heard "${transcript}" → Configured: ${updates.join(", ")}`
            : `Heard: "${transcript}". (Try: "5 hectares in Pune Kharif balanced")`
        );
      };

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      recognition.onerror = (err: any) => {
        setIsListening(false);
        setVoiceError(`Voice notice: ${err.error || "Could not hear audio"}`);
        setTimeout(() => setVoiceError(null), 4000);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognition.start();
    } catch {
      setIsListening(false);
      setVoiceError("Could not initialize microphone.");
      setTimeout(() => setVoiceError(null), 4000);
    }
  };

  // ── Submit ───────────────────────────────────────────────────────────────
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (selectedCrops.length === 0) {
      setError("Please select at least one crop.");
      return;
    }
    setLoading(true);
    setError(null);

    const safeLandArea = isNaN(landAreaHa) || landAreaHa <= 0 ? 1 : landAreaHa;
    const safeSurvivalBudget = isNaN(survivalBudget) || survivalBudget < 0 ? 0 : survivalBudget;

    const payload: PortfolioRequest = {
      state,
      district,
      season,
      land_area_ha:        safeLandArea,
      survival_budget_inr: safeSurvivalBudget,
      irrigation_type:     irrigationType,
      risk_aversion:       riskAversion,
      available_crops:     selectedCrops,
      // Optional overrides — null means auto-detect
      rainfall_mm:     rainfallOverride !== "" ? parseFloat(rainfallOverride) : undefined,
      temperature_c:   tempOverride     !== "" ? parseFloat(tempOverride)     : undefined,
      climate_regime:  regimeOverride   !== "" ? regimeOverride               : undefined,
    };

    try {
      const data = await optimizePortfolio(payload);
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error.");
    } finally {
      setLoading(false);
    }
  };

  // ── Multi-season rotation plan ───────────────────────────────────────────
  const handleRotation = async () => {
    if (selectedCrops.length < 2) { setError("Select at least two crops for a rotation plan."); return; }
    setRotLoading(true);
    setError(null);

    const safeLandArea = isNaN(landAreaHa) || landAreaHa <= 0 ? 1 : landAreaHa;
    const safeSurvivalBudget = isNaN(survivalBudget) || survivalBudget < 0 ? 0 : survivalBudget;

    try {
      const data = await optimizeRotation({
        state, district, season,
        land_area_ha: safeLandArea,
        survival_budget_inr: safeSurvivalBudget,
        irrigation_type: irrigationType,
        risk_aversion: riskAversion,
        available_crops: selectedCrops,
        rainfall_mm:    rainfallOverride !== "" ? parseFloat(rainfallOverride) : undefined,
        temperature_c:  tempOverride     !== "" ? parseFloat(tempOverride)     : undefined,
        climate_regime: regimeOverride   !== "" ? regimeOverride               : undefined,
        n_seasons: 3,
      });
      setRotation(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error.");
    } finally {
      setRotLoading(false);
    }
  };


  // ── Derived ─────────────────────────────────────────────────────────────
  const budgetMet     = result ? result.worst_case_income >= survivalBudget : null;
  const regimeEmoji   = climatePreview ? (REGIME_EMOJI[climatePreview.climate_regime] ?? "🌍") : null;
  const regimeLabel   = climatePreview?.climate_regime?.replace("_", " ") ?? "";

  // ── Render ───────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gray-50 font-sans">
      {/* Top nav */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-emerald-600 flex items-center justify-center text-white font-bold text-sm">A</div>
          <span className="font-bold text-gray-900 tracking-tight">AgriVest</span>
          <span className="text-xs text-gray-400 hidden sm:inline">Climate-Driven Yield Regression & Risk-Optimized Crop Portfolio</span>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className={`w-2 h-2 rounded-full ${apiOnline ? "bg-emerald-500 animate-pulse" : "bg-red-400"}`} />
          <span className={apiOnline ? "text-emerald-700 font-medium" : "text-red-500"}>
            {apiOnline ? "API Connected" : "API Offline"}
          </span>
        </div>
      </header>

      <div className="flex flex-col md:flex-row min-h-[calc(100vh-57px)]">
        {/* ============================================================
            SIDEBAR — Inputs
            ============================================================ */}
        <aside className="w-full md:w-80 lg:w-96 bg-white border-b md:border-b-0 md:border-r border-gray-200 p-6 md:sticky md:top-[57px] md:h-[calc(100vh-57px)] md:overflow-y-auto flex-shrink-0">
          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <h2 className="text-base font-bold text-gray-900">Farm Configuration</h2>
              <p className="text-xs text-gray-500 mt-0.5">Climate is auto-detected from district history + live ENSO data.</p>
            </div>

            {/* ── Voice Assistant (Farmer-Friendly Voice Input) ── */}
            <div className="bg-gradient-to-br from-emerald-800 to-teal-900 text-white rounded-xl p-3.5 shadow-sm space-y-2.5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-base ${
                    isListening ? "bg-red-500 animate-ping" : "bg-white/15"
                  }`}>
                    🎙️
                  </div>
                  <div>
                    <p className="text-[11px] font-bold uppercase tracking-wider text-emerald-300">Voice Assistant</p>
                    <p className="text-xs text-white/90 font-medium">Speak your farm setup</p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={handleVoiceInput}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold shadow-sm transition-all flex items-center gap-1.5 ${
                    isListening
                      ? "bg-red-600 text-white animate-pulse"
                      : "bg-white text-emerald-950 hover:bg-emerald-50 active:scale-95"
                  }`}
                >
                  {isListening ? (
                    <>
                      <span className="w-2 h-2 rounded-full bg-white animate-ping" />
                      Listening…
                    </>
                  ) : (
                    "Tap to Speak"
                  )}
                </button>
              </div>
              {voiceFeedback && (
                <div className="bg-black/30 rounded-lg p-2 text-[11px] text-emerald-100 leading-relaxed border border-emerald-500/30">
                  {voiceFeedback}
                </div>
              )}
              {voiceError && (
                <div className="bg-rose-950/70 rounded-lg p-2 text-[11px] text-rose-200 border border-rose-500/30">
                  {voiceError}
                </div>
              )}
              {!voiceFeedback && !voiceError && !isListening && (
                <p className="text-[10px] text-emerald-200/75">
                  💡 Tip: Say <em>&ldquo;5 hectares in Pune Kharif balanced risk&rdquo;</em>
                </p>
              )}
            </div>

            {/* ── Location ─────────────────────────────────────── */}
            <fieldset className="space-y-3">
              <legend className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Location</legend>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">State</label>
                <input type="text" value={state} onChange={e => setState(e.target.value)}
                  placeholder="e.g. Maharashtra"
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900 bg-white placeholder:text-gray-400 focus:ring-2 focus:ring-emerald-500 focus:border-transparent font-medium" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">District</label>
                <input type="text" value={district} onChange={e => setDistrict(e.target.value)}
                  placeholder="e.g. Pune"
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900 bg-white placeholder:text-gray-400 focus:ring-2 focus:ring-emerald-500 focus:border-transparent font-medium" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Season</label>
                <select value={season} onChange={e => setSeason(e.target.value)}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900 bg-white focus:ring-2 focus:ring-emerald-500 focus:border-transparent font-medium">
                  {seasonList.map(s => <option key={s} value={s} className="text-gray-900 bg-white">{s}</option>)}
                </select>
              </div>
            </fieldset>

            {/* ── Live Climate Sync Card ────────────────────────── */}
            <div className={`rounded-xl p-3 text-sm transition-all ${
              climatePreviewLoading ? "bg-gray-100 animate-pulse"
              : climatePreview ? "bg-emerald-50 ring-1 ring-emerald-200"
              : "bg-gray-100"
            }`}>
              {climatePreviewLoading ? (
                <p className="text-gray-400 text-xs">⟳ Fetching climate data...</p>
              ) : climatePreview ? (
                <div className="space-y-1">
                  <p className="font-semibold text-emerald-800 flex items-center gap-1.5">
                    {regimeEmoji} Live Sync: <span className="font-bold">{regimeLabel}</span> Detected
                  </p>
                  <p className="text-gray-700 text-xs">
                    District Baseline: <strong>{climatePreview.rainfall_mm} mm</strong>, <strong>{climatePreview.temperature_c}°C</strong>
                  </p>
                  {climatePreview.enso_rain_adj_pct !== 0 && (
                    <p className="text-gray-500 text-xs">
                      {climatePreview.climate_regime === "El_Nino" ? "↓" : "↑"} ENSO adjustment: {climatePreview.enso_rain_adj_pct > 0 ? "+" : ""}{climatePreview.enso_rain_adj_pct}% rainfall
                      {climatePreview.enso_temp_adj_c !== 0 ? `, +${climatePreview.enso_temp_adj_c}°C` : ""}
                    </p>
                  )}
                  <p className="text-gray-400 text-xs">ONI = {climatePreview.oni_value} · {climatePreview.baseline_source}</p>
                </div>
              ) : (
                <p className="text-gray-400 text-xs">Enter district + season above to fetch live climate data.</p>
              )}
            </div>

            {/* ── Irrigation ───────────────────────────────────── */}
            <fieldset className="space-y-2">
              <legend className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Irrigation Access</legend>
              {IRRIGATION_OPTIONS.map(opt => (
                <label key={opt.value}
                  className={`flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-all text-sm ${
                    irrigationType === opt.value
                      ? "border-emerald-500 bg-emerald-50 text-emerald-800"
                      : "border-gray-200 text-gray-700 hover:border-gray-300"
                  }`}>
                  <div className="flex items-center gap-2">
                    <input type="radio" name="irrigation" value={opt.value}
                      checked={irrigationType === opt.value}
                      onChange={() => setIrrigationType(opt.value)}
                      className="accent-emerald-600" />
                    {opt.label}
                  </div>
                  <span className="text-xs font-mono bg-white rounded px-1.5 py-0.5 ring-1 ring-gray-200 text-gray-600">{opt.badge}</span>
                </label>
              ))}
            </fieldset>

            {/* ── Farm Financials & Risk Preference ─────────────── */}
            <fieldset className="space-y-3">
              <legend className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Farm Financials & Risk Strategy</legend>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Total Land Area (ha)</label>
                <input
                  type="number"
                  min={0.1}
                  max={10000}
                  step={0.5}
                  value={isNaN(landAreaHa) ? "" : landAreaHa}
                  onChange={e => {
                    const v = e.target.value;
                    setLandAreaHa(v === "" ? NaN : parseFloat(v));
                  }}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900 bg-white placeholder:text-gray-400 focus:ring-2 focus:ring-emerald-500 focus:border-transparent font-medium"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Survival Budget (INR)</label>
                <input
                  type="number"
                  min={0}
                  step={1000}
                  value={isNaN(survivalBudget) ? "" : survivalBudget}
                  onChange={e => {
                    const v = e.target.value;
                    setSurvivalBudget(v === "" ? NaN : parseFloat(v));
                  }}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900 bg-white placeholder:text-gray-400 focus:ring-2 focus:ring-emerald-500 focus:border-transparent font-medium"
                />
                <p className="text-xs text-gray-400 mt-1">Minimum acceptable revenue floor to avoid bankruptcy.</p>
              </div>

              {/* ── Visual Strategy Selection (Farmer-First) ── */}
              <div className="pt-1 space-y-2.5">
                <div className="flex items-center justify-between">
                  <label className="block text-sm font-medium text-gray-700">Crop Strategy</label>
                  <span className={`text-[11px] px-2.5 py-0.5 rounded-full font-semibold ${
                    riskAversion >= 2.0 ? "bg-blue-100 text-blue-800" :
                    riskAversion >= 0.8 ? "bg-emerald-100 text-emerald-800" :
                    "bg-amber-100 text-amber-800"
                  }`}>
                    {riskAversion >= 2.0 ? "🛡️ Safety First" : riskAversion >= 0.8 ? "⚖️ Balanced Growth" : "🚀 High Return"}
                  </span>
                </div>

                {/* 3 Visual Strategy Cards */}
                <div className="space-y-2">
                  {/* Defensive Card */}
                  <div
                    onClick={() => setRiskAversion(2.5)}
                    className={`p-3 rounded-xl border cursor-pointer transition-all flex items-start gap-3 ${
                      riskAversion >= 2.0
                        ? "border-blue-500 bg-blue-50/70 ring-1 ring-blue-400 shadow-sm"
                        : "border-gray-200 bg-white hover:border-gray-300"
                    }`}
                  >
                    <div className="w-8 h-8 rounded-lg bg-blue-100 text-blue-700 flex items-center justify-center text-lg flex-shrink-0">
                      🛡️
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <p className="text-xs font-bold text-gray-900">Safety First (Low Risk)</p>
                        <span className="text-[10px] font-semibold text-blue-700 bg-blue-100 px-1.5 py-0.5 rounded">Defensive</span>
                      </div>
                      <p className="text-[11px] text-gray-500 mt-0.5 leading-snug">
                        Guarantees survival floor during drought; relies on high PMFBY insurance & stable staple crops.
                      </p>
                    </div>
                  </div>

                  {/* Balanced Card */}
                  <div
                    onClick={() => setRiskAversion(1.5)}
                    className={`p-3 rounded-xl border cursor-pointer transition-all flex items-start gap-3 ${
                      riskAversion >= 0.8 && riskAversion < 2.0
                        ? "border-emerald-500 bg-emerald-50/70 ring-1 ring-emerald-400 shadow-sm"
                        : "border-gray-200 bg-white hover:border-gray-300"
                    }`}
                  >
                    <div className="w-8 h-8 rounded-lg bg-emerald-100 text-emerald-700 flex items-center justify-center text-lg flex-shrink-0">
                      ⚖️
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <p className="text-xs font-bold text-gray-900">Balanced Growth</p>
                        <span className="text-[10px] font-semibold text-emerald-700 bg-emerald-100 px-1.5 py-0.5 rounded">Recommended</span>
                      </div>
                      <p className="text-[11px] text-gray-500 mt-0.5 leading-snug">
                        Smart diversification: hedges climate and mandi market swings with a mix of pulses and cash crops.
                      </p>
                    </div>
                  </div>

                  {/* Aggressive Card */}
                  <div
                    onClick={() => setRiskAversion(0.0)}
                    className={`p-3 rounded-xl border cursor-pointer transition-all flex items-start gap-3 ${
                      riskAversion < 0.8
                        ? "border-amber-500 bg-amber-50/70 ring-1 ring-amber-400 shadow-sm"
                        : "border-gray-200 bg-white hover:border-gray-300"
                    }`}
                  >
                    <div className="w-8 h-8 rounded-lg bg-amber-100 text-amber-700 flex items-center justify-center text-lg flex-shrink-0">
                      🚀
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <p className="text-xs font-bold text-gray-900">High Return (Aggressive)</p>
                        <span className="text-[10px] font-semibold text-amber-700 bg-amber-100 px-1.5 py-0.5 rounded">Max Profit</span>
                      </div>
                      <p className="text-[11px] text-gray-500 mt-0.5 leading-snug">
                        Maximizes projected cash-in-hand by prioritizing top-revenue commercial crops.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Visual spectrum fine-tuning bar (No lambda numbers!) */}
                <div className="pt-2">
                  <div className="flex items-center justify-between text-[11px] text-gray-500 font-medium mb-1">
                    <span>Fine-Tune Balance:</span>
                    <span className="text-[11px] text-gray-600 font-semibold">
                      {riskAversion >= 2.0 ? "High Protection" : riskAversion >= 0.8 ? "Balanced Mix" : "Peak Return Focus"}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    step={1}
                    value={lambdaToSliderPos(riskAversion)}
                    onChange={e => setRiskAversion(sliderPosToLambda(parseInt(e.target.value, 10)))}
                    className="w-full accent-emerald-600 cursor-pointer h-2 bg-gradient-to-r from-amber-400 via-emerald-400 to-blue-500 rounded-lg appearance-none"
                  />
                  <div className="flex justify-between text-[10px] text-gray-500 mt-1">
                    <span className="text-amber-700 font-semibold">🚀 High Profit</span>
                    <span className="text-emerald-700 font-semibold">⚖️ Balanced</span>
                    <span className="text-blue-700 font-semibold">🛡️ Low Risk</span>
                  </div>
                </div>
              </div>
            </fieldset>

            {/* ── Crop Selection ────────────────────────────────── */}
            <fieldset className="space-y-2">
              <legend className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Crops to Consider</legend>
              <div className="flex flex-wrap gap-1.5 max-h-36 overflow-y-auto">
                {(availableCrops.length > 0 ? availableCrops
                  : ["Rice","Wheat","Maize","Jowar","Bajra","Gram","Groundnut","Cotton",
                     "Soyabean","Sugarcane","Mustard","Sunflower","Arhar/Tur","Urad",
                     "Moong(Green Gram)","Potato","Onion","Tomato","Banana","Coconut"]
                ).map(crop => {
                  const isProxied = Boolean(proxiedCrops[crop]);
                  const isSelected = selectedCrops.includes(crop);
                  return (
                    <button type="button" key={crop} onClick={() => toggleCrop(crop)}
                      className={`text-xs px-2.5 py-1 rounded-full border font-medium transition-all ${
                        isSelected
                          ? "bg-emerald-600 text-white border-emerald-600"
                          : isProxied
                          ? "bg-amber-50 text-amber-800 border-amber-300 hover:border-amber-400"
                          : "bg-white text-gray-600 border-gray-300 hover:border-emerald-400"
                      }`}>
                      {crop}{isProxied ? " *" : ""}
                    </button>
                  );
                })}
              </div>
              {selectedCrops.length > 0 && (
                <p className="text-xs text-emerald-700 font-medium">
                  {selectedCrops.length} selected: {selectedCrops.join(", ")}
                </p>
              )}
              {selectedCrops.some(c => proxiedCrops[c]) && (
                <div className="bg-amber-50 rounded-lg p-2.5 text-[11px] text-amber-800 border border-amber-200 space-y-1">
                  <p className="font-semibold flex items-center gap-1">
                    <span>⚠️</span> Note on Proxied Horticultural Data:
                  </p>
                  {selectedCrops.filter(c => proxiedCrops[c]).map(c => (
                    <p key={c} className="text-amber-700 text-[10px]">
                      • <strong>{c}</strong>: {proxiedCrops[c]}
                    </p>
                  ))}
                </div>
              )}
              <div className="flex gap-2">
                <input type="text" value={cropInput} onChange={e => setCropInput(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && (e.preventDefault(), addCustomCrop())}
                  placeholder="Custom crop name..."
                  className="flex-1 rounded-lg border border-gray-300 px-3 py-1.5 text-xs text-gray-900 bg-white placeholder:text-gray-400 focus:ring-2 focus:ring-emerald-500 focus:border-transparent font-medium" />
                <button type="button" onClick={addCustomCrop}
                  className="text-xs bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg px-3 py-1.5 font-medium">Add</button>
              </div>
            </fieldset>

            {/* ── Advanced Manual Overrides (collapsed) ────────── */}
            <div className="border border-dashed border-gray-300 rounded-xl overflow-hidden">
              <button type="button" onClick={() => setShowOverrides(v => !v)}
                className="w-full flex items-center justify-between px-4 py-2.5 text-xs font-semibold text-gray-600 hover:bg-gray-50 transition-colors">
                <span>⚙ Advanced Manual Overrides</span>
                <span className="text-gray-400">{showOverrides ? "▲" : "▼"}</span>
              </button>
              {showOverrides && (
                <div className="px-4 pb-4 space-y-3 bg-gray-50">
                  <p className="text-xs text-gray-400 pt-2">
                    Leave blank to use auto-detected values. Override only for testing specific scenarios.
                  </p>
                  <div>
                    <label className="block text-xs font-medium text-gray-600 mb-1">Rainfall Override (mm)</label>
                    <input type="number" min={0} max={5000} value={rainfallOverride}
                      onChange={e => setRainfallOverride(e.target.value)}
                      placeholder={climatePreview ? `Auto: ${climatePreview.rainfall_mm}` : "Auto-detected"}
                      className="w-full rounded-lg border border-gray-300 px-3 py-2 text-xs text-gray-900 bg-white placeholder:text-gray-400 focus:ring-2 focus:ring-amber-400 focus:border-transparent font-medium" />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-600 mb-1">Temperature Override (°C)</label>
                    <input type="number" min={0} max={60} step={0.1} value={tempOverride}
                      onChange={e => setTempOverride(e.target.value)}
                      placeholder={climatePreview ? `Auto: ${climatePreview.temperature_c}` : "Auto-detected"}
                      className="w-full rounded-lg border border-gray-300 px-3 py-2 text-xs text-gray-900 bg-white placeholder:text-gray-400 focus:ring-2 focus:ring-amber-400 focus:border-transparent font-medium" />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-600 mb-1">ENSO Regime Override</label>
                    <select value={regimeOverride} onChange={e => setRegimeOverride(e.target.value)}
                      className="w-full rounded-lg border border-gray-300 px-3 py-2 text-xs text-gray-900 bg-white focus:ring-2 focus:ring-amber-400 focus:border-transparent font-medium">
                      <option value="" className="text-gray-900 bg-white">— Auto-detected —</option>
                      <option value="Normal" className="text-gray-900 bg-white">Normal</option>
                      <option value="El_Nino" className="text-gray-900 bg-white">El Niño</option>
                      <option value="La_Nina" className="text-gray-900 bg-white">La Niña</option>
                    </select>
                  </div>
                  {(rainfallOverride || tempOverride || regimeOverride) && (
                    <button type="button"
                      onClick={() => { setRainfallOverride(""); setTempOverride(""); setRegimeOverride(""); }}
                      className="text-xs text-red-500 hover:underline">
                      Clear overrides → restore auto-detect
                    </button>
                  )}

                  {/* Mathematical Model Specification details */}
                  <div className="pt-3 border-t border-gray-200 mt-2 space-y-1.5">
                    <p className="text-xs font-bold text-gray-700">🔬 Mathematical Specification (Markowitz QP)</p>
                    <div className="bg-white p-2.5 rounded-lg border border-gray-200 text-[11px] space-y-1 font-mono text-gray-600">
                      <div className="flex justify-between">
                        <span>Risk Penalty (λ):</span>
                        <span className="font-bold text-emerald-700">{riskAversion.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Mathematical Objective:</span>
                        <span>max wᵀμ − λ·wᵀΣw</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Covariance Matrix:</span>
                        <span>28-Yr Regime Panel + Agmarknet</span>
                      </div>
                      <div className="flex justify-between">
                        <span>SLSQP Constraints:</span>
                        <span>Σw=1, 0≤w≤1, wᵀy_wc≥B</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* ── Submit ───────────────────────────────────────── */}
            <button type="submit" disabled={loading}
              className="w-full bg-emerald-600 hover:bg-emerald-700 disabled:bg-emerald-400 text-white font-semibold rounded-xl py-3 px-4 transition-colors flex items-center justify-center gap-2">
              {loading ? (
                <>
                  <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeDasharray="60" strokeDashoffset="20" />
                  </svg>
                  <span>Running SciPy SLSQP…</span>
                </>
              ) : (
                "⚡ Optimise Portfolio"
              )}
            </button>
            <button type="button" onClick={handleRotation} disabled={rotLoading}
              className="w-full bg-lime-600 hover:bg-lime-700 disabled:bg-lime-400 text-white font-semibold rounded-xl py-2.5 px-4 transition-colors text-sm">
              {rotLoading ? "Planning rotation…" : "🔄 Plan 3-Season Rotation"}
            </button>
          </form>
        </aside>

        {/* ============================================================
            MAIN CONTENT — Results
            ============================================================ */}
        <main className="flex-1 p-6 space-y-6">
          {/* Error banner */}
          {error && (
            <div className="bg-rose-50 border border-rose-200 rounded-xl p-4 text-sm text-rose-700 flex gap-3">
              <span className="text-xl">⚠</span>
              <div>
                <p className="font-semibold">Optimization Failed</p>
                <p className="whitespace-pre-wrap mt-1 text-xs">{error}</p>
              </div>
            </div>
          )}

          {/* Partial-success banner */}
          {result && !result.budget_met && (
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-sm text-amber-800 flex gap-3">
              <span className="text-xl">⚠️</span>
              <div>
                <p className="font-semibold">Budget Constraint Not Fully Met</p>
                <p className="text-xs mt-1">{result.message}</p>
              </div>
            </div>
          )}

          {/* Welcome state */}
          {!result && !loading && (
            <div className="flex flex-col items-center justify-center h-full min-h-64 text-center space-y-3 opacity-60">
              <div className="w-16 h-16 rounded-2xl bg-emerald-100 flex items-center justify-center text-3xl">🌾</div>
              <h3 className="text-lg font-bold text-gray-700">AgriVest Agro-Climatic Advisory Engine</h3>
              <p className="text-sm text-gray-500 max-w-sm">
                Configure your farm location and irrigation access. Climate is auto-detected. Click
                <strong> Optimise Portfolio</strong> to get your AI-powered crop allocation.
              </p>
            </div>
          )}

          {/* KPI Cards */}
          {result && (
            <>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <MetricCard
                  label   = "Expected Income"
                  value   = {INR(result.expected_income)}
                  subLabel= {`Net profit: ${INR(result.net_expected_profit)}`}
                  icon    = "💰"
                  color   = "green" />
                <MetricCard
                  label   = "Worst-Case Income"
                  value   = {INR(result.worst_case_income)}
                  subLabel= {budgetMet === false ? `⚠ Below budget of ${INR(survivalBudget)}` : `✓ Exceeds survival budget`}
                  icon    = "🛡️"
                  color   = {budgetMet === false ? "rose" : "blue"}
                  warn    = {budgetMet === false} />
                <MetricCard
                  label   = "Portfolio Volatility"
                  value   = {INR(result.portfolio_volatility_inr ?? 0)}
                  subLabel= {`Risk-Adjusted Sharpe: ${result.sharpe_ratio ?? 0}`}
                  icon    = "📉"
                  color   = "blue" />
                <MetricCard
                  label   = "Total PMFBY Premium"
                  value   = {INR(result.total_premium)}
                  subLabel= {`${Object.values(result.crop_details).filter(d => d.pmfby_payout > 0).length} crops insured`}
                  icon    = "📑"
                  color   = "amber" />
              </div>

              {/* Allocation chart + table */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <AllocationChart
                  weights = {result.weights}
                  areaHa  = {result.area_ha} />
                <CropTable
                  cropDetails = {result.crop_details}
                  weights     = {result.weights}
                  areaHa      = {result.area_ha} />
              </div>

              {/* Explainability panel */}
              <ExplainabilityPanel
                result         = {result}
                survivalBudget = {survivalBudget}
                totalAreaHa    = {landAreaHa} />
            </>
          )}
          {rotation && <RotationPanel data={rotation} />}
        </main>
      </div>
    </div>
  );
}

