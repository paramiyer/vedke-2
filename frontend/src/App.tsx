import { useEffect, useMemo, useRef, useState, type ChangeEventHandler } from "react";

type Tab = "build" | "view";
type StageStatus = "idle" | "running" | "done" | "error";

type Stage = {
  id: string;
  title: string;
  hint: string;
  status: StageStatus;
  output?: string;
};

type ArtifactPreview = {
  path: string;
  exists: boolean;
  bytes: number;
  content: string;
  truncated: boolean;
};

type BuildForm = {
  experimentName: string;
  youtubeUrl: string;
  pdfPath: string;
  pages: string;
  aiModel: string;
  asrMode: string;
  languageCode: string;
  topTrim: string;
  bottomTrim: string;
  dropRules: string;
  minConfidence: string;
  maxEditCost: string;
  anchorToken: string;
};

type StageStatusMap = Record<"input" | "asr" | "pdf" | "align" | "gate", StageStatus>;

type ExperimentSummary = {
  experiment: string;
  stageStatus: StageStatusMap & { karaokeDone?: boolean };
};

type ExperimentSnapshot = {
  experiment: string;
  form: Partial<BuildForm>;
  stageStatus: StageStatusMap & { karaokeDone?: boolean };
  files: string[];
};

type KaraokeToken = {
  tokenId: string;
  token: string;
  x: number;
  y: number;
  w: number;
  h: number;
  start_s: number;
  end_s: number;
};

type KaraokeViewPayload = {
  experiment: string;
  pages: number[];
  tokensByPage: Record<string, KaraokeToken[]>;
  pageImageUrls?: Record<string, string>;
  pageImageData?: Record<string, string>;
  pageBoundsByPage?: Record<string, { maxRight: number; maxBottom: number }>;
  trimTopRatio?: number;
  trimBottomRatio?: number;
  audioUrl: string;
  pageImageBaseUrl: string;
  files: Record<string, string>;
};

type ReviewColumn = { key: string; label: string };

const INITIAL_FORM: BuildForm = {
  experimentName: "",
  youtubeUrl: "",
  pdfPath: "data/ganapatiaccent.pdf",
  pages: "2,3,4",
  aiModel: "saaras:v3",
  asrMode: "verbatim",
  languageCode: "sa-IN",
  topTrim: "0.08",
  bottomTrim: "0.06",
  dropRules: "2:1-3",
  minConfidence: "0.78",
  maxEditCost: "0.32",
  anchorToken: ""
};

const FIELD_HELP: Record<string, string> = {
  experimentName:
    "Required. Used as folder slug under data/. Example: ganapati_trial_01. All artifacts are saved under this folder.",
  youtubeUrl: "Required. Source link used to fetch audio for ASR pipeline.",
  pdfPath: "Required. Source PDF path used as the truth text reference.",
  pages: "Required. Page syntax like 2,3,4 or 2-5. Wrong pages will affect truth extraction.",
  storageLocation:
    "Generated paths are fixed: data/<experiment>/input and data/<experiment>/out. This is locked to protect downstream steps.",
  aiModel: "AI model profile used for Sarvam transcription. saaras:v3 is the default production profile.",
  asrMode: "ASR output mode. Use verbatim for exact transcript output.",
  languageCode: "Language locale passed to ASR (example: sa-IN).",
  topTrim: "0 to <1. Higher value trims more from page top.",
  bottomTrim: "0 to <1. Higher value trims more from page bottom.",
  dropRules: "Optional line-drop rules after extraction. Format: page:lineRange (example: 2:1-3).",
  minConfidence: "0 to 1. Higher = stricter auto-accept and more review routing.",
  maxEditCost: "0 to 1. Lower = tighter text matching, higher = more tolerant.",
  anchorToken: "Alignment anchor tokenId from pdf_tokens.json (example: P0008_T0007)."
};

const STAGE_TEMPLATES: Stage[] = [
  { id: "input", title: "Source Intake", hint: "Experiment folder + source acquisition", status: "idle" },
  { id: "asr", title: "ASR Preparation", hint: "YouTube audio + Sarvam output normalization", status: "idle" },
  { id: "pdf", title: "PDF Truth Build", hint: "OCR token extraction with cleanup rules (word|punc + x/y/w/h)", status: "idle" },
  { id: "align", title: "Alignment Config", hint: "LLM timestamp alignment + CSV review loop", status: "idle" },
  { id: "gate", title: "Quality Gates", hint: "Guardrail checks and review routing", status: "idle" }
];

const AI_MODELS = ["saaras:v3"];
const API_BASE = "http://localhost:8787";
const REVIEW_COLUMNS_ORDER: ReviewColumn[] = [
  { key: "segment_index", label: "Seg" },
  { key: "audio_text", label: "Audio Text" },
  { key: "kept_tokens_text", label: "OCR Kept Text" },
  { key: "audio_start_s", label: "Start" },
  { key: "audio_end_s", label: "End" },
  { key: "coarse_start_token_id", label: "Coarse Start" },
  { key: "coarse_end_token_id", label: "Coarse End" },
  { key: "status", label: "Status" },
  { key: "coverage_ratio_estimate", label: "Coverage" },
  { key: "left_out_eligible_count_within_span", label: "Left Out" },
  { key: "kept_token_ids", label: "Kept Token IDs" },
  { key: "dropped_candidate_token_ids", label: "Dropped Token IDs" },
  { key: "previous_segment_end_token_id", label: "Prev End" },
  { key: "notes", label: "Notes" },
  { key: "match_ratio", label: "Match" },
  { key: "manual_fallback", label: "Fallback" },
];

function slugify(value: string): string {
  return value
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

async function parseApiResponse(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text.trim()) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { ok: false, message: text };
  }
}

async function fetchWithTimeout(url: string, init: RequestInit, timeoutMs = 120000): Promise<Response> {
  if (timeoutMs <= 0) {
    return fetch(url, init);
  }
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    window.clearTimeout(timer);
  }
}

const ALIGNMENT_TIMEOUT_MS = 0;

function parseSimpleCsvRow(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && i + 1 < line.length && line[i + 1] === '"') {
        cur += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }
    if (ch === "," && !inQuotes) {
      out.push(cur);
      cur = "";
      continue;
    }
    cur += ch;
  }
  out.push(cur);
  return out;
}

function parseCsvTable(csvText: string): Array<Record<string, string>> {
  const raw = csvText.trim();
  if (!raw) return [];
  const lines = raw.split(/\r?\n/).filter((x) => x.trim().length > 0);
  if (lines.length < 2) return [];
  const header = parseSimpleCsvRow(lines[0]);
  const rows: Array<Record<string, string>> = [];
  for (const line of lines.slice(1)) {
    const cells = parseSimpleCsvRow(line);
    const row: Record<string, string> = {};
    for (let i = 0; i < header.length; i += 1) {
      row[header[i]] = cells[i] ?? "";
    }
    rows.push(row);
  }
  return rows;
}

type RenderRect = { left: number; top: number; width: number; height: number };

function mapOcrTokenToRenderRect(params: {
  token: Pick<KaraokeToken, "x" | "y" | "w" | "h">;
  naturalSize?: { w: number; h: number };
  pageBounds?: { maxRight: number; maxBottom: number };
}): RenderRect {
  const { token, naturalSize, pageBounds } = params;
  if (!naturalSize || naturalSize.w <= 0 || naturalSize.h <= 0) {
    return { left: token.x, top: token.y, width: token.w, height: token.h };
  }

  const boundW = Math.max(1, Number(pageBounds?.maxRight || 0));
  const boundH = Math.max(1, Number(pageBounds?.maxBottom || 0));

  const scaleX = naturalSize.w / boundW;
  const scaleY = naturalSize.h / boundH;

  return {
    left: token.x * scaleX,
    top: token.y * scaleY,
    width: token.w * scaleX,
    height: token.h * scaleY,
  };
}

async function readFileAsText(file: File): Promise<string> {
  return file.text();
}

function App() {
  const [tab, setTab] = useState<Tab>("build");
  const [form, setForm] = useState<BuildForm>(INITIAL_FORM);
  const [experimentMode, setExperimentMode] = useState<"new" | "load">("new");
  const [selectedExperiment, setSelectedExperiment] = useState<string>("");
  const [experimentOptions, setExperimentOptions] = useState<ExperimentSummary[]>([]);
  const [loadingExperiments, setLoadingExperiments] = useState<boolean>(false);
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [openInfo, setOpenInfo] = useState<string | null>(null);
  const [activeStage, setActiveStage] = useState<string>("input");
  const [stages, setStages] = useState<Stage[]>(STAGE_TEMPLATES);
  const [events, setEvents] = useState<string[]>([]);
  const folderPickerRef = useRef<HTMLInputElement | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [alignmentReviewCsv, setAlignmentReviewCsv] = useState<string>("");
  const [leftOutCsv, setLeftOutCsv] = useState<string>("");
  const [alignmentFeedback, setAlignmentFeedback] = useState<string>("");
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState<boolean>(false);
  const [showManualAlignModal, setShowManualAlignModal] = useState<boolean>(false);
  const [manualPrompt, setManualPrompt] = useState<string>("");
  const [manualPromptLoading, setManualPromptLoading] = useState<boolean>(false);
  const [manualEnrichedJson, setManualEnrichedJson] = useState<string>("");
  const [manualReviewCsv, setManualReviewCsv] = useState<string>("");
  const [manualLeftOutCsv, setManualLeftOutCsv] = useState<string>("");
  const [manualStatus, setManualStatus] = useState<string>("");
  const [manualRecs, setManualRecs] = useState<string[]>([]);
  const [isManualImporting, setIsManualImporting] = useState<boolean>(false);
  const [promptCopied, setPromptCopied] = useState<boolean>(false);
  const [showExpandedReview, setShowExpandedReview] = useState<boolean>(false);
  const [isKaraokeBuilding, setIsKaraokeBuilding] = useState<boolean>(false);
  const [viewExperiment, setViewExperiment] = useState<string>("");
  const [viewPayload, setViewPayload] = useState<KaraokeViewPayload | null>(null);
  const [playbackSeconds, setPlaybackSeconds] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [viewMessage, setViewMessage] = useState<string>("");
  const [imageLoadErrors, setImageLoadErrors] = useState<Record<string, string>>({});
  const [playNonce, setPlayNonce] = useState<number>(0);
  const [pageNaturalSize, setPageNaturalSize] = useState<Record<string, { w: number; h: number }>>({});
  const [currentViewPage, setCurrentViewPage] = useState<number | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const tokenRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const karaokeScrollRef = useRef<HTMLDivElement | null>(null);
  const pageImgRefs = useRef<Record<string, HTMLImageElement | null>>({});
  const pageCanvasRefs = useRef<Record<string, HTMLCanvasElement | null>>({});
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const experimentSlug = useMemo(() => slugify(form.experimentName), [form.experimentName]);
  const inputDir = experimentSlug ? `data/${experimentSlug}/input` : "data/<experiment>/input";
  const outDir = experimentSlug ? `data/${experimentSlug}/out` : "data/<experiment>/out";
  const experimentExists = experimentSlug.length > 0 && experimentOptions.some((e) => e.experiment === experimentSlug);

  const validations = useMemo(() => {
    return [
      { key: "exp", label: "Experiment name is set", ok: experimentSlug.length > 0 },
      { key: "yt", label: "YouTube URL is set", ok: form.youtubeUrl.trim().length > 0 },
      { key: "pdf", label: "PDF path is set", ok: form.pdfPath.trim().length > 4 },
      { key: "pages", label: "Page spec is set", ok: form.pages.trim().length > 0 },
      {
        key: "threshold",
        label: "Confidence threshold in (0,1]",
        ok: Number(form.minConfidence) > 0 && Number(form.minConfidence) <= 1
      }
    ];
  }, [form, experimentSlug]);

  const isReadyToRunAll = validations.every((v) => v.ok);
  const alignmentTableRows = useMemo(() => parseCsvTable(alignmentReviewCsv), [alignmentReviewCsv]);
  const leftOutTableRows = useMemo(() => parseCsvTable(leftOutCsv), [leftOutCsv]);
  const reviewColumns = useMemo(() => {
    if (alignmentTableRows.length === 0) return [] as ReviewColumn[];
    const keys = new Set<string>();
    alignmentTableRows.forEach((row) => Object.keys(row).forEach((k) => keys.add(k)));
    const ordered = REVIEW_COLUMNS_ORDER.filter((c) => keys.has(c.key));
    const known = new Set(ordered.map((c) => c.key));
    const extras = Array.from(keys)
      .filter((k) => !known.has(k))
      .map((k) => ({ key: k, label: k }));
    return [...ordered, ...extras];
  }, [alignmentTableRows]);
  const compactReviewColumns = useMemo(() => reviewColumns.slice(0, 8), [reviewColumns]);
  const allStagesDone = useMemo(() => stages.every((s) => s.status === "done"), [stages]);
  const karaokeReadyExperiments = useMemo(
    () => experimentOptions.filter((e) => Boolean(e.stageStatus.karaokeDone)).map((e) => e.experiment),
    [experimentOptions]
  );
  const flattenedKaraokeTokens = useMemo(() => {
    if (!viewPayload) return [] as Array<KaraokeToken & { page: number; key: string; rank: number }>;
    const out: Array<KaraokeToken & { page: number; key: string; rank: number }> = [];
    viewPayload.pages.forEach((page, pageRank) => {
      const recs = Array.isArray(viewPayload.tokensByPage[String(page)]) ? viewPayload.tokensByPage[String(page)] : [];
      recs.forEach((t, idx) => out.push({ ...t, page, key: `${page}:${t.tokenId || idx}`, rank: pageRank * 100000 + idx }));
    });
    out.sort((a, b) => (a.start_s !== b.start_s ? a.start_s - b.start_s : a.rank - b.rank));
    return out;
  }, [viewPayload]);
  const activeKaraokeToken = useMemo(() => {
    if (flattenedKaraokeTokens.length === 0) return null;
    return (
      flattenedKaraokeTokens.find((t) => playbackSeconds >= t.start_s && playbackSeconds <= t.end_s) ||
      flattenedKaraokeTokens.find((t) => playbackSeconds < t.start_s) ||
      null
    );
  }, [flattenedKaraokeTokens, playbackSeconds]);
  const tokenRankByKey = useMemo(() => {
    const map = new Map<string, number>();
    flattenedKaraokeTokens.forEach((t, idx) => map.set(t.key, idx));
    return map;
  }, [flattenedKaraokeTokens]);
  const activeTokenRank = useMemo(() => {
    if (!activeKaraokeToken) return -1;
    return tokenRankByKey.get(activeKaraokeToken.key) ?? -1;
  }, [activeKaraokeToken, tokenRankByKey]);

  const extractAlignmentCsv = (artifacts: ArtifactPreview[]): string => {
    const hit = artifacts.find((a) => a.path.endsWith("pdf_tokens_segment_mapping_review.csv"));
    return hit?.content || "";
  };
  const extractLeftOutCsv = (artifacts: ArtifactPreview[]): string => {
    const hit = artifacts.find((a) => a.path.endsWith("pdf_tokens_left_out_non_punctuation.csv"));
    return hit?.content || "";
  };

  const summarizeArtifactsForOutput = (payload: Record<string, unknown>, artifacts: ArtifactPreview[]) => {
    return JSON.stringify(
      {
        inputDir: payload.inputDir,
        outDir: payload.outDir,
        files: payload.files,
        artifactPaths: artifacts.map((a) => ({ path: a.path, exists: a.exists, bytes: a.bytes, truncated: a.truncated })),
        logs: Array.isArray(payload.logs)
          ? payload.logs.map((x: { command?: string; status?: number }) => ({
              command: String(x.command || ""),
              status: Number(x.status ?? -1),
            }))
          : [],
      },
      null,
      2
    );
  };

  const stageAvailable = (stageIndex: number) => {
    if (stageIndex === 0) return true;
    return stages[stageIndex - 1].status === "done";
  };

  const applyLoadedStageStatus = (statusMap: StageStatusMap) => {
    const ordered: Array<keyof StageStatusMap> = ["input", "asr", "pdf", "align", "gate"];
    setStages((prev) =>
      prev.map((s) => {
        const key = s.id as keyof StageStatusMap;
        return { ...s, status: statusMap[key] || "idle" };
      })
    );
    const firstPending = ordered.find((key) => statusMap[key] !== "done") || "gate";
    setActiveStage(firstPending);
  };

  const refreshExperimentOptions = async () => {
    setLoadingExperiments(true);
    try {
      const response = await fetchWithTimeout(`${API_BASE}/api/experiments`, { method: "GET" }, 120000);
      const payload = (await parseApiResponse(response)) as Record<string, unknown>;
      if (!response.ok || !payload.ok) return;
      const options = Array.isArray(payload.experiments)
        ? payload.experiments.filter((x): x is ExperimentSummary => typeof x === "object" && x !== null) as ExperimentSummary[]
        : [];
      setExperimentOptions(options);
    } finally {
      setLoadingExperiments(false);
    }
  };

  const loadExperiment = async (slug: string) => {
    if (!slug) return;
    const response = await fetchWithTimeout(
      `${API_BASE}/api/experiment?name=${encodeURIComponent(slug)}`,
      { method: "GET" },
      120000
    );
    const payload = (await parseApiResponse(response)) as Record<string, unknown>;
    if (!response.ok || !payload.ok) {
      const message = String(payload?.message || "failed to load experiment");
      setEvents((prev) => [`${new Date().toLocaleTimeString()} • LOAD failed: ${message}`, ...prev]);
      return;
    }
    const snapshot = (payload.snapshot || {}) as ExperimentSnapshot;
    const loadedForm = snapshot.form || {};
    setForm((prev) => ({
      ...prev,
      experimentName: String(loadedForm.experimentName || slug),
      youtubeUrl: String(loadedForm.youtubeUrl || ""),
      pdfPath: String(loadedForm.pdfPath || ""),
      pages: String(loadedForm.pages || ""),
      anchorToken: String(loadedForm.anchorToken || ""),
    }));
    if (snapshot.stageStatus) {
      applyLoadedStageStatus(snapshot.stageStatus);
    }
    setEvents((prev) => [`${new Date().toLocaleTimeString()} • Loaded experiment ${slug}`, ...prev]);
  };

  useEffect(() => {
    void refreshExperimentOptions();
  }, []);

  useEffect(() => {
    if (tab !== "view") return;
    if (!viewExperiment && karaokeReadyExperiments.length > 0) {
      setViewExperiment(karaokeReadyExperiments[0]);
    }
  }, [tab, karaokeReadyExperiments, viewExperiment]);

  useEffect(() => {
    if (!isPlaying) return;
    let timer = 0;
    const tick = () => {
      const a = audioRef.current;
      if (a) setPlaybackSeconds(a.currentTime);
      timer = window.setTimeout(tick, 60);
    };
    tick();
    return () => window.clearTimeout(timer);
  }, [isPlaying]);

  useEffect(() => {
    if (!activeKaraokeToken) return;
    const container = karaokeScrollRef.current;
    const el = tokenRefs.current[activeKaraokeToken.key];
    if (!el || !container) return;
    const containerRect = container.getBoundingClientRect();
    const tokenRect = el.getBoundingClientRect();
    const pinOffset = Math.max(120, container.clientHeight * 0.28);
    const delta = tokenRect.top - containerRect.top;
    const maxScroll = Math.max(0, container.scrollHeight - container.clientHeight);
    const target = Math.max(0, Math.min(maxScroll, container.scrollTop + delta - pinOffset));
    container.scrollTo({ top: target, behavior: "smooth" });
  }, [activeKaraokeToken?.key]);

  useEffect(() => {
    if (!viewPayload || !currentViewPage) return;
    const pageKey = String(currentViewPage);
    const img = pageImgRefs.current[pageKey];
    const canvas = pageCanvasRefs.current[pageKey];
    if (!img || !canvas || !img.complete) return;
    const natural = pageNaturalSize[pageKey];
    if (!natural || natural.w <= 0 || natural.h <= 0) return;

    canvas.width = natural.w;
    canvas.height = natural.h;
    canvas.style.width = `${natural.w}px`;
    canvas.style.height = `${natural.h}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, natural.w, natural.h);

    if (!offscreenCanvasRef.current) {
      offscreenCanvasRef.current = document.createElement("canvas");
    }
    const off = offscreenCanvasRef.current;
    off.width = natural.w;
    off.height = natural.h;
    const offCtx = off.getContext("2d");
    if (!offCtx) return;
    offCtx.clearRect(0, 0, natural.w, natural.h);
    offCtx.drawImage(img, 0, 0, natural.w, natural.h);

    const pageTokens = Array.isArray(viewPayload.tokensByPage[pageKey]) ? viewPayload.tokensByPage[pageKey] : [];
    const pageBounds = viewPayload.pageBoundsByPage?.[pageKey];

    for (let idx = 0; idx < pageTokens.length; idx += 1) {
      const t = pageTokens[idx];
      const key = `${currentViewPage}:${t.tokenId || idx}`;
      const rank = tokenRankByKey.get(key) ?? Number.MAX_SAFE_INTEGER;
      const isCurrent = activeKaraokeToken?.key === key && playbackSeconds >= t.start_s && playbackSeconds <= t.end_s;
      const isCompleted = activeTokenRank >= 0 && rank < activeTokenRank;
      if (!isCurrent && !isCompleted) continue;

      const progress = isCurrent
        ? Math.max(0, Math.min(1, (playbackSeconds - t.start_s) / Math.max(0.001, t.end_s - t.start_s)))
        : 1;
      const mapped = mapOcrTokenToRenderRect({
        token: t,
        naturalSize: natural,
        pageBounds,
      });
      const left = Math.max(0, Math.floor(mapped.left));
      const top = Math.max(0, Math.floor(mapped.top));
      const revealWidth = Math.max(1, Math.floor(mapped.width * progress));
      const width = Math.min(natural.w - left, revealWidth);
      const height = Math.min(natural.h - top, Math.max(1, Math.floor(mapped.height)));
      if (width <= 0 || height <= 0) continue;

      const source = offCtx.getImageData(left, top, width, height);
      const px = source.data;
      const tint = isCurrent ? { r: 16, g: 111, b: 94 } : { r: 38, g: 122, b: 86 };
      const alphaBoost = isCurrent ? 0.95 : 0.6;
      for (let i = 0; i < px.length; i += 4) {
        const a = px[i + 3];
        if (a < 20) {
          px[i + 3] = 0;
          continue;
        }
        const lum = 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2];
        if (lum > 192) {
          px[i + 3] = 0;
          continue;
        }
        const strength = ((192 - lum) / 192) * alphaBoost;
        px[i] = Math.round(px[i] * (1 - strength) + tint.r * strength);
        px[i + 1] = Math.round(px[i + 1] * (1 - strength) + tint.g * strength);
        px[i + 2] = Math.round(px[i + 2] * (1 - strength) + tint.b * strength);
        px[i + 3] = Math.round(255 * strength);
      }
      ctx.putImageData(source, left, top);
    }
  }, [
    viewPayload,
    currentViewPage,
    playbackSeconds,
    activeKaraokeToken?.key,
    activeTokenRank,
    pageNaturalSize,
    tokenRankByKey,
  ]);

  const openManualAlignModal = async () => {
    if (isRunning || activeStage !== "align") return;
    if (form.anchorToken.trim().length === 0) {
      setEvents((prev) => [`${new Date().toLocaleTimeString()} • ALIGN manual override blocked: missing anchor token`, ...prev]);
      return;
    }
    setShowManualAlignModal(true);
    setManualPromptLoading(true);
    setManualStatus("");
    setManualRecs([]);
    try {
      const response = await fetchWithTimeout(`${API_BASE}/api/alignment-manual-prompt`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ form }),
      }, 120000);
      const payload = (await parseApiResponse(response)) as Record<string, unknown>;
      if (!response.ok || !payload.ok) {
        const message = String(payload?.message || "failed to load manual alignment prompt");
        setManualStatus(message);
        return;
      }
      setManualPrompt(String(payload.prompt || ""));
    } catch (error) {
      setManualStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setManualPromptLoading(false);
    }
  };

  const importManualAlignment = async () => {
    if (isManualImporting || isRunning) return;
    if (!manualEnrichedJson.trim() || !manualReviewCsv.trim()) {
      setManualStatus("Upload both enriched JSON and review CSV files.");
      return;
    }
    setIsManualImporting(true);
    setManualStatus("");
    setManualRecs([]);
    try {
      const response = await fetchWithTimeout(`${API_BASE}/api/alignment-manual-import`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          form,
          uploaded: {
            enrichedJson: manualEnrichedJson,
            reviewCsv: manualReviewCsv,
            leftOutCsv: manualLeftOutCsv,
          },
        }),
      }, 120000);
      const payload = (await parseApiResponse(response)) as Record<string, unknown>;
      if (!response.ok || !payload.ok) {
        const message = String(payload?.message || "manual alignment validation failed");
        setManualStatus(message);
        const validator = (payload.validator || {}) as Record<string, unknown>;
        const recs = Array.isArray(validator.recommendations)
          ? validator.recommendations.filter((x): x is string => typeof x === "string")
          : [];
        setManualRecs(recs);
        return;
      }
      const artifacts = Array.isArray(payload.artifacts)
        ? payload.artifacts.filter((a): a is ArtifactPreview => typeof a === "object" && a !== null) as ArtifactPreview[]
        : [];
      setStages((prev) =>
        prev.map((s) =>
          s.id === "align"
            ? {
                ...s,
                status: "done",
                output: summarizeArtifactsForOutput(payload, artifacts),
              }
            : s
        )
      );
      setAlignmentReviewCsv(extractAlignmentCsv(artifacts));
      setLeftOutCsv(extractLeftOutCsv(artifacts));
      const validator = (payload.validator || {}) as Record<string, unknown>;
      const recs = Array.isArray(validator.recommendations)
        ? validator.recommendations.filter((x): x is string => typeof x === "string")
        : [];
      setManualRecs(recs);
      setManualStatus("Manual upload validated and Stage 4 marked complete.");
      setEvents((prev) => [`${new Date().toLocaleTimeString()} • ALIGN completed (manual override)`, ...prev]);
    } catch (error) {
      setManualStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setIsManualImporting(false);
    }
  };

  const copyManualPrompt = async () => {
    if (!manualPrompt.trim()) return;
    try {
      await navigator.clipboard.writeText(manualPrompt);
      setPromptCopied(true);
      window.setTimeout(() => setPromptCopied(false), 1200);
    } catch {
      setManualStatus("Clipboard copy failed. You can still select and copy manually.");
    }
  };

  const runKaraokeBuild = async () => {
    if (isKaraokeBuilding || isRunning) return;
    if (!allStagesDone) return;
    setIsKaraokeBuilding(true);
    try {
      const response = await fetchWithTimeout(`${API_BASE}/api/karaoke-build`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ form }),
      }, 120000);
      const payload = (await parseApiResponse(response)) as Record<string, unknown>;
      if (!response.ok || !payload.ok) {
        const message = String(payload?.message || "Karoke build failed");
        setEvents((prev) => [`${new Date().toLocaleTimeString()} • KAROKE failed: ${message}`, ...prev]);
        return;
      }
      setEvents((prev) => [`${new Date().toLocaleTimeString()} • KAROKE build completed`, ...prev]);
      await refreshExperimentOptions();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setEvents((prev) => [`${new Date().toLocaleTimeString()} • KAROKE failed: ${message}`, ...prev]);
    } finally {
      setIsKaraokeBuilding(false);
    }
  };

  const playKaraokeForExperiment = async () => {
    if (!viewExperiment) return;
    setViewMessage("");
    try {
      const response = await fetchWithTimeout(
        `${API_BASE}/api/karaoke-view?experiment=${encodeURIComponent(viewExperiment)}`,
        { method: "GET" },
        120000
      );
      const payload = (await parseApiResponse(response)) as Record<string, unknown>;
      if (!response.ok || !payload.ok) {
        setViewMessage(String(payload?.message || "Failed to load karaoke payload"));
        return;
      }
      setViewPayload(payload as unknown as KaraokeViewPayload);
      setImageLoadErrors({});
      setPageNaturalSize({});
      const typedPayload = payload as unknown as KaraokeViewPayload;
      setCurrentViewPage(Array.isArray(typedPayload.pages) && typedPayload.pages.length > 0 ? typedPayload.pages[0] : null);
      setPlayNonce(Date.now());
      setPlaybackSeconds(0);
      window.setTimeout(async () => {
        const a = audioRef.current;
        if (!a) return;
        a.currentTime = 0;
        try {
          await a.play();
          setIsPlaying(true);
        } catch {
          setViewMessage("Audio play was blocked by browser. Click play in audio control.");
        }
      }, 100);
    } catch (error) {
      setViewMessage(error instanceof Error ? error.message : String(error));
    }
  };

  const runStage = async (id: string): Promise<boolean> => {
    const stageIndex = stages.findIndex((s) => s.id === id);
    if (stageIndex === -1 || !stageAvailable(stageIndex) || isRunning) return false;
    if (id === "align" && form.anchorToken.trim().length === 0) {
      const message = "anchorToken is required for Stage 4 alignment";
      setStages((prev) =>
        prev.map((s) =>
          s.id === id
            ? {
                ...s,
                status: "error",
                output: message,
              }
            : s
        )
      );
      setEvents((prev) => [`${new Date().toLocaleTimeString()} • ${id.toUpperCase()} failed: ${message}`, ...prev]);
      return false;
    }

    setIsRunning(true);
    setStages((prev) => prev.map((s) => (s.id === id ? { ...s, status: "running" } : s)));

    try {
      if (id === "align") {
        const startResp = await fetchWithTimeout(
          `${API_BASE}/api/run-stage-async`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ stage: id, form }),
          },
          120000
        );
        const startPayload = (await parseApiResponse(startResp)) as Record<string, unknown>;
        if (!startResp.ok || !startPayload.ok) {
          const message = String(startPayload?.message || "alignment async start failed");
          setStages((prev) =>
            prev.map((s) =>
              s.id === id
                ? {
                    ...s,
                    status: "error",
                    output: JSON.stringify(startPayload, null, 2),
                  }
                : s
            )
          );
          setEvents((prev) => [`${new Date().toLocaleTimeString()} • ${id.toUpperCase()} failed: ${message}`, ...prev]);
          return false;
        }
        const jobId = String(startPayload.jobId || "");
        if (!jobId) {
          throw new Error("Missing jobId from async alignment start");
        }
        setEvents((prev) => [`${new Date().toLocaleTimeString()} • ALIGN started (job ${jobId.slice(0, 8)})`, ...prev]);

        const maxWaitMs = 45 * 60 * 1000;
        const pollEveryMs = 2500;
        const started = Date.now();
        let finalPayload: Record<string, unknown> | null = null;
        while (Date.now() - started < maxWaitMs) {
          // eslint-disable-next-line no-await-in-loop
          await new Promise((r) => window.setTimeout(r, pollEveryMs));
          // eslint-disable-next-line no-await-in-loop
          const pollResp = await fetchWithTimeout(
            `${API_BASE}/api/job-status?jobId=${encodeURIComponent(jobId)}`,
            { method: "GET" },
            120000
          );
          // eslint-disable-next-line no-await-in-loop
          const pollPayload = (await parseApiResponse(pollResp)) as Record<string, unknown>;
          if (!pollResp.ok) continue;
          const state = String(pollPayload.state || "");
          if (state === "running") continue;
          finalPayload = pollPayload;
          break;
        }
        if (!finalPayload) {
          throw new Error("Alignment job did not finish within 45 minutes");
        }
        if (!finalPayload.ok || String(finalPayload.state || "") !== "done") {
          const message = String(finalPayload?.message || "alignment job failed");
          setStages((prev) =>
            prev.map((s) =>
              s.id === id
                ? {
                    ...s,
                    status: "error",
                    output: JSON.stringify(finalPayload, null, 2),
                  }
                : s
            )
          );
          setEvents((prev) => [`${new Date().toLocaleTimeString()} • ${id.toUpperCase()} failed: ${message}`, ...prev]);
          return false;
        }
        const artifacts = Array.isArray(finalPayload.artifacts)
          ? finalPayload.artifacts.filter((a): a is ArtifactPreview => typeof a === "object" && a !== null) as ArtifactPreview[]
          : [];
        setStages((prev) =>
          prev.map((s) =>
            s.id === id
              ? {
                  ...s,
                  status: "done",
                  output: summarizeArtifactsForOutput(finalPayload, artifacts),
                }
              : s
          )
        );
        setAlignmentReviewCsv(extractAlignmentCsv(artifacts));
        setLeftOutCsv(extractLeftOutCsv(artifacts));
        setEvents((prev) => [`${new Date().toLocaleTimeString()} • ${id.toUpperCase()} completed`, ...prev]);
        return true;
      }

      const response = await fetchWithTimeout(`${API_BASE}/api/run-stage`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stage: id, form }),
      }, id === "align" ? ALIGNMENT_TIMEOUT_MS : 120000);
      const payload = (await parseApiResponse(response)) as Record<string, unknown>;
      if (!response.ok || !payload.ok) {
        const message = String(payload?.message || "stage execution failed");
        setStages((prev) =>
          prev.map((s) =>
            s.id === id
              ? {
                  ...s,
                  status: "error",
                  output: JSON.stringify(payload, null, 2),
                }
              : s
          )
        );
        setEvents((prev) => [`${new Date().toLocaleTimeString()} • ${id.toUpperCase()} failed: ${message}`, ...prev]);
        return false;
      }
      const artifacts = Array.isArray(payload.artifacts)
        ? payload.artifacts.filter((a): a is ArtifactPreview => typeof a === "object" && a !== null) as ArtifactPreview[]
        : [];
      setStages((prev) =>
        prev.map((s) =>
          s.id === id
            ? {
                ...s,
                status: "done",
                output: summarizeArtifactsForOutput(payload, artifacts),
              }
            : s
        )
      );
      if (id === "align") {
        setAlignmentReviewCsv(extractAlignmentCsv(artifacts));
        setLeftOutCsv(extractLeftOutCsv(artifacts));
      }
      if (id === "input") {
        void refreshExperimentOptions();
      }
      setEvents((prev) => [`${new Date().toLocaleTimeString()} • ${id.toUpperCase()} completed`, ...prev]);
      return true;
    } catch (error) {
      const message = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
      const hint =
        message.includes("AbortError")
          ? `Request timed out. Check API server on ${API_BASE} and retry.`
          : `If this is a network error, ensure frontend API server is running on ${API_BASE}.`;
      setStages((prev) =>
        prev.map((s) =>
          s.id === id
            ? {
                ...s,
                status: "error",
                output: `${message}\n\n${hint}`,
              }
            : s
        )
      );
      setEvents((prev) => [`${new Date().toLocaleTimeString()} • ${id.toUpperCase()} failed: ${message}`, ...prev]);
      return false;
    } finally {
      setIsRunning(false);
    }
  };

  const runAll = async () => {
    if (!isReadyToRunAll || isRunning) return;
    for (const stage of STAGE_TEMPLATES) {
      // eslint-disable-next-line no-await-in-loop
      const ok = await runStage(stage.id);
      if (!ok) break;
    }
  };

  const resetFrontendState = () => {
    if (isRunning) return;
    setExperimentMode("new");
    setSelectedExperiment("");
    setForm(INITIAL_FORM);
    setShowAdvanced(false);
    setOpenInfo(null);
    setActiveStage("input");
    setStages(STAGE_TEMPLATES);
    setEvents([]);
    setAlignmentReviewCsv("");
    setLeftOutCsv("");
    setAlignmentFeedback("");
  };

  const submitAlignmentFeedback = async () => {
    if (isSubmittingFeedback || isRunning) return;
    if (!alignmentFeedback.trim()) return;
    setIsSubmittingFeedback(true);
    try {
      const response = await fetchWithTimeout(`${API_BASE}/api/alignment-feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ form, feedback: alignmentFeedback }),
      }, ALIGNMENT_TIMEOUT_MS);
      const payload = (await parseApiResponse(response)) as Record<string, unknown>;
      if (!response.ok || !payload.ok) {
        const message = String(payload?.message || "alignment feedback execution failed");
        setEvents((prev) => [`${new Date().toLocaleTimeString()} • ALIGN FEEDBACK failed: ${message}`, ...prev]);
        return;
      }
      const artifacts = Array.isArray(payload.artifacts)
        ? payload.artifacts.filter((a): a is ArtifactPreview => typeof a === "object" && a !== null) as ArtifactPreview[]
        : [];
      setAlignmentReviewCsv(extractAlignmentCsv(artifacts));
      setLeftOutCsv(extractLeftOutCsv(artifacts));
      setEvents((prev) => [`${new Date().toLocaleTimeString()} • ALIGN FEEDBACK completed`, ...prev]);
    } catch (error) {
      const message = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
      setEvents((prev) => [`${new Date().toLocaleTimeString()} • ALIGN FEEDBACK failed: ${message}`, ...prev]);
    } finally {
      setIsSubmittingFeedback(false);
    }
  };

  const pickPdfFromFolder = () => {
    folderPickerRef.current?.click();
  };

  const onFolderPicked: ChangeEventHandler<HTMLInputElement> = (event) => {
    const files = Array.from(event.target.files || []);
    const firstPdf = files.find((file) => file.name.toLowerCase().endsWith(".pdf"));
    if (!firstPdf) {
      setEvents((prev) => [`${new Date().toLocaleTimeString()} • No PDF found in selected folder`, ...prev]);
      return;
    }
    const relativePath = (firstPdf as File & { webkitRelativePath?: string }).webkitRelativePath || firstPdf.name;
    setForm((f) => ({ ...f, pdfPath: relativePath }));
    setEvents((prev) => [`${new Date().toLocaleTimeString()} • PDF selected from folder: ${relativePath}`, ...prev]);
  };

  useEffect(() => {
    if (!experimentSlug) return;
    if (!activeStage) return;
    const load = async () => {
      try {
        const response = await fetchWithTimeout(
          `${API_BASE}/api/stage-artifacts?experiment=${encodeURIComponent(experimentSlug)}&stage=${encodeURIComponent(activeStage)}`,
          { method: "GET" },
          120000
        );
        const payload = (await parseApiResponse(response)) as Record<string, unknown>;
        if (!response.ok || !payload.ok) return;
        const artifacts = Array.isArray(payload.artifacts)
          ? payload.artifacts.filter((a): a is ArtifactPreview => typeof a === "object" && a !== null) as ArtifactPreview[]
          : [];
        if (activeStage === "align") {
          setAlignmentReviewCsv(extractAlignmentCsv(artifacts));
          setLeftOutCsv(extractLeftOutCsv(artifacts));
        }
      } catch {
        // keep UI stable if artifact fetch fails
      }
    };
    void load();
  }, [activeStage, experimentSlug]);

  return (
    <div className="app-shell">
      <div className="bg-glow one" />
      <div className="bg-glow two" />
      <header className="topbar">
        <h1>Vedke Orchestrator</h1>
        <div className="tab-strip">
          <button className={tab === "build" ? "tab active" : "tab"} onClick={() => setTab("build")}>
            Build
          </button>
          <button className={tab === "view" ? "tab active" : "tab"} onClick={() => setTab("view")}>
            View
          </button>
        </div>
      </header>

      {tab === "build" ? (
        <main className="build-layout">
          <aside className="stage-rail card">
            <h2>Pipeline Control</h2>
            <p className="muted">Run each stage, inspect output, then proceed.</p>
            {stages.map((stage, idx) => (
              <button
                key={stage.id}
                className={`stage-pill ${activeStage === stage.id ? "selected" : ""} ${stage.status}`}
                onClick={() => setActiveStage(stage.id)}
                disabled={!stageAvailable(idx)}
              >
                <span>
                  {idx + 1}. {stage.title}
                </span>
                <small>{stage.status}</small>
              </button>
            ))}
            <button className="primary-run" onClick={runAll} disabled={!isReadyToRunAll || isRunning}>
              Run Full Build
            </button>
            <button className="ghost" onClick={resetFrontendState} disabled={isRunning}>
              Reset Frontend
            </button>
          </aside>

          <section className="builder card">
            <h2>Build Inputs</h2>
            <div className="stage-heading">
              <h3>{stages.find((s) => s.id === activeStage)?.title}</h3>
              <p>{stages.find((s) => s.id === activeStage)?.hint}</p>
            </div>

            <div className="section-block">
              <h3>Mandatory Inputs</h3>
              <div className="mode-switch-row">
                <label>
                  <span className="field-title">Experiment Mode</span>
                  <select
                    value={experimentMode}
                    onChange={async (e) => {
                      const nextMode = e.target.value as "new" | "load";
                      setExperimentMode(nextMode);
                      if (nextMode === "new") {
                        setSelectedExperiment("");
                        setForm(INITIAL_FORM);
                        setStages(STAGE_TEMPLATES);
                        setActiveStage("input");
                      } else {
                        await refreshExperimentOptions();
                      }
                    }}
                  >
                    <option value="new">Create New Experiment</option>
                    <option value="load">Load Existing Experiment</option>
                  </select>
                </label>
                {experimentMode === "load" ? (
                  <label>
                    <span className="field-title">Existing Experiments</span>
                    <select
                      value={selectedExperiment}
                      onChange={async (e) => {
                        const slug = e.target.value;
                        setSelectedExperiment(slug);
                        if (slug) await loadExperiment(slug);
                      }}
                    >
                      <option value="">{loadingExperiments ? "Loading..." : "Select experiment"}</option>
                      {experimentOptions.map((x) => (
                        <option key={x.experiment} value={x.experiment}>
                          {x.experiment}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}
              </div>
              <div className="form-grid">
                <label>
                  <FieldTitle title="Experiment Name" fieldKey="experimentName" openInfo={openInfo} onInfoToggle={setOpenInfo} />
                  <input
                    value={form.experimentName}
                    onChange={(e) => setForm((f) => ({ ...f, experimentName: e.target.value }))}
                    placeholder="ganapati_trial_01"
                    readOnly={experimentMode === "load"}
                  />
                  {openInfo === "experimentName" ? <InfoBlob text={FIELD_HELP.experimentName} /> : null}
                </label>
                <label>
                  <FieldTitle title="YouTube URL" fieldKey="youtubeUrl" openInfo={openInfo} onInfoToggle={setOpenInfo} />
                  <input
                    value={form.youtubeUrl}
                    onChange={(e) => setForm((f) => ({ ...f, youtubeUrl: e.target.value }))}
                    placeholder="https://www.youtube.com/watch?v=..."
                    readOnly={experimentMode === "load"}
                  />
                  {openInfo === "youtubeUrl" ? <InfoBlob text={FIELD_HELP.youtubeUrl} /> : null}
                </label>
                <label>
                  <FieldTitle title="PDF Path" fieldKey="pdfPath" openInfo={openInfo} onInfoToggle={setOpenInfo} />
                  <div className="pdf-picker-field">
                    <input
                      value={form.pdfPath}
                      readOnly
                      placeholder="Select a folder to locate PDF"
                      className="pdf-readonly"
                    />
                    <button
                      type="button"
                      className="pick-btn-inline"
                      onClick={pickPdfFromFolder}
                      aria-label="Pick PDF folder"
                      title="Pick PDF folder"
                      disabled={experimentMode === "load"}
                    >
                      📁
                    </button>
                  </div>
                  <input
                    ref={folderPickerRef}
                    type="file"
                    style={{ display: "none" }}
                    onChange={onFolderPicked}
                    {...({ webkitdirectory: "true", directory: "true", multiple: true } as Record<string, string | boolean>)}
                  />
                  {openInfo === "pdfPath" ? <InfoBlob text={FIELD_HELP.pdfPath} /> : null}
                </label>
                <label>
                  <FieldTitle title="Page Selection" fieldKey="pages" openInfo={openInfo} onInfoToggle={setOpenInfo} />
                  <input
                    value={form.pages}
                    onChange={(e) => setForm((f) => ({ ...f, pages: e.target.value }))}
                    readOnly={experimentMode === "load"}
                  />
                  {openInfo === "pages" ? <InfoBlob text={FIELD_HELP.pages} /> : null}
                </label>
              </div>

              {experimentExists ? (
                <p className="overwrite-warning">
                  Warning: `data/{experimentSlug}` already exists. Existing input/output artifacts may be overwritten.
                </p>
              ) : null}

              <div className="location-box">
                <FieldTitle title="Artifact Locations" fieldKey="storageLocation" openInfo={openInfo} onInfoToggle={setOpenInfo} />
                <code>{inputDir}</code>
                <code>{outDir}</code>
                {openInfo === "storageLocation" ? <InfoBlob text={FIELD_HELP.storageLocation} /> : null}
              </div>
            </div>

            <div className="section-block">
              <button className="advanced-toggle" onClick={() => setShowAdvanced((v) => !v)}>
                {showAdvanced ? "Hide Advanced Parameters" : "Show Advanced Parameters"}
              </button>
              {showAdvanced ? (
                <div className="form-grid advanced">
                  <label>
                    <FieldTitle title="AI Model" fieldKey="aiModel" openInfo={openInfo} onInfoToggle={setOpenInfo} />
                    <select value={form.aiModel} onChange={(e) => setForm((f) => ({ ...f, aiModel: e.target.value }))}>
                      {AI_MODELS.map((model) => (
                        <option key={model} value={model}>
                          {model}
                        </option>
                      ))}
                    </select>
                    {openInfo === "aiModel" ? <InfoBlob text={FIELD_HELP.aiModel} /> : null}
                  </label>
                  <label>
                    <FieldTitle title="ASR Mode" fieldKey="asrMode" openInfo={openInfo} onInfoToggle={setOpenInfo} />
                    <select value={form.asrMode} onChange={(e) => setForm((f) => ({ ...f, asrMode: e.target.value }))}>
                      <option value="verbatim">verbatim</option>
                      <option value="translit">translit</option>
                    </select>
                    {openInfo === "asrMode" ? <InfoBlob text={FIELD_HELP.asrMode} /> : null}
                  </label>
                  <label>
                    <FieldTitle title="Language Code" fieldKey="languageCode" openInfo={openInfo} onInfoToggle={setOpenInfo} />
                    <input value={form.languageCode} onChange={(e) => setForm((f) => ({ ...f, languageCode: e.target.value }))} />
                    {openInfo === "languageCode" ? <InfoBlob text={FIELD_HELP.languageCode} /> : null}
                  </label>
                  <label>
                    <FieldTitle title="Top Trim Ratio" fieldKey="topTrim" openInfo={openInfo} onInfoToggle={setOpenInfo} />
                    <input value={form.topTrim} onChange={(e) => setForm((f) => ({ ...f, topTrim: e.target.value }))} />
                    {openInfo === "topTrim" ? <InfoBlob text={FIELD_HELP.topTrim} /> : null}
                  </label>
                  <label>
                    <FieldTitle title="Bottom Trim Ratio" fieldKey="bottomTrim" openInfo={openInfo} onInfoToggle={setOpenInfo} />
                    <input value={form.bottomTrim} onChange={(e) => setForm((f) => ({ ...f, bottomTrim: e.target.value }))} />
                    {openInfo === "bottomTrim" ? <InfoBlob text={FIELD_HELP.bottomTrim} /> : null}
                  </label>
                  <label>
                    <FieldTitle title="Drop Rules" fieldKey="dropRules" openInfo={openInfo} onInfoToggle={setOpenInfo} />
                    <input value={form.dropRules} onChange={(e) => setForm((f) => ({ ...f, dropRules: e.target.value }))} />
                    {openInfo === "dropRules" ? <InfoBlob text={FIELD_HELP.dropRules} /> : null}
                  </label>
                  <label>
                    <FieldTitle title="Min Confidence" fieldKey="minConfidence" openInfo={openInfo} onInfoToggle={setOpenInfo} />
                    <input value={form.minConfidence} onChange={(e) => setForm((f) => ({ ...f, minConfidence: e.target.value }))} />
                    {openInfo === "minConfidence" ? <InfoBlob text={FIELD_HELP.minConfidence} /> : null}
                  </label>
                  <label>
                    <FieldTitle title="Max Edit Cost" fieldKey="maxEditCost" openInfo={openInfo} onInfoToggle={setOpenInfo} />
                    <input value={form.maxEditCost} onChange={(e) => setForm((f) => ({ ...f, maxEditCost: e.target.value }))} />
                    {openInfo === "maxEditCost" ? <InfoBlob text={FIELD_HELP.maxEditCost} /> : null}
                  </label>
                <label>
                  <FieldTitle title="Anchor Token" fieldKey="anchorToken" openInfo={openInfo} onInfoToggle={setOpenInfo} />
                    <input
                      value={form.anchorToken}
                      onChange={(e) => setForm((f) => ({ ...f, anchorToken: e.target.value }))}
                      placeholder="P0008_T0007"
                    />
                    {openInfo === "anchorToken" ? <InfoBlob text={FIELD_HELP.anchorToken} /> : null}
                </label>
                </div>
              ) : null}
            </div>

            <div className="action-row">
              <button className="ghost" onClick={resetFrontendState} disabled={isRunning}>
                Reset Frontend
              </button>
              <button className="primary-run" onClick={runKaraokeBuild} disabled={!allStagesDone || isRunning || isKaraokeBuilding}>
                {isKaraokeBuilding ? "Building..." : "Karoke Build"}
              </button>
              {activeStage === "align" ? (
                <button className="ghost" onClick={openManualAlignModal} disabled={isRunning}>
                  Manual Override
                </button>
              ) : null}
              <button className="run-one" onClick={() => runStage(activeStage)} disabled={isRunning}>
                Run This Stage
              </button>
            </div>
          </section>

          <aside className="insights card">
            <h2>Interim Outputs</h2>
            <div className="validation-box">
              <h3>Validation</h3>
              {validations.map((v) => (
                <p key={v.key} className={v.ok ? "ok" : "bad"}>
                  {v.ok ? "✓" : "•"} {v.label}
                </p>
              ))}
            </div>

            <div className="output-box">
              <h3>Stage Artifact</h3>
              <pre>{stages.find((s) => s.id === activeStage)?.output || "Run the stage to view output preview."}</pre>
            </div>

            {activeStage === "align" ? (
              <div className="output-box">
                <div className="output-head">
                  <h3>Alignment Review CSV</h3>
                  <button className="ghost" onClick={() => setShowExpandedReview(true)} disabled={alignmentTableRows.length === 0}>
                    Enlarge
                  </button>
                </div>
                {alignmentTableRows.length === 0 ? (
                  <p className="muted">Run Alignment Config to generate review table.</p>
                ) : (
                  <div className="table-wrap">
                    <table className="review-table">
                      <thead>
                        <tr>
                          {compactReviewColumns.map((col) => (
                            <th key={col.key}>{col.label}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {alignmentTableRows.map((row) => (
                          <tr key={`${row.segment_index || "na"}-${row.coarse_start_token_id || "na"}-${row.coarse_end_token_id || "na"}`}>
                            {compactReviewColumns.map((col) => (
                              <td key={col.key}>{row[col.key] || "-"}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                <h3>Left Out Eligible OCR Tokens</h3>
                {leftOutTableRows.length === 0 ? (
                  <p className="muted">No left-out CSV found yet.</p>
                ) : (
                  <div className="table-wrap">
                    <table className="review-table">
                      <thead>
                        <tr>
                          <th>Page</th>
                          <th>Token ID</th>
                          <th>Token</th>
                          <th>Kind</th>
                          <th>Width</th>
                        </tr>
                      </thead>
                      <tbody>
                        {leftOutTableRows.map((row) => (
                          <tr key={`${row.page}-${row.tokenId}`}>
                            <td>{row.page || "-"}</td>
                            <td>{row.tokenId || "-"}</td>
                            <td>{row.token || "-"}</td>
                            <td>{row.kind || "-"}</td>
                            <td>{row.w || "-"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                <h3>Alignment Feedback</h3>
                <textarea
                  rows={5}
                  value={alignmentFeedback}
                  onChange={(e) => setAlignmentFeedback(e.target.value)}
                  placeholder="Example: in seg 35 move first token match by one token ahead"
                />
                <button className="run-one" onClick={submitAlignmentFeedback} disabled={isSubmittingFeedback || isRunning}>
                  {isSubmittingFeedback ? "Applying Feedback..." : "Apply Feedback + Rebuild Alignment"}
                </button>
              </div>
            ) : null}

            <div className="events-box">
              <h3>Execution Log</h3>
              {events.length === 0 ? <p className="muted">No stage runs yet.</p> : events.map((e) => <p key={e}>{e}</p>)}
            </div>
          </aside>
        </main>
      ) : (
        <main className="view-placeholder card">
          <h2>View Tab</h2>
          <div className="view-controls">
            <select value={viewExperiment} onChange={(e) => setViewExperiment(e.target.value)}>
              <option value="">Select karaoke-ready experiment</option>
              {karaokeReadyExperiments.map((exp) => (
                <option key={exp} value={exp}>
                  {exp}
                </option>
              ))}
            </select>
            <button className="run-one" onClick={playKaraokeForExperiment} disabled={!viewExperiment}>
              Play
            </button>
          </div>
          {viewMessage ? <p className="muted">{viewMessage}</p> : null}
          {viewPayload ? (
            <div className="karaoke-player-wrap">
              <audio
                ref={audioRef}
                src={`${API_BASE}${viewPayload.audioUrl}`}
                controls
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                onEnded={() => setIsPlaying(false)}
                onTimeUpdate={() => {
                  const a = audioRef.current;
                  if (a) setPlaybackSeconds(a.currentTime);
                }}
              />
              <div className="view-controls">
                <button
                  className="ghost"
                  onClick={() => {
                    const pages = viewPayload.pages || [];
                    const idx = pages.findIndex((p) => p === currentViewPage);
                    if (idx > 0) setCurrentViewPage(pages[idx - 1]);
                  }}
                  disabled={!currentViewPage || !viewPayload.pages.includes(currentViewPage) || viewPayload.pages.indexOf(currentViewPage) <= 0}
                >
                  Prev Page
                </button>
                <p className="muted">
                  Page {currentViewPage ?? "-"} / {viewPayload.pages[viewPayload.pages.length - 1] ?? "-"}
                </p>
                <button
                  className="ghost"
                  onClick={() => {
                    const pages = viewPayload.pages || [];
                    const idx = pages.findIndex((p) => p === currentViewPage);
                    if (idx >= 0 && idx + 1 < pages.length) setCurrentViewPage(pages[idx + 1]);
                  }}
                  disabled={
                    !currentViewPage ||
                    !viewPayload.pages.includes(currentViewPage) ||
                    viewPayload.pages.indexOf(currentViewPage) >= viewPayload.pages.length - 1
                  }
                >
                  Next Page
                </button>
              </div>
              <div className="karaoke-pages" ref={karaokeScrollRef}>
                {(currentViewPage ? [currentViewPage] : []).map((page) => {
                  const pageTokens = Array.isArray(viewPayload.tokensByPage[String(page)]) ? viewPayload.tokensByPage[String(page)] : [];
                  const natural = pageNaturalSize[String(page)];
                  const pageBounds = viewPayload.pageBoundsByPage?.[String(page)];
                  const imageUrl =
                    viewPayload.pageImageUrls?.[String(page)] ||
                    `${viewPayload.pageImageBaseUrl}${page}`;
                  const fullImageUrl = `${API_BASE}${imageUrl}${imageUrl.includes("?") ? "&" : "?"}v=${playNonce}`;
                  const imgSrc = viewPayload.pageImageData?.[String(page)] || fullImageUrl;
                  return (
                    <div key={page} className="karaoke-page-card">
                      <p className="muted">Page {page}</p>
                      <div
                        className="karaoke-page-layer"
                        style={
                          natural
                            ? { width: `${natural.w}px`, height: `${natural.h}px` }
                            : undefined
                        }
                      >
                        <img
                          src={imgSrc}
                          alt={`page-${page}`}
                          ref={(el) => {
                            pageImgRefs.current[String(page)] = el;
                          }}
                          onLoad={(e) => {
                            const el = e.currentTarget;
                            const w = Number(el.naturalWidth || 0);
                            const h = Number(el.naturalHeight || 0);
                            if (w > 0 && h > 0) {
                              setPageNaturalSize((prev) => ({ ...prev, [String(page)]: { w, h } }));
                            }
                          }}
                          onError={() =>
                            setImageLoadErrors((prev) => ({
                              ...prev,
                              [String(page)]: imgSrc,
                            }))
                          }
                        />
                        <canvas
                          className="karaoke-canvas-overlay"
                          ref={(el) => {
                            pageCanvasRefs.current[String(page)] = el;
                          }}
                        />
                        <div className="karaoke-token-overlay">
                          {pageTokens.map((t, idx) => {
                            const key = `${page}:${t.tokenId || idx}`;
                            const rank = tokenRankByKey.get(key) ?? Number.MAX_SAFE_INTEGER;
                            const isCurrent =
                              activeKaraokeToken?.key === key && playbackSeconds >= t.start_s && playbackSeconds <= t.end_s;
                            const isCompleted = activeTokenRank >= 0 && rank < activeTokenRank;
                            const progress = isCurrent
                              ? Math.max(0, Math.min(1, (playbackSeconds - t.start_s) / Math.max(0.001, t.end_s - t.start_s)))
                              : isCompleted
                                ? 1
                                : 0;
                            const mapped = mapOcrTokenToRenderRect({
                              token: t,
                              naturalSize: natural,
                              pageBounds,
                            });
                            const stripeHeight = Math.max(3, mapped.height * 0.18);
                            return (
                              <div
                                key={key}
                                ref={(el) => {
                                  tokenRefs.current[key] = el;
                                }}
                                className={`karaoke-token ${isCompleted ? "completed" : ""} ${isCurrent ? "active" : ""}`}
                                style={{
                                  left: `${mapped.left}px`,
                                  top: `${mapped.top + mapped.height - stripeHeight}px`,
                                  width: `${mapped.width}px`,
                                  height: `${stripeHeight}px`,
                                  ["--reveal" as string]: `${Math.round(progress * 100)}%`,
                                }}
                                title={`${t.token} (${t.start_s.toFixed(2)}-${t.end_s.toFixed(2)})`}
                              />
                            );
                          })}
                        </div>
                      </div>
                      {imageLoadErrors[String(page)] ? (
                        <p className="muted">
                          Image load failed for page {page}: {imageLoadErrors[String(page)]}
                        </p>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <p>Karaoke text-audio alignment view will be implemented next.</p>
          )}
        </main>
      )}

      {showExpandedReview ? (
        <div className="modal-backdrop" onClick={() => setShowExpandedReview(false)}>
          <div className="modal-card modal-card-wide" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <h3>Alignment Review CSV (Expanded)</h3>
              <button className="ghost" onClick={() => setShowExpandedReview(false)}>
                Close
              </button>
            </div>
            <div className="table-wrap table-wrap-xl">
              <table className="review-table">
                <thead>
                  <tr>
                    {reviewColumns.map((col) => (
                      <th key={col.key}>{col.label}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {alignmentTableRows.map((row) => (
                    <tr key={`${row.segment_index || "na"}-${row.coarse_start_token_id || "na"}-${row.coarse_end_token_id || "na"}-expanded`}>
                      {reviewColumns.map((col) => (
                        <td key={col.key}>{row[col.key] || "-"}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ) : null}

      {showManualAlignModal ? (
        <div className="modal-backdrop" onClick={() => setShowManualAlignModal(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <h3>Stage 4 Manual Override</h3>
              <button className="ghost" onClick={() => setShowManualAlignModal(false)}>
                Close
              </button>
            </div>
            <p className="muted">
              Use this mode to run alignment in ChatGPT manually, then upload outputs for Python guardrail validation.
            </p>
            <ol className="manual-steps">
              <li>Open ChatGPT and paste the prompt below.</li>
              <li>Attach `0.json` and `pdf_tokens.json` from this experiment out folder.</li>
              <li>Run the prompt and download generated files.</li>
              <li>Upload files below and click Validate + Import.</li>
            </ol>
            <div className="modal-head">
              <h3>Prompt</h3>
              <button className="ghost" onClick={copyManualPrompt} disabled={!manualPrompt.trim()}>
                {promptCopied ? "Copied" : "Copy Prompt"}
              </button>
            </div>
            {manualPromptLoading ? (
              <p className="muted">Loading prompt...</p>
            ) : (
              <textarea className="manual-prompt" rows={14} value={manualPrompt} readOnly />
            )}
            <div className="manual-upload-grid">
              <label>
                Enriched JSON (required)
                <input
                  type="file"
                  accept=".json,application/json"
                  onChange={async (e) => {
                    const f = e.target.files?.[0];
                    if (!f) return;
                    setManualEnrichedJson(await readFileAsText(f));
                  }}
                />
              </label>
              <label>
                Review CSV (required)
                <input
                  type="file"
                  accept=".csv,text/csv"
                  onChange={async (e) => {
                    const f = e.target.files?.[0];
                    if (!f) return;
                    setManualReviewCsv(await readFileAsText(f));
                  }}
                />
              </label>
              <label>
                Left-Out CSV (optional)
                <input
                  type="file"
                  accept=".csv,text/csv"
                  onChange={async (e) => {
                    const f = e.target.files?.[0];
                    if (!f) return;
                    setManualLeftOutCsv(await readFileAsText(f));
                  }}
                />
              </label>
            </div>
            {manualStatus ? <p className="manual-status">{manualStatus}</p> : null}
            {manualRecs.length > 0 ? (
              <div className="manual-recs">
                <h4>Recommendations</h4>
                {manualRecs.map((r) => (
                  <p key={r}>- {r}</p>
                ))}
              </div>
            ) : null}
            <div className="action-row">
              <button className="ghost" onClick={() => setShowManualAlignModal(false)}>
                Cancel
              </button>
              <button className="run-one" onClick={importManualAlignment} disabled={isManualImporting || isRunning}>
                {isManualImporting ? "Validating..." : "Validate + Import"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function FieldTitle({
  title,
  fieldKey,
  openInfo,
  onInfoToggle
}: {
  title: string;
  fieldKey: string;
  openInfo: string | null;
  onInfoToggle: (key: string | null) => void;
}) {
  return (
    <span className="field-title">
      {title}
      <button type="button" className="info-btn" onClick={() => onInfoToggle(openInfo === fieldKey ? null : fieldKey)}>
        i
      </button>
    </span>
  );
}

function InfoBlob({ text }: { text: string }) {
  return <small className="info-blob">{text}</small>;
}

export default App;
