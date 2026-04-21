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
  audioInputMode: "youtube" | "mp3";
  youtubeUrl: string;
  mp3FileData: string;
  mp3FileName: string;
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
  anchorSegmentIndex: string;
  lastToken: string;
  minSeconds: string;
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

type ReviewEditorToken = {
  tokenId: string;
  token: string;
  charCount: number;
  w: number;
  matraCount: number;
  globalIdx: number;
  isEligible: boolean;
  hasTimestamp: boolean;
  timingSegmentIndex: number | undefined;
};

type ReviewEditorRow = {
  segmentIndex: number;
  audioText: string;
  audioStartS: number;
  audioEndS: number;
  status: string;
  coverage: string;
  coarseStart: string;
  coarseEnd: string;
  keptTokens: ReviewEditorToken[];
  leftOutTokens: ReviewEditorToken[];
  timeReductionS: number;
};

// Mātrā counter — mirrors the logic in server.mjs countMatras().
// short vowel in open syllable = 1 mātrā (laghu); long vowel or closed syllable = 2 mātrās (guru).
function countMatras(token: string): number {
  const SHORT_V_SIGNS = new Set([0x093F, 0x0941, 0x0943, 0x0946, 0x094A]);
  const LONG_V_SIGNS  = new Set([0x093E, 0x0940, 0x0942, 0x0944, 0x0947, 0x0948, 0x094B, 0x094C]);
  const SHORT_IND     = new Set([0x0905, 0x0907, 0x0909, 0x090B, 0x090C]);
  const LONG_IND      = new Set([0x0906, 0x0908, 0x090A, 0x0960, 0x090F, 0x0910, 0x0913, 0x0914]);
  const VIRAMA = 0x094D, ANUSVARA = 0x0902, VISARGA = 0x0903;
  const cps = [...token].map(c => c.codePointAt(0) as number);
  const syllables: { heavy: boolean }[] = [];
  let i = 0;
  while (i < cps.length) {
    const cp = cps[i];
    if (SHORT_IND.has(cp)) {
      const next = cps[i + 1];
      syllables.push({ heavy: next === ANUSVARA || next === VISARGA });
      i++;
    } else if (LONG_IND.has(cp)) {
      syllables.push({ heavy: true });
      i++;
    } else if (cp >= 0x0915 && cp <= 0x0939) {
      const next = cps[i + 1];
      if (next === VIRAMA) {
        if (syllables.length > 0) syllables[syllables.length - 1].heavy = true;
        i += 2;
      } else if (SHORT_V_SIGNS.has(next)) {
        const next2 = cps[i + 2];
        syllables.push({ heavy: next2 === ANUSVARA || next2 === VISARGA });
        i += 2;
      } else if (LONG_V_SIGNS.has(next)) {
        syllables.push({ heavy: true });
        i += 2;
      } else if (next === ANUSVARA || next === VISARGA) {
        syllables.push({ heavy: true });
        i += 2;
      } else {
        syllables.push({ heavy: false });
        i++;
      }
    } else if (cp === ANUSVARA || cp === VISARGA) {
      if (syllables.length > 0) syllables[syllables.length - 1].heavy = true;
      i++;
    } else {
      i++;
    }
  }
  return Math.max(1, syllables.reduce((s, syl) => s + (syl.heavy ? 2 : 1), 0));
}

const INITIAL_FORM: BuildForm = {
  experimentName: "",
  audioInputMode: "youtube",
  youtubeUrl: "",
  mp3FileData: "",
  mp3FileName: "",
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
  anchorToken: "",
  anchorSegmentIndex: "0",
  lastToken: "",
  minSeconds: "0.4"
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
  anchorToken: "Alignment anchor tokenId from pdf_tokens.json (example: P0008_T0007).",
  anchorSegmentIndex: "0-based index of the audio segment that contains the anchor token. Default 0 means anchor is in the first segment. Set to 1 if the first segment starts with audio words that have no OCR match (e.g. an introductory syllable before the first OCR token).",
  lastToken: "Optional. Last OCR tokenId to include in alignment (example: P0010_T0042). If left blank, all tokens from the anchor token to the end of the PDF are considered. If set, only tokens between anchor and last token (inclusive) are sent to the LLM.",
  minSeconds: "Minimum time (seconds) guaranteed to each token during Review Realign timestamp redistribution. Tokens that would receive less than this are pinned to this floor, and the remaining duration is redistributed among longer tokens. Default 0.4s."
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
  const [showAlignReviewEditor, setShowAlignReviewEditor] = useState(false);
  const [reviewEditorRows, setReviewEditorRows] = useState<ReviewEditorRow[]>([]);
  const [isRealigning, setIsRealigning] = useState(false);
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
  const [karaokeZoom, setKaraokeZoom] = useState<number>(1.0);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const tokenRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const karaokeScrollRef = useRef<HTMLDivElement | null>(null);
  const pageImgRefs = useRef<Record<string, HTMLImageElement | null>>({});
  const pageCanvasRefs = useRef<Record<string, HTMLCanvasElement | null>>({});
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  // Tracks the last line we scrolled to so we only reposition on line transitions.
  const lastScrolledLineKey = useRef<string | null>(null);

  const experimentSlug = useMemo(() => slugify(form.experimentName), [form.experimentName]);
  const inputDir = experimentSlug ? `data/${experimentSlug}/input` : "data/<experiment>/input";
  const outDir = experimentSlug ? `data/${experimentSlug}/out` : "data/<experiment>/out";
  const experimentExists = experimentSlug.length > 0 && experimentOptions.some((e) => e.experiment === experimentSlug);

  const validations = useMemo(() => {
    return [
      { key: "exp", label: "Experiment name is set", ok: experimentSlug.length > 0 },
      {
        key: "yt",
        label: form.audioInputMode === "mp3" ? "MP3 file selected" : "YouTube URL is set",
        ok: form.audioInputMode === "mp3" ? form.mp3FileData.length > 0 : form.youtubeUrl.trim().length > 0,
      },
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

  // Line key: page + OCR-space y quantized to 200px bins.
  // Sanskrit OCR tokens on the same visual line have bounding-box tops that
  // vary by up to ~70px (complex stacking characters, diacritics). Adjacent
  // lines are ~135–185px apart at 300 DPI. A 200px bin safely groups all
  // tokens on the same visual line while keeping adjacent lines in separate bins.
  const activeLineKey = useMemo(() => {
    if (!activeKaraokeToken) return null;
    return `${activeKaraokeToken.page}:${Math.round(activeKaraokeToken.y / 200) * 200}`;
  }, [activeKaraokeToken]);

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
      audioInputMode: (loadedForm.audioInputMode === "mp3" ? "mp3" : "youtube") as "youtube" | "mp3",
      youtubeUrl: String(loadedForm.youtubeUrl || ""),
      pdfPath: String(loadedForm.pdfPath || ""),
      pages: String(loadedForm.pages || ""),
      anchorToken: String(loadedForm.anchorToken || ""),
      anchorSegmentIndex: String(loadedForm.anchorSegmentIndex ?? "0"),
      lastToken: String(loadedForm.lastToken || ""),
      minSeconds: String(loadedForm.minSeconds ?? "0.4"),
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

  // Reset scroll gate when zoom changes so the active line re-anchors at the new scale.
  useEffect(() => {
    lastScrolledLineKey.current = null;
  }, [karaokeZoom]);

  // Rolling karaoke scroll: reposition on line change.
  // Uses the page image element position + token.y * scaleY * zoom to compute the
  // scroll target — no dependency on token anchor div positions, which were
  // unreliable when flex-shrink compressed page cards into nested scroll contexts.
  // Returns early when pageNaturalSize is not yet set; the dep on pageNaturalSize
  // causes a retry as soon as the image loads, and lastScrolledLineKey is only
  // committed after all guards pass so no line is silently skipped.
  useEffect(() => {
    if (!activeKaraokeToken || !activeLineKey || !viewPayload) return;
    if (activeLineKey === lastScrolledLineKey.current) return;

    const container = karaokeScrollRef.current;
    const img = pageImgRefs.current[String(activeKaraokeToken.page)];
    const natural = pageNaturalSize[String(activeKaraokeToken.page)];
    const pageBounds = viewPayload.pageBoundsByPage?.[String(activeKaraokeToken.page)];
    if (!container || !img || !natural || !pageBounds) return;

    lastScrolledLineKey.current = activeLineKey;

    const scaleY = natural.h / Math.max(1, pageBounds.maxBottom);
    const tokenRenderedY = activeKaraokeToken.y * scaleY * karaokeZoom;
    const containerTop = container.getBoundingClientRect().top;
    const imgTop = img.getBoundingClientRect().top;
    const imgAbsoluteTop = imgTop - containerTop + container.scrollTop;
    const tokenAbsoluteTop = imgAbsoluteTop + tokenRenderedY;
    const pinOffset = container.clientHeight * 0.38;
    const maxScroll = Math.max(0, container.scrollHeight - container.clientHeight);
    const target = Math.max(0, Math.min(maxScroll, tokenAbsoluteTop - pinOffset));
    container.scrollTo({ top: target, behavior: "smooth" });
  }, [activeLineKey, pageNaturalSize, viewPayload, karaokeZoom]);

  // Keep the page label in sync with whichever page the active token is on.
  useEffect(() => {
    if (activeKaraokeToken?.page != null) {
      setCurrentViewPage(activeKaraokeToken.page);
    }
  }, [activeKaraokeToken?.page]);

  // Draw highlights on every page's canvas. Runs on every playback tick so all
  // pages stay in sync as the audio progresses (completed tokens accumulate).
  useEffect(() => {
    if (!viewPayload) return;
    if (!offscreenCanvasRef.current) {
      offscreenCanvasRef.current = document.createElement("canvas");
    }
    const off = offscreenCanvasRef.current;

    for (const pageNum of viewPayload.pages) {
      const pageKey = String(pageNum);
      const img = pageImgRefs.current[pageKey];
      const canvas = pageCanvasRefs.current[pageKey];
      if (!img || !canvas || !img.complete) continue;
      const natural = pageNaturalSize[pageKey];
      if (!natural || natural.w <= 0 || natural.h <= 0) continue;

      canvas.width = natural.w;
      canvas.height = natural.h;
      canvas.style.width = `${natural.w * karaokeZoom}px`;
      canvas.style.height = `${natural.h * karaokeZoom}px`;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;

      off.width = natural.w;
      off.height = natural.h;
      const offCtx = off.getContext("2d");
      if (!offCtx) continue;
      offCtx.clearRect(0, 0, natural.w, natural.h);
      offCtx.drawImage(img, 0, 0, natural.w, natural.h);

      const pageTokens = Array.isArray(viewPayload.tokensByPage[pageKey]) ? viewPayload.tokensByPage[pageKey] : [];
      const pageBounds = viewPayload.pageBoundsByPage?.[pageKey];

      // Determine which tokens on this page are active or completed
      const tokenStates: Array<{ isCurrent: boolean; isCompleted: boolean; progress: number }> = pageTokens.map((t, idx) => {
        const key = `${pageNum}:${t.tokenId || idx}`;
        const rank = tokenRankByKey.get(key) ?? Number.MAX_SAFE_INTEGER;
        const isCurrent = activeKaraokeToken?.key === key && playbackSeconds >= t.start_s && playbackSeconds <= t.end_s;
        const isCompleted = activeTokenRank >= 0 && rank < activeTokenRank;
        const progress = isCurrent
          ? Math.max(0, Math.min(1, (playbackSeconds - t.start_s) / Math.max(0.001, t.end_s - t.start_s)))
          : 1;
        return { isCurrent, isCompleted, progress };
      });

      const pageHasActivity = tokenStates.some((s) => s.isCurrent || s.isCompleted);

      // Clear canvas — matched token regions are painted below; all other areas
      // remain transparent so the background <img> shows through unmodified.
      ctx.clearRect(0, 0, natural.w, natural.h);

      // Paint each active/completed token with bright green tint.
      for (let idx = 0; idx < pageTokens.length; idx += 1) {
        const { isCurrent, isCompleted, progress } = tokenStates[idx];
        if (!isCurrent && !isCompleted) continue;
        const t = pageTokens[idx];

        const mapped = mapOcrTokenToRenderRect({ token: t, naturalSize: natural, pageBounds });
        const left = Math.max(0, Math.floor(mapped.left));
        const top = Math.max(0, Math.floor(mapped.top));
        const revealWidth = Math.max(1, Math.floor(mapped.width * progress));
        const width = Math.min(natural.w - left, revealWidth);
        const height = Math.min(natural.h - top, Math.max(1, Math.floor(mapped.height)));
        if (width <= 0 || height <= 0) continue;

        // Read from the undimmed offscreen copy so token pixels are full-brightness
        const source = offCtx.getImageData(left, top, width, height);
        const px = source.data;
        // Current word: bright lime-green; completed words: softer green
        const tint = isCurrent ? { r: 60, g: 210, b: 80 } : { r: 50, g: 180, b: 70 };
        const alphaBoost = isCurrent ? 0.88 : 0.52;
        for (let i = 0; i < px.length; i += 4) {
          const a = px[i + 3];
          if (a < 20) { px[i + 3] = 0; continue; }
          const lum = 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2];
          if (lum > 192) { px[i + 3] = 0; continue; }
          const strength = ((192 - lum) / 192) * alphaBoost;
          px[i] = Math.round(px[i] * (1 - strength) + tint.r * strength);
          px[i + 1] = Math.round(px[i + 1] * (1 - strength) + tint.g * strength);
          px[i + 2] = Math.round(px[i + 2] * (1 - strength) + tint.b * strength);
          px[i + 3] = 255;
        }
        ctx.putImageData(source, left, top);
      }
    }
  }, [
    viewPayload,
    playbackSeconds,
    activeKaraokeToken?.key,
    activeTokenRank,
    pageNaturalSize,
    tokenRankByKey,
    karaokeZoom,
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

  const openAlignReviewEditor = async () => {
    if (alignmentTableRows.length === 0) return;
    try {
      const response = await fetchWithTimeout(
        `${API_BASE}/api/alignment-enriched-tokens?experiment=${encodeURIComponent(form.experimentName)}`,
        { method: "GET" },
        120000
      );
      const data = (await parseApiResponse(response)) as Record<string, unknown>;
      if (!response.ok || !data.ok) {
        setEvents((prev) => [`${new Date().toLocaleTimeString()} • ALIGN EDITOR failed: ${String(data?.message || "fetch failed")}`, ...prev]);
        return;
      }
      const pages = (data.pages || {}) as Record<string, Array<Record<string, unknown>>>;
      const savedTimeReductions = (data.segmentTimeReductions || {}) as Record<string, number>;
      const flat: ReviewEditorToken[] = [];
      for (const pgKey of Object.keys(pages).sort((a, b) => Number(a) - Number(b))) {
        for (const tok of pages[pgKey]) {
          flat.push({
            tokenId: String(tok.tokenId || ""),
            token: String(tok.token || ""),
            charCount: Number(tok.char_count ?? (tok.token ? String(tok.token).length : 0)),
            w: Number(tok.w ?? 0),
            matraCount: countMatras(String(tok.token || "")),
            globalIdx: flat.length,
            isEligible: !!tok.is_eligible,
            // Source of truth: enriched file's timestamps, not the CSV's kept_token_ids
            hasTimestamp: "start_time_seconds" in tok,
            timingSegmentIndex: typeof tok.timing_segment_index === "number" ? (tok.timing_segment_index as number) : undefined,
          });
        }
      }
      const tokenGlobalIdx = new Map<string, number>();
      for (const t of flat) tokenGlobalIdx.set(t.tokenId, t.globalIdx);

      // Build keptTokens per segment directly from the enriched file
      const keptBySegment = new Map<number, ReviewEditorToken[]>();
      for (const tok of flat) {
        if (tok.hasTimestamp && tok.timingSegmentIndex !== undefined) {
          const arr = keptBySegment.get(tok.timingSegmentIndex) ?? [];
          arr.push(tok);
          keptBySegment.set(tok.timingSegmentIndex, arr);
        }
      }
      // Sort each segment's kept tokens by reading order
      for (const arr of keptBySegment.values()) arr.sort((a, b) => a.globalIdx - b.globalIdx);

      // All eligible tokens without timestamps → candidates for left-out assignment
      const unmatched = flat.filter((t) => t.isEligible && !t.hasTimestamp);

      const rows: ReviewEditorRow[] = [];
      for (const row of alignmentTableRows) {
        const segIdx = Number(row.segment_index ?? 0);
        const coarseStartId = String(row.coarse_start_token_id || "");
        const coarseEndId = String(row.coarse_end_token_id || "");
        rows.push({
          segmentIndex: segIdx,
          audioText: String(row.audio_text || ""),
          audioStartS: Number(row.audio_start_s ?? 0),
          audioEndS: Number(row.audio_end_s ?? 0),
          status: String(row.status || ""),
          coverage: String(row.coverage_ratio_estimate || ""),
          coarseStart: coarseStartId,
          coarseEnd: coarseEndId,
          keptTokens: keptBySegment.get(segIdx) ?? [],
          leftOutTokens: [],
          timeReductionS: Number(savedTimeReductions[String(segIdx)] ?? 0),
        });
      }

      // Assign each unmatched eligible token to the segment that owns the position
      // it appears in, based on OCR reading order.
      // Strategy: build a sorted list of all kept tokens (globalIdx → rowIndex),
      // then for each left-out token find the last kept token whose globalIdx < it —
      // the left-out token belongs to that kept token's segment.
      if (rows.length > 0) {
        // All kept tokens across all segments, sorted by reading order
        const allKept: { globalIdx: number; rowIdx: number }[] = [];
        for (let i = 0; i < rows.length; i++) {
          for (const kt of rows[i].keptTokens) {
            allKept.push({ globalIdx: kt.globalIdx, rowIdx: i });
          }
        }
        allKept.sort((a, b) => a.globalIdx - b.globalIdx);

        for (const tok of unmatched) {
          // Find the last kept token that comes before (or at) this token in reading order
          let targetRowIdx = 0; // default: first segment
          for (const kt of allKept) {
            if (kt.globalIdx <= tok.globalIdx) targetRowIdx = kt.rowIdx;
            else break;
          }
          rows[targetRowIdx].leftOutTokens.push(tok);
        }
        for (const row of rows) row.leftOutTokens.sort((a, b) => a.globalIdx - b.globalIdx);
      }

      setReviewEditorRows(rows);
      setShowAlignReviewEditor(true);
    } catch (error) {
      setEvents((prev) => [`${new Date().toLocaleTimeString()} • ALIGN EDITOR error: ${error instanceof Error ? error.message : String(error)}`, ...prev]);
    }
  };

  function moveTokenInEditor(fromRow: number, fromCol: "kept" | "left", tokenId: string, toRow: number, toCol: "kept" | "left") {
    setReviewEditorRows((rows) => {
      const next = rows.map((r) => ({ ...r, keptTokens: [...r.keptTokens], leftOutTokens: [...r.leftOutTokens] }));
      const srcRow = next[fromRow];
      const token = (fromCol === "kept" ? srcRow.keptTokens : srcRow.leftOutTokens).find((t) => t.tokenId === tokenId);
      if (!token) return rows;
      if (fromCol === "kept") srcRow.keptTokens = srcRow.keptTokens.filter((t) => t.tokenId !== tokenId);
      else srcRow.leftOutTokens = srcRow.leftOutTokens.filter((t) => t.tokenId !== tokenId);
      const tgtRow = next[toRow];
      if (toCol === "kept") tgtRow.keptTokens = [...tgtRow.keptTokens, token].sort((a, b) => a.globalIdx - b.globalIdx);
      else tgtRow.leftOutTokens = [...tgtRow.leftOutTokens, token].sort((a, b) => a.globalIdx - b.globalIdx);
      return next;
    });
  }

  const realignFromEditor = async () => {
    setIsRealigning(true);
    try {
      const segments = reviewEditorRows.map((row) => ({
        segmentIndex: row.segmentIndex,
        keptTokenIds: row.keptTokens.map((t) => t.tokenId),
        audioStartS: row.audioStartS,
        audioEndS: row.audioEndS,
        timeReductionS: row.timeReductionS ?? 0,
      }));
      const response = await fetchWithTimeout(`${API_BASE}/api/alignment-realign`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ form: { experimentName: form.experimentName, minSeconds: form.minSeconds }, segments }),
      }, 120000);
      const data = (await parseApiResponse(response)) as Record<string, unknown>;
      if (!response.ok || !data.ok) {
        setEvents((prev) => [`${new Date().toLocaleTimeString()} • REALIGN failed: ${String(data?.message || "error")}`, ...prev]);
        return;
      }
      if (typeof data.alignmentReviewCsv === "string" && data.alignmentReviewCsv) {
        setAlignmentReviewCsv(data.alignmentReviewCsv);
      }
      setEvents((prev) => [`${new Date().toLocaleTimeString()} • REALIGN completed`, ...prev]);
      setShowAlignReviewEditor(false);
      // Refresh the karaoke view payload so dropped-token time is immediately
      // redistributed and the stale highlighting is gone.
      if (viewExperiment === form.experimentName) {
        try {
          const viewResp = await fetchWithTimeout(
            `${API_BASE}/api/karaoke-view?experiment=${encodeURIComponent(viewExperiment)}`,
            { method: "GET" },
            120000
          );
          const viewData = (await parseApiResponse(viewResp)) as Record<string, unknown>;
          if (viewResp.ok && viewData.ok) {
            setViewPayload(viewData as unknown as KaraokeViewPayload);
            setImageLoadErrors({});
            setPageNaturalSize({});
          }
        } catch (_) { /* non-fatal: user can switch to View tab to reload manually */ }
      }
    } catch (error) {
      setEvents((prev) => [`${new Date().toLocaleTimeString()} • REALIGN error: ${error instanceof Error ? error.message : String(error)}`, ...prev]);
    } finally {
      setIsRealigning(false);
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
                  <span className="field-title">Audio Input</span>
                  <div style={{ display: "flex", gap: "0.5rem", marginBottom: "0.4rem" }}>
                    <button
                      type="button"
                      className={form.audioInputMode === "youtube" ? "primary-run" : "ghost"}
                      style={{ padding: "0.2rem 0.7rem", fontSize: "0.85rem" }}
                      onClick={() => setForm((f) => ({ ...f, audioInputMode: "youtube" }))}
                      disabled={experimentMode === "load"}
                    >
                      YouTube URL
                    </button>
                    <button
                      type="button"
                      className={form.audioInputMode === "mp3" ? "primary-run" : "ghost"}
                      style={{ padding: "0.2rem 0.7rem", fontSize: "0.85rem" }}
                      onClick={() => setForm((f) => ({ ...f, audioInputMode: "mp3" }))}
                      disabled={experimentMode === "load"}
                    >
                      Upload MP3
                    </button>
                  </div>
                  {form.audioInputMode === "youtube" ? (
                    <>
                      <input
                        value={form.youtubeUrl}
                        onChange={(e) => setForm((f) => ({ ...f, youtubeUrl: e.target.value }))}
                        placeholder="https://www.youtube.com/watch?v=..."
                        readOnly={experimentMode === "load"}
                      />
                      {openInfo === "youtubeUrl" ? <InfoBlob text={FIELD_HELP.youtubeUrl} /> : null}
                    </>
                  ) : (
                    <>
                      <input
                        type="file"
                        accept="audio/mpeg,audio/mp3,.mp3"
                        disabled={experimentMode === "load"}
                        onChange={(e) => {
                          const file = e.target.files?.[0];
                          if (!file) return;
                          const reader = new FileReader();
                          reader.onload = () => {
                            const b64 = (reader.result as string).split(",")[1] ?? "";
                            setForm((f) => ({ ...f, mp3FileData: b64, mp3FileName: file.name }));
                          };
                          reader.readAsDataURL(file);
                        }}
                      />
                      {form.mp3FileName ? (
                        <span style={{ fontSize: "0.8rem", color: "#888" }}>{form.mp3FileName}</span>
                      ) : null}
                    </>
                  )}
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
                <label>
                  <FieldTitle title="Last Token" fieldKey="lastToken" openInfo={openInfo} onInfoToggle={setOpenInfo} />
                    <input
                      value={form.lastToken}
                      onChange={(e) => setForm((f) => ({ ...f, lastToken: e.target.value }))}
                      placeholder="Optional — leave blank to use all tokens after anchor"
                    />
                    {openInfo === "lastToken" ? <InfoBlob text={FIELD_HELP.lastToken} /> : null}
                </label>
                <label>
                  <FieldTitle title="Anchor Segment Index" fieldKey="anchorSegmentIndex" openInfo={openInfo} onInfoToggle={setOpenInfo} />
                  <input
                    type="number"
                    min="0"
                    value={form.anchorSegmentIndex}
                    onChange={(e) => setForm((f) => ({ ...f, anchorSegmentIndex: e.target.value }))}
                    placeholder="0"
                  />
                  {openInfo === "anchorSegmentIndex" ? <InfoBlob text={FIELD_HELP.anchorSegmentIndex} /> : null}
                </label>
                <label>
                  <FieldTitle title="Min Token Seconds" fieldKey="minSeconds" openInfo={openInfo} onInfoToggle={setOpenInfo} />
                  <input
                    type="number"
                    min="0.1"
                    step="0.1"
                    value={form.minSeconds}
                    onChange={(e) => setForm((f) => ({ ...f, minSeconds: e.target.value }))}
                    placeholder="0.4"
                  />
                  {openInfo === "minSeconds" ? <InfoBlob text={FIELD_HELP.minSeconds} /> : null}
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
                  <button className="ghost" onClick={openAlignReviewEditor} disabled={alignmentTableRows.length === 0}>
                    Review &amp; Realign
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
                <p className="muted">
                  Page {currentViewPage ?? "-"} / {viewPayload.pages[viewPayload.pages.length - 1] ?? "-"}
                </p>
                <div className="zoom-controls">
                  <button
                    className="zoom-btn"
                    onClick={() => setKaraokeZoom((z) => Math.max(0.25, Math.round((z - 0.25) * 100) / 100))}
                    disabled={karaokeZoom <= 0.25}
                  >−</button>
                  <span className="zoom-label">{Math.round(karaokeZoom * 100)}%</span>
                  <button
                    className="zoom-btn"
                    onClick={() => setKaraokeZoom((z) => Math.min(2.0, Math.round((z + 0.25) * 100) / 100))}
                    disabled={karaokeZoom >= 2.0}
                  >+</button>
                </div>
              </div>
              <div className="karaoke-pages" ref={karaokeScrollRef}>
                {viewPayload.pages.map((page) => {
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
                            ? { width: `${natural.w * karaokeZoom}px`, height: `${natural.h * karaokeZoom}px` }
                            : undefined
                        }
                      >
                        <img
                          src={imgSrc}
                          alt={`page-${page}`}
                          ref={(el) => {
                            pageImgRefs.current[String(page)] = el;
                          }}
                          style={natural ? { width: `${natural.w * karaokeZoom}px`, height: `${natural.h * karaokeZoom}px` } : undefined}
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
                            const mapped = mapOcrTokenToRenderRect({
                              token: t,
                              naturalSize: natural,
                              pageBounds,
                            });
                            // Invisible div sized to the full token bbox — used only
                            // as a scroll anchor for the rolling karaoke effect.
                            return (
                              <div
                                key={key}
                                ref={(el) => {
                                  tokenRefs.current[key] = el;
                                }}
                                className="karaoke-token-anchor"
                                style={{
                                  left: `${mapped.left * karaokeZoom}px`,
                                  top: `${mapped.top * karaokeZoom}px`,
                                  width: `${mapped.width * karaokeZoom}px`,
                                  height: `${mapped.height * karaokeZoom}px`,
                                }}
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

      {showAlignReviewEditor ? (
        <div className="modal-backdrop" onClick={() => setShowAlignReviewEditor(false)}>
          <div className="modal-card modal-card-wide" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <h3>Alignment Review Editor</h3>
              <div style={{ display: "flex", gap: 8 }}>
                <button className="run-one" onClick={realignFromEditor} disabled={isRealigning}>
                  {isRealigning ? "Realigning..." : "Realign"}
                </button>
                <button className="ghost" onClick={() => setShowAlignReviewEditor(false)}>Close</button>
              </div>
            </div>
            <p className="muted" style={{ margin: "6px 0 10px", fontSize: 13 }}>
              Move tokens between Kept and Left Out, or between segments. Click Realign to redistribute timestamps.
            </p>
            <div className="table-wrap table-wrap-xl">
              <table className="review-table review-editor-table">
                <thead>
                  <tr>
                    <th style={{width:40}}>Seg</th>
                    <th style={{width:180}}>Audio Text</th>
                    <th>OCR Kept Tokens</th>
                    <th>OCR Left Out Tokens</th>
                    <th style={{width:90}}>Time Reduction (s)</th>
                    <th style={{width:55}}>Start</th>
                    <th style={{width:55}}>End</th>
                    <th style={{width:65}} title="Effective duration ÷ kept token count (s/token)">s/token</th>
                    <th style={{width:65}} title="Effective duration ÷ total pixel width of kept tokens (s/px)">s/px</th>
                    <th style={{width:65}} title="Effective duration ÷ total mātrā count of kept tokens (s/mātrā) — phonetic beat weight">s/mātrā</th>
                    <th style={{width:60}}>Status</th>
                    <th style={{width:65}}>Coverage</th>
                  </tr>
                </thead>
                <tbody>
                  {reviewEditorRows.map((row, rowIdx) => (
                    <tr key={row.segmentIndex}>
                      <td>{row.segmentIndex}</td>
                      <td style={{ fontSize: 12, color: "var(--muted)" }}>{row.audioText}</td>
                      <td>
                        <div className="token-cell">
                          {row.keptTokens.map((tok) => (
                            <div key={tok.tokenId} className="token-card token-kept" title={tok.tokenId}>
                              <span>{tok.token}</span>
                              <div className="token-card-actions">
                                {rowIdx > 0 && (
                                  <button title="Move to previous segment" onClick={() => moveTokenInEditor(rowIdx, "kept", tok.tokenId, rowIdx - 1, "kept")}>↑</button>
                                )}
                                {rowIdx < reviewEditorRows.length - 1 && (
                                  <button title="Move to next segment" onClick={() => moveTokenInEditor(rowIdx, "kept", tok.tokenId, rowIdx + 1, "kept")}>↓</button>
                                )}
                                <button title="Move to left out" onClick={() => moveTokenInEditor(rowIdx, "kept", tok.tokenId, rowIdx, "left")}>✕</button>
                              </div>
                            </div>
                          ))}
                        </div>
                      </td>
                      <td>
                        <div className="token-cell">
                          {row.leftOutTokens.map((tok) => (
                            <div key={tok.tokenId} className="token-card token-leftout" title={tok.tokenId}>
                              <span>{tok.token}</span>
                              <div className="token-card-actions">
                                <button title="Add to kept" onClick={() => moveTokenInEditor(rowIdx, "left", tok.tokenId, rowIdx, "kept")}>+</button>
                              </div>
                            </div>
                          ))}
                        </div>
                      </td>
                      <td style={{textAlign:"center", minWidth: 160}}>
                        <input
                          type="number"
                          step={0.5}
                          value={row.timeReductionS}
                          style={{width:70, textAlign:"center", fontSize:12}}
                          onChange={(e) => {
                            const val = parseFloat(e.target.value) || 0;
                            setReviewEditorRows((rows) =>
                              rows.map((r, i) => i === rowIdx ? { ...r, timeReductionS: val } : r)
                            );
                          }}
                        />
                      </td>
                      <td style={{fontSize:12}}>{row.audioStartS}</td>
                      <td style={{fontSize:12}}>{row.audioEndS}</td>
                      <td style={{fontSize:12, textAlign:"center", color:"var(--muted)"}}>
                        {(() => {
                          const eff = (row.audioEndS - row.audioStartS) - row.timeReductionS;
                          const n = row.keptTokens.length;
                          return n > 0 ? (eff / n).toFixed(3) : "-";
                        })()}
                      </td>
                      <td style={{fontSize:12, textAlign:"center", color:"var(--muted)"}}>
                        {(() => {
                          const eff = (row.audioEndS - row.audioStartS) - row.timeReductionS;
                          const totalW = row.keptTokens.reduce((s, t) => s + t.w, 0);
                          return totalW > 0 ? (eff / totalW).toFixed(4) : "-";
                        })()}
                      </td>
                      <td style={{fontSize:12, textAlign:"center", color:"var(--muted)"}}>
                        {(() => {
                          const eff = (row.audioEndS - row.audioStartS) - row.timeReductionS;
                          const totalM = row.keptTokens.reduce((s, t) => s + t.matraCount, 0);
                          return totalM > 0 ? (eff / totalM).toFixed(4) : "-";
                        })()}
                      </td>
                      <td style={{fontSize:12}}>{row.status}</td>
                      <td style={{fontSize:12}}>{row.coverage}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ) : null}

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
