import { useMemo, useRef, useState, type ChangeEventHandler } from "react";

type Tab = "build" | "view";
type StageStatus = "idle" | "running" | "done" | "error";

type Stage = {
  id: string;
  title: string;
  hint: string;
  status: StageStatus;
  output?: string;
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
};

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
  maxEditCost: "0.32"
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
  maxEditCost: "0 to 1. Lower = tighter text matching, higher = more tolerant."
};

const STAGE_TEMPLATES: Stage[] = [
  { id: "input", title: "Source Intake", hint: "Experiment folder + source acquisition", status: "idle" },
  { id: "asr", title: "ASR Preparation", hint: "YouTube audio + Sarvam output normalization", status: "idle" },
  { id: "pdf", title: "PDF Truth Build", hint: "Agent 4 token-level extraction (word|punc + x/y/w/h)", status: "idle" },
  { id: "align", title: "Alignment Config", hint: "Guardrail config for merge/split-aware matching", status: "idle" },
  { id: "gate", title: "Quality Gates", hint: "Guardrail checks and review routing", status: "idle" }
];

const AI_MODELS = ["saaras:v3"];
const API_BASE = "http://localhost:8787";

function slugify(value: string): string {
  return value
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function readKnownExperiments(): string[] {
  try {
    const raw = window.localStorage.getItem("vedke_known_experiments");
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((x) => typeof x === "string");
  } catch {
    return [];
  }
}

function writeKnownExperiments(slugs: string[]) {
  window.localStorage.setItem("vedke_known_experiments", JSON.stringify(slugs));
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
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    window.clearTimeout(timer);
  }
}

function App() {
  const [tab, setTab] = useState<Tab>("build");
  const [form, setForm] = useState<BuildForm>(INITIAL_FORM);
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [openInfo, setOpenInfo] = useState<string | null>(null);
  const [activeStage, setActiveStage] = useState<string>("input");
  const [stages, setStages] = useState<Stage[]>(STAGE_TEMPLATES);
  const [events, setEvents] = useState<string[]>([]);
  const [knownExperiments, setKnownExperiments] = useState<string[]>(() => readKnownExperiments());
  const folderPickerRef = useRef<HTMLInputElement | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(false);

  const experimentSlug = useMemo(() => slugify(form.experimentName), [form.experimentName]);
  const inputDir = experimentSlug ? `data/${experimentSlug}/input` : "data/<experiment>/input";
  const outDir = experimentSlug ? `data/${experimentSlug}/out` : "data/<experiment>/out";
  const experimentExists = experimentSlug.length > 0 && knownExperiments.includes(experimentSlug);

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

  const stageAvailable = (stageIndex: number) => {
    if (stageIndex === 0) return true;
    return stages[stageIndex - 1].status === "done";
  };

  const trackExperimentSlug = (slug: string) => {
    if (!slug) return;
    if (knownExperiments.includes(slug)) return;
    const next = [...knownExperiments, slug];
    setKnownExperiments(next);
    writeKnownExperiments(next);
  };

  const runStage = async (id: string): Promise<boolean> => {
    const stageIndex = stages.findIndex((s) => s.id === id);
    if (stageIndex === -1 || !stageAvailable(stageIndex) || isRunning) return false;

    setIsRunning(true);
    setStages((prev) => prev.map((s) => (s.id === id ? { ...s, status: "running" } : s)));

    try {
      const response = await fetchWithTimeout(`${API_BASE}/api/run-stage`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stage: id, form }),
      });
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
      const logs = Array.isArray(payload.logs)
        ? payload.logs.map((x: { command?: string; status?: number }) => ({
            command: String(x.command || ""),
            status: Number(x.status ?? -1),
          }))
        : [];
      setStages((prev) =>
        prev.map((s) =>
          s.id === id
            ? {
                ...s,
                status: "done",
                output: JSON.stringify(
                  {
                    inputDir: payload.inputDir,
                    outDir: payload.outDir,
                    files: payload.files,
                    logs,
                  },
                  null,
                  2
                ),
              }
            : s
        )
      );
      if (id === "input") trackExperimentSlug(experimentSlug);
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
    setForm(INITIAL_FORM);
    setShowAdvanced(false);
    setOpenInfo(null);
    setActiveStage("input");
    setStages(STAGE_TEMPLATES);
    setEvents([]);
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
              <div className="form-grid">
                <label>
                  <FieldTitle title="Experiment Name" fieldKey="experimentName" openInfo={openInfo} onInfoToggle={setOpenInfo} />
                  <input
                    value={form.experimentName}
                    onChange={(e) => setForm((f) => ({ ...f, experimentName: e.target.value }))}
                    placeholder="ganapati_trial_01"
                  />
                  {openInfo === "experimentName" ? <InfoBlob text={FIELD_HELP.experimentName} /> : null}
                </label>
                <label>
                  <FieldTitle title="YouTube URL" fieldKey="youtubeUrl" openInfo={openInfo} onInfoToggle={setOpenInfo} />
                  <input
                    value={form.youtubeUrl}
                    onChange={(e) => setForm((f) => ({ ...f, youtubeUrl: e.target.value }))}
                    placeholder="https://www.youtube.com/watch?v=..."
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
                  <input value={form.pages} onChange={(e) => setForm((f) => ({ ...f, pages: e.target.value }))} />
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
                </div>
              ) : null}
            </div>

            <div className="action-row">
              <button className="ghost" onClick={resetFrontendState} disabled={isRunning}>
                Reset Frontend
              </button>
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

            <div className="events-box">
              <h3>Execution Log</h3>
              {events.length === 0 ? <p className="muted">No stage runs yet.</p> : events.map((e) => <p key={e}>{e}</p>)}
            </div>
          </aside>
        </main>
      ) : (
        <main className="view-placeholder card">
          <h2>View Tab</h2>
          <p>Karaoke text-audio alignment view will be implemented next.</p>
        </main>
      )}
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
