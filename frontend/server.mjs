import http from "node:http";
import { randomUUID } from "node:crypto";
import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import url from "node:url";

const __filename = url.fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "..");
const PORT = 8787;
const jobs = new Map();

function slugify(value) {
  return String(value || "")
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function sendJson(res, status, payload) {
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS"
  });
  res.end(JSON.stringify(payload));
}

function runCommand(command, args) {
  const rendered = [command, ...args].join(" ");
  // eslint-disable-next-line no-console
  console.log(`[api] run:start ${rendered}`);
  const result = spawnSync(command, args, {
    cwd: REPO_ROOT,
    encoding: "utf-8"
  });
  // eslint-disable-next-line no-console
  console.log(`[api] run:end status=${result.status ?? 1} cmd=${rendered}`);
  if (result.stdout) {
    // eslint-disable-next-line no-console
    console.log(`[api] stdout:\n${result.stdout}`);
  }
  if (result.stderr) {
    // eslint-disable-next-line no-console
    console.error(`[api] stderr:\n${result.stderr}`);
  }
  return {
    ok: result.status === 0,
    status: result.status ?? 1,
    stdout: (result.stdout || "").slice(0, 4000),
    stderr: (result.stderr || "").slice(0, 4000)
  };
}

function runCommandRaw(command, args) {
  const rendered = [command, ...args].join(" ");
  // eslint-disable-next-line no-console
  console.log(`[api] run-raw:start ${rendered}`);
  const result = spawnSync(command, args, {
    cwd: REPO_ROOT,
    encoding: "utf-8",
    maxBuffer: 20 * 1024 * 1024
  });
  // eslint-disable-next-line no-console
  console.log(`[api] run-raw:end status=${result.status ?? 1} cmd=${rendered}`);
  if (result.stdout) {
    // eslint-disable-next-line no-console
    console.log(`[api] raw-stdout-bytes=${Buffer.byteLength(result.stdout, "utf-8")}`);
  }
  if (result.stderr) {
    // eslint-disable-next-line no-console
    console.error(`[api] raw-stderr:\n${result.stderr}`);
  }
  return {
    ok: result.status === 0,
    status: result.status ?? 1,
    stdout: result.stdout || "",
    stderr: result.stderr || ""
  };
}

function runCommandNoTrim(command, args) {
  const rendered = [command, ...args].join(" ");
  // eslint-disable-next-line no-console
  console.log(`[api] run-full:start ${rendered}`);
  const result = spawnSync(command, args, {
    cwd: REPO_ROOT,
    encoding: "utf-8",
    maxBuffer: 20 * 1024 * 1024
  });
  // eslint-disable-next-line no-console
  console.log(`[api] run-full:end status=${result.status ?? 1} cmd=${rendered}`);
  return {
    ok: result.status === 0,
    status: result.status ?? 1,
    stdout: result.stdout || "",
    stderr: result.stderr || ""
  };
}

function runCommandAsync(command, args) {
  return new Promise((resolve) => {
    const rendered = [command, ...args].join(" ");
    // eslint-disable-next-line no-console
    console.log(`[api] run-async:start ${rendered}`);
    const child = spawn(command, args, {
      cwd: REPO_ROOT,
      stdio: ["ignore", "pipe", "pipe"]
    });
    let stdout = "";
    let stderr = "";
    child.stdout?.setEncoding("utf-8");
    child.stderr?.setEncoding("utf-8");
    child.stdout?.on("data", (chunk) => {
      stdout += chunk;
      // eslint-disable-next-line no-console
      console.log(`[api] async-stdout ${String(chunk)}`);
    });
    child.stderr?.on("data", (chunk) => {
      stderr += chunk;
      // eslint-disable-next-line no-console
      console.error(`[api] async-stderr ${String(chunk)}`);
    });
    child.on("close", (code) => {
      // eslint-disable-next-line no-console
      console.log(`[api] run-async:end status=${code ?? 1} cmd=${rendered}`);
      resolve({
        ok: code === 0,
        status: code ?? 1,
        stdout: stdout.slice(0, 4000),
        stderr: stderr.slice(0, 4000)
      });
    });
    child.on("error", (err) => {
      // eslint-disable-next-line no-console
      console.error(`[api] run-async:error ${rendered} err=${String(err)}`);
      resolve({
        ok: false,
        status: 1,
        stdout: stdout.slice(0, 4000),
        stderr: `${stderr}\n${String(err)}`.slice(0, 4000)
      });
    });
  });
}

function pythonCmd() {
  if (process.platform === "win32") {
    const venvPy = path.join(REPO_ROOT, ".venv", "Scripts", "python.exe");
    return fs.existsSync(venvPy) ? venvPy : "python";
  }
  const venvPy = path.join(REPO_ROOT, ".venv", "bin", "python3");
  return fs.existsSync(venvPy) ? venvPy : "python3";
}

function walkFiles(dir, maxItems = 300) {
  const out = [];
  if (!fs.existsSync(dir)) return out;
  const stack = [dir];
  while (stack.length > 0 && out.length < maxItems) {
    const current = stack.pop();
    if (!current) break;
    const entries = fs.readdirSync(current, { withFileTypes: true });
    for (const e of entries) {
      const full = path.join(current, e.name);
      if (e.isDirectory()) {
        stack.push(full);
      } else {
        out.push(path.relative(REPO_ROOT, full));
      }
      if (out.length >= maxItems) break;
    }
  }
  out.sort();
  return out;
}

function parseBody(req) {
  return new Promise((resolve, reject) => {
    let data = "";
    req.on("data", (chunk) => {
      data += chunk;
      if (data.length > 20_000_000) {
        reject(new Error("Request body too large"));
      }
    });
    req.on("end", () => {
      if (!data) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(data));
      } catch (err) {
        reject(err);
      }
    });
  });
}

function stageArtifactPaths(stage, experiment, inputDir, outDir) {
  const base = `data/${experiment}`;
  if (stage === "input") {
    return [`${base}/out/source_intake_manifest.json`];
  }
  if (stage === "asr") {
    return [`${base}/out/status.json`, `${base}/out/manifest.json`, `${base}/out/0.json`];
  }
  if (stage === "pdf") {
    return [
      `${base}/out/pdf_tokens.json`,
      `${base}/out/pdf_tokens_cleaned.json`,
      `${base}/out/check_extraction.json`
    ];
  }
  if (stage === "align" || stage === "align_feedback") {
    return [
      `${base}/out/alignment_config.json`,
      `${base}/out/alignment_llm_manifest.json`,
      `${base}/out/alignment_manual_validation.json`,
      `${base}/out/pdf_tokens_segment_mapping_review.csv`,
      `${base}/out/pdf_tokens_left_out_non_punctuation.csv`,
      `${base}/out/pdf_tokens_enriched_with_timestamps.json`
    ];
  }
  if (stage === "gate") {
    return [`${base}/out/quality_gates.json`];
  }
  return [path.relative(REPO_ROOT, inputDir), path.relative(REPO_ROOT, outDir)];
}

function readArtifactPreview(relativePath) {
  const abs = path.resolve(REPO_ROOT, relativePath);
  if (!abs.startsWith(REPO_ROOT)) {
    return { path: relativePath, exists: false, bytes: 0, content: "", truncated: false };
  }
  if (!fs.existsSync(abs) || !fs.statSync(abs).isFile()) {
    return { path: relativePath, exists: false, bytes: 0, content: "", truncated: false };
  }
  const bytes = fs.statSync(abs).size;
  const isCsv = relativePath.toLowerCase().endsWith(".csv");
  const limit = isCsv ? 300_000 : 40_000;
  const text = fs.readFileSync(abs, "utf-8");
  const truncated = text.length > limit;
  return {
    path: relativePath,
    exists: true,
    bytes,
    content: truncated ? `${text.slice(0, limit)}\n\n...truncated...` : text,
    truncated
  };
}

function readTextIfExists(filePath) {
  if (!fs.existsSync(filePath)) return "";
  try {
    return fs.readFileSync(filePath, "utf-8").trim();
  } catch {
    return "";
  }
}

function parsePageSpec(raw) {
  const text = String(raw || "").trim();
  if (!text) return [];
  const out = new Set();
  for (const chunkRaw of text.split(",")) {
    const chunk = chunkRaw.trim();
    if (!chunk) continue;
    if (chunk.includes("-")) {
      const [a, b] = chunk.split("-", 2).map((x) => Number(x.trim()));
      if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
      const start = Math.min(a, b);
      const end = Math.max(a, b);
      for (let p = start; p <= end; p += 1) out.add(p);
    } else {
      const p = Number(chunk);
      if (Number.isFinite(p)) out.add(p);
    }
  }
  return Array.from(out).sort((a, b) => a - b);
}

function deriveStageStatus(experiment) {
  const inputDir = path.join(REPO_ROOT, "data", experiment, "input");
  const outDir = path.join(REPO_ROOT, "data", experiment, "out");
  return {
    input: fs.existsSync(path.join(outDir, "source_intake_manifest.json")) ? "done" : "idle",
    asr: fs.existsSync(path.join(outDir, "0.json")) ? "done" : "idle",
    pdf: fs.existsSync(path.join(outDir, "pdf_tokens.json")) ? "done" : "idle",
    align:
      fs.existsSync(path.join(outDir, "alignment_config.json")) &&
      fs.existsSync(path.join(outDir, "pdf_tokens_enriched_with_timestamps.json"))
        ? "done"
        : "idle",
    gate: fs.existsSync(path.join(outDir, "quality_gates.json")) ? "done" : "idle",
    karaokeDone: fs.existsSync(path.join(outDir, "karaoke_status.json")),
    inputDir: path.relative(REPO_ROOT, inputDir),
    outDir: path.relative(REPO_ROOT, outDir)
  };
}

function loadExperimentSnapshot(experiment) {
  const inputDir = path.join(REPO_ROOT, "data", experiment, "input");
  const outDir = path.join(REPO_ROOT, "data", experiment, "out");
  const yt = readTextIfExists(path.join(inputDir, "yt_link.txt"));
  const pages = readTextIfExists(path.join(inputDir, "pages.txt"));
  const pdfLocal = readTextIfExists(path.join(inputDir, "source_pdf_local.txt"));
  const pdf = pdfLocal || readTextIfExists(path.join(inputDir, "source_pdf.txt"));
  const alignManifestPath = path.join(outDir, "alignment_llm_manifest.json");
  let anchorToken = "";
  let anchorSegmentIndex = "0";
  if (fs.existsSync(alignManifestPath)) {
    try {
      const parsed = JSON.parse(fs.readFileSync(alignManifestPath, "utf-8"));
      if (parsed && typeof parsed.anchor_token === "string") anchorToken = parsed.anchor_token.trim();
      if (parsed && typeof parsed.anchor_segment_index === "number") anchorSegmentIndex = String(parsed.anchor_segment_index);
    } catch {
      anchorToken = "";
    }
  }
  return {
    experiment,
    form: {
      experimentName: experiment,
      youtubeUrl: yt,
      pdfPath: pdf,
      pages,
      anchorToken,
      anchorSegmentIndex
    },
    stageStatus: deriveStageStatus(experiment),
    files: [...walkFiles(inputDir), ...walkFiles(outDir)]
  };
}

function listExperiments() {
  const dataDir = path.join(REPO_ROOT, "data");
  if (!fs.existsSync(dataDir)) return [];
  const entries = fs.readdirSync(dataDir, { withFileTypes: true });
  const slugs = entries.filter((e) => e.isDirectory()).map((e) => e.name).sort();
  return slugs.map((slug) => ({
    experiment: slug,
    stageStatus: deriveStageStatus(slug)
  }));
}

function runKaraokeBuild(form) {
  const experiment = slugify(form.experimentName);
  if (!experiment) return { ok: false, message: "Experiment name is required" };
  const inputDir = path.join(REPO_ROOT, "data", experiment, "input");
  const outDir = path.join(REPO_ROOT, "data", experiment, "out");
  const required = [
    path.join(outDir, "quality_gates.json"),
    path.join(outDir, "pdf_tokens_enriched_with_timestamps.json"),
    path.join(inputDir, "audio.mp3"),
    path.join(inputDir, "pages.txt")
  ];
  for (const f of required) {
    if (!fs.existsSync(f)) return { ok: false, message: `Missing required file: ${path.relative(REPO_ROOT, f)}` };
  }
  const sourcePdf = path.join(inputDir, "source.pdf");
  if (!fs.existsSync(sourcePdf)) {
    return { ok: false, message: `Missing required file: ${path.relative(REPO_ROOT, sourcePdf)}` };
  }
  const pages = parsePageSpec(readTextIfExists(path.join(inputDir, "pages.txt")));
  if (pages.length === 0) return { ok: false, message: "pages.txt has no valid pages" };

  const pagesDir = path.join(outDir, "karaoke_pages");
  fs.mkdirSync(pagesDir, { recursive: true });

  // If Stage 3 (pdf_truth) already produced frozen page images for all pages,
  // reuse them rather than re-rendering from the PDF.  This preserves the
  // canonical coordinate space established at OCR time.
  const missingPages = pages.filter((p) => {
    const png = path.join(pagesDir, `page_${String(p).padStart(4, "0")}.png`);
    return !fs.existsSync(png);
  });

  const logs = [];
  if (missingPages.length > 0) {
    // Only regenerate pages that are missing.  Clear any stale PNGs for those
    // pages first so we don't mix artifacts from different render passes.
    for (const p of missingPages) {
      const stale = path.join(pagesDir, `page_${String(p).padStart(4, "0")}.png`);
      if (fs.existsSync(stale)) fs.unlinkSync(stale);
    }
    for (const p of missingPages) {
      const prefix = path.join(pagesDir, `page_${String(p).padStart(4, "0")}`);
      const args = ["-f", String(p), "-singlefile", "-png", sourcePdf, prefix];
      const r = runCommandNoTrim("pdftoppm", args);
      logs.push({ command: ["pdftoppm", ...args].join(" "), ...r });
      if (!r.ok) {
        return {
          ok: false,
          message: "Failed generating karaoke page images",
          logs,
          experiment,
          inputDir: path.relative(REPO_ROOT, inputDir),
          outDir: path.relative(REPO_ROOT, outDir)
        };
      }
    }
  }

  const statusPath = path.join(outDir, "karaoke_status.json");
  const payload = {
    ok: true,
    experiment,
    status: "done",
    builtAt: new Date().toISOString(),
    pages,
    sourcePdf: path.relative(REPO_ROOT, sourcePdf),
    audio: path.relative(REPO_ROOT, path.join(inputDir, "audio.mp3")),
    enriched: path.relative(REPO_ROOT, path.join(outDir, "pdf_tokens_enriched_with_timestamps.json")),
    pagesDir: path.relative(REPO_ROOT, pagesDir)
  };
  fs.writeFileSync(statusPath, `${JSON.stringify(payload, null, 2)}\n`, "utf-8");
  return {
    ok: true,
    stage: "karaoke",
    experiment,
    logs,
    inputDir: path.relative(REPO_ROOT, inputDir),
    outDir: path.relative(REPO_ROOT, outDir),
    files: [...walkFiles(inputDir), ...walkFiles(outDir)]
  };
}

function getKaraokeViewPayload(experiment) {
  const inputDir = path.join(REPO_ROOT, "data", experiment, "input");
  const outDir = path.join(REPO_ROOT, "data", experiment, "out");
  const statusPath = path.join(outDir, "karaoke_status.json");
  const enrichedPath = path.join(outDir, "pdf_tokens_enriched_with_timestamps.json");
  if (!fs.existsSync(statusPath)) throw new Error("karaoke_status.json not found");
  if (!fs.existsSync(enrichedPath)) throw new Error("pdf_tokens_enriched_with_timestamps.json not found");
  const status = JSON.parse(fs.readFileSync(statusPath, "utf-8"));
  const enrichedRaw = JSON.parse(fs.readFileSync(enrichedPath, "utf-8"));
  const enriched = enrichedRaw?.enriched_pdf_tokens && typeof enrichedRaw.enriched_pdf_tokens === "object" ? enrichedRaw.enriched_pdf_tokens : enrichedRaw;
  const pagesObj = enriched?.pages && typeof enriched.pages === "object" ? enriched.pages : {};
  const trimTopRatio = Number(enriched?.extraction?.sourceFilter?.trim?.top_ratio || 0);
  const trimBottomRatio = Number(enriched?.extraction?.sourceFilter?.trim?.bottom_ratio || 0);
  const pages = Array.isArray(status.pages) ? status.pages.map((x) => Number(x)).filter((x) => Number.isFinite(x)) : [];

  // ---------------------------------------------------------------------------
  // Resolve canonical page dimensions (OCR source space, px).
  //
  // Token x/y/w/h coordinates are defined in the coordinate space of the
  // full-page PNG rendered at 300 DPI by pdftoppm during Stage 3.  We MUST
  // use those exact page dimensions—not the max token extent—as the reference
  // when mapping token coordinates to any display space.
  //
  // Priority:
  //   1. pdf_tokens.json "pageImageSizes" (written by stage_orchestrator since
  //      the coordinate-fix was applied)
  //   2. check_extraction.json "results[page].image_size_px" (always present;
  //      fixes rendering for runs produced before the coordinate-fix)
  //   3. Fallback: compute from max token extent (old behaviour, inaccurate).
  // ---------------------------------------------------------------------------
  let canonicalPageDims = {};
  // Priority 1: pdf_tokens.json pageImageSizes
  const pdfTokensPath = path.join(outDir, "pdf_tokens.json");
  if (fs.existsSync(pdfTokensPath)) {
    try {
      const pt = JSON.parse(fs.readFileSync(pdfTokensPath, "utf-8"));
      if (pt.pageImageSizes && typeof pt.pageImageSizes === "object") {
        canonicalPageDims = pt.pageImageSizes;
      }
    } catch (_) { /* ignore parse errors */ }
  }
  // Priority 2: check_extraction.json results[page].image_size_px
  if (Object.keys(canonicalPageDims).length === 0) {
    const checkExtrPath = path.join(outDir, "check_extraction.json");
    if (fs.existsSync(checkExtrPath)) {
      try {
        const ce = JSON.parse(fs.readFileSync(checkExtrPath, "utf-8"));
        const ceResults = ce.results || {};
        for (const [pno, pdata] of Object.entries(ceResults)) {
          if (pdata && pdata.image_size_px && pdata.image_size_px.w > 0 && pdata.image_size_px.h > 0) {
            canonicalPageDims[String(pno)] = { w: pdata.image_size_px.w, h: pdata.image_size_px.h };
          }
        }
      } catch (_) { /* ignore parse errors */ }
    }
  }

  const tokensByPage = {};
  const pageImageUrls = {};
  const pageImageData = {};
  const pageBoundsByPage = {};
  for (const p of pages) {
    const recs = pagesObj[String(p)] || [];
    if (!Array.isArray(recs)) continue;

    // Determine canonical page bounds for coordinate mapping.
    // Use stored OCR source image dimensions when available; fall back to
    // computing from token extents only if no better source exists.
    const dims = canonicalPageDims[String(p)];
    if (dims && dims.w > 0 && dims.h > 0) {
      pageBoundsByPage[String(p)] = { maxRight: dims.w, maxBottom: dims.h };
    } else {
      // Legacy fallback: compute from token extent (inaccurate but safe).
      let maxRight = 0;
      let maxBottom = 0;
      recs.forEach((r) => {
        const x = Number(r?.x || 0);
        const y = Number(r?.y || 0);
        const w = Number(r?.w || 0);
        const h = Number(r?.h || 0);
        if (Number.isFinite(x) && Number.isFinite(w)) maxRight = Math.max(maxRight, x + w);
        if (Number.isFinite(y) && Number.isFinite(h)) maxBottom = Math.max(maxBottom, y + h);
      });
      pageBoundsByPage[String(p)] = { maxRight, maxBottom };
    }

    tokensByPage[String(p)] = recs
      .filter((r) => r && typeof r === "object" && Number.isFinite(Number(r.start_time_seconds)) && Number.isFinite(Number(r.end_time_seconds)))
      .map((r) => ({
        tokenId: String(r.tokenId || ""),
        token: String(r.token || ""),
        x: Number(r.x || 0),
        y: Number(r.y || 0),
        w: Number(r.w || 0),
        h: Number(r.h || 0),
        start_s: Number(r.start_time_seconds),
        end_s: Number(r.end_time_seconds)
      }));
    pageImageUrls[String(p)] = `/api/karaoke-page?experiment=${encodeURIComponent(experiment)}&page=${encodeURIComponent(String(p))}`;
    const pagePng = path.join(outDir, "karaoke_pages", `page_${String(p).padStart(4, "0")}.png`);
    if (fs.existsSync(pagePng)) {
      const b64 = fs.readFileSync(pagePng).toString("base64");
      pageImageData[String(p)] = `data:image/png;base64,${b64}`;
    }
  }
  return {
    experiment,
    pages,
    tokensByPage,
    pageImageUrls,
    pageImageData,
    pageBoundsByPage,
    trimTopRatio,
    trimBottomRatio,
    audioUrl: `/api/karaoke-audio?experiment=${encodeURIComponent(experiment)}`,
    pageImageBaseUrl: `/api/karaoke-page?experiment=${encodeURIComponent(experiment)}&page=`,
    files: {
      audio: path.relative(REPO_ROOT, path.join(inputDir, "audio.mp3")),
      enriched: path.relative(REPO_ROOT, enrichedPath),
      status: path.relative(REPO_ROOT, statusPath)
    }
  };
}

function runStage(stage, form) {
  const experiment = slugify(form.experimentName);
  if (!experiment) {
    return { ok: false, message: "Experiment name is required" };
  }
  const inputDir = path.join(REPO_ROOT, "data", experiment, "input");
  const outDir = path.join(REPO_ROOT, "data", experiment, "out");
  const pages = String(form.pages || "").trim();
  const pdf = String(form.pdfPath || "").trim();
  const yt = String(form.youtubeUrl || "").trim();
  const anchorToken = String(form.anchorToken || "").trim();

  const commands = [];
  if (stage === "input") {
    commands.push([
      pythonCmd(),
      [
        "-m",
        "scripts.stage_orchestrator",
        "--stage",
        "source_intake",
        "--experiment",
        experiment,
        "--youtube-url",
        yt,
        "--pdf",
        pdf,
        "--pages",
        pages
      ]
    ]);
  } else if (stage === "asr") {
    const mode = String(form.asrMode || "verbatim");
    const prepareArgs = ["-m", "scripts.prepare_asr_audio", "--experiment", experiment, "--url", yt, "--overwrite"];
    const sttArgs = [
      "-m",
      "scripts.run_sarvam_stt",
      "--experiment",
      experiment,
      "--with-timestamps",
      "--model",
      String(form.aiModel || "saaras:v3"),
      "--language-code",
      String(form.languageCode || "sa-IN"),
      "--num-speakers",
      "1",
      "--mode",
      mode
    ];

    const prep = runCommand(pythonCmd(), prepareArgs);
    const logs = [{ command: [pythonCmd(), ...prepareArgs].join(" "), ...prep }];
    if (!prep.ok) {
      return {
        ok: false,
        message: `Command failed: ${pythonCmd()}`,
        logs,
        experiment,
        inputDir: path.relative(REPO_ROOT, inputDir),
        outDir: path.relative(REPO_ROOT, outDir),
        files: [...walkFiles(inputDir), ...walkFiles(outDir)]
      };
    }

    const stt = runCommand(pythonCmd(), sttArgs);
    logs.push({ command: [pythonCmd(), ...sttArgs].join(" "), ...stt });
    if (!stt.ok) {
      return {
        ok: false,
        message: `Command failed: ${pythonCmd()}`,
        logs,
        experiment,
        inputDir: path.relative(REPO_ROOT, inputDir),
        outDir: path.relative(REPO_ROOT, outDir),
        files: [...walkFiles(inputDir), ...walkFiles(outDir)]
      };
    }

    return {
      ok: true,
      stage,
      experiment,
      inputDir: path.relative(REPO_ROOT, inputDir),
      outDir: path.relative(REPO_ROOT, outDir),
      logs,
      files: [...walkFiles(inputDir), ...walkFiles(outDir)]
    };
  } else if (stage === "pdf") {
    commands.push([
      pythonCmd(),
      [
        "-m",
        "scripts.stage_orchestrator",
        "--stage",
        "pdf_truth",
        "--experiment",
        experiment,
        "--pdf",
        pdf,
        "--pages",
        pages,
        "--trim-top-ratio",
        String(form.topTrim || "0.08"),
        "--trim-bottom-ratio",
        String(form.bottomTrim || "0.06")
      ]
    ]);
  } else if (stage === "align") {
    if (!anchorToken) {
      return { ok: false, message: "anchorToken is required for alignment stage" };
    }
    const anchorSeg = String(Number.isFinite(parseInt(form.anchorSegmentIndex, 10)) ? parseInt(form.anchorSegmentIndex, 10) : 0);
    commands.push([
      pythonCmd(),
      [
        "-m",
        "scripts.stage_orchestrator",
        "--stage",
        "alignment_config",
        "--experiment",
        experiment,
        "--min-confidence",
        String(form.minConfidence || "0.78"),
        "--max-edit-cost",
        String(form.maxEditCost || "0.32"),
        "--anchor-token",
        anchorToken,
        "--anchor-segment",
        anchorSeg
      ]
    ]);
  } else if (stage === "gate") {
    commands.push([
      pythonCmd(),
      [
        "-m",
        "scripts.stage_orchestrator",
        "--stage",
        "quality_gates",
        "--experiment",
        experiment,
        "--min-confidence",
        String(form.minConfidence || "0.78"),
        "--max-edit-cost",
        String(form.maxEditCost || "0.32")
      ]
    ]);
  } else {
    return { ok: false, message: `Unknown stage: ${stage}` };
  }

  const logs = [];
  for (const [cmd, args] of commands) {
    const r = runCommand(cmd, args);
    logs.push({ command: [cmd, ...args].join(" "), ...r });
    if (!r.ok) {
      return {
        ok: false,
        message: `Command failed: ${cmd}`,
        logs,
        experiment,
        inputDir: path.relative(REPO_ROOT, inputDir),
        outDir: path.relative(REPO_ROOT, outDir),
        files: [...walkFiles(inputDir), ...walkFiles(outDir)]
      };
    }
  }

  const artifacts = stageArtifactPaths(stage, experiment, inputDir, outDir).map((p) => readArtifactPreview(p));
  return {
    ok: true,
    stage,
    experiment,
    inputDir: path.relative(REPO_ROOT, inputDir),
    outDir: path.relative(REPO_ROOT, outDir),
    logs,
    files: [...walkFiles(inputDir), ...walkFiles(outDir)],
    artifacts
  };
}

function runAlignmentFeedback(form, feedback) {
  const experiment = slugify(form.experimentName);
  if (!experiment) {
    return { ok: false, message: "Experiment name is required" };
  }
  if (!String(feedback || "").trim()) {
    return { ok: false, message: "feedback is required" };
  }
  const inputDir = path.join(REPO_ROOT, "data", experiment, "input");
  const outDir = path.join(REPO_ROOT, "data", experiment, "out");
  const args = ["-m", "scripts.run_alignment_llm", "--experiment", experiment, "--feedback", String(feedback)];
  const r = runCommand(pythonCmd(), args);
  const logs = [{ command: [pythonCmd(), ...args].join(" "), ...r }];
  if (!r.ok) {
    return {
      ok: false,
      message: `Command failed: ${pythonCmd()}`,
      logs,
      experiment,
      inputDir: path.relative(REPO_ROOT, inputDir),
      outDir: path.relative(REPO_ROOT, outDir),
      files: [...walkFiles(inputDir), ...walkFiles(outDir)]
    };
  }
  return {
    ok: true,
    stage: "align_feedback",
    experiment,
    inputDir: path.relative(REPO_ROOT, inputDir),
    outDir: path.relative(REPO_ROOT, outDir),
    logs,
    files: [...walkFiles(inputDir), ...walkFiles(outDir)],
    artifacts: stageArtifactPaths("align_feedback", experiment, inputDir, outDir).map((p) => readArtifactPreview(p))
  };
}

function buildResultFromLogs(stage, experiment, inputDir, outDir, logs) {
  const artifacts = stageArtifactPaths(stage, experiment, inputDir, outDir).map((p) => readArtifactPreview(p));
  return {
    ok: true,
    stage,
    experiment,
    inputDir: path.relative(REPO_ROOT, inputDir),
    outDir: path.relative(REPO_ROOT, outDir),
    logs,
    files: [...walkFiles(inputDir), ...walkFiles(outDir)],
    artifacts
  };
}

function startAsyncAlignmentJob(form) {
  const experiment = slugify(form.experimentName);
  if (!experiment) {
    return { ok: false, message: "Experiment name is required" };
  }
  const inputDir = path.join(REPO_ROOT, "data", experiment, "input");
  const outDir = path.join(REPO_ROOT, "data", experiment, "out");
  const anchorToken = String(form.anchorToken || "").trim();
  if (!anchorToken) {
    return { ok: false, message: "anchorToken is required for alignment stage" };
  }
  const args = [
    "-m",
    "scripts.stage_orchestrator",
    "--stage",
    "alignment_config",
    "--experiment",
    experiment,
    "--min-confidence",
    String(form.minConfidence || "0.78"),
    "--max-edit-cost",
    String(form.maxEditCost || "0.32"),
    "--anchor-token",
    anchorToken
  ];
  const jobId = randomUUID();
  jobs.set(jobId, {
    ok: true,
    state: "running",
    stage: "align",
    experiment,
    createdAt: new Date().toISOString()
  });
  void runCommandAsync(pythonCmd(), args).then((r) => {
    const logs = [{ command: [pythonCmd(), ...args].join(" "), ...r }];
    if (!r.ok) {
      jobs.set(jobId, {
        ok: false,
        state: "failed",
        stage: "align",
        experiment,
        message: `Command failed: ${pythonCmd()}`,
        logs,
        inputDir: path.relative(REPO_ROOT, inputDir),
        outDir: path.relative(REPO_ROOT, outDir),
        files: [...walkFiles(inputDir), ...walkFiles(outDir)],
        completedAt: new Date().toISOString()
      });
      return;
    }
    jobs.set(jobId, {
      ...buildResultFromLogs("align", experiment, inputDir, outDir, logs),
      state: "done",
      completedAt: new Date().toISOString()
    });
  });
  return { ok: true, jobId, state: "running", stage: "align", experiment };
}

function runAlignmentManualPrompt(form) {
  const experiment = slugify(form.experimentName);
  if (!experiment) return { ok: false, message: "Experiment name is required" };
  const anchorToken = String(form.anchorToken || "").trim();
  if (!anchorToken) return { ok: false, message: "anchorToken is required for manual alignment prompt" };
  const inputDir = path.join(REPO_ROOT, "data", experiment, "input");
  const outDir = path.join(REPO_ROOT, "data", experiment, "out");
  const args = [
    "-m",
    "scripts.run_alignment_llm",
    "--experiment",
    experiment,
    "--anchor-token",
    anchorToken,
    "--emit-prompt-only"
  ];
  const r = runCommandRaw(pythonCmd(), args);
  if (!r.ok) {
    return {
      ok: false,
      message: `Command failed: ${pythonCmd()}`,
      logs: [{ command: [pythonCmd(), ...args].join(" "), ...r }],
      experiment,
      inputDir: path.relative(REPO_ROOT, inputDir),
      outDir: path.relative(REPO_ROOT, outDir),
      files: [...walkFiles(inputDir), ...walkFiles(outDir)]
    };
  }
  return { ok: true, experiment, prompt: r.stdout || "" };
}

function runAlignmentManualImport(form, uploaded) {
  const experiment = slugify(form.experimentName);
  if (!experiment) return { ok: false, message: "Experiment name is required" };
  const anchorToken = String(form.anchorToken || "").trim();
  if (!anchorToken) return { ok: false, message: "anchorToken is required for manual alignment import" };
  const enrichedJson = String(uploaded?.enrichedJson || "");
  const reviewCsv = String(uploaded?.reviewCsv || "");
  const leftOutCsv = String(uploaded?.leftOutCsv || "");
  if (!enrichedJson.trim()) return { ok: false, message: "Uploaded enriched JSON is required" };
  if (!reviewCsv.trim()) return { ok: false, message: "Uploaded review CSV is required" };

  const inputDir = path.join(REPO_ROOT, "data", experiment, "input");
  const outDir = path.join(REPO_ROOT, "data", experiment, "out");
  fs.mkdirSync(inputDir, { recursive: true });
  fs.mkdirSync(outDir, { recursive: true });
  const tempDir = path.join(inputDir, "_manual_upload");
  fs.mkdirSync(tempDir, { recursive: true });
  const enrichedPath = path.join(tempDir, "pdf_tokens_enriched_with_timestamps.upload.json");
  const reviewPath = path.join(tempDir, "pdf_tokens_segment_mapping_review.upload.csv");
  const leftOutPath = path.join(tempDir, "pdf_tokens_left_out_non_punctuation.upload.csv");
  fs.writeFileSync(enrichedPath, enrichedJson, "utf-8");
  fs.writeFileSync(reviewPath, reviewCsv, "utf-8");
  if (leftOutCsv.trim()) fs.writeFileSync(leftOutPath, leftOutCsv, "utf-8");

  const args = [
    "-m",
    "scripts.validate_alignment_manual",
    "--experiment",
    experiment,
    "--anchor-token",
    anchorToken,
    "--enriched-json",
    enrichedPath,
    "--review-csv",
    reviewPath,
    "--commit"
  ];
  if (leftOutCsv.trim()) {
    args.push("--left-out-csv", leftOutPath);
  }
  const r = runCommand(pythonCmd(), args);
  const logs = [{ command: [pythonCmd(), ...args].join(" "), ...r }];
  let validatorPayload = {};
  if (r.stdout && r.stdout.trim()) {
    try {
      validatorPayload = JSON.parse(r.stdout);
    } catch {
      validatorPayload = { raw: r.stdout };
    }
  }
  const validationPath = path.join(outDir, "alignment_manual_validation.json");
  const validationPayload = {
    ok: r.ok,
    experiment,
    anchorToken,
    uploaded: {
      enriched_json: enrichedPath,
      review_csv: reviewPath,
      left_out_csv: leftOutCsv.trim() ? leftOutPath : "",
    },
    validator: validatorPayload,
    logs,
    updatedAt: new Date().toISOString(),
  };
  fs.writeFileSync(validationPath, `${JSON.stringify(validationPayload, null, 2)}\n`, "utf-8");

  if (!r.ok) {
    return {
      ok: false,
      message: `Manual alignment guardrails failed`,
      logs,
      validator: validatorPayload,
      validation: path.relative(REPO_ROOT, validationPath),
      experiment,
      inputDir: path.relative(REPO_ROOT, inputDir),
      outDir: path.relative(REPO_ROOT, outDir),
      files: [...walkFiles(inputDir), ...walkFiles(outDir)]
    };
  }
  const alignmentConfigPath = path.join(outDir, "alignment_config.json");
  const payload = {
    stage: "alignment_config",
    experiment,
    strategy: "manual_chatgpt_upload_with_guardrails",
    inputs: {
      sarvam: path.join(outDir, "0.json"),
      pdf_tokens: path.join(outDir, "pdf_tokens.json"),
      anchor_token: anchorToken
    },
    outputs: {
      enriched_pdf_tokens: path.join(outDir, "pdf_tokens_enriched_with_timestamps.json"),
      review_csv: path.join(outDir, "pdf_tokens_segment_mapping_review.csv"),
      left_out_csv: path.join(outDir, "pdf_tokens_left_out_non_punctuation.csv")
    },
    validation: validatorPayload
  };
  fs.writeFileSync(alignmentConfigPath, `${JSON.stringify(payload, null, 2)}\n`, "utf-8");
  return {
    ...buildResultFromLogs("align", experiment, inputDir, outDir, logs),
    ok: true,
    stage: "align",
    mode: "manual_override",
    validator: validatorPayload,
    validation: path.relative(REPO_ROOT, validationPath),
  };
}

const server = http.createServer(async (req, res) => {
  const parsedUrl = new URL(req.url || "/", `http://localhost:${PORT}`);
  const reqPath = parsedUrl.pathname;

  if (req.method === "OPTIONS") {
    sendJson(res, 200, {});
    return;
  }

  if (req.method === "GET" && reqPath === "/favicon.ico") {
    res.writeHead(204, { "Access-Control-Allow-Origin": "*" });
    res.end();
    return;
  }

  if (req.method === "POST" && reqPath === "/api/run-stage") {
    try {
      const body = await parseBody(req);
      const stage = String(body.stage || "");
      const form = body.form || {};
      const result = runStage(stage, form);
      sendJson(res, result.ok ? 200 : 400, result);
    } catch (err) {
      sendJson(res, 500, { ok: false, message: String(err) });
    }
    return;
  }

  // Accept legacy misspelling (`karoke`) for backward compatibility with stale frontend bundles.
  if (
    req.method === "POST" &&
    (reqPath === "/api/karaoke-build" || reqPath === "/api/karoke-build")
  ) {
    try {
      const body = await parseBody(req);
      const form = body.form || {};
      const result = runKaraokeBuild(form);
      sendJson(res, result.ok ? 200 : 400, result);
    } catch (err) {
      sendJson(res, 500, { ok: false, message: String(err) });
    }
    return;
  }

  if (req.method === "POST" && reqPath === "/api/run-stage-async") {
    try {
      const body = await parseBody(req);
      const stage = String(body.stage || "");
      const form = body.form || {};
      if (stage !== "align") {
        sendJson(res, 400, { ok: false, message: "run-stage-async currently supports only align stage" });
        return;
      }
      const result = startAsyncAlignmentJob(form);
      sendJson(res, result.ok ? 200 : 400, result);
    } catch (err) {
      sendJson(res, 500, { ok: false, message: String(err) });
    }
    return;
  }

  if (req.method === "GET" && reqPath === "/api/job-status") {
    const u = parsedUrl;
    const jobId = String(u.searchParams.get("jobId") || "");
    if (!jobId) {
      sendJson(res, 400, { ok: false, message: "jobId query param required" });
      return;
    }
    const job = jobs.get(jobId);
    if (!job) {
      sendJson(res, 404, { ok: false, message: `job not found: ${jobId}` });
      return;
    }
    sendJson(res, 200, job);
    return;
  }

  if (req.method === "POST" && reqPath === "/api/alignment-feedback") {
    try {
      const body = await parseBody(req);
      const form = body.form || {};
      const feedback = String(body.feedback || "");
      const result = runAlignmentFeedback(form, feedback);
      sendJson(res, result.ok ? 200 : 400, result);
    } catch (err) {
      sendJson(res, 500, { ok: false, message: String(err) });
    }
    return;
  }

  if (req.method === "POST" && reqPath === "/api/alignment-manual-prompt") {
    try {
      const body = await parseBody(req);
      const form = body.form || {};
      const result = runAlignmentManualPrompt(form);
      sendJson(res, result.ok ? 200 : 400, result);
    } catch (err) {
      sendJson(res, 500, { ok: false, message: String(err) });
    }
    return;
  }

  if (req.method === "POST" && reqPath === "/api/alignment-manual-import") {
    try {
      const body = await parseBody(req);
      const form = body.form || {};
      const uploaded = body.uploaded || {};
      const result = runAlignmentManualImport(form, uploaded);
      sendJson(res, result.ok ? 200 : 400, result);
    } catch (err) {
      sendJson(res, 500, { ok: false, message: String(err) });
    }
    return;
  }

  if (req.method === "GET" && reqPath === "/api/files") {
    const u = parsedUrl;
    const experiment = slugify(u.searchParams.get("experiment") || "");
    if (!experiment) {
      sendJson(res, 400, { ok: false, message: "experiment query param required" });
      return;
    }
    const inputDir = path.join(REPO_ROOT, "data", experiment, "input");
    const outDir = path.join(REPO_ROOT, "data", experiment, "out");
    sendJson(res, 200, {
      ok: true,
      experiment,
      inputDir: path.relative(REPO_ROOT, inputDir),
      outDir: path.relative(REPO_ROOT, outDir),
      files: [...walkFiles(inputDir), ...walkFiles(outDir)]
    });
    return;
  }

  if (req.method === "GET" && reqPath === "/api/experiments") {
    sendJson(res, 200, { ok: true, experiments: listExperiments() });
    return;
  }

  if (req.method === "GET" && reqPath === "/api/karaoke-view") {
    try {
      const u = parsedUrl;
      const experiment = slugify(u.searchParams.get("experiment") || "");
      console.log(`[api] karaoke-view:get experiment=${experiment}`);
      if (!experiment) {
        sendJson(res, 400, { ok: false, message: "experiment query param required" });
        return;
      }
      const payload = getKaraokeViewPayload(experiment);
      sendJson(res, 200, { ok: true, ...payload });
    } catch (err) {
      sendJson(res, 400, { ok: false, message: String(err) });
    }
    return;
  }

  if (req.method === "GET" && reqPath === "/api/karaoke-audio") {
    const u = parsedUrl;
    const experiment = slugify(u.searchParams.get("experiment") || "");
    if (!experiment) {
      sendJson(res, 400, { ok: false, message: "experiment query param required" });
      return;
    }
    const audioPath = path.join(REPO_ROOT, "data", experiment, "input", "audio.mp3");
    if (!fs.existsSync(audioPath)) {
      sendJson(res, 404, { ok: false, message: "audio file not found" });
      return;
    }
    const fileSize = fs.statSync(audioPath).size;
    const rangeHeader = req.headers["range"];
    if (rangeHeader) {
      // Support HTTP Range requests so the browser can buffer and seek without
      // re-fetching the entire file from the beginning (which caused the audio
      // to reset to t=0 mid-playback).
      const [startStr, endStr] = rangeHeader.replace(/bytes=/, "").split("-");
      const start = parseInt(startStr, 10);
      const end = endStr ? parseInt(endStr, 10) : fileSize - 1;
      const chunkSize = end - start + 1;
      res.writeHead(206, {
        "Content-Range": `bytes ${start}-${end}/${fileSize}`,
        "Accept-Ranges": "bytes",
        "Content-Length": chunkSize,
        "Content-Type": "audio/mpeg",
        "Access-Control-Allow-Origin": "*"
      });
      fs.createReadStream(audioPath, { start, end }).pipe(res);
    } else {
      res.writeHead(200, {
        "Content-Length": fileSize,
        "Accept-Ranges": "bytes",
        "Content-Type": "audio/mpeg",
        "Access-Control-Allow-Origin": "*"
      });
      fs.createReadStream(audioPath).pipe(res);
    }
    return;
  }

  if (req.method === "GET" && (reqPath === "/api/karaoke-page" || reqPath === "/api/karoke-page")) {
    const u = parsedUrl;
    const experiment = slugify(u.searchParams.get("experiment") || "");
    const page = Number(u.searchParams.get("page") || "0");
    if (!experiment || !Number.isFinite(page) || page <= 0) {
      sendJson(res, 400, { ok: false, message: "experiment and positive page are required" });
      return;
    }
    const p = path.join(REPO_ROOT, "data", experiment, "out", "karaoke_pages", `page_${String(page).padStart(4, "0")}.png`);
    console.log(`[api] karaoke-page:get experiment=${experiment} page=${page} path=${p}`);
    if (!fs.existsSync(p)) {
      console.log(`[api] karaoke-page:missing experiment=${experiment} page=${page}`);
      sendJson(res, 404, { ok: false, message: "karaoke page image not found; run Karoke Build first" });
      return;
    }
    res.writeHead(200, {
      "Content-Type": "image/png",
      "Access-Control-Allow-Origin": "*"
    });
    fs.createReadStream(p).pipe(res);
    return;
  }

  if (req.method === "GET" && reqPath === "/api/karaoke-page-data") {
    const u = parsedUrl;
    const experiment = slugify(u.searchParams.get("experiment") || "");
    const page = Number(u.searchParams.get("page") || "0");
    if (!experiment || !Number.isFinite(page) || page <= 0) {
      sendJson(res, 400, { ok: false, message: "experiment and positive page are required" });
      return;
    }
    const p = path.join(REPO_ROOT, "data", experiment, "out", "karaoke_pages", `page_${String(page).padStart(4, "0")}.png`);
    console.log(`[api] karaoke-page-data:get experiment=${experiment} page=${page} path=${p}`);
    if (!fs.existsSync(p)) {
      console.log(`[api] karaoke-page-data:missing experiment=${experiment} page=${page}`);
      sendJson(res, 404, { ok: false, message: "karaoke page image not found; run Karoke Build first" });
      return;
    }
    const b64 = fs.readFileSync(p).toString("base64");
    sendJson(res, 200, { ok: true, mime: "image/png", data: b64 });
    return;
  }

  if (req.method === "GET" && reqPath === "/api/experiment") {
    const u = parsedUrl;
    const experiment = slugify(u.searchParams.get("name") || "");
    if (!experiment) {
      sendJson(res, 400, { ok: false, message: "name query param required" });
      return;
    }
    const inputDir = path.join(REPO_ROOT, "data", experiment, "input");
    const outDir = path.join(REPO_ROOT, "data", experiment, "out");
    if (!fs.existsSync(inputDir) && !fs.existsSync(outDir)) {
      sendJson(res, 404, { ok: false, message: `experiment not found: ${experiment}` });
      return;
    }
    sendJson(res, 200, { ok: true, snapshot: loadExperimentSnapshot(experiment) });
    return;
  }

  if (req.method === "GET" && reqPath === "/api/stage-artifacts") {
    const u = parsedUrl;
    const experiment = slugify(u.searchParams.get("experiment") || "");
    const stage = String(u.searchParams.get("stage") || "");
    if (!experiment || !stage) {
      sendJson(res, 400, { ok: false, message: "experiment and stage query params required" });
      return;
    }
    const inputDir = path.join(REPO_ROOT, "data", experiment, "input");
    const outDir = path.join(REPO_ROOT, "data", experiment, "out");
    const artifacts = stageArtifactPaths(stage, experiment, inputDir, outDir).map((p) => readArtifactPreview(p));
    sendJson(res, 200, { ok: true, experiment, stage, artifacts });
    return;
  }

  if (req.method === "GET" && reqPath === "/api/alignment-enriched-tokens") {
    const u = parsedUrl;
    const experiment = slugify(u.searchParams.get("experiment") || "");
    if (!experiment) {
      sendJson(res, 400, { ok: false, message: "experiment required" });
      return;
    }
    const enrichedPath = path.join(REPO_ROOT, "data", experiment, "out", "pdf_tokens_enriched_with_timestamps.json");
    if (!fs.existsSync(enrichedPath)) {
      sendJson(res, 404, { ok: false, message: "enriched file not found" });
      return;
    }
    try {
      const content = JSON.parse(fs.readFileSync(enrichedPath, "utf-8"));
      sendJson(res, 200, { ok: true, ...content });
    } catch (err) {
      sendJson(res, 500, { ok: false, message: String(err) });
    }
    return;
  }

  if (req.method === "POST" && reqPath === "/api/alignment-realign") {
    try {
      const body = await parseBody(req);
      const form = body.form || {};
      const segments = body.segments || [];
      const experiment = slugify(form.experimentName || "");
      if (!experiment) {
        sendJson(res, 400, { ok: false, message: "experimentName required" });
        return;
      }

      const enrichedPath = path.join(REPO_ROOT, "data", experiment, "out", "pdf_tokens_enriched_with_timestamps.json");
      if (!fs.existsSync(enrichedPath)) {
        sendJson(res, 404, { ok: false, message: "enriched file not found" });
        return;
      }

      const enriched = JSON.parse(fs.readFileSync(enrichedPath, "utf-8"));

      // Build flat token map: tokenId -> token record
      const tokenMap = new Map();
      for (const toks of Object.values(enriched.pages)) {
        for (const tok of toks) tokenMap.set(tok.tokenId, tok);
      }

      // Clear all existing timestamps and reset match state
      for (const tok of tokenMap.values()) {
        delete tok.start_time_seconds;
        delete tok.end_time_seconds;
        delete tok.timing_segment_index;
        delete tok.timing_match_ratio;
        delete tok.timing_manual_fallback;
        delete tok.matched_segment_index;
        // Explicitly mark all eligible tokens as unmatched; stamping below sets kept ones to true
        if (tok.is_eligible) tok.is_matched = false;
        else delete tok.is_matched;
      }

      const minSeconds = Math.max(0.05, parseFloat(form.minSeconds) || 0.4);

      // Redistribute timestamps per segment using floor-and-redistribute algorithm
      for (const seg of segments) {
        const { segmentIndex, keptTokenIds, audioStartS, audioEndS, timeReductionS } = seg;
        if (!keptTokenIds || keptTokenIds.length === 0) continue;
        // Subtract user-specified time reduction (extra audio words not in OCR)
        // before distributing the remaining duration across kept tokens.
        const reduction = parseFloat(timeReductionS) || 0;
        const duration = Math.max(0.1, (audioEndS - audioStartS) - reduction);
        const toks = keptTokenIds.map(id => tokenMap.get(id)).filter(Boolean);
        if (toks.length === 0) continue;

        // Compute char weights
        const weights = toks.map(t => t.char_count || t.token.length || 1);

        // Iterative floor-and-redistribute:
        // Pin any token whose proportional time < minSeconds, then redistribute
        // the remaining duration among free tokens. Repeat until stable.
        const allocated = new Array(toks.length).fill(null); // null = free, number = pinned
        let remainingDuration = duration;
        let changed = true;
        while (changed) {
          changed = false;
          const freeIndices = toks.map((_, i) => i).filter(i => allocated[i] === null);
          if (freeIndices.length === 0) break;
          const freeWeightTotal = freeIndices.reduce((s, i) => s + weights[i], 0);
          if (freeWeightTotal <= 0) break;
          for (const i of freeIndices) {
            const proportional = (weights[i] / freeWeightTotal) * remainingDuration;
            if (proportional < minSeconds) {
              allocated[i] = minSeconds;
              remainingDuration -= minSeconds;
              changed = true;
            }
          }
          // Safety: if pinning everything exhausts duration, fall back to equal split
          if (remainingDuration <= 0) {
            const equalShare = duration / toks.length;
            for (let i = 0; i < toks.length; i++) allocated[i] = equalShare;
            break;
          }
        }
        // Assign remaining free tokens proportionally from remaining duration
        const stillFree = toks.map((_, i) => i).filter(i => allocated[i] === null);
        if (stillFree.length > 0) {
          const freeWeightTotal = stillFree.reduce((s, i) => s + weights[i], 0);
          for (const i of stillFree) {
            allocated[i] = freeWeightTotal > 0
              ? (weights[i] / freeWeightTotal) * remainingDuration
              : remainingDuration / stillFree.length;
          }
        }

        // Stamp tokens with monotonic timestamps from allocations
        let cursor = audioStartS;
        for (let i = 0; i < toks.length; i++) {
          const tok = toks[i];
          const dur = allocated[i] ?? (duration / toks.length);
          tok.start_time_seconds = Math.round(cursor * 1000) / 1000;
          tok.end_time_seconds = Math.round((cursor + dur) * 1000) / 1000;
          tok.timing_segment_index = segmentIndex;
          tok.timing_match_ratio = 0.0;
          tok.timing_manual_fallback = true;
          tok.is_matched = true;
          tok.matched_segment_index = segmentIndex;
          cursor += dur;
        }
      }

      // Persist time reductions so the review editor can restore them on reopen.
      enriched.segmentTimeReductions = enriched.segmentTimeReductions || {};
      for (const seg of segments) {
        const r = parseFloat(seg.timeReductionS) || 0;
        if (r !== 0) enriched.segmentTimeReductions[String(seg.segmentIndex)] = r;
        else delete enriched.segmentTimeReductions[String(seg.segmentIndex)];
      }

      fs.writeFileSync(enrichedPath, JSON.stringify(enriched, null, 2), "utf-8");

      // Read back updated review CSV if it exists
      const reviewCsvPath = path.join(REPO_ROOT, "data", experiment, "out", "pdf_tokens_segment_mapping_review.csv");
      let alignmentReviewCsv = "";
      if (fs.existsSync(reviewCsvPath)) alignmentReviewCsv = fs.readFileSync(reviewCsvPath, "utf-8");

      sendJson(res, 200, { ok: true, experiment, alignmentReviewCsv });
    } catch (err) {
      sendJson(res, 500, { ok: false, message: String(err) });
    }
    return;
  }

  sendJson(res, 404, { ok: false, message: "Not found" });
});

server.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`Stage runner API listening on http://localhost:${PORT}`);
});
