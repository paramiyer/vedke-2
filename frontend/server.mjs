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
  if (fs.existsSync(alignManifestPath)) {
    try {
      const parsed = JSON.parse(fs.readFileSync(alignManifestPath, "utf-8"));
      if (parsed && typeof parsed.anchor_token === "string") anchorToken = parsed.anchor_token.trim();
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
      anchorToken
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
        anchorToken
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
  if (req.method === "OPTIONS") {
    sendJson(res, 200, {});
    return;
  }

  if (req.method === "POST" && req.url === "/api/run-stage") {
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

  if (req.method === "POST" && req.url === "/api/run-stage-async") {
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

  if (req.method === "GET" && req.url?.startsWith("/api/job-status")) {
    const u = new URL(req.url, `http://localhost:${PORT}`);
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

  if (req.method === "POST" && req.url === "/api/alignment-feedback") {
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

  if (req.method === "POST" && req.url === "/api/alignment-manual-prompt") {
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

  if (req.method === "POST" && req.url === "/api/alignment-manual-import") {
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

  if (req.method === "GET" && req.url?.startsWith("/api/files")) {
    const u = new URL(req.url, `http://localhost:${PORT}`);
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

  if (req.method === "GET" && req.url === "/api/experiments") {
    sendJson(res, 200, { ok: true, experiments: listExperiments() });
    return;
  }

  if (req.method === "GET" && req.url?.startsWith("/api/experiment")) {
    const u = new URL(req.url, `http://localhost:${PORT}`);
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

  if (req.method === "GET" && req.url?.startsWith("/api/stage-artifacts")) {
    const u = new URL(req.url, `http://localhost:${PORT}`);
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

  sendJson(res, 404, { ok: false, message: "Not found" });
});

server.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`Stage runner API listening on http://localhost:${PORT}`);
});
