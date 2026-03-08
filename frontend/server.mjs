import http from "node:http";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import url from "node:url";

const __filename = url.fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "..");
const PORT = 8787;

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
  const result = spawnSync(command, args, {
    cwd: REPO_ROOT,
    encoding: "utf-8"
  });
  return {
    ok: result.status === 0,
    status: result.status ?? 1,
    stdout: result.stdout || "",
    stderr: result.stderr || ""
  };
}

function pythonCmd() {
  return process.platform === "win32" ? "python" : "python3";
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
      if (data.length > 2_000_000) {
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
        String(form.maxEditCost || "0.32")
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

  return {
    ok: true,
    stage,
    experiment,
    inputDir: path.relative(REPO_ROOT, inputDir),
    outDir: path.relative(REPO_ROOT, outDir),
    logs,
    files: [...walkFiles(inputDir), ...walkFiles(outDir)]
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

  sendJson(res, 404, { ok: false, message: "Not found" });
});

server.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`Stage runner API listening on http://localhost:${PORT}`);
});
