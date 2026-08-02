import { spawn } from "node:child_process";
import path from "node:path";

const mode = process.argv[2];
if (!new Set(["dev", "build", "start"]).has(mode)) {
  console.error("Uso: node scripts/run-vinext.mjs <dev|build|start>");
  process.exit(2);
}

const executable = path.join(
  process.cwd(),
  "node_modules",
  ".bin",
  process.platform === "win32" ? "vinext.CMD" : "vinext",
);

const child = spawn(executable, [mode], {
  stdio: "inherit",
  shell: process.platform === "win32",
  env: {
    ...process.env,
    WRANGLER_LOG_PATH: path.join(".wrangler", "wrangler.log"),
  },
});

child.on("error", (error) => {
  console.error(`Não foi possível iniciar o vinext: ${error.message}`);
  process.exit(1);
});

child.on("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  process.exit(code ?? 1);
});
