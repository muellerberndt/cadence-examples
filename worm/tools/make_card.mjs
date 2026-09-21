#!/usr/bin/env node
// Photograph tools/card.html into web/card.jpg: 1200 x 630, a JPEG of a few hundred kilobytes at most,
// because X drops large card images and then remembers the failure for that address.
//   node worm/tools/make_card.mjs          needs Google Chrome; CHROME=/path/to/chrome overrides where it is
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { readFile, writeFile, mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, extname, normalize } from "node:path";

const root = new URL("..", import.meta.url).pathname;
const types = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json", ".png": "image/png" };
const server = createServer(async (req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname)).replace(/^\/+/, "");
  try { const body = await readFile(join(root, path)); res.writeHead(200, { "content-type": types[extname(path)] ?? "application/octet-stream" }); res.end(body); }
  catch { res.writeHead(404); res.end(); }
});
await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
const port = server.address().port, debug = 9400 + Math.floor(Math.random() * 400);
const chrome = spawn(process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  ["--headless=new", `--remote-debugging-port=${debug}`, "--window-size=1200,630", "--hide-scrollbars",
   `--user-data-dir=${await mkdtemp(join(tmpdir(), "worm-card-"))}`, "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let ws;
for (let k = 0; k < 60 && !ws; k++) {
  try { ws = (await (await fetch(`http://127.0.0.1:${debug}/json/list`)).json()).find((x) => x.type === "page")?.webSocketDebuggerUrl; } catch {}
  if (!ws) await sleep(200);
}
const sock = new WebSocket(ws); await new Promise((r) => (sock.onopen = r));
let id = 0; const pending = new Map();
sock.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); } };
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); sock.send(JSON.stringify({ id: i, method, params })); });
await send("Emulation.setDeviceMetricsOverride", { width: 1200, height: 630, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url: `http://127.0.0.1:${port}/tools/card.html` });
for (let k = 0; k < 100; k++) { if ((await send("Runtime.evaluate", { expression: "document.title" })).result.result.value === "ready") break; await sleep(200); }
await sleep(300);
const shot = await send("Page.captureScreenshot", { format: "jpeg", quality: 93 });
await writeFile(join(root, "web/card.jpg"), Buffer.from(shot.result.data, "base64"));
console.log("web/card.jpg written");
sock.close(); chrome.kill(); server.close(); process.exit(0);
