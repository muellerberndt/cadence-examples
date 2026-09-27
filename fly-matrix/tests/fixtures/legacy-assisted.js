// Replay immutable subset receipts with their exact archived solver, never by
// replacing a historical hash with the current demo's implementation.
import { createHash } from "node:crypto";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";

export function legacyAssistedFixture() {
  const source = fileURLToPath(new URL("../../", import.meta.url));
  const root = mkdtempSync(join(tmpdir(), "fly-legacy-source-"));
  const cleanup = () => rmSync(root, { recursive: true, force: true });
  process.once("exit", cleanup);
  try {
    for (const name of ["assisted_evidence.json", "assisted_evidence_1024.json"]) {
      const receipt = JSON.parse(readFileSync(join(source, "receipts", name), "utf8"));
      for (const pin of receipt.sources) {
        const src = pin.path === "web/brain.js"
          ? join(source, "tests/fixtures/legacy-assisted/brain.js") : join(source, pin.path);
        assert.equal(createHash("sha256").update(readFileSync(src)).digest("hex"), pin.sha256,
          `Historical source unavailable: ${pin.path}`);
        const dest = join(root, pin.path);
        mkdirSync(dirname(dest), { recursive: true });
        // Large immutable payloads are linked, not duplicated. Code is copied
        // so relative module imports resolve inside the archived source tree.
        if (pin.path.startsWith("web/data/")) {
          try { symlinkSync(src, dest); } catch (e) { if (e.code !== "EEXIST") throw e; }
        } else copyFileSync(src, dest);
      }
    }
    copyFileSync(join(source, "tools/verify_assisted_evidence.mjs"), join(root, "tools/verify_assisted_evidence.mjs"));
    symlinkSync(join(source, "receipts"), join(root, "receipts"), "dir");
    return { root, cleanup };
  } catch (error) { cleanup(); throw error; }
}
