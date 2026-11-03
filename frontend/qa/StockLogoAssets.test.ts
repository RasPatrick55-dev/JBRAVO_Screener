import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import assets from "../src/components/ui/stockLogoAssets.json";

describe("Stored Logo.dev asset byte contract", () => {
  it("covers exactly the 24 observed ticker mappings without duplicate paths", () => {
    const expected="ALGT AMC APH AVT CRC CRM DT ELAN ENVA ETSY FULT GDS IESC MOD MUR PATH RTX RYTM SGI SHLS STRL WMG WULF YPF".split(" ");
    expect(Object.keys(assets).sort()).toEqual(expected);
    expect(new Set(Object.values(assets).map(asset=>asset.src)).size).toBe(24);
  });
  it.each(Object.entries(assets))("verifies %s original PNG hash, dimensions and exact provider ticker binding", (ticker,asset) => {
    const bytes=readFileSync("public"+asset.src);
    expect(createHash("sha256").update(bytes).digest("hex")).toBe(asset.sha256);
    expect(bytes.subarray(0,8).toString("hex")).toBe("89504e470d0a1a0a");
    expect(bytes.readUInt32BE(16)).toBe(asset.width);
    expect(bytes.readUInt32BE(20)).toBe(asset.height);
    expect(asset.src).toBe(`/assets/company-logos/${ticker}-${asset.sha256}.png`);
    expect(asset.ticker).toBe(ticker);expect(asset.resolution.value).toBe(ticker);
    expect(asset.resolution.type).toBe("ticker");expect(asset.provider).toBe("Logo.dev");
    expect(asset.providerEtag).toMatch(/^[a-f0-9]{64}$/);
    expect(asset).not.toHaveProperty("name");expect(asset).not.toHaveProperty("domain");
    expect(JSON.stringify(asset)).not.toMatch(/https:|token=|sk_|signed_url/i);
  });
  it("binds deployed PNG bytes and dimensions to the manifest content hash", () => {
    const bytes=readFileSync("public"+assets.STRL.src);
    expect(createHash("sha256").update(bytes).digest("hex")).toBe(assets.STRL.sha256);
    expect(bytes.length).toBe(5535);
    expect(bytes.subarray(0,8).toString("hex")).toBe("89504e470d0a1a0a");
    expect(bytes.readUInt32BE(16)).toBe(256);expect(bytes.readUInt32BE(20)).toBe(256);
    expect(assets.STRL.src).toContain(assets.STRL.sha256);
  });
});
