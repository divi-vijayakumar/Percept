import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { OrderedAudioBuffer, pcm16Energy } from "../src/vad.js";

const fixture = JSON.parse(
  readFileSync(new URL("./fixtures/audio_stream.golden.json", import.meta.url), "utf8")
);

function expectSubset(actual, expected) {
  for (const [key, value] of Object.entries(expected)) {
    if (typeof value === "number") expect(actual[key]).toBeCloseTo(value, 12);
    else expect(actual[key]).toEqual(value);
  }
}

describe("shared audio energy vectors", () => {
  for (const vector of fixture.energy_vectors) {
    it(vector.name, () => {
      expect(pcm16Energy(vector.samples)).toBeCloseTo(vector.expected, 12);
      if (vector.chunks) {
        const weighted =
          vector.chunks.reduce((acc, chunk) => acc + pcm16Energy(chunk) * chunk.length, 0) /
          vector.chunks.reduce((acc, chunk) => acc + chunk.length, 0);
        expect(weighted).toBeCloseTo(vector.expected, 12);
      }
    });
  }
});

describe("shared ordered audio timeline vectors", () => {
  for (const vector of fixture.timeline_vectors) {
    it(vector.name, () => {
      const timeline = new OrderedAudioBuffer({ seconds: 10 });
      const results = vector.units.map((unit) => timeline.push(unit));
      vector.expected.forEach((expected, index) => expectSubset(results[index], expected));
      expect(timeline.units().map((unit) => unit.seq)).toEqual(vector.order);
      if (vector.directions) {
        expect(timeline.units().map((unit) => unit.direction)).toEqual(vector.directions);
      }
      if (vector.participants) {
        expect(timeline.units().map((unit) => unit.participant_id)).toEqual(vector.participants);
      }
      if (vector.boundary_kinds) {
        expect(timeline.boundaries().map((boundary) => boundary.kind)).toEqual(
          vector.boundary_kinds
        );
      }
    });
  }

  it("stays count-bounded under deterministic overload", () => {
    const timeline = new OrderedAudioBuffer({ seconds: 1000, maxUnits: 8 });
    let evicted = 0;
    for (let seq = 0; seq < 1000; seq++) {
      evicted += timeline.push({ seq, t_start: seq, t_end: seq + 0.1 }).evicted;
    }
    expect(timeline.units()).toHaveLength(8);
    expect(evicted).toBe(992);
    expect(timeline.units()[0].seq).toBe(992);
  });

  it("stays byte-bounded and rejects a single oversized unit before mutation", () => {
    const timeline = new OrderedAudioBuffer({
      seconds: 1000,
      maxUnits: 8,
      maxBytes: 4,
    });
    expect(timeline.push({ seq: 0, t_start: 0, t_end: 0.1, byte_length: 2 }).evicted).toBe(0);
    expect(timeline.push({ seq: 1, t_start: 0.1, t_end: 0.2, byte_length: 2 }).evicted).toBe(0);
    const result = timeline.push({
      seq: 2,
      t_start: 0.2,
      t_end: 0.3,
      byte_length: 2,
    });
    expect(result.evicted).toBe(1);
    expect(result.evicted_bytes).toBe(2);
    expect(timeline.units().map((unit) => unit.seq)).toEqual([1, 2]);
    expect(() =>
      timeline.push({ seq: 3, t_start: 0.3, t_end: 0.4, byte_length: 5 })
    ).toThrow(/byte cap/);
    expect(() =>
      timeline.push({
        seq: 3,
        t_start: 0.3,
        t_end: 0.4,
        byte_length: 1,
        pcm: new Uint8Array(4),
      })
    ).toThrow(/disagrees/);
    expect(timeline.units().map((unit) => unit.seq)).toEqual([1, 2]);
  });

  it("remembers duplicates after their retained unit is evicted", () => {
    const timeline = new OrderedAudioBuffer({ seconds: 100, maxUnits: 2 });
    for (let seq = 0; seq < 3; seq++) {
      timeline.push({ seq, t_start: seq * 0.1, t_end: (seq + 1) * 0.1 });
    }
    expect(timeline.units().map((unit) => unit.seq)).toEqual([1, 2]);
    const duplicate = timeline.push({ seq: 0, t_start: 0, t_end: 0.1 });
    expect(duplicate.accepted).toBe(false);
    expect(duplicate.duplicate).toBe(true);
  });

  it("retains immutable metadata and returns defensive copies", () => {
    const boundary = { kind: "onset", t: 0.05 };
    const frame = { sample_rate: 16000, channels: 1, sample_format: "s16le" };
    const timeline = new OrderedAudioBuffer();
    timeline.push({
      seq: 0,
      t_start: 0,
      t_end: 0.1,
      sample_rate: 16000,
      channels: 1,
      sample_format: "s16le",
      byte_length: 2,
      boundary_marker: "kept",
      boundaries: [boundary],
      frame,
    });
    boundary.t = 9;
    frame.sample_rate = 8000;
    const firstRead = timeline.units();
    expect(firstRead[0].boundaries[0].t).toBe(0.05);
    expect(firstRead[0].frame.sample_rate).toBe(16000);
    expect(Object.isFrozen(firstRead[0])).toBe(true);
    expect(Object.isFrozen(firstRead[0].boundaries)).toBe(true);
    expect(Object.isFrozen(firstRead[0].boundaries[0])).toBe(true);
    expect(Object.isFrozen(firstRead[0].frame)).toBe(true);
    expect(() => {
      firstRead[0].boundaries[0].t = 7;
    }).toThrow(TypeError);
    expect(() => {
      firstRead[0].frame.sample_rate = 4000;
    }).toThrow(TypeError);
    expect(timeline.units()[0].boundaries[0].t).toBe(0.05);
    expect(timeline.units()[0].frame.sample_rate).toBe(16000);
  });

  it("canonicalizes frame-only format metadata and validates PCM alignment", () => {
    const timeline = new OrderedAudioBuffer();
    timeline.push({
      seq: 0,
      t_start: 0,
      t_end: 0.1,
      frame: {
        sample_rate: 16000,
        channels: 2,
        sample_format: "s16le",
        pcm: new Uint8Array(4),
      },
    });
    const retained = timeline.units()[0];
    expect(retained.sample_rate).toBe(16000);
    expect(retained.channels).toBe(2);
    expect(retained.sample_format).toBe("s16le");
    expect(retained.byte_length).toBe(4);

    expect(() =>
      new OrderedAudioBuffer().push({
        seq: 0,
        t_start: 0,
        t_end: 0.1,
        frame: {
          sample_rate: 16000,
          channels: 2,
          sample_format: "s16le",
          pcm: new Uint8Array(2),
        },
      })
    ).toThrow(/aligned/);
    expect(() =>
      new OrderedAudioBuffer().push({
        seq: 0,
        t_start: 0,
        t_end: 0.1,
        frame: {
          sample_rate: 0,
          channels: 1,
          sample_format: "s16le",
          pcm: new Uint8Array(2),
        },
      })
    ).toThrow(/sample_rate/);
  });

  it("deep-copies top-level and frame PCM on ingress and every egress", () => {
    const topPcm = new Uint8Array([1, 2]);
    const framePcm = new Uint8Array([3, 4]);
    const timeline = new OrderedAudioBuffer();
    timeline.push({
      seq: 0,
      t_start: 0,
      t_end: 0.1,
      sample_rate: 16000,
      channels: 1,
      sample_format: "s16le",
      byte_length: 2,
      pcm: topPcm,
    });
    timeline.push({
      seq: 1,
      t_start: 0.1,
      t_end: 0.2,
      frame: {
        sample_rate: 16000,
        channels: 1,
        sample_format: "s16le",
        pcm: framePcm,
      },
    });
    topPcm[0] = 9;
    framePcm[0] = 9;
    const firstRead = timeline.units();
    expect([...firstRead[0].pcm]).toEqual([1, 2]);
    expect([...firstRead[1].frame.pcm]).toEqual([3, 4]);
    firstRead[0].pcm[0] = 8;
    firstRead[1].frame.pcm[0] = 8;
    const secondRead = timeline.units();
    expect([...secondRead[0].pcm]).toEqual([1, 2]);
    expect([...secondRead[1].frame.pcm]).toEqual([3, 4]);
    expect(secondRead[0].pcm).not.toBe(firstRead[0].pcm);
    expect(secondRead[1].frame.pcm).not.toBe(firstRead[1].frame.pcm);
  });

  it("uses half-open windows for units and boundaries", () => {
    const timeline = new OrderedAudioBuffer();
    timeline.push({
      seq: 0,
      t_start: 0,
      t_end: 0.1,
      boundaries: [
        { kind: "onset", t: 0 },
        { kind: "endpoint", t: 0.1 },
      ],
    });
    expect(timeline.window(0.1, 0.2)).toEqual([]);
    expect(timeline.boundaries(0, 0.1).map((boundary) => boundary.kind)).toEqual(["onset"]);
  });

  it.each([
    {},
    { seconds: Number.NaN },
    { maxUnits: 0 },
    { maxUnits: Number.MAX_SAFE_INTEGER },
    { maxBytes: 0 },
  ])("rejects invalid buffer config %#", (config) => {
    if (Object.keys(config).length === 0) return;
    expect(() => new OrderedAudioBuffer(config)).toThrow();
  });

  it.each([
    { seq: -1, t_start: 0, t_end: 0.1 },
    { seq: Number.MAX_SAFE_INTEGER + 1, t_start: 0, t_end: 0.1 },
    { seq: 0, t_start: Number.NaN, t_end: 0.1 },
    { seq: 0, t_start: 0.2, t_end: 0.1 },
    { seq: 0, t_start: 0, t_end: 0.1, energy: 1.1 },
    { seq: 0, t_start: 0, t_end: 0.1, sample_rate: 0 },
    { seq: 0, t_start: 0, t_end: 0.1, channels: 0 },
    { seq: 0, t_start: 0, t_end: 0.1, event_t: Number.POSITIVE_INFINITY },
    {
      seq: 0,
      t_start: 0,
      t_end: 0.1,
      boundaries: [{ kind: "onset", t: Number.NaN }],
    },
  ])("rejects malformed unit %#", (unit) => {
    expect(() => new OrderedAudioBuffer().push(unit)).toThrow();
  });
});
