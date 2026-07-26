import { describe, it, expect } from "vitest";
import {
  CONTRACT_VERSION,
  ContractError,
  makeBox,
  parseBox,
  makeWatchCondition,
  parseWatchCondition,
  makeWatchSpec,
  parseWatchSpec,
  makeTier0Signal,
  parseTier0Signal,
  makeBoundary,
  parseBoundary,
  makeEdgeCapabilities,
  parseEdgeCapabilities,
  makeBoundaryReport,
  parseBoundaryReport,
  makeIncidentSnippet,
  parseIncidentSnippet,
  makeIncident,
  parseIncident,
  makeSubjectRef,
  parseSubjectRef,
  provisionalSubject,
  PROVISIONAL_KEY_PREFIX,
  motionScore,
  makeFrame,
  frameEnergy,
  EnergyVad,
  endpointEnergySequence,
  SileroVadAdapter,
  Edge,
  MotionPeriodicity,
  AcousticOnset,
  SkillBank,
  wakeScore,
  shouldWake,
  registerSkill,
  makeDetector,
  manifestOf,
  listManifests,
  registered,
  detectorsInCategory,
  signalOf,
  CATEGORY_MOTION,
  CATEGORY_AUDIO,
  CATEGORY_FRAME,
  CATEGORY_PRESENCE,
  CATEGORY_POSE,
  CATEGORY_AUDIO_EVENT,
  CATEGORY_CHANGE,
  CATEGORY_SEMANTIC_RELEVANCE,
  openness,
  PoseOpenness,
  registerPoseSkill,
  PrivacyScreen,
  makePrivacyVerdict,
  PRIVACY_CATEGORIES,
  PRIVACY_ACTIONS,
  PRIVACY_SCREEN_VERSION,
  scanText,
  PrivacyImageAdapter,
  PrivacyAudioHead,
} from "../src/index.js";

// Package-root re-export surface. A consumer doing `import { X } from "percept-edge"`
// (which resolves to src/index.js) must be able to reach every symbol the package
// advertises as public — including the SubjectRef surface, which historically was
// exported from contract.js but historically not re-exported at the package root.
describe("package root exports (src/index.js)", () => {
  it("re-exports the SubjectRef surface", () => {
    expect(typeof makeSubjectRef).toBe("function");
    expect(typeof parseSubjectRef).toBe("function");
    expect(typeof provisionalSubject).toBe("function");
    expect(PROVISIONAL_KEY_PREFIX).toBe("UN_");
  });

  it("builds and parses a provisional SubjectRef reached only via the root import", () => {
    const ref = provisionalSubject("ep-123");
    expect(ref.resolution).toBe("provisional");
    expect(ref.key).toBe(`${PROVISIONAL_KEY_PREFIX}ep-123`);

    const parsed = parseSubjectRef(JSON.stringify(ref), { path: "subject" });
    expect(parsed).toEqual(ref);
  });

  it("still re-exports the rest of the previously-public surface", () => {
    expect(typeof CONTRACT_VERSION).not.toBe("undefined");
    expect(typeof ContractError).toBe("function");
    expect(typeof makeBox).toBe("function");
    expect(typeof parseBox).toBe("function");
    expect(typeof makeWatchCondition).toBe("function");
    expect(typeof parseWatchCondition).toBe("function");
    expect(typeof makeWatchSpec).toBe("function");
    expect(typeof parseWatchSpec).toBe("function");
    expect(typeof makeTier0Signal).toBe("function");
    expect(typeof parseTier0Signal).toBe("function");
    expect(typeof makeBoundary).toBe("function");
    expect(typeof parseBoundary).toBe("function");
    expect(typeof makeEdgeCapabilities).toBe("function");
    expect(typeof parseEdgeCapabilities).toBe("function");
    expect(typeof makeBoundaryReport).toBe("function");
    expect(typeof parseBoundaryReport).toBe("function");
    expect(typeof makeIncidentSnippet).toBe("function");
    expect(typeof parseIncidentSnippet).toBe("function");
    expect(typeof makeIncident).toBe("function");
    expect(typeof parseIncident).toBe("function");
    expect(typeof motionScore).toBe("function");
    expect(typeof makeFrame).toBe("function");
    expect(typeof frameEnergy).toBe("function");
    expect(typeof EnergyVad).toBe("function");
    expect(typeof endpointEnergySequence).toBe("function");
    expect(typeof SileroVadAdapter).toBe("function");
    expect(typeof Edge).toBe("function");
    expect(typeof MotionPeriodicity).toBe("function");
    expect(typeof AcousticOnset).toBe("function");
    expect(typeof SkillBank).toBe("function");
    expect(typeof wakeScore).toBe("function");
    expect(typeof shouldWake).toBe("function");
    expect(typeof registerSkill).toBe("function");
    expect(typeof makeDetector).toBe("function");
    expect(typeof manifestOf).toBe("function");
    expect(typeof listManifests).toBe("function");
    expect(typeof registered).toBe("function");
    expect(typeof detectorsInCategory).toBe("function");
    expect(typeof signalOf).toBe("function");
    expect(typeof CATEGORY_MOTION).toBe("string");
    expect(typeof CATEGORY_AUDIO).toBe("string");
    expect(typeof CATEGORY_FRAME).toBe("string");
    expect(typeof CATEGORY_PRESENCE).toBe("string");
    expect(typeof CATEGORY_POSE).toBe("string");
    expect(typeof CATEGORY_AUDIO_EVENT).toBe("string");
    expect(typeof CATEGORY_CHANGE).toBe("string");
    expect(typeof CATEGORY_SEMANTIC_RELEVANCE).toBe("string");
    expect(typeof openness).toBe("function");
    expect(typeof PoseOpenness).toBe("function");
    expect(typeof registerPoseSkill).toBe("function");
    expect(typeof PrivacyScreen).toBe("function");
    expect(typeof makePrivacyVerdict).toBe("function");
    expect(typeof PRIVACY_CATEGORIES).not.toBe("undefined");
    expect(typeof PRIVACY_ACTIONS).not.toBe("undefined");
    expect(typeof PRIVACY_SCREEN_VERSION).not.toBe("undefined");
    expect(typeof scanText).toBe("function");
    expect(typeof PrivacyImageAdapter).toBe("function");
    expect(typeof PrivacyAudioHead).toBe("function");
  });
});
