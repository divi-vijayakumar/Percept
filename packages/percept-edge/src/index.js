// @percept/edge — on-device reactive edge for the Percept SDK.
//
// Public surface: the frozen wire-contract, the motion gate, the energy VAD,
// and the Edge class that wires the WatchSpec-down / Tier0Signal-up loop.

export {
  ContractError,
  makeBox,
  parseBox,
  makeWatchCondition,
  parseWatchCondition,
  makeWatchSpec,
  parseWatchSpec,
  makeTier0Signal,
  parseTier0Signal,
} from "./contract.js";

export { motionScore, makeFrame } from "./motion.js";

export {
  frameEnergy,
  EnergyVad,
  endpointEnergySequence,
  SileroVadAdapter,
} from "./vad.js";

export { Edge } from "./edge.js";

export {
  makeBoundary,
  MotionPeriodicity,
  AcousticOnset,
  SkillBank,
  registerSkill,
  makeDetector,
  manifestOf,
  listManifests,
  registered,
  signalOf,
} from "./skills.js";
