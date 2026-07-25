// percept-edge — on-device reactive edge for the Percept SDK.
//
// Public surface: the frozen wire-contract, the motion gate, the energy VAD,
// and the Edge class that wires the WatchSpec-down / Tier0Signal-up loop.

export {
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
} from "./contract.js";

export { motionScore, makeFrame } from "./motion.js";

export {
  frameEnergy,
  pcm16Energy,
  OrderedAudioBuffer,
  EnergyVad,
  endpointEnergySequence,
  SileroVadAdapter,
} from "./vad.js";

export { Edge } from "./edge.js";

export {
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
} from "./skills.js";

export { openness, PoseOpenness, registerPoseSkill } from "./pose.js";

export {
  PrivacyScreen,
  makePrivacyVerdict,
  PRIVACY_CATEGORIES,
  PRIVACY_ACTIONS,
  PRIVACY_SCREEN_VERSION,
} from "./privacy.js";
export { scanText } from "./privacy-text.js";
export { PrivacyImageAdapter } from "./privacy-image.js";
export { PrivacyAudioHead } from "./privacy-audio.js";
