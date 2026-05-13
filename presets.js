// Named visual configurations tuned per track.
// Each preset maps uniform keys + 'harmSmoothingSetting' to their tuned values.

export const presets = {
  'martin-jarl': {
    iHarmRotation:      1.5,
    iHarmSensLow:       0.03,
    iHarmSensHigh:      0.10,
    iHarmSat:           0.65,
    iTransientStrength: 0.0,
    iSubFloor:          0.60,
    iSubStrength:       0.15,
    iSubExp:            2.5,
    harmSmoothingSetting: 1.0,
  },
};

export const DEFAULT_PRESET = 'martin-jarl';
