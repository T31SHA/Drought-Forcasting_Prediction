// Shape of src/data/forecast.json, written by `python -m pipeline.train`.

export interface Metric {
  model: string;
  horizon: number;
  n: number;
  rmse: number;
  mae: number;
  r2: number;
  explained_variance: number;
  bias: number;
  skill_vs_persistence: number;
  drought_months: number;
  drought_hits: number;
  drought_misses: number;
  drought_false_alarms: number;
  pod: number | null;
  far: number | null;
  coverage_80?: number | null;
  coverage_95?: number;
  interval_width_80?: number;
  interval_width_95?: number;
}

export interface Driver {
  feature: string;
  label: string;
  value: number;
  contribution: number;
}

export interface HorizonForecast {
  horizon: number;
  target_date: string;
  model: string;
  prediction: number;
  class: string;
  intervals: Record<"80" | "95", [number, number]>;
  p_drought: number;
  drivers: Driver[];
}

export interface Backtest {
  dates: string[];
  observed: number[];
  selected: string;
  predictions: Record<string, number[]>;
  radius: Record<"80" | "95", number>;
}

export interface Payload {
  meta: {
    author: string;
    location: { name: string; lat: number; lon: number };
    data_start: string;
    data_end: string;
    target: string;
    test_window: [string, string];
    dev_end: string;
    n_trials: number;
    cv_folds: number;
  };
  observed: { dates: string[]; spei1: (number | null)[]; spei3: (number | null)[]; spei6: (number | null)[]; spei12: (number | null)[]; spei24: (number | null)[] };
  current: { date: string; spei12: number; class: string };
  forecast: HorizonForecast[];
  metrics: Metric[];
  cv_metrics: Metric[];
  legacy: { horizon: number; rmse: number; reported_rmse: number; r2: number; skill_vs_persistence: number }[];
  season: { model: string; horizon: number; season: string; n: number; rmse: number; mae: number; bias: number }[];
  regime_cv: { model: string; horizon: number; regime: string; n: number; rmse: number; mae: number; bias: number }[];
  importance: Record<string, { feature: string; label: string; value: number }[]>;
  backtest: Record<string, Backtest>;
  bootstrap: Record<string, { rmse_90ci: Record<string, [number, number]>; p_lower: Record<string, number> }>;
}
