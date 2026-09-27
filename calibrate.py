import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, lsq_linear

from equations import bytes_moved, energy, flops, latency


def _usable_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[~frame["oom"]].copy()


def _training_rows(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    data = _usable_rows(frame)
    mask = ~data["is_validation"].astype(bool)
    for column in columns:
        values = data[column].to_numpy(dtype=float)
        mask &= np.isfinite(values) & (values > 0)
    return data.loc[mask]


def fit_latency(frame: pd.DataFrame) -> dict[str, float]:
    train = _training_rows(frame, ["latency_s"])
    image_size = train["image_size"].to_numpy(dtype=float)
    batch = train["batch"].to_numpy(dtype=float)
    observed = train["latency_s"].to_numpy(dtype=float)

    initial = np.log([10e-6, 300e9, 8e12])
    lower = np.log([1e-9, 1e6, 1e6])
    upper = np.log([1e-2, 1e15, 1e16])

    def residual(log_parameters: np.ndarray) -> np.ndarray:
        tau, bandwidth, throughput = np.exp(log_parameters)
        theta = {
            "tau_s": tau,
            "bandwidth_bytes_per_s": bandwidth,
            "throughput_flops_per_s": throughput,
        }
        return (latency(image_size, batch, theta) - observed) / observed

    result = least_squares(residual, initial, bounds=(lower, upper))
    tau, bandwidth, throughput = np.exp(result.x)
    return {
        "tau_s": float(tau),
        "bandwidth_bytes_per_s": float(bandwidth),
        "throughput_flops_per_s": float(throughput),
    }


def fit_energy(
    frame: pd.DataFrame,
    latency_theta: dict[str, float],
) -> dict[str, object]:
    train = _training_rows(frame, ["energy_j"])
    image_size = train["image_size"].to_numpy(dtype=float)
    batch = train["batch"].to_numpy(dtype=float)
    observed = train["energy_j"].to_numpy(dtype=float)

    features = np.column_stack(
        [
            latency(image_size, batch, latency_theta),
            flops(image_size, batch),
            bytes_moved(image_size, batch),
        ]
    )
    relative_design = features / observed[:, None]
    column_scale = np.linalg.norm(relative_design, axis=0)

    result = lsq_linear(
        relative_design / column_scale,
        np.ones_like(observed),
        bounds=(0, np.inf),
    )
    idle_power, joules_per_flop, joules_per_byte = result.x / column_scale
    return {
        "idle_power_w": float(idle_power),
        "joules_per_flop": float(joules_per_flop),
        "joules_per_byte": float(joules_per_byte),
        "latency": latency_theta,
    }


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    relative_error = np.abs(predicted - observed) / observed
    return {
        "count": int(observed.size),
        "mape_percent": float(100 * np.mean(relative_error)),
        "rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
    }


def calibration_metrics(
    frame: pd.DataFrame,
    latency_theta: dict[str, float],
    energy_theta: dict[str, object],
) -> dict[str, dict[str, dict[str, float | int]]]:
    data = _usable_rows(frame)
    metrics: dict[str, dict[str, dict[str, float | int]]] = {
        "latency": {},
        "energy": {},
    }
    for split_name, validation_value in (("train", False), ("validation", True)):
        split = data.loc[data["is_validation"].astype(bool) == validation_value]
        for metric_name, column, predict in (
            ("latency", "latency_s", lambda s, b: latency(s, b, latency_theta)),
            ("energy", "energy_j", lambda s, b: energy(s, b, energy_theta)),
        ):
            observed = split[column].to_numpy(dtype=float)
            finite = np.isfinite(observed) & (observed > 0)
            if not finite.any():
                metrics[metric_name][split_name] = {
                    "count": 0,
                    "mape_percent": float("nan"),
                    "rmse": float("nan"),
                }
                continue
            image_size = split["image_size"].to_numpy(dtype=float)[finite]
            batch = split["batch"].to_numpy(dtype=float)[finite]
            predicted = np.asarray(predict(image_size, batch), dtype=float)
            metrics[metric_name][split_name] = _metrics(observed[finite], predicted)
    return metrics


def calibrate(frame: pd.DataFrame) -> dict[str, object]:
    latency_theta = fit_latency(frame)
    energy_theta = fit_energy(frame, latency_theta)
    return {
        "latency": latency_theta,
        "energy": energy_theta,
        "metrics": calibration_metrics(frame, latency_theta, energy_theta),
    }


def save_calibration(frame: pd.DataFrame, output_path: str | Path) -> dict[str, object]:
    result = calibrate(frame)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("measurements", nargs="?", default="results/measurements.csv")
    parser.add_argument("--output", default="results/theta.json")
    args = parser.parse_args()
    save_calibration(pd.read_csv(args.measurements), args.output)


if __name__ == "__main__":
    main()
