"""Run inference on IBM Quantum hardware using a trained QPY circuit."""

from __future__ import annotations

import json
import os

import numpy as np
from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from config import (
    C_DEPTH,
    HARDWARE_RAW_RESULT_PATH,
    IBM_API_KEY,
    IBM_BACKEND,
    IBM_CHANNEL,
    IBM_INSTANCE_CRN,
    IBM_SHOTS,
    NQUBIT,
    PCA_DIM,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    WEIGHT_QPY_PATH,
)
from data_utils import load_pt_features
from qcl_classification import QclClassification
from qcl_utils import min_max_scaling


def load_trained_circuit(qpy_path: str):
    """Load a single trained circuit from a QPY file."""

    with open(qpy_path, "rb") as f:
        circuits = qpy.load(f)

    if not circuits:
        raise ValueError(f"No circuits found in QPY file: {qpy_path}")
    if len(circuits) > 1:
        print(
            f"[warning] Multiple circuits found in {qpy_path}; using the first entry for inference."
        )

    return circuits[0]


def _build_sampling_circuit(model: QclClassification, x_sample: np.ndarray):
    """Encode a single sample and attach measurements for hardware execution."""

    x_scaled = min_max_scaling(np.asarray([x_sample]))[0]
    input_gate = model.create_input_gate(x_scaled)

    circuit = input_gate.compose(model.output_gate)
    circuit.measure_all()
    return circuit


def _quasi_to_probs(quasi_dist, num_class: int):
    """Convert a quasi-distribution into class probabilities."""

    return np.array([quasi_dist.get(cls, 0.0) for cls in range(num_class)])


def run_hardware_inference(sample_count: int = 5):
    """Execute the trained circuit on IBM Quantum hardware.

    Parameters
    ----------
    sample_count : int, default 5
        Number of test samples to send to the hardware backend.
    """

    if not IBM_API_KEY:
        raise ValueError("Set IBM_API_KEY in config.py before running hardware inference.")

    _, x_test, _, y_test = load_pt_features(TRAIN_DATA_PATH, TEST_DATA_PATH, PCA_DIM)
    num_class = len(np.unique(y_test))

    circuit = load_trained_circuit(WEIGHT_QPY_PATH)
    model = QclClassification(NQUBIT, C_DEPTH, num_class)
    model.set_output_gate(circuit)

    service = QiskitRuntimeService(
        channel=IBM_CHANNEL,
        token=IBM_API_KEY,
        instance=IBM_INSTANCE_CRN,
    )

    predictions = []
    raw_results = []

    # IBM_BACKEND が名前(str)なら backend オブジェクトにする
    backend = service.backend(IBM_BACKEND) if isinstance(IBM_BACKEND, str) else IBM_BACKEND

    # ISA 回路に変換するための pass manager（2024-03-04 以降必須）
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)

    # Session は使わない（open plan だと session 作成が弾かれる）
    sampler = SamplerV2(mode=backend)

    for idx in range(min(sample_count, len(x_test))):
        qc = _build_sampling_circuit(model, x_test[idx])

        # ★ ISA に合わせて変換してから投げる
        isa_qc = pm.run(qc)

        job = sampler.run([isa_qc], shots=IBM_SHOTS)
        res = job.result()

        # SamplerV2 は quasi_dists ではなく counts 相当を返す
        counts = res[0].data.meas.get_counts()  # 例: {"00": 510, "11": 490}

        # 既存の _quasi_to_probs を活かすため、int-key の確率辞書に変換
        shots = sum(counts.values()) if counts else 1
        quasi = {int(bitstr, 2): c / shots for bitstr, c in counts.items()}

        probs = _quasi_to_probs(quasi, num_class)
        pred_label = int(np.argmax(probs))

        predictions.append(pred_label)
        raw_results.append(
            {
                "sample_index": idx,
                "counts": counts,
                "shots": shots,
                "quasi_distribution": quasi,
                "probabilities": probs.tolist(),
                "predicted_label": pred_label,
                "true_label": int(y_test[idx]),
            }
        )
        print(
            f"sample {idx}: predicted={pred_label}, true={int(y_test[idx])}, distribution={probs}"
        )

    os.makedirs(os.path.dirname(HARDWARE_RAW_RESULT_PATH) or ".", exist_ok=True)
    with open(HARDWARE_RAW_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, ensure_ascii=False, indent=2)
    print(f"Raw hardware data saved to {HARDWARE_RAW_RESULT_PATH}")

    return predictions


if __name__ == "__main__":
    print(
        f"Starting hardware inference using backend '{IBM_BACKEND}' with {IBM_SHOTS} shots per circuit."
    )
    run_hardware_inference()
