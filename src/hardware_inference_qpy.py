"""Run inference on IBM Quantum hardware using a trained QPY circuit."""

from __future__ import annotations

import numpy as np
from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session

from config import (
    C_DEPTH,
    IBM_API_KEY,
    IBM_BACKEND,
    IBM_CHANNEL,
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

    service = QiskitRuntimeService(channel=IBM_CHANNEL, token=IBM_API_KEY)

    predictions = []
    with Session(service=service, backend=IBM_BACKEND) as session:
        sampler = Sampler(session=session, options={"shots": IBM_SHOTS})

        for idx in range(min(sample_count, len(x_test))):
            qc = _build_sampling_circuit(model, x_test[idx])
            job = sampler.run([qc])
            quasi = job.result().quasi_dists[0]
            probs = _quasi_to_probs(quasi, num_class)
            pred_label = int(np.argmax(probs))

            predictions.append(pred_label)
            print(
                f"sample {idx}: predicted={pred_label}, true={int(y_test[idx])}, distribution={probs}"
            )

    return predictions


if __name__ == "__main__":
    print(
        f"Starting hardware inference using backend '{IBM_BACKEND}' with {IBM_SHOTS} shots per circuit."
    )
    run_hardware_inference()
