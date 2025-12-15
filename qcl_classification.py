import numpy as np
from config import (
    GRAD_CLIP_NORM,
    LR_DECAY_STEPS,
    LR_DECAY_RATE,
)
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from sklearn.metrics import log_loss
from scipy.optimize import minimize
from qcl_utils import create_time_evol_gate, min_max_scaling, softmax

from tqdm import tqdm


class QclClassification:
    """Solve classification problems using quantum circuit learning (Qiskit backend)."""

    def __init__(self, nqubit, c_depth, num_class=None):
        """
        :param nqubit: Number of qubits; ``2**nqubit`` must be at least the number of classes.
        :param c_depth: Circuit depth.
        :param num_class: Number of classes. ``None`` lets ``fit`` infer this from ``y_list``.
        """
        self.nqubit = nqubit
        self.c_depth = c_depth

        self.input_x_list = []  # List of normalized inputs
        self.theta = []  # Flat list of parameters θ

        self.output_gate = None  # Parameterized circuit
        self.theta_params = None

        self.num_class = num_class  # Number of classes

        self.obs = None
        if self.num_class is not None and self.num_class <= self.nqubit:
            self._initialize_observable()

    def _initialize_observable(self):
        """Prepare observables based on ``num_class`` using ``Z`` expectation values."""
        if self.num_class is None:
            return
        if self.num_class > self.nqubit:
            raise ValueError("num_class exceeds number of qubits for observable-based decoding")
        obs = []
        for i in range(self.num_class):
            pauli = ["I"] * self.nqubit
            pauli[self.nqubit - i - 1] = "Z"  # Qiskit orders qubits left-to-right
            obs.append(SparsePauliOp.from_list([("").join(pauli), 1.0]))
        self.obs = obs

    def create_input_gate(self, x):
        """Generate the quantum circuit that encodes the input ``x``."""
        # Elements of x are assumed to lie within [-1, 1]
        u = QuantumCircuit(self.nqubit)

        angle_y = np.arcsin(x)
        angle_z = np.arccos(x**2)

        for i in range(self.nqubit):
            idx = i % len(x)
            u.ry(angle_y[idx], i)
            u.rz(angle_z[idx], i)

        return u

    def set_input_state(self, x_list):
        """Store the list of encoded input features."""
        x_list_normalized = min_max_scaling(x_list)  # Scale each feature of x to [-1, 1]
        self.input_x_list = list(x_list_normalized)

    def create_initial_output_gate(self):
        """Assemble the output gate ``U_out`` and initialize its parameters."""
        time_evol_gate = create_time_evol_gate(self.nqubit)
        theta = 2.0 * np.pi * np.random.rand(self.c_depth, self.nqubit, 3)
        self.theta = theta.flatten()

        param_count = self.c_depth * self.nqubit * 3
        self.theta_params = ParameterVector("theta", param_count)

        circuit = QuantumCircuit(self.nqubit)
        param_idx = 0
        for _ in range(self.c_depth):
            circuit.append(time_evol_gate, list(range(self.nqubit)))
            for i in range(self.nqubit):
                circuit.rx(self.theta_params[param_idx], i)
                param_idx += 1
                circuit.rz(self.theta_params[param_idx], i)
                param_idx += 1
                circuit.rx(self.theta_params[param_idx], i)
                param_idx += 1
        self.output_gate = circuit

    def update_output_gate(self, theta):
        """Update parameters ``θ``."""
        self.theta = theta

    def set_output_gate(self, circuit):
        """Assign a fixed, non-parameterized output circuit for inference."""

        self.output_gate = circuit
        self.theta_params = None
        self.theta = []

    def get_output_gate_parameter(self):
        """Retrieve the parameters ``θ`` for ``U_out``."""
        return np.array(self.theta)

    def _bind_output_gate(self, theta):
        if self.output_gate is None:
            raise RuntimeError("Output gate is not initialized.")

        if self.theta_params is None:
            return self.output_gate

        bind_map = {param: theta[idx] for idx, param in enumerate(self.theta_params)}
        return self.output_gate.assign_parameters(bind_map)

    def _state_for_x(self, x, theta):
        input_gate = self.create_input_gate(x)
        output_gate = self._bind_output_gate(theta)
        circuit = QuantumCircuit(self.nqubit)
        circuit.compose(input_gate, inplace=True)
        circuit.compose(output_gate, inplace=True)
        return Statevector.from_instruction(circuit)

    def pred(self, theta):
        """Compute the model outputs for ``x_list`` using observable expectations."""

        if len(self.input_x_list) == 0:
            raise RuntimeError("Input states are not set. Call set_input_state first.")

        res = []
        for x in self.input_x_list:
            state = self._state_for_x(x, theta)
            r = [np.real(state.expectation_value(o)) for o in self.obs]
            r = softmax(r)
            res.append(r.tolist())
        return np.array(res)

    def pred_amplitude(self, x_list):
        """Return class probabilities using state amplitudes."""

        x_scaled = min_max_scaling(x_list)
        res = []
        for x in x_scaled:
            state = self._state_for_x(x, self.theta)
            probs = []
            for cls in range(self.num_class):
                amp = state.data[cls]
                probs.append(abs(amp) ** 2)
            res.append(softmax(np.log(np.array(probs) + 1e-10)))
        return np.array(res)

    def cost_func(self, theta):
        """Compute the cost function value.

        :param theta: List of rotation-gate angles ``θ``.
        """

        y_pred = self.pred(theta)

        # cross-entropy loss
        loss = log_loss(self.y_list, y_pred)

        return loss

    # for BFGS
    def B_grad(self, theta):
        # Return the list of dB/dθ values via parameter shift
        theta_plus = [theta.copy() + np.eye(len(theta))[i] * np.pi / 2. for i in range(len(theta))]
        theta_minus = [theta.copy() - np.eye(len(theta))[i] * np.pi / 2. for i in range(len(theta))]

        grad = []
        for i in tqdm(range(len(theta)), desc="param", leave=False):
            grad.append((self.pred(theta_plus[i]) - self.pred(theta_minus[i])) / 2.)

        return np.array(grad)

    # for BFGS
    def cost_func_grad(self, theta):
        y_minus_t = self.pred(theta) - self.y_list
        B_gr_list = self.B_grad(theta)
        grad = [np.sum(y_minus_t * B_gr) for B_gr in B_gr_list]
        return np.array(grad)

    def fit(self, x_list, y_list, maxiter=1000):
        """
        :param x_list: Training inputs ``x``.
        :param y_list: Training targets ``y``.
        :param maxiter: Number of iterations for ``scipy.optimize.minimize``.
        :return: Loss value after training.
        :return: Optimized parameter values ``θ``.
        """

        # Determine num_class from y_list if not preset
        if self.num_class is None:
            self.num_class = y_list.shape[1]
        if self.num_class > 2 ** self.nqubit:
            raise ValueError("num_class exceeds representable classes for given qubits")
        if self.obs is None and self.num_class <= self.nqubit:
            self._initialize_observable()
        # Ensure that y_list matches num_class in dimensionality
        if y_list.shape[1] != self.num_class:
            raise ValueError("y_list and num_class mismatch")

        # Prepare initial states
        self.set_input_state(x_list)

        # Create a random U_out
        self.create_initial_output_gate()
        theta_init = self.theta

        # Store ground-truth labels
        self.y_list = y_list

        # for callbacks
        self.n_iter = 0
        self.maxiter = maxiter

        print("Initial parameter:")
        print()
        print(f"Initial value of cost function:  {self.cost_func(self.theta):.4f}")
        print()
        print('============================================================')
        print("Iteration count...")
        result = minimize(self.cost_func,
                          self.theta,
                          # method='Nelder-Mead',
                          method='BFGS',
                          jac=self.cost_func_grad,
                          options={"maxiter":maxiter},
                          callback=self.callbackF)
        theta_opt = self.theta
        print('============================================================')
        print()
        print("Optimized parameter:")
        print()
        print(f"Final value of cost function:  {self.cost_func(self.theta):.4f}")
        print()
        return result, theta_init, theta_opt

    def fit_backprop_inner_product(self, x_list, y_label, lr=0.1, n_iter=100):
        raise NotImplementedError("Gradient backpropagation is not implemented for the Qiskit backend.")

    def fit_llp_inner_product(
        self,
        x_list,
        bag_sampler,
        teacher_probs,
        lr=0.1,
        n_iter=100,
        loss="ce",
        n_jobs=1,
        return_history=False,
    ):
        """Train using bag-level label proportions with parameter-shift gradients."""

        teacher = (
            teacher_probs.numpy() if hasattr(teacher_probs, "numpy") else teacher_probs
        )

        if self.num_class is None:
            self.num_class = teacher.shape[1]
        if self.num_class > 2 ** self.nqubit:
            raise ValueError("num_class exceeds representable classes for given qubits")

        x_scaled = min_max_scaling(x_list)
        input_gates = [self.create_input_gate(x) for x in x_scaled]

        self.create_initial_output_gate()

        bag_list = [indices for indices in bag_sampler if len(indices) > 0]

        losses = []
        best_loss = float("inf")
        best_step = 0
        best_params = self.get_output_gate_parameter().copy()

        def predict_probs(theta_values, gates):
            probs = []
            for gate in gates:
                bound_gate = self._bind_output_gate(theta_values)
                circuit = QuantumCircuit(self.nqubit)
                circuit.compose(gate, inplace=True)
                circuit.compose(bound_gate, inplace=True)
                state = Statevector.from_instruction(circuit)
                p = np.abs(state.data) ** 2
                probs.append(p[: self.num_class])
            return probs

        for step in tqdm(range(n_iter),desc="[train]"):
            total_grad = np.zeros_like(self.theta)
            total_loss = 0.0

            for bag_indices, t_probs in zip(bag_list, teacher):
                gates = [input_gates[idx] for idx in bag_indices]
                preds = predict_probs(self.theta, gates)
                bag_pred = np.mean(preds, axis=0)

                eps = 1e-10
                if loss == "ce":
                    bag_loss = -np.sum(t_probs * np.log(bag_pred + eps))
                    dL_dp = -(t_probs / (bag_pred + eps))
                elif loss == "kl":
                    bag_loss = np.sum(t_probs * np.log(t_probs / (bag_pred + eps)))
                    dL_dp = -t_probs / (bag_pred + eps)
                else:
                    raise ValueError("loss must be 'ce' or 'kl'")

                total_loss += bag_loss

                # Parameter-shift gradients
                for j in range(len(self.theta)):
                    theta_plus = self.theta.copy()
                    theta_minus = self.theta.copy()
                    theta_plus[j] += np.pi / 2
                    theta_minus[j] -= np.pi / 2

                    preds_plus = predict_probs(theta_plus, gates)
                    preds_minus = predict_probs(theta_minus, gates)
                    bag_pred_plus = np.mean(preds_plus, axis=0)
                    bag_pred_minus = np.mean(preds_minus, axis=0)

                    grad_pred = (bag_pred_plus - bag_pred_minus) / 2.0
                    total_grad[j] += np.dot(dL_dp, grad_pred)

            total_grad /= len(bag_list)
            total_loss /= len(bag_list)

            # Gradient clipping
            if GRAD_CLIP_NORM is not None:
                norm = np.linalg.norm(total_grad)
                if norm > GRAD_CLIP_NORM:
                    total_grad = total_grad / norm * GRAD_CLIP_NORM

            # Parameter update
            self.theta = self.theta - lr * total_grad

            if step % 10 == 0 or step == n_iter - 1:
                losses.append(total_loss)
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_step = step
                    best_params = self.theta.copy()
                print(f"[{step:03d}] loss={total_loss:.6f}")

            if (step + 1) % LR_DECAY_STEPS == 0:
                lr *= LR_DECAY_RATE

        self.theta = best_params
        if return_history:
            return self.theta, losses, best_step
        return self.theta

    def callbackF(self, theta):
        self.n_iter += 1
        # print parameter and cost function every (maxiter/10) times
        if self.n_iter % (self.maxiter / 10) == 0:
            print('%7d-th step' % self.n_iter)
            print(theta)
            print('')
            print(f"value of cost function:  {self.cost_func(theta):.4f}")


if __name__ == '__main__':
    # Usage example
    from data_utils import create_fixed_proportion_batches
    from config import BAG_SIZE
    import sys
    sys.path.append('..')

    nq = 4
    cd = 1
    num_class = 4
    xc = QclClassification(nq, cd, num_class)

    data0 = np.random.normal(loc=0.0, scale=2.0, size=(1000, 5))
    data1 = np.random.normal(loc=3.0, scale=2.0, size=(1000, 5))
    xc.set_input_state(np.concatenate([data0, data1]))

    y0 = np.zeros(1000)
    y1 = np.ones(1000)
    label = np.concatenate([y0, y1])
    sample_index_batch, label_prop_batch = create_fixed_proportion_batches(label, BAG_SIZE)
    tp = []
    for label_prop in label_prop_batch:
        tp.append(np.array([1.0 - label_prop[1], label_prop[1]]))
    tp = np.array(tp)

    max_iter = 50
    xc.fit_llp_inner_product(np.concatenate([data0, data1]), sample_index_batch, tp, n_iter=max_iter, lr=0.1, loss='ce')
    # xc.fit(np.concatenate([data0, data1]), tp, maxiter=max_iter)
