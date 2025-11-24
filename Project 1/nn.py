import numpy as np


class NeuralNetwork:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size=1,
        activation="relu",
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        batch_size=32,
        target_loss=None,
        max_epochs=1000,
        seed=None,
        l2_lambda=0.0,
        dropout_rate=0.0,
        lr_decay=None,
        early_stopping_patience=None,
    ):
        if seed is not None:
            np.random.seed(seed)

        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1).")

        self.activation_name = activation
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.base_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.l2_lambda = max(0.0, l2_lambda)
        self.dropout_rate = dropout_rate
        self._dropout_mask = None

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.target_loss = target_loss
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self._patience_counter = 0

        if activation == "relu":
            limit1 = np.sqrt(2.0 / input_size)
        else:
            limit1 = np.sqrt(1.0 / input_size)
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * limit1

        limit2 = np.sqrt(1.0 / hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * limit2

        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

        self.m_w_ih = np.zeros_like(self.weights_input_hidden)
        self.v_w_ih = np.zeros_like(self.weights_input_hidden)
        self.m_b_h = np.zeros_like(self.bias_hidden)
        self.v_b_h = np.zeros_like(self.bias_hidden)

        self.m_w_ho = np.zeros_like(self.weights_hidden_output)
        self.v_w_ho = np.zeros_like(self.weights_hidden_output)
        self.m_b_o = np.zeros_like(self.bias_output)
        self.v_b_o = np.zeros_like(self.bias_output)

        self.t = 0

        self.history = {
            "epoch": [],
            "loss_full": [],
            "loss_batches": [],
            "class_error_full": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "val_loss": [],
            "lr": [],
            "w_ih_norm": [],
            "w_ho_norm": [],
        }

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        result = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
        return np.clip(result, 1e-7, 1 - 1e-7)

    def sigmoid_derivative_from_output(self, y):
        return y * (1.0 - y)

    def relu(self, x):
        return np.maximum(0.0, x)

    def relu_derivative_from_pre(self, x):
        return (x > 0.0).astype(float)

    def feedforward(self, X, training=False):
        self.z_hidden = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        if self.activation_name == "sigmoid":
            self.a_hidden = self.sigmoid(self.z_hidden)
        elif self.activation_name == "relu":
            self.a_hidden = self.relu(self.z_hidden)
        else:
            raise ValueError("Unsupported activation function")

        if training and self.dropout_rate > 0.0:
            keep_prob = 1.0 - self.dropout_rate
            self._dropout_mask = (
                np.random.rand(*self.a_hidden.shape) < keep_prob
            ).astype(self.a_hidden.dtype) / keep_prob
            self.a_hidden *= self._dropout_mask
        else:
            self._dropout_mask = None

        self.z_output = (
            np.dot(self.a_hidden, self.weights_hidden_output) + self.bias_output
        )
        self.y_hat = self.sigmoid(self.z_output)
        return self.y_hat

    def backward_batch(self, X_batch, y_batch):
        m = X_batch.shape[0]
        y_hat = self.feedforward(X_batch, training=True)

        dL_dy_hat = (y_hat - y_batch) / m
        delta_output = dL_dy_hat * self.sigmoid_derivative_from_output(y_hat)

        grad_w_ho = np.dot(self.a_hidden.T, delta_output)
        grad_b_o = np.sum(delta_output, axis=0, keepdims=True)

        hidden_error = np.dot(delta_output, self.weights_hidden_output.T)
        if self.activation_name == "sigmoid":
            hidden_delta = hidden_error * self.sigmoid_derivative_from_output(
                self.a_hidden
            )
        else:
            hidden_delta = hidden_error * self.relu_derivative_from_pre(self.z_hidden)

        if self._dropout_mask is not None:
            hidden_delta *= self._dropout_mask

        grad_w_ih = np.dot(X_batch.T, hidden_delta)
        grad_b_h = np.sum(hidden_delta, axis=0, keepdims=True)

        if self.l2_lambda > 0.0:
            grad_w_ho += (self.l2_lambda / m) * self.weights_hidden_output
            grad_w_ih += (self.l2_lambda / m) * self.weights_input_hidden

        self.t += 1

        self.m_w_ho = self.beta1 * self.m_w_ho + (1 - self.beta1) * grad_w_ho
        self.v_w_ho = self.beta2 * self.v_w_ho + (1 - self.beta2) * (grad_w_ho**2)
        m_hat_w_ho = self.m_w_ho / (1 - self.beta1**self.t)
        v_hat_w_ho = self.v_w_ho / (1 - self.beta2**self.t)
        self.weights_hidden_output -= (
            self.learning_rate * m_hat_w_ho / (np.sqrt(v_hat_w_ho) + self.epsilon)
        )

        self.m_b_o = self.beta1 * self.m_b_o + (1 - self.beta1) * grad_b_o
        self.v_b_o = self.beta2 * self.v_b_o + (1 - self.beta2) * (grad_b_o**2)
        m_hat_b_o = self.m_b_o / (1 - self.beta1**self.t)
        v_hat_b_o = self.v_b_o / (1 - self.beta2**self.t)
        self.bias_output -= (
            self.learning_rate * m_hat_b_o / (np.sqrt(v_hat_b_o) + self.epsilon)
        )

        self.m_w_ih = self.beta1 * self.m_w_ih + (1 - self.beta1) * grad_w_ih
        self.v_w_ih = self.beta2 * self.v_w_ih + (1 - self.beta2) * (grad_w_ih**2)
        m_hat_w_ih = self.m_w_ih / (1 - self.beta1**self.t)
        v_hat_w_ih = self.v_w_ih / (1 - self.beta2**self.t)
        self.weights_input_hidden -= (
            self.learning_rate * m_hat_w_ih / (np.sqrt(v_hat_w_ih) + self.epsilon)
        )

        self.m_b_h = self.beta1 * self.m_b_h + (1 - self.beta1) * grad_b_h
        self.v_b_h = self.beta2 * self.v_b_h + (1 - self.beta2) * (grad_b_h**2)
        m_hat_b_h = self.m_b_h / (1 - self.beta1**self.t)
        v_hat_b_h = self.v_b_h / (1 - self.beta2**self.t)
        self.bias_hidden -= (
            self.learning_rate * m_hat_b_h / (np.sqrt(v_hat_b_h) + self.epsilon)
        )

        batch_mse = 0.5 * np.mean((y_hat - y_batch) ** 2)
        return batch_mse

    def _compute_full_metrics(self, X, y):
        y_hat = self.feedforward(X, training=False)
        mse = 0.5 * np.mean((y_hat - y) ** 2)
        y_pred_labels = (y_hat >= 0.5).astype(int)
        class_error = 1.0 - np.mean(y_pred_labels == y)
        return mse, class_error

    def train(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        verbose=True,
        reset_optimizer=True,
        clear_history=True,
    ):
        X, y = self._validate_inputs(X, y)

        if (X_val is None) ^ (y_val is None):
            raise ValueError("Provide both X_val and y_val or neither.")
        if X_val is not None:
            X_val, y_val = self._validate_inputs(X_val, y_val)

        if clear_history:
            self._reset_history()
        if reset_optimizer:
            self._reset_optimizer_state()
        self.learning_rate = self.base_learning_rate
        self._patience_counter = 0

        N = X.shape[0]
        best_loss = np.inf
        best_state = None
        best_epoch = -1
        stop_reason = "max_epochs"

        if verbose:
            print("=" * 60)
            print("ROZPOCZĘCIE TRENINGU (Adam Optimizer)")
            print("=" * 60)

        for epoch in range(self.max_epochs):
            self.learning_rate = self._compute_lr(epoch)
            perm = np.random.permutation(N)
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            batch_losses = []

            for start in range(0, N, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                if X_batch.shape[0] == 0:
                    continue

                batch_mse = self.backward_batch(X_batch, y_batch)
                batch_losses.append(batch_mse)

            epoch_loss, epoch_class_error = self._compute_full_metrics(X, y)
            train_accuracy = 1.0 - epoch_class_error

            if X_val is not None and y_val is not None:
                val_loss, val_class_error = self._compute_full_metrics(X_val, y_val)
                val_accuracy = 1.0 - val_class_error
            else:
                val_loss = np.nan
                val_accuracy = np.nan

            monitored_loss = val_loss if np.isfinite(val_loss) else epoch_loss
            if monitored_loss < best_loss:
                best_loss = monitored_loss
                best_state = self._snapshot_weights()
                best_epoch = epoch
                self._patience_counter = 0
            else:
                self._patience_counter += 1

            self._log_epoch(
                epoch,
                epoch_loss,
                epoch_class_error,
                batch_losses,
                train_accuracy,
                val_loss,
                val_accuracy,
            )

            show_epoch = epoch < 5 or epoch % 100 == 0 or epoch >= self.max_epochs - 5

            if verbose and show_epoch:
                if np.isfinite(val_accuracy):
                    print(
                        f"Epoch {epoch:4d} | Loss: {epoch_loss:.6f} | "
                        f"Train Acc: {train_accuracy*100:.2f}% | "
                        f"Val Acc: {val_accuracy*100:.2f}%"
                    )
                else:
                    print(
                        f"Epoch {epoch:4d} | Loss: {epoch_loss:.6f} | "
                        f"Accuracy: {train_accuracy*100:.2f}%"
                    )

            if (
                self.early_stopping_patience is not None
                and self._patience_counter >= self.early_stopping_patience
            ):
                stop_reason = (
                    f"early_stopping (patience={self.early_stopping_patience})"
                )
                if verbose:
                    print(
                        f"\n[Epoch {epoch}] Validation loss hasn't improved for {self.early_stopping_patience} epochs. Stopping."
                    )
                break

            if self.target_loss is not None and epoch_loss <= self.target_loss:
                stop_reason = f"target_loss ({self.target_loss}) reached"
                if verbose:
                    print(
                        f"\n[Epoch {epoch}] Target loss {self.target_loss} reached. Stopping."
                    )
                break

        if best_state is not None:
            self._restore_weights(best_state)

        if verbose:
            print("\n" + "=" * 60)
            print("ZAKOŃCZENIE TRENINGU")
            print("=" * 60)
            print(f"Przyczyna zatrzymania: {stop_reason}")
            print(f"Liczba wykonanych epok: {epoch + 1} / {self.max_epochs}")
            print(
                f"Najlepszy (monitorowany) loss: {best_loss:.6f} (epoch {best_epoch})"
            )

            if X_val is not None and y_val is not None:
                final_val_loss, final_val_err = self._compute_full_metrics(X_val, y_val)
                final_val_accuracy = 1.0 - final_val_err
                print(f"\nKOŃCOWE METRYKI WALIDACYJNE:")
                print(f"  Loss: {final_val_loss:.6f}")
                print(f"  Accuracy: {final_val_accuracy*100:.2f}%")

            final_train_loss, final_train_err = self._compute_full_metrics(X, y)
            final_train_accuracy = 1.0 - final_train_err
            print(f"\nKOŃCOWE METRYKI TRENINGOWE:")
            print(f"  Loss: {final_train_loss:.6f}")
            print(f"  Accuracy: {final_train_accuracy*100:.2f}%")
            print("=" * 60)

        return self.history

    def _log_epoch(
        self,
        epoch,
        epoch_loss,
        epoch_class_error,
        batch_losses,
        train_accuracy,
        val_loss,
        val_accuracy,
    ):
        self.history["epoch"].append(epoch)
        self.history["loss_full"].append(epoch_loss)
        self.history["class_error_full"].append(epoch_class_error)
        self.history["loss_batches"].append(
            np.mean(batch_losses) if batch_losses else np.nan
        )
        self.history["train_accuracy"].append(train_accuracy)
        self.history["val_loss"].append(val_loss)
        self.history["val_accuracy"].append(val_accuracy)
        self.history["lr"].append(self.learning_rate)
        self.history["w_ih_norm"].append(
            float(np.linalg.norm(self.weights_input_hidden))
        )
        self.history["w_ho_norm"].append(
            float(np.linalg.norm(self.weights_hidden_output))
        )

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self.feedforward(X, training=False)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def _reset_optimizer_state(self):
        self.m_w_ih.fill(0.0)
        self.v_w_ih.fill(0.0)
        self.m_b_h.fill(0.0)
        self.v_b_h.fill(0.0)
        self.m_w_ho.fill(0.0)
        self.v_w_ho.fill(0.0)
        self.m_b_o.fill(0.0)
        self.v_b_o.fill(0.0)
        self.t = 0

    def _reset_history(self):
        for key in self.history:
            self.history[key] = []

    def _validate_inputs(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError("X must be a 2-D array.")
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim == 2 and y.shape[1] != 1:
            raise ValueError("y must be 1-D or have shape (n_samples, 1).")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        return X, y

    def _compute_lr(self, epoch):
        if not self.lr_decay:
            return self.base_learning_rate

        decay = 0.0
        step = 1

        if isinstance(self.lr_decay, dict):
            decay = float(self.lr_decay.get("decay", 0.0))
            step = int(self.lr_decay.get("step", 1))
        elif isinstance(self.lr_decay, (list, tuple)) and len(self.lr_decay) == 2:
            decay = float(self.lr_decay[0])
            step = int(self.lr_decay[1])
        elif isinstance(self.lr_decay, (int, float)):
            decay = float(self.lr_decay)
        else:
            return self.base_learning_rate

        step = max(step, 1)
        if decay <= 0.0:
            return self.base_learning_rate

        return self.base_learning_rate / (1.0 + decay * (epoch // step))

    def _snapshot_weights(self):
        return {
            "weights_input_hidden": self.weights_input_hidden.copy(),
            "weights_hidden_output": self.weights_hidden_output.copy(),
            "bias_hidden": self.bias_hidden.copy(),
            "bias_output": self.bias_output.copy(),
        }

    def _restore_weights(self, state):
        self.weights_input_hidden = state["weights_input_hidden"]
        self.weights_hidden_output = state["weights_hidden_output"]
        self.bias_hidden = state["bias_hidden"]
        self.bias_output = state["bias_output"]
