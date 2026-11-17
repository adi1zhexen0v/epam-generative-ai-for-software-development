from data_generator import generate_data, max_of_three
from neuron_network import NeuronNetwork
from config import N_TRAIN, N_VAL, N_TEST, TRAIN_SEED, VAL_SEED, TEST_SEED, CHECK_EVERY, TOLERANCE

def main() -> None:
    X_train, y_train = generate_data(N_TRAIN, seed=TRAIN_SEED)
    X_val, y_val = generate_data(N_VAL, seed=VAL_SEED)
    X_test, y_test = generate_data(N_TEST, seed=TEST_SEED)

    network = NeuronNetwork()
    network.train(training_inputs=X_train, training_outputs=y_train, val_inputs=X_val, val_outputs=y_val)

    train_loss = network.history["train_loss"]
    val_loss = network.history["val_loss"]
    print(f"Epochs run: {len(train_loss) * CHECK_EVERY}")
    print(f"Train MSE: {train_loss[0]:.6f} -> {train_loss[-1]:.6f}")
    print(f"Val MSE:   {val_loss[0]:.6f} -> {min(val_loss):.6f} (best)")

    print(f"Test accuracy (|error| <= {TOLERANCE}): {(network.evaluate(X_test, y_test) * 100):.2f}%")
    print(f"Test MAE: {network.mae(X_test, y_test):.4f}")
    for x1, x2, x3 in [(0.5, -0.2, 0.1), (1, 1, 1), (-1, -1, -1), (-0.7, -0.3, -0.9), (0.2, 0.8, -0.6), (0, 0, 0)]:
        print(f"({x1}, {x2}, {x3}) -> {network.predict([x1, x2, x3])[0]:.3f} (true {max_of_three(x1, x2, x3)})")

if __name__ == "__main__":
    main()
