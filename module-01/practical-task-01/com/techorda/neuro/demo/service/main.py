from data_generator import generate_data
from neuron_network import NeuronNetwork
from config import N_TEST, TEST_SEED, CHECK_EVERY

def main() -> None:
    X_test, y_test = generate_data(N_TEST, seed=TEST_SEED)

    network = NeuronNetwork()

    train_loss = network.history["train_loss"]
    val_loss = network.history["val_loss"]
    print(f"Epochs run: {len(train_loss) * CHECK_EVERY}")
    print(f"Train BCE: {train_loss[0]:.4f} -> {train_loss[-1]:.4f}")
    print(f"Val BCE:   {val_loss[0]:.4f} -> {min(val_loss):.4f} (best)")

    print(f"Test accuracy: {(network.evaluate(X_test, y_test) * 100):.2f}%")
    for x, y in [(0, 0), (1, 1), (4, -2), (3, 4), (6, 0), (9, 9)]:
        print(f"({x}, {y}) -> {network.classify(x, y)}")

if __name__ == "__main__":
    main()
