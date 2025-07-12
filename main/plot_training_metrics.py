import json
import matplotlib.pyplot as plt
import os

def load_metrics(log_dir):
    log_history_path = os.path.join(log_dir, "trainer_state.json")
    with open(log_history_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    log_history = state["log_history"]
    epochs = []
    train_loss = []
    eval_loss = []
    for entry in log_history:
        if "epoch" in entry:
            if "loss" in entry:
                epochs.append(entry["epoch"])
                train_loss.append(entry["loss"])
            if "eval_loss" in entry:
                eval_loss.append((entry["epoch"], entry["eval_loss"]))
    return epochs, train_loss, eval_loss

def plot_metrics(epochs, train_loss, eval_loss):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Training Loss", marker="o")
    if eval_loss:
        eval_epochs, eval_losses = zip(*eval_loss)
        plt.plot(eval_epochs, eval_losses, label="Validation Loss", marker="x")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    log_dir = "dnd_treasure_gpt2"
    epochs, train_loss, eval_loss = load_metrics(log_dir)
    plot_metrics(epochs, train_loss, eval_loss) 