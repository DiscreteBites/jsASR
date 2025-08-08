import pandas as pd

def print_histories( 
    val_key: str, best_val: float, best_epoch: int, 
    all_histories: list[pd.DataFrame]
):
    # Combine all chunks
    full_history = pd.concat(all_histories, ignore_index=True)

    # Print final values
    print("\nFinal epoch metrics:")
    print(full_history.tail(1).T)

    # Print best values for each metric
    print("\nBest values for each metric:")
    best_vals = {
        col: full_history[col].max() if "acc" in col else full_history[col].min()
        for col in full_history.columns
    }
    for k, v in best_vals.items():
        print(f"{k}: {v:.4f}")

    print(f"\nBest {val_key}: {best_val:.4f} at epoch {best_epoch}")

    return