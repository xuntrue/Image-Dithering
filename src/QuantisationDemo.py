import textwrap
import matplotlib.pyplot as plt
import numpy as np

def quantise(values: np.ndarray, levels: int) -> tuple[np.ndarray, np.ndarray]:
    step = 2.0 / (levels - 1)
    indices = np.clip(np.round((values + 1) / step), 0, levels - 1).astype(int)
    return indices * step - 1, indices

def show_quantisation(levels: int = 8, num_samples: int = 32, cycles: float = 2.0) -> None:
    """ Three plots: continuous input, quantised overlay, stored samples """
    if levels < 2:
        raise ValueError("levels must be at least 2")

    # A dense axis stands in for "continuous"
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * cycles * t)
    quantised, _ = quantise(signal, levels)

    # What is actually stored: sample the wave, then keep only the level index
    t_samples = np.linspace(0, 1, num_samples, endpoint=False)
    _, stored = quantise(np.sin(2 * np.pi * cycles * t_samples), levels)
    bits = max(1, int(np.ceil(np.log2(levels))))
    level_values = np.linspace(-1, 1, levels)

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    # (1) Continuous input
    ax = axes[0]
    ax.plot(t, signal, color="tab:blue", linewidth=2)
    #ax.set_title("1. Continuous input (infinitely many possible values)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.25, 1.25)

    # (2) Quantised wave (red) over the original (blue)
    ax = axes[1]
    ax.plot(t, signal, color="tab:blue", linewidth=2, alpha=1.0, label="Original")
    ax.plot(t, quantised, color="red", linewidth=2, alpha=0.6, label=f"Quantised ({levels} levels)")
    if levels <= 16:
        for i, lv in enumerate(level_values):
            ax.axhline(lv, color="gray", linestyle=":", linewidth=0.8,
                       label="Allowed levels" if i == 0 else None)
    #ax.set_title("2. Every value is snapped to the nearest allowed level")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.25, 1.25)
    ax.legend(loc="upper right", fontsize=8)
    
    """
    # (3) What is stored in memory: integers, not amplitudes
    ax = axes[2]
    ax.plot(t_samples, stored, "ro")                                      # Scatter plot of stored integers
    #ax.stem(t_samples, stored, linefmt="r-", markerfmt="ro", basefmt=" ") # Stem plot to show the discrete nature of stored values
    #ax.set_title(f"3. Stored in memory: {num_samples} samples x {bits} bit(s) = {num_samples * bits} bits")
    ax.set_ylabel("Stored level (integer)")
    ax.set_xlabel("Time")
    ax.set_ylim(-0.5, levels - 0.5)
    if levels <= 16:
        ax.set_yticks(range(levels))
    """

    # (3) Quantisation error across the whole wave
    ax = axes[2]
    step = 2.0 / (levels - 1)
    error = abs(signal - quantised)  # error = |value - quantised(value)|
    rms = np.sqrt(np.mean(error ** 2))
    ax.plot(t, error, color="purple", linewidth=1.5)
    ax.fill_between(t, error, color="purple", alpha=0.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(step / 2, color="gray", linestyle="--", linewidth=1,
               label=f"+/- step/2 = {step / 2:.3f}")
    ax.axhline(-step / 2, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(0, step * 0.75)
    ax.set_title(f"4. Quantisation error (original - quantised), RMS = {rms:.3f}")
    ax.set_ylabel("Absolute Error")
    ax.set_xlabel("Time")
    ax.legend(loc="upper right", fontsize=8)

    if True:  # toggle major + minor gridlines on/off
        for ax in axes:
            ax.set_axisbelow(True)    # draw gridlines behind the data
            ax.minorticks_on()        # minor gridlines need minor ticks to exist
            ax.grid(which="major", linestyle="-", alpha=0.5)
            ax.grid(which="minor", linestyle=":", alpha=0.3)

    #wrapped = textwrap.fill(f"Stored integers: {stored.tolist()}", width=110)
    #fig.text(0.5, 0.01, wrapped, ha="center", va="bottom", fontsize=9)
    #plt.tight_layout(rect=(0, 0.02 + 0.02 * (wrapped.count("\n") + 1), 1, 1))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    bits = 2
    levels = 2**bits + 1
    show_quantisation(levels=levels, num_samples=16+1)