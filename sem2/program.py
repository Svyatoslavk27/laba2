import numpy as np
import matplotlib
matplotlib.use("Agg")          # без GUI — тільки збереження у файл
import matplotlib.pyplot as plt
from scipy.stats import shapiro, pearsonr
import os


# ПАРАМЕТРИ МОДЕЛЮВАННЯ

T = 1.0
DT = 0.01
K = 150                    # кількість реалізацій для оцінки середнього та дисперсії
M_VALUES_TASK1 = [1000, 3000, 10000]
M_TASK23 = 3000            # робоче значення M для завдань 2 і 3
SEED = 42

# Папка для збереження графіків 
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)   # створюємо, якщо не існує

rng = np.random.default_rng(SEED)
t_grid = np.arange(0.0, T + DT, DT)



# ДОПОМІЖНІ ФУНКЦІЇ

def shapiro_test(x: np.ndarray, max_size: int = 5000) -> tuple[float, float]:
    x = np.asarray(x).ravel()
    if x.size > max_size:
        idx = np.linspace(0, x.size - 1, max_size, dtype=int)
        x = x[idx]
    stat, p_value = shapiro(x)
    return float(stat), float(p_value)


def monte_carlo_paths(path_generator, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = np.array([path_generator() for _ in range(k)])
    mean_est = np.mean(paths, axis=0)
    var_est = np.var(paths, axis=0, ddof=1)
    return paths, mean_est, var_est


def plot_paths_mean_var(
    paths: np.ndarray,
    mean_est: np.ndarray,
    var_est: np.ndarray,
    title: str,
    filename: str,                         # ← ім'я файлу для збереження
    theoretical_var: np.ndarray | None = None,
) -> None:
    """
    Малює кілька реалізацій процесу, оцінку середнього та оцінку дисперсії.
    Зберігає графік у папку PLOTS_DIR під ім'ям filename.
    """
    plt.figure(figsize=(10, 6))

    n_show = min(5, len(paths))
    for i in range(n_show):
        plt.plot(t_grid, paths[i], linewidth=1.2, alpha=0.9)

    plt.plot(t_grid, mean_est, linewidth=2.5, label="Оцінка середнього")
    plt.plot(t_grid, var_est,  linewidth=2.5, label="Оцінка дисперсії")

    if theoretical_var is not None:
        plt.plot(t_grid, theoretical_var, "--", linewidth=2,
                 label="Теоретична дисперсія Var(W(t)) = t")

    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("Значення")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [збережено] {save_path}")


def print_header(text: str) -> None:
    print("\n" + "=" * 90)
    print(text)
    print("=" * 90)



# РОЗКЛАД W1(M, t)

def prepare_w1_basis(M: int, t: np.ndarray) -> np.ndarray:
    k = np.arange(1, M + 1, dtype=float)
    coeff = np.sqrt(2.0) / (np.pi * (k - 0.5))
    basis = np.sin(np.pi * np.outer(k - 0.5, t))
    return coeff[:, None] * basis


def simulate_w1(prepared_basis: np.ndarray, rng: np.random.Generator,
                eta: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    M = prepared_basis.shape[0]
    if eta is None:
        eta = rng.standard_normal(M)
    path = np.sum(eta[:, None] * prepared_basis, axis=0)
    return path, eta



# РОЗКЛАД W2(M, t)

def prepare_w2_basis(M: int, t: np.ndarray) -> np.ndarray:
    i = np.arange(1, M + 1, dtype=float)
    basis = np.sqrt(2.0) * np.sin(np.pi * np.outer(i, t)) / (np.pi * i)[:, None]
    return basis


def simulate_w2(prepared_basis: np.ndarray, t: np.ndarray, rng: np.random.Generator,
                eta0: float | None = None, eta: np.ndarray | None = None) -> tuple[np.ndarray, float, np.ndarray]:
    M = prepared_basis.shape[0]
    if eta0 is None:
        eta0 = float(rng.standard_normal())
    if eta is None:
        eta = rng.standard_normal(M)
    path = eta0 * t + np.sum(eta[:, None] * prepared_basis, axis=0)
    return path, eta0, eta



# РОЗКЛАД W3(M, t)

def prepare_w3_basis(M: int, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(1, M + 1, dtype=float)
    sin_part = np.sqrt(2.0) * np.sin(2.0 * np.pi * np.outer(i, t)) / (2.0 * np.pi * i)[:, None]
    cos_part = np.sqrt(2.0) * (1.0 - np.cos(2.0 * np.pi * np.outer(i, t))) / (2.0 * np.pi * i)[:, None]
    return sin_part, cos_part


def simulate_w3(prepared_basis: tuple[np.ndarray, np.ndarray], t: np.ndarray,
                rng: np.random.Generator,
                eta0: float | None = None,
                eta1: np.ndarray | None = None,
                eta2: np.ndarray | None = None) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    sin_part, cos_part = prepared_basis
    M = sin_part.shape[0]
    if eta0 is None:
        eta0 = float(rng.standard_normal())
    if eta1 is None:
        eta1 = rng.standard_normal(M)
    if eta2 is None:
        eta2 = rng.standard_normal(M)
    path = eta0 * t + np.sum(eta1[:, None] * sin_part + eta2[:, None] * cos_part, axis=0)
    return path, eta0, eta1, eta2



# ЗАВДАННЯ 1

def task1():
    print_header("ЗАВДАННЯ 1. МОДЕЛЮВАННЯ W1(M,t) ДЛЯ M = 1000, 3000, 10000")

    for M in M_VALUES_TASK1:
        print(f"\n--- M = {M} ---")

        prepared_w1 = prepare_w1_basis(M, t_grid)

        eta_sample = rng.standard_normal(M)
        stat_eta, p_eta = shapiro_test(eta_sample)
        print(f"Shapiro-Wilk для eta_k: statistic = {stat_eta:.6f}, p-value = {p_eta:.6f}")

        single_path, _ = simulate_w1(prepared_w1, rng, eta=eta_sample)
        increments = np.diff(single_path)

        stat_inc, p_inc = shapiro_test(increments)
        print(f"Shapiro-Wilk для приростів W1: statistic = {stat_inc:.6f}, p-value = {p_inc:.6f}")

        paths, mean_est, var_est = monte_carlo_paths(
            lambda: simulate_w1(prepared_w1, rng)[0],
            K
        )

        print(f"Оцінка E[W(1)] ≈ {mean_est[-1]:.6f}")
        print(f"Оцінка Var[W(0.5)] ≈ {var_est[len(t_grid)//2]:.6f}, теоретично 0.5")
        print(f"Оцінка Var[W(1)] ≈ {var_est[-1]:.6f}, теоретично 1.0")

        plot_paths_mean_var(
            paths, mean_est, var_est,
            title=f"Завдання 1: W1(M,t), M={M}",
            filename=f"task1_W1_M{M}.png",
            theoretical_var=t_grid,
        )



# ЗАВДАННЯ 2

def task2():
    print_header("ЗАВДАННЯ 2. МОДЕЛЮВАННЯ W3(M,t)")

    M = M_TASK23
    prepared_w3 = prepare_w3_basis(M, t_grid)

    eta1 = rng.standard_normal(M)
    eta2 = rng.standard_normal(M)

    stat_eta1, p_eta1 = shapiro_test(eta1)
    stat_eta2, p_eta2 = shapiro_test(eta2)
    corr, _ = pearsonr(eta1, eta2)

    print(f"M = {M}")
    print(f"Shapiro-Wilk для eta1_i: statistic = {stat_eta1:.6f}, p-value = {p_eta1:.6f}")
    print(f"Shapiro-Wilk для eta2_i: statistic = {stat_eta2:.6f}, p-value = {p_eta2:.6f}")
    print(f"Коефіцієнт кореляції corr(eta1_i, eta2_i) = {corr:.6f}")

    single_path, _, _, _ = simulate_w3(prepared_w3, t_grid, rng, eta1=eta1, eta2=eta2)
    increments = np.diff(single_path)
    stat_inc, p_inc = shapiro_test(increments)
    print(f"Shapiro-Wilk для приростів W3: statistic = {stat_inc:.6f}, p-value = {p_inc:.6f}")

    paths, mean_est, var_est = monte_carlo_paths(
        lambda: simulate_w3(prepared_w3, t_grid, rng)[0],
        K
    )

    print(f"Оцінка E[W(1)] ≈ {mean_est[-1]:.6f}")
    print(f"Оцінка Var[W(0.5)] ≈ {var_est[len(t_grid)//2]:.6f}, теоретично 0.5")
    print(f"Оцінка Var[W(1)] ≈ {var_est[-1]:.6f}, теоретично 1.0")

    plot_paths_mean_var(
        paths, mean_est, var_est,
        title=f"Завдання 2: W3(M,t), M={M}",
        filename=f"task2_W3_M{M}.png",
        theoretical_var=t_grid,
    )



# ЗАВДАННЯ 3

def task3():
    print_header("ЗАВДАННЯ 3. МОДЕЛЮВАННЯ W2(M,t)")

    M = M_TASK23
    prepared_w2 = prepare_w2_basis(M, t_grid)

    eta_from_task1 = rng.standard_normal(M)
    eta0 = float(rng.standard_normal())

    single_path, _, _ = simulate_w2(prepared_w2, t_grid, rng, eta0=eta0, eta=eta_from_task1)
    increments = np.diff(single_path)
    stat_inc, p_inc = shapiro_test(increments)

    print(f"M = {M}")
    print(f"Shapiro-Wilk для приростів W2: statistic = {stat_inc:.6f}, p-value = {p_inc:.6f}")

    paths, mean_est, var_est = monte_carlo_paths(
        lambda: simulate_w2(prepared_w2, t_grid, rng)[0],
        K
    )

    print(f"Оцінка E[W(1)] ≈ {mean_est[-1]:.6f}")
    print(f"Оцінка Var[W(0.5)] ≈ {var_est[len(t_grid)//2]:.6f}, теоретично 0.5")
    print(f"Оцінка Var[W(1)] ≈ {var_est[-1]:.6f}, теоретично 1.0")

    plot_paths_mean_var(
        paths, mean_est, var_est,
        title=f"Завдання 3: W2(M,t), M={M}",
        filename=f"task3_W2_M{M}.png",
        theoretical_var=t_grid,
    )

    return prepared_w2



# ПОРІВНЯННЯ W1, W2, W3

def compare_all_three():
    print_header("ПОРІВНЯННЯ РЕАЛІЗАЦІЙ W1, W2, W3")

    M = M_TASK23

    prepared_w1 = prepare_w1_basis(M, t_grid)
    prepared_w2 = prepare_w2_basis(M, t_grid)
    prepared_w3 = prepare_w3_basis(M, t_grid)

    path_w1, _       = simulate_w1(prepared_w1, rng)
    path_w2, _, _    = simulate_w2(prepared_w2, t_grid, rng)
    path_w3, _, _, _ = simulate_w3(prepared_w3, t_grid, rng)

    plt.figure(figsize=(10, 6))
    plt.plot(t_grid, path_w1, label="W1(M,t)")
    plt.plot(t_grid, path_w2, label="W2(M,t)")
    plt.plot(t_grid, path_w3, label="W3(M,t)")
    plt.title(f"Порівняння трьох розкладів вінерівського процесу, M={M}")
    plt.xlabel("t")
    plt.ylabel("W(t)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "compare_W1_W2_W3.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [збережено] {save_path}")



# ГОЛОВНИЙ БЛОК

if __name__ == "__main__":
    task1()
    task2()
    task3()
    compare_all_three()

    print(f"\nУсі графіки збережено у папці: {os.path.abspath(PLOTS_DIR)}/")
    print("Файли:")
    for f in sorted(os.listdir(PLOTS_DIR)):
        print(f"  • {f}")
