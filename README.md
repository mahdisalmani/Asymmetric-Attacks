# 📄 Official Implementation

**Official implementation of the paper:**  
**_Rewriting the Budget: A General Framework for Black-Box Attacks Under Cost Asymmetry_**

This repository provides the codebase to reproduce experiments presented in the paper, including support for several state-of-the-art black-box adversarial attacks under asymmetric cost settings.

---

## 🔧 Running Attacks

You can run different attacks by specifying the desired configuration through `main.py`. The key parameters include the attack name, total query budget, cost settings, and device selection.

### ✅ Basic Usage

`python main.py --attack <ATTACK_NAME> --device <DEVICE> --total-cost <COST> [OPTIONS...]`

---

## 🚀 Supported Attacks & Examples

### 🔹 SURFREE

`python main.py --attack SURFREE --device "cuda:0" --total-cost 10000 --dimension-reduction-mode "Full" --dimension-reduction-factor 4.0 --search-cost 1 --query-cost 2`

### 🔹 OPT

`python main.py --attack OPT --device "cuda:0" --total-cost 15000 --search-cost 1 --query-cost 5`

### 🔹 HSJA

`python main.py --attack HSJA --device "cuda:0" --total-cost 10000 --initial-gradient-queries 30 --search-cost 1 --query-cost 2`

### 🔹 GeoDA

`python main.py --attack GEODA --device "cuda:0" --total-cost 150000 --initial-gradient-queries 30 --dimension-reduction-mode "Full" --dimension-reduction-factor 4.0 --search-cost 1 --query-cost 100`

### 🔹 CGBA

`python main.py --attack CGBA --device "cuda:0" --total-cost 10000 --initial-gradient-queries 30 --dimension-reduction-mode "Full" --dimension-reduction-factor 4.0 --search-cost 1 --query-cost 2`

---

## ⚙️ Key Arguments

- `--attack`: Attack name (`SURFREE`, `OPT`, `HSJA`, `GEODA`, `CGBA`)
- `--device`: Device to run on (e.g., `cuda:0`, `cpu`)
- `--total-cost`: Total allowed query budget
- `--search-cost`: Cost of a single query used during **Asymmetric Search**
- `--query-cost`: Cost of the final attack query; defines **asymmetry ratio** \( c^\star \)
- `--initial-gradient-queries`: Number of initial queries for gradient estimation (used in gradient-based attacks)
- `--dimension-reduction-mode`, `--dimension-reduction-factor`: Options to reduce the attack dimension (e.g., `"Full"`, `4.0`)

