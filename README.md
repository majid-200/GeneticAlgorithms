# A Practical Guide to Metaheuristic Algorithms in Python
This repository contains a collection of Python implementations for popular metaheuristic optimization algorithms, including Genetic Algorithms (GA), Particle Swarm Optimization (PSO), and Simulated Annealing (SA). These notebooks were created as a hands-on learning exercise to understand the core principles of these algorithms and their practical applications in solving complex optimization problems.

The implementations range from building the algorithms from scratch to utilizing powerful Python libraries, demonstrating their use in function optimization, creative problem-solving, and machine learning hyperparameter tuning.

## Key Concepts Covered

### 🧬 Genetic Algorithms (GA)
Inspired by Charles Darwin's theory of natural evolution, Genetic Algorithms are a search-based optimization technique. The core idea is to evolve a population of candidate solutions (individuals) toward better solutions over several generations. Key components include:
- **Population:** A set of potential solutions (chromosomes).
- **Fitness Function:** Evaluates how good a solution is.
- **Selection:** Chooses the fittest individuals to pass their genes to the next generation.
- **Crossover:** Combines the genetic information of two parents to create new offspring.
- **Mutation:** Introduces random variations into the offspring to maintain diversity and prevent premature convergence to local optima.

### 🐦 Particle Swarm Optimization (PSO)
PSO is a computational method inspired by the social behavior of bird flocking or fish schooling. The algorithm works by having a population (swarm) of candidate solutions (particles) that move around in the search space. Each particle's movement is influenced by:
- Its own best-known position (**personal best**, `pbest`).
- The entire swarm's best-known position (**global best**, `gbest`).
This collective movement allows the swarm to explore the solution space efficiently and converge towards the optimal solution.

### 🔥 Simulated Annealing (SA)
Simulated Annealing is a probabilistic technique for approximating the global optimum of a given function. The name comes from the process of annealing in metallurgy, where a material is heated and then slowly cooled to alter its physical properties. Key concepts include:
- **Temperature:** A parameter that controls the probability of accepting a worse solution. It starts high and is gradually lowered.
- **Cooling Schedule:** The rate at which the temperature decreases.
- **Acceptance Probability:** Initially, the algorithm is more likely to accept worse solutions (exploring the search space to escape local optima). As the temperature cools, it becomes more selective, only accepting better solutions (exploiting promising areas).

---

## Notebook Summaries

This repository contains the following notebooks, each exploring a different facet of metaheuristic algorithms.

### 1. Genetic Algorithm Implementations

#### `GA_two-dimensional inverted Gaussian function.ipynb`
This notebook provides a **from-scratch implementation** of a Genetic Algorithm to find the minimum of a two-dimensional inverted Gaussian function. It clearly demonstrates the core components of a GA, including:
- **Encoding:** Representing solutions as binary bitstrings.
- **Decoding:** Converting bitstrings back into numerical values.
- **Selection:** Using a tournament selection method.
- **Crossover & Mutation:** Standard single-point crossover and bit-flip mutation.

#### `GA_Camaflouge.ipynb` (Camouflage Problem)
This notebook demonstrates the evolutionary nature of GAs through a creative camouflage problem. The goal is to evolve a color or a 2D image patch to match a target. It's a great visual example of fitness-driven evolution.
- **Simulation 1:** Evolving a single RGB color value to match a target color.
- **Simulation 2:** Evolving a 16x16 pixel image patch to match a solid-color target grid.

#### `GA_Heart_Disease.ipynb`
A practical example of using a GA to optimize input parameters for a trained machine learning model. The objective is to find the combination of biking and smoking habits that minimizes the predicted risk of heart disease.
- A `RandomForestRegressor` is first trained on the dataset.
- This trained model then serves as the **fitness function** for the GA.
- The notebook contrasts a **from-scratch GA implementation** with one using the `geneticalgorithm2` library.

#### `GA_HyperParameter_Optimization.ipynb`
This notebook showcases a powerful application of GAs: automated machine learning and hyperparameter tuning using the **TPOT library**. It tackles two problems:
- **Regression Task:** Predicting the yield strength of steel by finding an optimal machine learning pipeline and its hyperparameters.
- **Classification Task:** Solving the classic Digits dataset, where TPOT searches for a high-accuracy classification pipeline.

#### `GA_Optimizing_Steel_Strength_using_Metaheuristic_algo.ipynb`
This notebook applies a GA to a materials science problem: optimizing the chemical composition of a steel alloy to maximize its yield strength.
- A Random Forest model is trained on the dataset to serve as the objective function.
- A **from-scratch GA** and the **`geneticalgorithm2` library** are used to find the optimal alloy composition.
- The results from both approaches are compared.

---

### 2. Particle Swarm Optimization Implementations

#### `Particle_Swarm_Optimization.ipynb`
A fundamental, **from-scratch implementation** of the Particle Swarm Optimization (PSO) algorithm.
- It solves a simple 3D optimization problem, aiming to find the minimum of the function `f(x, y, z) = (x-4)² + (y-5)² + (z+6)²`.
- The code is well-commented, explaining the initialization of particles and the update steps for velocity and position based on cognitive (`c1`) and social (`c2`) components.

#### `PSO_Optimizing_Steel_Strength.ipynb`
This notebook applies PSO to the same steel strength optimization problem.
- It demonstrates both a **from-scratch implementation** of PSO and a solution using the **`pyswarm` library**.
- The goal is to find the ideal alloy composition that maximizes the predicted yield strength from a pre-trained Random Forest model.

---

### 3. Simulated Annealing Implementation

#### `Simulated_Annealing.ipynb`
This notebook serves as a foundational introduction to the Simulated Annealing algorithm with a clear, **from-scratch implementation**.
- It visualizes the core concepts by solving a simple 3D function: `f(x, y, z) = (x-4)² + (y-5)² + (z+6)²`.
- It includes excellent visualizations, including a comparison of how different `cooling_rate` values affect the temperature schedule and final solution quality.
- A separate section provides a detailed 3D surface plot to visualize the algorithm's search path as it explores a more complex objective function, helping build intuition.

#### `SA_Optimizing_Steel_Strength.ipynb`
Applies Simulated Annealing (SA) to the steel strength optimization problem. This notebook provides:
- A **from-scratch implementation** of the SA algorithm, clearly showing the concepts of temperature, cooling schedule, and acceptance probability.
- A comparison using the **`differential_evolution`** optimizer from the `scipy` library, another powerful metaheuristic, to solve the same problem.

---

## How to Run
To explore these notebooks, please follow the steps below:

1.  **Clone the repository:**
    ```bash
    git https://github.com/majid-200/MetaheuristicAlgorithms.git
    cd MetaheuristicAlgorithms
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    A `requirements.txt` file is provided. You can install all dependencies with:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch Jupyter:**
    ```bash
    jupyter notebook
    ```
    Or, if you prefer JupyterLab:
    ```bash
    jupyter lab
    ```

5.  Navigate to the cloned directory in the Jupyter interface and open any of the `.ipynb` files to run the code.

## Requirements
The main libraries used in this project are listed in `requirements.txt`. Key dependencies include:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `geneticalgorithm2`
- `pyswarm`
- `tpot`

## Acknowledgements
This project was inspired by and developed following a YouTube playlist (https://youtube.com/playlist?list=PLZsOBAyNTZwZprhl9wyoEn6Hkv9JT9EZD&feature=shared) on metaheuristic algorithms. It serves as a practical application of the concepts taught in the series.
