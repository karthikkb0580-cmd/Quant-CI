# QuantCI System Architecture

## Architecture Overview

QuantCI follows a modular, local-first research architecture.

```mermaid
flowchart TB
    subgraph Data Layer
        A[LRTS Dataset]
        B[Test Execution Records]
        C[Commit Metadata]
    end

    subgraph Processing Layer
        D[Dataset Validation]
        E[Normalization]
        F[Temporal Splitting]
        G[Feature Engineering]
    end

    subgraph Intelligence Layer
        H[Logistic Regression]
        I[Random Forest]
        J[Risk Prediction]
    end

    subgraph Optimization Layer
        K[Candidate Selection]
        L[Greedy Solver]
        M[CP-SAT Solver]
        N[QUBO Formulation]
        O[Ising Conversion]
        P[QAOA Simulator]
        Q[Slack Repair]
    end

    subgraph Presentation Layer
        R[Evaluation Engine]
        S[Streamlit Dashboard]
        T[Research Reports]
    end

    A --> B
    A --> C
    B --> D
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    G --> I
    H --> J
    I --> J
    J --> K
    K --> L
    K --> M
    K --> N
    N --> O
    O --> P
    P --> Q
    L --> R
    M --> R
    Q --> R
    R --> S
    R --> T
```

## Design Principles

- Modular implementation
- Local-first execution
- Reproducible experimentation
- Temporal data separation
- Conservative execution constraints
- Independent solution verification
- Transparent research limitations

## Data Storage

Raw data, processed datasets, trained models and experiment artifacts are stored locally.

No external database or paid cloud service is required.

## Deployment

The application runs locally through Streamlit.

The research pipeline and dashboard operate within the same Python environment.