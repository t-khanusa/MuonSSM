# Assets Directory

This directory should contain the following images for the README:

## Required Images

### 1. muonssm_overview.png
**Description**: A high-level architecture diagram showing:
- The MuonSSM pipeline with three evaluation tasks (Language, Vision, Time-Series)
- The core innovation: Momentum + Newton-Schulz orthogonalization
- Comparison between standard SSM and MuonSSM recurrence

**Suggested layout**:
```
┌──────────────────────────────────────────────────────────────────┐
│                        MuonSSM Overview                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│   │  Language   │    │   Vision    │    │ Time-Series │          │
│   │  (FineWeb)  │    │ (ImageNet)  │    │   (HAR)     │          │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘          │
│          │                  │                  │                 │
│          └─────────────────-┼──────────────────┘                 │
│                             ▼                                    │
│   ┌──────────────────────────────────────────────────────┐       │
│   │                    MuonSSM Core                      │       │
│   │  ┌────────────┐  ┌────────────────────────┐          │       │
│   │  │  Momentum  │──│ Newton-Schulz Ortho.   │          │       │
│   │  │  (β, α)    │  │      (ns_steps)        │          │       │
│   │  └────────────┘  └────────────────────────┘          │       │
│   └──────────────────────────────────────────────────────┘       │
│                                                                  │
│   Standard SSM:  h_t = exp(δA)·h_{t-1} + δ·B·x                   │
│   MuonSSM:       v_t = β·v_{t-1} + α·NS(δ·B·x)                   |
|                  h_t = exp(δA)·h_{t-1} + v_t                     │
│                                                                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 2. results_comparison.png (optional)
**Description**: Performance comparison charts for:
- Language: Perplexity curves
- Vision: Top-1/Top-5 accuracy
- Time-Series: F1 scores

### How to Create

You can create these images using:
1. **draw.io** - Free online diagram tool
2. **Excalidraw** - Whiteboard-style diagrams
3. **PowerPoint/Keynote** - Presentation software
4. **Matplotlib/Seaborn** - For result plots in Python

Export as PNG with dimensions around 1200x600 pixels for optimal display.
