# Mobile AI Workshop Super Resolution
## Prequisites
- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Docker](https://www.docker.com/) and Docker Compose
- Git

## Setup
### 1. Create Conda Environment

Create conda environment:

```bash
conda env create -f environment.yml
conda activate mai-upres
uv sync --all-groups
```

### 2. Install dependencies

```bash
uv add --group [name] <dep>
```
