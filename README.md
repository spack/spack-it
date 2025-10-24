# spack-it

spack-it is an experimental framework that uses LLMs to automatically generate Spack package recipes from source repositories. It explores how retrieval, metadata analysis, and iterative refinement can assist in packaging scientific software. The current version supports repositories that use CMake as their build system and includes tools for repository inspection, context retrieval, recipe synthesis, and automated validation through Spack’s build pipeline. These artifacts accompany the "LLMs as Packagers of HPC Software" paper and contain the data, evaluation scripts, and generated outputs used in the study.

![Flow Chart](img/chart.png)
*Overview of the spack-it workflow. The agent coordinates tools for metadata extraction, retrieval, recipe generation, evaluation, and iterative repair. Retrieved examples are combined with repository metadata to guide generation, and failed builds trigger a repair loop that refines the recipe using feedback from Spack outputs.*

## Run Analysis Scripts

All analysis scripts are located in the `analysis/` directory. These scripts reproduce the figures and tables presented in the paper using the processed data stored in `results/`. Each script is self-contained and can be run independently to generate plots or summary statistics for a specific experiment (e.g., ablations, similarity metrics, etc).

## Running spack-it

There are a few prerequisites before using the end-to-end system.

### Python Requirements

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install -r rag_requirements.txt
```

### Spack
spack-it requires a current installation of [Spack](https://spack.io) in order to extract metadata from packages and perform other tasks. Because of this, all Python scripts must be executed with `spack python`.

### Extract Spack Recipe Corpus

We use existing Spack packages as reference data in spack-it. Running `spack python extraction/dump.py` will create the appropriate files in `data/`. Other scripts in this repository are set up to use the default location, but this can be configured to your needs.

### Neo4j
To get the E4S knowledge graph up and running, build the container image in the `neo4j/` folder with Docker, and run `spack python extraction/graph.py` to put the Spack package metadata into the database.

### Retrieval Augmented Generation (RAG)
To run the RAG component of spack-it, you will need to index the chunks of Spack packages that you wish to retrieve from. See `rag/index.py` for more information. You may need a GPU system to run this in a computationally efficient manner.

### Build Container
spack-it uses an isolated build container for each evaluation step of generated recipes. Build the container in `container/` and it will be automatically used when you run spack-it.

```bash
podman build -f container/Dockerfile -t builder .
```

### Environment

```
# your system ppython
export SPACK_PYTHON="./env/bin/python"
export PYTHONPATH=$PYTHONPATH:.
export NEO4J_URI="bolt://user:pass@localhost:7687"
export TEMPLATE_DIR="generate/templates"

# set to an openai LLM endpoint and key if necessary
export LLM_API_URL=""
export LLM_API_KEY=""
```

### Execution

To get started running the end-to-end system, you can run `spack python generate/generate.py`. It will output options available to you to run the spack-it workflow.

Please read through the script, or the spack-it paper to fully understand the options before you get started.

## Support

Please open an issue for any questions or comments.

## License

This repository is distributed under the terms of the MIT license. See [LICENSE](https://github.com/spack/spack-it/blob/main/LICENSE) for details.

SPDX-License-Identifier: MIT

LLNL-CODE-2013149