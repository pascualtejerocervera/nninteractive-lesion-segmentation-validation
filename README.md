# Unit of Knowledge - Template

Please replace this readme with information about you experiment. Include:

* The project's name
* TRAC ticket number
* Point to the jupyter notebook that will be your entrypoint for your unit of knowledge
* Author name
* **Instructions on building your Dockerfile:**
    * Include external paths to external experiment data directories that need to be added to the repository before building.
    * Describe if extra, special environment variables/arguments need to be defined in order to build your container.

## Converting the template into a Unit of Knowledge

1. Clone this repository and use it to create a new Unit of Knowledge repository on [https://github.com/DIAGNijmegen](https://github.com/DIAGNijmegen). For details see [SOP 7335](https://repos.diagnijmegen.nl/trac/ticket/7335).

2. Replace this README.md with a short description of your experiment (see above).

3. Add your experiment code to the new repository.

    - Modify `entrypoint.ipynb` and add a technical description of your experiment. Explain:
        - How to execute your code
        - The most important data structures of your experiment
        - How to apply your experiment to a dataset (if applicable)
        - See [SOP 7330](https://repos.diagnijmegen.nl/trac/ticket/7330)
    - Add the name of your jupyter notebook to the README.md, designate its name clearly.

4. Write your `prepare_data`-script according to [SOP 7334](https://repos.diagnijmegen.nl/trac/ticket/7334).

    - You can use the templates `prepare_data.py/sh.template`. Make sure both template files are deleted after you wrote your own `prepare_data`-script.
    - Verify that your `prepare_data`-script works by manually invoking it using the `test-prepare-data.py`-script that is available on the deep learning cluster. Details are described in the SOP.

5. Rename `Dockerfile.template` to `Dockerfile` and fill in the correct build rules. Test if your docker builds and commit it to the DIAG regsitry. See [SOP 7351](https://repos.diagnijmegen.nl/trac/ticket/7351).

6. Ask for validation of your Unit of Knowledge.

