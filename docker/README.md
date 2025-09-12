# Docker Usage for Prithvi-EO-Segmentation

## Build the Docker Image

```sh
docker build -t prithvi-eo-segmentation -f docker/Dockerfile .
```

## Run the Container (with Jupyter Notebook)

```sh
docker run -it --rm -p 8888:8888 -v $(pwd):/app prithvi-eo-segmentation jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token=''
```

- The `-v $(pwd):/app` mounts your project directory into the container.
- Access Jupyter at [http://localhost:8888](http://localhost:8888) in your browser.

## Run the Container (Interactive Bash)

```sh
docker run -it --rm -v $(pwd):/app prithvi-eo-segmentation bash
```

---

Edit the Dockerfile as needed for additional dependencies or custom entrypoints.
