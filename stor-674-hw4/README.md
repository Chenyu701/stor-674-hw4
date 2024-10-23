# taxi-example

## Get data
```
bash get_taxi.sh
```

## Explore data
Using the jupyter stacks project datascience notebook. Directions [here](https://jupyter-docker-stacks.readthedocs.io/en/latest/#example-2).
```
docker run -it --rm -p 10000:8888 -v "${PWD}":/home/jovyan/work quay.io/jupyter/datascience-notebook:2024-08-30
```
