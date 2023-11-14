# Preparation with Docker

## Build Docker image

```shell
cd docker
docker build -t InterpAny .
```

## Run Docker container

### RUN the container

```shell
docker run -it --name=interp -p 5001:5001 -p 8080:8080 --gpus all --shm-size=8g InterpAny
```
After this command finishes, the webapp demo will be available at http://localhost:8080


```shell
docker run -it --name=interp -p 5001:5001 -p 8080:8080 --gpus all --shm-size=8g InterpAny /bin/bash
``` 
This will create a container that all environment is set up, and you can play with the code in the container.
