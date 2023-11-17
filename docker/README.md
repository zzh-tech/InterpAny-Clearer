# Run with Docker Container

## Build Docker image
Make sure docker >= 19.03 and nvidia-container-toolkit >= 1.3 are installed.
```shell
docker build -t interpany:v0 -f docker/Dockerfile .
```

## Run Docker container

### RUN the container

```shell
docker run -it --name=interp -p 5001:5001 -p 8080:8080 --gpus all --shm-size=8g interpany:v0 
```
This command will build a container with the name `interp`, which serves a webapp on http://localhost:8080/ (only accessible from the local machine).

```shell
docker run -it --name=interp -p 5001:5001 -p 8080:8080 --gpus all --shm-size=8g interpany:v0 /bin/bash
``` 
With '/bin/bash/' augmentation, only the container with all the dependencies is built, and the project code is located at "/InterpAny-Clearer" in the container.
