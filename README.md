# DL Template for paper

## MAC
```
docker build -f docker/Dockerfile.cpu -t dl-dev .
uv pip sync requirements.txt
```

## Linux
```
docker build -f docker/Dockerfile.gpu -t dl-prod .
docker run --gpus all -v $(pwd):/workspace -w /workspace -it dl-prod
uv pip sync requirements.txt
```

## [Memo]Docker Initialize
```
docker builder prune
docker buildx prune --all
```