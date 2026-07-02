# Spark vLLM 컨테이너 실행 가이드

## vLLM 이미지 확인

최신 Spark용 vLLM 이미지는 NVIDIA NGC에서 확인합니다.

- [vLLM | NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm)

```bash
docker pull nvcr.io/nvidia/vllm:26.03-py3
```

## 컨테이너 실행 방식

아래의 **컨테이너 생성 방법**으로 한 번 생성해두면, 이후에는 `docker start`로 다시 실행할 수 있습니다.

모델 가중치는 `~/ai_models`에 저장되므로 컨테이너 자체 용량은 크지 않습니다. 따라서 사용이 끝날 때마다 컨테이너를 삭제할 필요는 없습니다.

```bash
docker start gpt-oss-120b
```

## 공통 옵션

| 옵션 | 설명 |
| --- | --- |
| `--gpus all` | 모든 GPU를 컨테이너에 할당 |
| `--ipc=host` | vLLM 실행에 필요한 shared memory 설정 |
| `--ulimit memlock=-1` | 메모리 lock 제한 해제 |
| `--ulimit stack=67108864` | stack size 설정 |
| `-p 8000:8000` | 호스트 8000 포트를 컨테이너 8000 포트에 매핑 |
| `-v ~/ai_models:/root/.cache/huggingface` | 모델 캐시 디렉터리 마운트 |
| `--gpu-memory-utilization` | GPU 메모리 사용 비율 |
| `--max-model-len` | 최대 컨텍스트 길이 |

---

## GPT-OSS 120B (NVFP4)

```bash
docker run -d --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 \
  -v ~/ai_models:/root/.cache/huggingface \
  --name gpt-oss-120b \
  nvcr.io/nvidia/vllm:26.03-py3 \
  vllm serve openai/gpt-oss-120b \
    --served-model-name openai/gpt-oss-120b \
    --dtype auto \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-model-len 65536 \
    --host 0.0.0.0 \
    --port 8000
```

실행:

```bash
docker start gpt-oss-120b
```

---

## Gemma 4 31B (NVFP4)

```bash
docker run -d --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 \
  -v ~/ai_models:/root/.cache/huggingface \
  --name gemma4-31b-nvfp4 \
  vllm/vllm-openai:gemma4-cu130 \
  nvidia/Gemma-4-31B-IT-NVFP4 \
    --served-model-name nvidia/Gemma-4-31B-IT-NVFP4 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 65536
```

실행:

```bash
docker start gemma4-31b-nvfp4
```

---

## Gemma 4 26B-A4B-it (BF16)

```bash
docker run -d --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 \
  -v ~/ai_models:/root/.cache/huggingface \
  --name gemma4-26b-a4b \
  vllm/vllm-openai:gemma4-cu130 \
  google/gemma-4-26B-A4B-it \
    --served-model-name google/gemma-4-26B-A4B-it \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 65536
```

실행:

```bash
docker start gemma4-26b-a4b
```

---

## 상태 확인

컨테이너 목록 확인:

```bash
docker ps -a
```

실행 로그 확인:

```bash
docker logs -f gpt-oss-120b
```

컨테이너 중지:

```bash
docker stop gpt-oss-120b
```

컨테이너 삭제:

```bash
docker rm gpt-oss-120b
```

> 컨테이너를 삭제해도 `~/ai_models`에 저장된 모델 가중치는 삭제되지 않습니다.
```
