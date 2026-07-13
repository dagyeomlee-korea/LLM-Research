# Qwen3.5 10개 모델 배치 분석 명세

## 목적

- `docs/qwen35_download_candidates.md`의 1차 10개 모델에 대해 텐서별 통계를 자동 산출한다.
- 중단 후 재실행해도 완료 모델은 건너뛰고 실패 모델만 다시 실행할 수 있게 한다.
- 완료 결과로 모델별 통계, 계열별 비교, 용도 기반 분류와 핵심 시사점을 작성한다.

## 입력

- JSON manifest: `repo_id`, `family`, `classification`, `path`, `expected_tensors`, `output_name`
- NAS의 단일 파일 BF16 safetensors 10개
- 분석 엔진: `numpy` 또는 기존 `python` streaming 엔진

## 처리

1. manifest의 경로·중복·예상 텐서 수를 검증한다.
2. 모델을 한 개씩 순차 분석해 NAS 메모리 사용을 제한한다.
3. 각 모델은 임시 JSONL에 기록하고 성공했을 때만 최종 JSONL로 교체한다.
4. 완료 marker에는 종료 코드, 결과 행 수, 시작·종료 시각을 기록한다.
5. 재실행 시 marker와 예상 행 수가 일치하는 모델은 건너뛴다.
6. 전체 완료 후 module type과 kurtosis 중심으로 모델별·계열별 통계를 집계한다.

## 성능 정책

- `numpy` 엔진은 safetensors를 mmap으로 열고 제한된 byte chunk만 배열로 변환한다.
- BF16은 uint16 상위 비트를 FP32 bit pattern으로 확장한 뒤 float64 누적에 사용한다.
- NumPy가 없으면 기존 Python 엔진을 사용할 수 있지만, 10개 배치에서는 NumPy를 필수로 검증한다.
- NAS가 2코어·3.7GiB RAM이므로 병렬 모델 분석은 사용하지 않는다.

## 출력

- 모델별 `{output_name}.jsonl`
- 모델별 `{output_name}.log`
- 모델별 `{output_name}.done.json`
- 배치 상태 `batch-summary.json`
- 최종 분석 보고: 결과값, 핵심 시사점, family/classification별 비교, 해석 한계

## 모델 분류

- `official_instruct`: 공식 instruction anchor
- `official_base`: 공식 base anchor
- `mirror_or_finetune_candidate`: 동일 형상 mirror/fine-tune 후보
- `domain_finetune`: 도메인 특화 fine-tune 후보
- `abliterated`: abliterated claim 검증 후보

분류는 repo 정보에 따른 작업 가설이며, 통계만으로 학습 계보를 확정하지 않는다.

## 완료 기준

- 10개 모델 모두 exit code 0이다.
- 각 JSONL 행 수가 safetensors 예상 텐서 수와 일치한다.
- 모든 행이 JSON으로 파싱되고 `repo_id`, `tensor_name`, `module_type`, `kurtosis` 필드를 가진다.
- 오류 metadata가 있는 행 수를 별도로 보고한다.
- 원본 safetensors는 변경하지 않는다.
