# ComfyUI-VideoPrompt-LLM

`MiniMax H3 Video LLM Loader`는 레퍼런스 영상의 시간순 프레임을 로컬 Vision LLM으로 분석해
MiniMax H3 공식 **기본(base) 영상 프롬프트** 형식의 영어 `STRING`을 출력하는 독립 ComfyUI
커스텀 노드입니다. 단순 캡셔닝보다 동작·연기·카메라·컷·시간적 연속성 보존에 초점을 둡니다.

> 이 저장소의 코드는 구현과 정적 검토까지만 완료된 상태입니다. 실제 ComfyUI 실행, Ollama/LM
> Studio 추론, 영상 분석 품질, GPU 메모리 및 MiniMax H3 생성은 사용자 환경에서 확인해야 합니다.

## 주요 기능

- ComfyUI 공식 `VIDEO` 객체의 `get_stream_source()`/`get_components()` 경로 지원
- 시간순 프레임, FPS, 예상 길이 및 프레임별 타임스탬프 보존
- Uniform / Every N Frames / Every N Seconds 샘플링
- 최대 분석 프레임 수와 분석용 긴 변 해상도 제한
- 큰 프레임 묶음을 12장 단위로 관찰한 뒤 시간순으로 통합
- Ollama 및 LM Studio(OpenAI 호환 로컬 서버) 멀티이미지 Vision 호출
- 모델 목록 새로고침, 실행 중단 요청, 수동 언로드
- Thinking On/Off, Seed/Seed Mode, 실행 후 모델 유지/시간 유지/언로드
- 선택적 ComfyUI 모델 언로드 후 LLM 실행
- Custom System Prompt와 자연어 Prompt 역할 분리
- Thinking/Result 스크롤 패널 및 최종 `result (STRING)` 출력
- localhost 및 사설/LAN IP만 허용하는 서버 URL 안전 정책

## 설치

ComfyUI의 `custom_nodes` 폴더에서:

```bash
git clone https://github.com/djwanho-sys/ComfyUI-VideoPrompt-LLM.git
cd ComfyUI-VideoPrompt-LLM
pip install -r requirements.txt
```

ComfyUI를 다시 시작한 뒤 `DJwanho / LLM` 카테고리에서
`MiniMax H3 Video LLM Loader`를 추가합니다. 의존성 설치와 ComfyUI 재시작은 사용자가 직접 수행해야 합니다.

## Ollama 연결

1. Provider를 `Ollama`로 선택합니다.
2. 노드가 기본 서버 `http://127.0.0.1:11434`를 내부에서 자동 사용합니다. 주소 입력란은 없습니다.
3. 노드를 추가하면 설치된 모델이 `Ollama Model` 드롭다운에 자동 표시됩니다. 필요하면 `Refresh Models`로 다시 조회합니다.
4. 반드시 **멀티이미지 Vision 입력을 지원하는 모델**을 선택합니다.

노드는 `/api/tags`, `/api/chat`, `/api/generate`를 사용합니다. 특정 모델을 하드코딩하지 않으며,
목록에 나타난다는 사실만으로 Vision 지원이 보장되지는 않습니다.

## LM Studio 연결

Provider를 `LM Studio`로 바꾸면 기본 주소 `http://127.0.0.1:1234`를 내부에서 자동 사용합니다. 모델 조회에는
`/v1/models`, 생성에는 `/v1/chat/completions`를 사용합니다. Thinking 제어 필드를 서버가 거부하면
한 번 제거해 재시도합니다. LM Studio 버전에 따라 `/api/v1/models/unload` 지원 여부가 다를 수 있습니다.

## VIDEO 입력과 IMAGE 배치 호환

- `video`: 핵심 입력입니다. 최신 ComfyUI `VIDEO` 객체에서 메타데이터와 프레임을 읽습니다.
- `image_frames`: 영상에서 이미 추출된 시간순 `IMAGE` 배치를 위한 보조 입력입니다. 인물 외형
  레퍼런스나 피사체 교체 입력이 아닙니다.
- 둘 다 연결되면 `video`가 우선합니다.
- `fps = 0`: VIDEO 메타데이터 FPS를 사용합니다. IMAGE 배치에는 24 FPS를 적용합니다.
- `fps > 0`: 메타데이터 대신 지정 FPS를 사용해 타임스탬프를 계산합니다.

파일 기반 VIDEO는 가능한 경우 PyAV 스트림에서 선택 프레임만 보존해 메모리 사용량을 줄입니다.
그 경로를 사용할 수 없는 VIDEO 구현은 `get_components()`의 프레임 배치로 폴백합니다.

## 프레임 샘플링

- `Uniform`: 전체 길이에서 `max_analysis_frames`만큼 균등 선택합니다.
- `Every N Frames`: `sampling_interval` 프레임마다 선택합니다.
- `Every N Seconds`: `sampling_interval`초마다 선택합니다.
- `analysis_resolution`: 종횡비를 유지한 채 분석 이미지의 긴 변 최대 크기를 제한합니다.

샘플 수가 많으면 12장씩 관찰 결과를 만든 뒤 최종 요청에서 통합합니다. 프레임 사이에 직접 보이지
않는 움직임은 확인된 사실로 단정하지 않도록 기본 지침에 명시되어 있습니다.

## 기본 MiniMax H3 프롬프트

Custom System Prompt가 비어 있으면 `prompts/minimax_h3_default.txt`가 자동 적용됩니다. 출력 순서는
공식 기본 가이드의 다음 세 필드를 유지합니다.

따라서 ComfyUI 기본 `Load Video` 노드의 `VIDEO` 출력을 연결하고 Vision 모델만 선택하면,
`System Prompt / Custom`과 `Prompt`가 모두 비어 있어도 기본 MiniMax H3 영상 프롬프트를 자동 생성합니다.

```text
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

첫 Shot에는 시간이 없고, 실제 컷이 관찰된 뒤의 Shot에만 컷 시간을 씁니다. 영상 프레임만 분석하므로
오디오·대사·음악을 만들지 않고, 두 오디오 필드에는 오디오 증거가 없음을 명시합니다. 분석 입력을
생성 모델에 연결된 `<Video 1>`로 선언하지 않으며 Full-Reference 6섹션 형식을 자동 사용하지 않습니다.

## Custom System Prompt 및 자연어 Prompt

- Custom System Prompt 비어 있음: 번들 MiniMax H3 기본 지침 사용
- Custom System Prompt 입력됨: 번들 지침을 **대체**하고 사용자 지침만 system 역할로 전달
- 자연어 Prompt만 입력됨: 기본 system 지침 + 사용자 요청
- 둘 다 비어 있음: 기본 MiniMax H3 프롬프트 자동 생성

Custom System Prompt는 다른 영상 모델 형식에도 사용할 수 있습니다. 이 경우 MiniMax H3 세 필드 검사는
적용하지 않습니다.

## 출력 연결

기본 출력은 `result (STRING)` 하나입니다. MiniMax H3 생성 노드의 텍스트/프롬프트 입력에 직접
연결할 수 있습니다. Thinking은 UI 미리보기 전용이며 STRING 출력에 섞이지 않습니다.

## 모델 및 메모리 설정

- `Thinking`: 제공자가 실제로 반환한 reasoning/thinking만 표시합니다.
- `Seed Mode`: Fixed, Increment, Decrement, Randomize
- `Model After Run`: Keep Loaded, Keep for N Minutes, Unload
- `Unload ComfyUI Models Setting`: `Unload Before LLM`을 선택한 경우에만 실행 전 ComfyUI 모델을 언로드합니다.
- `Stop LLM`: 현재 스트리밍 HTTP 응답을 닫아 중단을 요청합니다. 제공자 상태에 따라 즉시 중단되지 않을 수 있습니다.

## 알려진 제약 사항

- 프레임 기반 분석이며 원본 오디오를 읽거나 전송하지 않습니다.
- 샘플링 간격보다 빠른 동작, 미세한 눈동자/입술 변화는 누락될 수 있습니다.
- 모델별 최대 이미지 수와 컨텍스트 길이가 다릅니다. 필요하면 최대 프레임 수와 해상도를 낮추십시오.
- 모델 목록 조회는 Vision 기능 자체를 검증하지 않습니다.
- 네트워크 호스트명은 허용하지 않으며 localhost 또는 리터럴 사설 IP만 허용합니다.
- LM Studio의 reasoning 및 unload API는 버전/모델에 따라 다릅니다.
- 실제 추론 및 ComfyUI 호환성 테스트는 이 개발 작업 범위에 포함되지 않았습니다.

## 참고 및 라이선스

- MiniMax H3 공식 [Base Prompt Guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
- MiniMax H3 공식 [Full-Reference Guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
- UI 및 로컬 LLM 동작 참고: [Deno2026/comfyui-deno-custom-nodes](https://github.com/Deno2026/comfyui-deno-custom-nodes)

원본 Deno 저장소의 구현 코드를 복사하지 않고 독립적으로 작성했습니다. 자세한 고지는 `NOTICE.md`,
본 프로젝트 라이선스는 `LICENSE`를 확인하십시오.
