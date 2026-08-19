# TheRock

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit) [![Multi-arch CI](https://github.com/ROCm/TheRock/actions/workflows/multi_arch_ci.yml/badge.svg?branch=main&event=push)](https://github.com/ROCm/TheRock/actions/workflows/multi_arch_ci.yml?query=branch%3Amain) [![Ubuntu 26.04](https://img.shields.io/badge/Ubuntu-26.04%20LTS-E95420?logo=ubuntu&logoColor=white)](https://ubuntu.com) [![GCC 15](https://img.shields.io/badge/GCC-15.2-blue?logo=gnu)](https://gcc.gnu.org)

TheRock (The HIP Environment and ROCm Kit) is a lightweight open source build platform for HIP and ROCm. It is designed for ROCm contributors as well as developers, researchers, and advanced users who need access to the latest ROCm capabilities without the complexity of traditional package-based installations.

This fork provides out-of-the-box support for **Ubuntu 26.04 LTS (Resolute Raccoon)**, **GCC 15.x**, **CMake 4.x**, and next-generation AMD hardware including **AMD Strix Halo APUs (`gfx1151` / Radeon 8060S / 8050S)**.

---

## 🖥️ Verified Hardware & Testbed Environment

This fork has been completely compiled, tested, and validated on the following hardware platform:

| Component | Specification |
| :--- | :--- |
| **System / Model** | **GMKtec NucBox EVO-X2** (SKU: EVO-X2-001) |
| **APU / Processor** | **AMD Ryzen™ AI MAX+ 395** (16 Cores, 32 Threads, Strix Halo) |
| **Integrated Graphics** | **AMD Radeon™ 8060S Graphics** (40 Compute Units / 2560 SPs, RDNA 3.5, ISA: `gfx1151`) |
| **System Memory** | **128 GB LPDDR5X** Unified High-Speed Memory |
| **Operating System** | **Ubuntu 26.04 LTS (Resolute Raccoon)** |
| **Linux Kernel** | `Linux 7.0.0-29-generic` (x86_64) |
| **Host Toolchain** | GCC 15.2.0 (`gcc (Ubuntu 15.2.0-16ubuntu1) 15.2.0`) / G++ 15.2.0 |
| **Build Tools** | CMake 4.2.3, Ninja 1.12.1, Python 3.14, `uv` 0.x |

---

## 📖 쉬운 개념 이해 (비유로 배우는 빌드 방식)

ROCm 전체를 바닥부터 빌드하면 50개가 넘는 라이브러리를 만드느라 **약 5시간**이 걸립니다. 하지만 우리는 내가 원하는 작업에 꼭 필요한 것만 골라서 **30분 만에 뚝딱** 만들 수 있습니다!

```
🍕 파이썬 가상환경 (uv):
   파이썬 버전(3.14, 3.13)마다 서로 다른 재료가 섞이지 않도록 방을 깨끗하게 따로 나누는 것!

🍔 모듈형 프리셋 (Preset):
   뷔페의 50가지 음식을 다 차리느라 5시간 기다릴 필요 없이,
   "LLM 추론 세트", "비디오/미디어 세트", "기본 HIP 세트"처럼 딱 필요한 메뉴만 주문해서 30분 만에 받는 것!

🧀 토핑 추가 옵션 (--with-*):
   햄버거 세트에 치즈나 베이컨을 얹듯이,
   기본 LLM 세트에 내가 원하는 라이브러리(MIOpen, 프로파일러 등)만 플래그 하나로 쏙쏙 추가하는 것!
```

---

## 🚀 초간단 4단계 시작하기 (Step-by-Step Tutorial)

### 1단계: 필수 프로그램 설치 (처음 1회만 실행)

터미널을 열고 복사해서 붙여넣기만 하면 됩니다:

```bash
# 기본 컴파일러 및 빌드 도구 설치
sudo apt update
sudo apt install -y \
  build-essential gcc g++ gfortran git ninja-build cmake \
  pkg-config xxd automake libtool python3-dev libegl1-mesa-dev \
  libsqlite3-dev texinfo bison flex curl make ccache

# 초고속 파이썬 도구 uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# Mirage 에뮬레이터용 Rust 설치
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.95.0
source "$HOME/.cargo/env"
```

---

### 2단계: 저장소 다운로드 및 가상환경 생성

```bash
# 1. 저장소 클론 및 이동
mkdir -p ~/virtualenv/venv314 && cd ~/virtualenv/venv314
git clone https://github.com/analogbox/TheRock.git
cd TheRock

# 2. 파이썬 3.14 가상환경 생성 (uv로 1초 만에 완료!)
./therock-env setup-venv 3.14
```

---

### 3단계: 원하는 용도에 맞게 30분 만에 빌드하기!

원하는 목적에 따라 아래 명령어 중 **하나만 골라서 실행**하세요:

#### ⚡ 옵션 A: AI / LLM 모델 돌리기 (llama.cpp, vLLM, Ollama) - 추천!
> **소요 시간: 약 30 ~ 35분**  
> 행렬 연산(BLAS)과 텐서 가속기만 쏙 골라 빌드합니다.
```bash
./therock-env build --preset llm --python 3.14
```

#### ⚡ 옵션 B: 최소형 HIP 런타임 (가장 빠른 빌드)
> **소요 시간: 약 20 ~ 25분**  
> Clang 23 컴파일러와 기본 HIP 드라이버만 빠르게 만듭니다.
```bash
./therock-env build --preset hip --python 3.14
```

#### ⚡ 옵션 C: 영상/비디오 및 Vulkan 미디어 가속 (rocDecode + rocJPEG)
> **소요 시간: 약 25 ~ 35분**  
> 하드웨어 비디오 디코딩 및 그래픽스 가속기를 만듭니다.
```bash
./therock-env build --preset vulkan --python 3.14
```

#### ⚡ 옵션 D: 수학 & 과학 계산 (FFT, 행렬 분해, SOLVER)
> **소요 시간: 약 1.5 ~ 2시간**
```bash
./therock-env build --preset math --python 3.14
```

#### ⚡ 옵션 E: PyTorch AI 학습 풀 스택 (MIOpen + RCCL 포함)
> **소요 시간: 약 3.5 ~ 4.5시간**
```bash
./therock-env build --preset ai --python 3.14
```

---

### 4단계: 빌드된 환경 켜기 (Activate) 및 GPU 작동 확인

빌드가 끝나면 터미널에 나온 `source ...` 명령어를 실행하면 끝납니다!

```bash
# 1. 빌드된 LLM 환경 켜기 (파이썬 가상환경 + ROCm 경로가 한 번에 켜집니다)
source build_py314_llm_inference/activate_env.sh

# 2. 내 GPU가 정상 작동하는지 확인!
rocminfo
```

출력 결과에 내 그래픽 카드(**AMD Radeon 8060S / gfx1151**)가 나오면 완벽하게 성공입니다! 🎉

```
=====================    
HSA Agents               
=====================    
Agent 1: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S (CPU)
Agent 2: gfx1151 / AMD Radeon 8060S Graphics (GPU, 40 Compute Units)
```

---

## 🍕 프리셋 전체 메뉴판 (Preset List)

| 프리셋 이름 | 별칭 (단축어) | 포함된 라이브러리 | 추천 용도 | 예상 소요 시간 |
| :--- | :--- | :--- | :--- | :--- |
| **`hip`** | `hip` | AMD Clang + HIP Runtime + AMDSMI + `rocminfo` | 초경량 HIP C++ 개발 | **~20분** |
| **`llm-inference`** | **`llm`** | HIP + `rocBLAS` + `hipBLASLt` + `rocPRIM` + `hipTensor` | **vLLM, llama.cpp, Ollama, ExLlamaV2** | **~30분** |
| **`cv-vision`** | `vision`, `cv` | HIP + `RPP` + `rocDecode` + `rocJPEG` + AMD Mesa | OpenCV, 실시간 영상/비전 AI 전처리 | **~30분** |
| **`vulkan`** | `vulkan` | HIP + AMD Mesa (RADV Vulkan) + `rocDecode` + `rocJPEG` | Vulkan 그래픽스 & 비디오 가속 | **~30분** |
| **`math`** | `math` | HIP + `rocBLAS` + `rocFFT` + `rocRAND` + `rocSOLVER` | FFT 신호처리 및 수학 계산 | ~1.5시간 |
| **`hpc`** | `hpc` | HIP + Math 전체 + `rocALUTION` + `rocSPARSE` | 공학/물리 시뮬레이션 | ~1.5시간 |
| **`ai`** | `ai` | HIP + Math + `MIOpen` (CK) + `RCCL` + `hipDNN` | PyTorch / JAX 전체 학습 환경 | ~4시간 |
| **`profiler`** | `profiler` | `rocprofiler-sdk`, `rocprofiler-systems`, `rocgdb` | 성능 측정 및 GPU 디버깅 | ~40분 |
| **`full`** | `full` | 50개 이상 전체 ROCm 스택 풀 빌드 | 전체 배포본 생성 | ~5시간 |

---

## 🧀 토핑 추가 옵션 (--with-*) 사용법

기본 프리셋에 내가 원하는 기능만 살짝 얹어서 빌드할 수 있습니다:

```bash
# 예시 1: LLM 프리셋에 프로파일러(디버거) 추가하기
./therock-env build --preset llm --python 3.14 --with-profiler

# 예시 2: LLM 프리셋에 MIOpen(합성곱 AI 라이브러리) 추가하기
./therock-env build --preset llm --python 3.14 --with-miopen

# 예시 3: HIP 기본 프리셋에 rocFFT(고속 푸리에 변환) 추가하기
./therock-env build --preset hip --python 3.14 --with-fft

# 예시 4: 여러 개를 차례대로 밤새 연속 빌드하기 (배치 매트릭스 빌드)
./therock-env build-matrix --presets hip,llm,vulkan --python 3.14
```

### 지원하는 토핑 플래그 요약

* **`--with-miopen`**: 이미지/합성곱 딥러닝 라이브러리 MIOpen 추가
* **`--with-rccl`**: 다중 GPU 간 고속 통신 라이브러리 RCCL 추가
* **`--with-profiler`**: GPU 프로파일러(`rocprofv3`) 및 디버거(`rocgdb`) 추가
* **`--with-fft`**: 고속 푸리에 변환 `rocFFT` 추가
* **`--with-media`** / **`--with-vulkan`**: 비디오 디코더(`rocDecode`, `rocJPEG`) 및 Vulkan Mesa 추가
* **`--without-blas`**: 행렬 곱셈(BLAS) 제외

---

## 📁 빌드 목록 확인 및 관리

```bash
# 1. 생성된 모든 파이썬 가상환경 보기
./therock-env list-envs

# 2. 현재 빌드 완료된 모든 ROCm 빌드 폴더 및 용량 확인
./therock-env list-builds
```

---

## 📚 기술 문서 및 포팅 가이드

* [GCC 15 & Ubuntu 26.04 Technical Porting Guide](docs/GCC15_UBUNTU2604_PORTING_GUIDE.md): GCC 15 표준 헤더 수정, CMake 4.x 수명주기 호환, Meson 링커 스크립트 수정에 대한 상세 기술 문서.
* [Development Guide](docs/development/development_guide.md): TheRock 소스코드 개발자 매뉴얼.
* [Supported GPUs](SUPPORTED_GPUS.md): AMD GPU 아키텍처 지원 로드맵.
