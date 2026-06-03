# AIVO Irodori Test Suite

## 概要

本リポジトリは、FastAPI + Streamlit を用いた Irodori-TTS 通常版および VoiceDesign v3向け長文朗読・オーディオブック検証フロントエンドツールです。
Irodori-TTSは前の文脈で声のトーンや感情表現が変わることがあります。そのため長文を分割すると声のトーンが変わる可能性があります。その分割処理の設定をここで確認できます。
生成音声全体の上限時間、目標チャンク、絶対上限（多少の上限に幅を持たせるけど譲れないライン）、文字数上限、チャンク結合時の無音処理設定、sanitize_symbols、……！？　？？？などを正規化のオンオフ。
また、各々のPCスペックでどこまで分割処理をすれば時間がかからないかのテストツールにもなっています。

<img width="938" height="686" alt="image" src="https://github.com/user-attachments/assets/4b907075-68b1-4053-bbda-ff44eb0cc0e9" />




### 対応環境

- Windows
- CUDA / PyTorch
- NVIDIA GPU

### リポジトリに含まれないもの（各自で用意するもの）

- Irodori-TTS 本体
- モデルのウェイト（学習済み重み）
- speaker.safetensors（話者埋め込みファイル）
- 生成された音声ファイル
- 個人の実験データ

### クイックスタート

- vendor/Irodori-TTS を各自で配置してください（詳細は Upstream を参照）。
- irodori_embed_profiles/ の下に speaker.safetensors ファイルを配置してください。
- 通常版スイートの起動: start_irodori_suite.bat
- VoiceDesign版スイートの起動: start_voicedesign_suite.bat

### 使用ポート

- 通常版 API: 8010
- 通常版 UI: 8501
- VoiceDesign API: 8020
- VoiceDesign UI: 8520

### 注意事項

- speaker.safetensors および生成された音声の権利関係については、利用者の責任において確認してください。
- 本リポジトリには、音声資産やトレーニング済みの特定話者は含まれていません。
- 研究および個人のテスト目的のみに使用してください。
- ライセンスは未設定です。

### 検証で得られた知見

- VoiceDesign v3 における caption（プロンプト）入力は、話し方のスタイルに大きく影響します。
- caption="narration"（ナレーション）と指定することで、長文の朗読時にトーンを安定させる効果が期待できます。
- 日本語の小説を読み上げる際、sanitize_symbols=false に設定した方がより自然に聞こえる場合があります。
- 一部のスピーカー埋め込みでは、予期しない音色（声質の変化）が発生することがあります。

### アップストリーム（本家リポジトリ）

Irodori-TTS のコアは以下の別リポジトリで管理されています。


- [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)
- [Aratako/Irodori-TTS-600M-v3-VoiceDesign](https://huggingface.co/Aratako/Irodori-TTS-600M-v3-VoiceDesign)
## English

This repository is a lightweight front-end validation tool built with FastAPI and Streamlit, specifically designed for long-form reading and audiobook generation using Irodori-TTS (Standard and VoiceDesign v3).

### Key Technical Challenges & Features

- **Long-Form Consistency & Context Control**
  In Irodori-TTS, the emotional tone and vocal expression of the generated speech can fluctuate depending on the preceding context. When processing long texts, dividing the text into smaller segments (chunking) often causes noticeable shifts in vocal delivery. This tool provides a dedicated cockpit to configure, test, and optimize chunking parameters to maintain a consistent narration tone from start to finish.

- **Advanced Chunking & Text Normalization Configurator**
  You can fine-tune and experiment with critical production parameters via the Web UI, including:
  - **Total Audio Duration Limit (max_seconds)**: Sets the absolute cap for the entire generated track.
  - **Target Chunk Length (target_chunk_seconds)**: Defines the preferred duration for each individual text block.
  - **Strict Hard Limits**: Establishes a non-negotiable upper bound while allowing for flexible chunking thresholds.
  - **Character Count Caps**: Manages strict limits on text input length per chunk.
  - **Silent Interval Control**: Adjusts and fine-tunes silence padding when stitching generated chunks together.
  - **Text Normalization / Sanitization toggle (`sanitize_symbols=false`)**: Easily toggle text/symbol normalization on or off to preserve the natural rhythm of Japanese novels.

- **Hardware Performance & Benchmarking Tool**
  Beyond voice generation, this suite serves as a local benchmarking tool. It allows users to stress-test their specific hardware configuration (such as VRAM efficiency and generation speed) to determine the optimal chunk size that yields the fastest processing time without overloading the system.

### Supported Environment

- Windows
- CUDA / PyTorch
- NVIDIA GPU

### Not Included

- Irodori-TTS itself
- Model weights
- speaker.safetensors
- Generated audio
- Personal experiment data

### Quick Start

- Prepare `vendor/Irodori-TTS` on your own
- Place speaker.safetensors files under `irodori_embed_profiles/`
- Standard suite: `start_irodori_suite.bat`
- VoiceDesign suite: `start_voicedesign_suite.bat`

### Ports

- Standard API: `8010`
- Standard UI: `8501`
- VoiceDesign API: `8020`
- VoiceDesign UI: `8520`

### Notes

- Users are responsible for checking the rights of speaker.safetensors and generated audio
- This repository does not include audio assets or trained speakers
- For research and personal testing only
- License not set

### Interesting Findings

- Caption input in VoiceDesign v3 affects speaking style
- caption="narration" can help stabilize long-form reading in some cases
- sanitize_symbols=false can sound more natural for Japanese novels
- some speaker embeddings may yield unexpected timbre

### Upstream

The Irodori-TTS core lives in separate repositories.

- [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)
- [Aratako/Irodori-TTS-600M-v3-VoiceDesign](https://huggingface.co/Aratako/Irodori-TTS-600M-v3-VoiceDesign)
