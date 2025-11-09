# AI 应用开发概览 — PPT 内容（逐页稿）

说明：下面是为一份 26 页左右的分享准备的 PPT 内容（每一页为一张幻灯片）。包含：自顶向下的技术名词分类架构、各术语定义、如何开发 AI 应用、常用开发框架（Java / Python）、以及最简化代码案例（Python + Java 两个示例）。每页包含建议要点和演讲备注（Speaker Notes）。

---

幻灯片 1 — 封面
- 标题：AI 应用开发概览
- 副标题：术语分类、开发流程、框架与最简示例
- 演讲者：firewolf2010
- 日期：2025-11-09

Speaker notes:
- 简短介绍自己、目标听众与本次分享目标：让听众能快速掌握 AI 应用开发的全景图与实战起点。

---

幻灯片 2 — 内容提要（Outline）
- 自顶向下的技术名词分类架构（架构图）
- 各技术名词解释（按层级）
- 开发一个 AI 应用的步骤与要点
- 如何使用开发框架（流程与示例）
- 常见框架一览（Java / Python）
- 最简化 AI 应用代码案例
- 实践建议与资源

Speaker notes:
- 快速说明每部分的目的与输出。

---

幻灯片 3 — 自顶向下的总体架构图（文字版）
- 层级（自上而下）：
  1. 应用层（UI / 客户端 / Chatbot）
  2. 服务编排层（API Gateway / Orchestration）
  3. 推理与模型层（LLM / 模型服务 / 模型集成）
  4. 数据层（训练数据 / 索引 / 向量数据库）
  5. 平台与基础设施层（容器 / GPU / 节点）
  6. MLOps 与治理（版本管理 / CI/CD / 监控 / 安全）
- 在每一层汇聚相关术语（下一页展示分层详情）

Speaker notes:
- 建议用一张可视化图（矩形分层或环形），重点展示术语如何归类到不同层。

---

幻灯片 4 — 架构图（分层列出具体技术名词）
- 应用层：
  - Chatbot, UI, API Client, 微交互（Autosuggest）
- 服务编排层：
  - API Gateway, Orchestrator, Serverless, Edge inference
- 推理与模型层：
  - LLM, Transformer, Tokenization, Embeddings, Fine-tuning, Quantization, Model Serving（TorchServe, Triton）
- 数据层：
  - 数据集（训练/验证/测试）、向量索引（FAISS, Annoy）、RAG（检索增强生成）、知识库、ETL
- 平台与基础设施层：
  - GPU/TPU/CPU、Kubernetes、Docker、云服务（SageMaker, Vertex AI）、Batch/Streaming
- MLOps 与治理：
  - 模型版本控制、CI/CD、监控（latency/throughput/accuracy）、A/B 测试、合规与隐私

Speaker notes:
- 可用不同颜色标识每层，箭头表示从数据到用户的流动。

---

幻灯片 5 — 术语分组（按功能模块快速列表）
- 核心模型术语：Transformer, Attention, LLM, GPT, BERT
- 数据与表示：Tokenization, Subword, Embedding, Vector
- 训练与微调：Pre-training, Fine-tuning, Transfer learning, LoRA
- 推理优化：Quantization, Pruning, Distillation, Batching
- 生成与检索：RAG, Retriever, Retriever-Reader, Vector DB
- 工程与部署：Model Serving, API, Orchestration, Autoscaling
- 评估与监控：Perplexity, BLEU, ROUGE, F1, Drift detection

Speaker notes:
- 强调这些术语会在后面逐个解释。

---

幻灯片 6 — 模型与架构（示例）
- Transformer：
  - 一种基于自注意力机制（self-attention）的神经网络架构，擅长序列建模与并行计算。
- Attention（自注意力）：
  - 计算序列中不同位置之间的相关性权重，允许模型关注输入的不同部分。
- LLM（Large Language Model）：
  - 大规模预训练语言模型，通常以 Transformer 为基础，参数量从百万到百亿/万亿级。
- GPT / BERT 区别：
  - GPT：自回归生成模型（生成任务强）
  - BERT：双向编码器，擅长理解与表示（NLP 下游任务强）

Speaker notes:
- 用直观例子说明 GPT 更擅长“写”，BERT 更擅长“理解”。

---

幻灯片 7 — 数据与表示（示例）
- Tokenization：
  - 将文本拆分成 token（子词、字符或单词），常见方法：BPE、WordPiece、Unigram。
- Embedding（向量表示）：
  - 将离散 token 映射到连续向量空间，便于神经网络处理。
- Vector DB / 向量索引（FAISS, Annoy, HNSW）：
  - 存储和检索高维向量的数据库 / 索引，支持近似最近邻搜索（ANN）。
- RAG（Retrieval-Augmented Generation）：
  - 检索相关文档并将其作为上下文输入生成模型，提高知识性与准确率。

Speaker notes:
- 举例：用向量索引实现文档检索来辅助问答。

---

幻灯片 8 — 训练与微调（示例）
- Pre-training：
  - 在大规模通用语料上训练模型以学习通用表示。
- Fine-tuning：
  - 在特定任务/领域数据上继续训练模型以适配下游任务。
- LoRA（Low-Rank Adapters）：
  - 一种参数高效微调方法，仅训练部分低秩矩阵，节省存储与计算。
- Transfer learning：
  - 从预训练模型迁移知识到新任务，显著降低训练成本和数据需求。

Speaker notes:
- 说明 LoRA 的优势：轻量、便于多任务、多模型版本共存。

---

幻灯片 9 — 推理优化（示例）
- Quantization（量化）：
  - 将模型参数从浮点转为低精度（int8/4），降低内存与加速推理。
- Pruning（剪枝）：
  - 移除低重要性的权重或神经元，缩小模型体积。
- Distillation（知识蒸馏）：
  - 用大模型（teacher）训练小模型（student），保留性能减少成本。
- Batching / Pipeline parallelism：
  - 通过批处理或流水并行提高吞吐量。

Speaker notes:
- 解释量化可能带来的精度-性能折中。

---

幻灯片 10 — 工程与部署（示例）
- Model Serving（如 TorchServe, Triton）：
  - 专注于模型部署与高效推理的服务框架。
- Orchestration（Kubernetes）：
  - 管理容器化部署、伸缩和服务发现。
- Edge inference：
  - 在边缘设备上推理（响应更快，数据不出本地）。
- Monitoring / Observability：
  - 监控延迟、错误率、输入分布漂移与模型性能。

Speaker notes:
- 强调 CI/CD 与自动化（例如模型注册、自动回滚）在生产中的重要性。

---

幻灯片 11 — 如何开发一个 AI 应用（高层步骤）
1. 定义目标与场景（功能、性能、合规要求）
2. 数据收集与清洗（标注、质量评估、隐私处理）
3. 选择模型与方法（预训练模型、微调或从头训练）
4. 快速原型（小规模验证、baseline）
5. 优化（量化/蒸馏/剪枝/缓存）
6. 构建 API 与服务（Model Serving、API Gateway）
7. 部署（容器、K8s、云服务或边缘）
8. 监控与维护（数据漂移、性能、模型更新）

Speaker notes:
- 强调迭代、验证以及与业务指标的紧密联动。

---

幻灯片 12 — 如何使用开发框架（流程化建议）
- 1. 环境准备：虚拟环境、依赖管理、硬件（GPU/TPU）
- 2. 快速尝试：使用预训练模型做小样例（验证思路）
- 3. 数据流水线：建立数据加载、增强、批处理
- 4. 训练/微调：设置检查点、早停、超参搜索
- 5. 导出 & 优化：ONNX、TorchScript、量化
- 6. 包装服务：FastAPI/Flask/Gradio + Model Server
- 7. 自动化：CI/CD + 测试（单元/集成/性能）
- 8. 观测：日志、指标、告警、回滚策略

Speaker notes:
- 给出每一步可选工具（下一页分语言列出）。

---

幻灯片 13 — 常见开发框架 — Python 生态（按用途）
- 深度学习 / 模型训练：
  - PyTorch, TensorFlow, JAX
- 预训练模型与 NLP：
  - Hugging Face Transformers, SentenceTransformers
- 微调与加速：
  - PEFT, BitsAndBytes（量化加速）
- 应用与服务：
  - FastAPI, Flask, Starlette, Uvicorn
- 快速展示 / Demo：
  - Streamlit, Gradio
- 向量检索与 RAG：
  - FAISS, Annoy, Milvus, Weaviate, Chroma
- 高阶工具 / LLM 应用：
  - LangChain, LlamaIndex, Jina
- 部署与推理：
  - TorchServe, NVIDIA Triton, ONNX Runtime, BentoML

Speaker notes:
- 举例：使用 Hugging Face + FastAPI + FAISS 完成 RAG 问答。

---

幻灯片 14 — 常见开发框架 — Java 生态（按用途）
- 深度学习 / 模型调用：
  - DJL (Deep Java Library) — 支持 PyTorch/TensorFlow/ONNX 后端
  - TensorFlow Java
  - DeepLearning4J (DL4J)
- 模型推理：
  - ONNX Runtime Java
- 应用框架：
  - Spring Boot（构建服务 API）
- 向量检索 / 数据：
  - 使用外部向量 DB（Milvus、Weaviate 等）通过客户端访问

Speaker notes:
- Java 常用于企业后端系统集成，适配已有 Java 堆栈。

---

幻灯片 15 — Java / Python 框架对比（简要）
- Python 优势：
  - 丰富的研究与实用库、快速原型、社区支持强
- Java 优势：
  - 企业级成熟生态、与现有后端系统集成容易、生产可靠性高
- 常见做法：
  - 使用 Python 做模型开发与导出（ONNX），Java 负责服务与生产集成（调用 ONNX/推理服务）

Speaker notes:
- 推荐模式：模型训练在 Python，导出后用 Java 或独立推理服务提供 API.