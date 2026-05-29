# RAG 商品分类系统

基于 BGE 语义检索和大语言模型的四级商品分类预测系统。

## 特点

- **混合检索**：BGE 语义检索 + 品牌精确匹配，提升准确率
- **防幻觉设计**：严格的 Prompt 工程，禁止 LLM 瞎编
- **多 LLM 支持**：兼容 OpenAI、Yi、智谱等 OpenAI 格式 API
- **完整评估**：内置准确率评估、错误分析工具

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API

复制配置模板并填入你的 API 信息：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# ============ LLM API 配置 ============

# API 提供商：openai | yi | custom
LLM_PROVIDER=openai

# 模型名称
# OpenAI: gpt-5.5
# Yi: yi-large, yi-medium
# 自定义: 根据你的 API 提供商填写
LLM_MODEL=glm

# API 密钥
OPENAI_API_KEY=sk-your-api-key-here

# API 地址（重要！）
# - OpenAI：留空，会自动使用官方地址
# - Yi / 智谱 / 国内 API：必须填写完整地址
# - 本地模型（如 LMStudio）：填写 http://localhost:11434/v1
API_BASE_URL=

# ============ 检索配置 ============

# 检索返回数量
RETRIEVAL_TOP_K=3

# 候选数量（用于品牌加权前）
RETRIEVAL_CANDIDATE_K=8

# 品匹配置分（匹配品牌时加的分）
BRAND_BOOST_SCORE=0.15

# ============ BGE 模型配置 ============

# BGE 模型名称（首次运行会自动下载）
BGE_MODEL_NAME=BAAI/bge-base-zh-v1.5
```

### 3. 构建知识库

首次运行需要构建向量索引（会自动下载 BGE 模型，约 400MB）：

```bash
python main.py build
```

### 4. 使用

**交互式分类：**
```bash
python main.py interactive
```

**批量分类：**
```bash
python main.py classify data/商品.csv output/结果.csv
```

**评估准确率：**
```bash
python main.py evaluate data/test.csv
```

## 常用 API 配置示例

### OpenAI 官方

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-xxx
API_BASE_URL=
```

### 零一万物 Yi

```env
LLM_PROVIDER=custom
LLM_MODEL=yi-large
OPENAI_API_KEY=your-yi-api-key
API_BASE_URL=https://api.lingyiwanwu.com/v1
```

### 智谱 AI

```env
LLM_PROVIDER=custom
LLM_MODEL=glm-4
OPENAI_API_KEY=your-zhipu-api-key
API_BASE_URL=https://open.bigmodel.cn/api/paas/v4
```

### 本地模型 (LMStudio / Ollama)

```env
LLM_PROVIDER=custom
LLM_MODEL=local-model
OPENAI_API_KEY=not-needed
API_BASE_URL=http://localhost:11434/v1
```

## 系统架构

```
商品名称
   │
   ├──→ [BGE 编码] → 向量检索 → Top-K 候选分类
   │                      │
   └──→ [品牌提取] ──────→ 品牌加权 → 重新排序
                              │
                              ↓
                         [LLM 分类] → 四级分类结果
```

## 项目结构

```
llmrag/
├── main.py                 # 主入口
├── src/
│   ├── config.py           # 配置管理
│   ├── knowledge_base.py   # 知识库构建
│   ├── vector_store.py    # BGE 向量存储
│   ├── retriever.py        # 混合检索器
│   ├── llm_classifier.py   # LLM 分类器
│   └── evaluator.py        # 评估模块
├── data/
│   └── 品类定义详情.csv     # 品类知识库（源数据）
├── knowledge_base/         # 知识库文件（生成）
├── bge_vector_index/       # 向量索引（生成）
└── output/                 # 分类结果输出
```

## 规模化处理

### 300 万商品处理方案

| 配置 | 处理速度 | 300万耗时 | API 成本估算 |
|------|----------|-----------|--------------|
| 单线程 | ~0.8 件/秒 | ~43 天 | 高 |
| 4 并发 | ~3 件/秒 | ~12 天 | 中 |
| 20 并发 | ~15 件/秒 | ~2.3 天 | 中 |

### 并发配置

编辑 `.env`：

```env
# 并发线程数（根据 API 限速调整）
MAX_WORKERS=20

# 批量大小
BATCH_SIZE=50

# 启用缓存（重复商品直接返回结果）
ENABLE_CACHE=true
CACHE_SIZE=10000
```

### 成本优化建议

1. **启用缓存** - 重复商品直接命中，无需调用 API
2. **本地模型** - 使用 LMStudio/LocalAI 零成本推理
3. **分批处理** - 将 300 万拆分为多个小批次，避免单次失败重跑

## 性能

| 指标 | 数值 |
|------|------|
| 四级分类准确率 | ~92% |
| 单次分类耗时 | ~1.2 秒（含 LLM 调用） |
| 检索耗时 | ~50ms（BGE 向量检索） |
| 支持商品数 | 300 万+ SKU |

## License

MIT
