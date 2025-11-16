# 部署指南 (Deployment Guide)

## 快速开始

### 本地开发

```bash
# 1. 克隆项目
git clone https://github.com/lcd955/AI_recommendation_for_finaical.git
cd AI_recommendation_for_finaical

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动服务
python -m src.api.main
```

### Docker部署

```bash
docker build -t ai-financial:latest .
docker run -p 8000:8000 ai-financial:latest
```

访问 http://localhost:8000/docs 查看API文档。

详细部署说明请参考项目README.md
