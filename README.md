# Lia — Literary Intelligence Assistant

Agente de IA para análise analítica de livros e avaliações, com foco em confiabilidade, observabilidade e operação em produção.

- **Dataset:** ~212K livros · 3M avaliações · 1M usuários únicos (Amazon Books)
- **Stack:** Python · SQLite · Anthropic / OpenAI · Streamlit
- **Padrão:** ReAct (Reasoning + Acting) com multi-provider e fast-path determinístico

---

## Arquitetura

```text
Entrada do usuário
 └─ guardrails.py      → validação e bloqueio de inputs maliciosos
 └─ router.py          → roteamento determinístico (fast-path, sem LLM)
 └─ agent.py           → loop ReAct: raciocínio + chamada de ferramentas
     ├─ llm_client.py  → abstração OpenAI / Anthropic com retry e fallback
     ├─ tools.py       → fachada das 12 ferramentas analíticas
     └─ repository.py  → queries SQLite com cache LRU e tabelas pré-computadas
 └─ observability.py   → logs JSONL, tokens, custo e latência por turno
```

### Componentes

| Arquivo | Responsabilidade |
|---|---|
| `agent.py` | Loop ReAct, retry, pruning de contexto, orçamento de tokens |
| `llm_client.py` | Desacoplamento de provider, retry com jitter, fallback automático |
| `tools.py` | 12 ferramentas analíticas registradas no TOOL_REGISTRY |
| `repository.py` | Acesso SQLite centralizado, cache LRU (TTL 5min, 256 entradas) |
| `router.py` | Roteamento determinístico de intenção antes do LLM |
| `guardrails.py` | Validação de entrada/saída, detecção de prompt injection |
| `observability.py` | Logs JSONL estruturados, rastreamento de tokens e custo |
| `evals.py` | Suíte de avaliação de regressão funcional |
| `config.py` | Configuração centralizada via variáveis de ambiente |
| `main.py` | CLI interativa com health check e comandos operacionais |
| `streamlit_app.py` | Interface visual com status em tempo real e métricas de sessão |
| `setup_db.py` | Carga dos CSVs no SQLite com índices e tabelas pré-computadas |
| `utils.py` | Funções auxiliares (load_env, require_env) |

---

## Ferramentas Analíticas (12)

| Ferramenta | Descrição |
|---|---|
| `get_dataset_totals` | Totais rápidos do dataset (fast-path) |
| `get_dataset_overview` | Visão completa: livros, autores, categorias, top gêneros |
| `search_books` | Busca por título, autor ou categoria |
| `get_book_analysis` | Análise detalhada de um livro: reviews, preço, score |
| `get_author_stats` | Estatísticas de um autor: títulos, score médio, categorias |
| `get_category_stats` | Análise de uma categoria literária |
| `get_top_rated_books` | Ranking de livros por score mínimo e número de reviews |
| `get_review_timeline` | Evolução temporal de avaliações |
| `get_user_profile` | Perfil de um revisor: histórico, padrões, score médio |
| `get_price_analysis` | ROI e impacto financeiro da automação |
| `analyze_price` | Distribuição de preços, faixas × score, livros mais caros |
| `analyze_review_quality` | Qualidade das reviews: comprimento, revisores ativos, amostras |

### Tabelas pré-computadas (SQLite)

Criadas automaticamente pelo `setup_db.py` para garantir performance < 200ms:

| Tabela | Conteúdo |
|---|---|
| `stats_review_length` | Distribuição de comprimento das reviews por faixa |
| `stats_top_reviewers` | Revisores com ≥ 20 avaliações, score médio e livros únicos |
| `stats_price_summary` | Totais e médias de preço em USD (total, with_price, avg, min, max) |
| `stats_price_bands` | Distribuição por faixa de preço em USD × score médio |

---

## Instalação

```bash
pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env   # Linux/macOS
```

Edite `.env` com suas chaves de API, depois carregue o banco (execução única, ~10–20 min):

```bash
python setup_db.py
```

### Iniciar

```bash
# CLI interativa
python main.py

# Interface visual (Streamlit)
streamlit run streamlit_app.py
```

---

## Variáveis de Ambiente

**Obrigatória (pelo menos uma):**

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

**Opcionais:**

```env
LLM_PROVIDER=openai              # provider padrão (openai | anthropic)
CLAUDE_MODEL=claude-sonnet-4-6
OPENAI_MODEL=gpt-4o-mini
AGENT_MAX_ITERATIONS=6
MAX_OUTPUT_TOKENS=4096
AGENT_TEMPERATURE=0
LLM_TIMEOUT_SECONDS=45           # timeout por chamada de API
LOG_LEVEL=INFO
BRL_PER_USD=5.80
QUERY_CACHE_TTL=300
GUARDRAILS=true
```

> O agente detecta automaticamente qual provider usar com base nas chaves disponíveis,
> sem necessidade de configuração manual.

---

## Comandos da CLI

| Comando | Ação |
|---|---|
| `/overview` | Panorama do dataset |
| `/roi` | Impacto financeiro da automação |
| `/evals` | Executa suíte de avaliação de regressão |
| `/stats` | Métricas da sessão (tokens, custo, latência) |
| `/health` | Health check do sistema |
| `/reset` | Limpa o histórico da conversa |
| `/help` | Ajuda |
| `/sair` | Encerra |

---

## Multi-provider & Fallback

- **Provider primário:** configurado via `LLM_PROVIDER` (padrão: `openai`)
- **Fallback automático:** em caso de erro de quota, rate-limit, timeout ou chave inválida, o agente troca de provider automaticamente
- **Retry com jitter:** backoff exponencial com variação aleatória para evitar thundering herd
- **Timeout configurável:** `LLM_TIMEOUT_SECONDS` (padrão: 45s)

---

## Performance

Otimizações aplicadas que garantem respostas dentro da meta de latência:

| Otimização | Impacto |
|---|---|
| Fast-path determinístico | Responde sem LLM para intenções simples (~400ms) |
| Cache LRU nas queries SQL | TTL 5min, 256 entradas — hit rate esperado ≥ 40% |
| Cache LRU de respostas | 128 entradas por sessão para queries repetidas |
| Tabelas pré-computadas | `analyze_review_quality`: 334s → 152ms |
| Índices otimizados | `analyze_price`: 103s → 2ms |
| Pruning de contexto | Janela de contexto mantida em ≤ 75% do limite |

---

## Observabilidade

Logs em `logs/` no formato **JSONL**. Cada turno registra:

- ferramentas chamadas e latência por ferramenta
- tokens de entrada/saída e custo estimado (USD e BRL)
- cache hit/miss nas queries SQL
- eventos de guardrail
- pruning de contexto
- provider usado e se houve fallback

Resultados de evals salvos em `artifacts/latest_evals.json`.

---

## Guardrails

- Bloqueio de prompt injection (padrões conhecidos)
- Limite de tamanho de entrada
- Validação de parâmetros de ferramentas
- Detecção de vazamento no output (chaves, tokens de sistema)

---

## Testes & Evals

Testes unitários em `tests/` (não requerem banco nem API key):

```bash
python -m pytest tests/ -v
# ou sem pytest instalado:
python -m unittest discover tests/ -v
```

Suíte de avaliação do agente (requer banco e API key):

```bash
# via CLI
python main.py  →  /evals

# via Streamlit: botão "Rodar Health Check" no painel lateral
```

Resultados salvos em `artifacts/latest_evals.json`.

---

## Roadmap

- **Q3 2026:** Sumarização generativa de reviews · RAG com catálogo editorial · Dashboard com visualizações · API REST
- **Q4 2026:** Recomendação por perfil de leitor · Alertas de tendências · Deploy em cloud
- **2027:** Agente com memória de longo prazo · Previsão de demanda · Fine-tuning editorial
