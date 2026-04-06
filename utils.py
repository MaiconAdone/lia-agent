"""
utils.py — Utilitários compartilhados do Book Analytics Agent.

Inclui loader de .env sem dependências externas (stdlib pura),
evitando a necessidade de instalar python-dotenv.
"""

import os
import re
import sys


# ─────────────────────────────────────────
# Loader de .env
# ─────────────────────────────────────────

def load_env(env_path: str | None = None) -> dict[str, str]:
    """
    Carrega variáveis de um arquivo .env para os.environ.

    Suporta:
      - KEY=value
      - KEY="value com espaços"
      - KEY='value'
      - # comentários (linhas e inline)
      - Linhas em branco

    Retorna dict com as chaves carregadas (sem sobrescrever variáveis já
    definidas no ambiente — variável de sistema tem prioridade).
    """
    if env_path is None:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

    loaded: dict[str, str] = {}

    if not os.path.exists(env_path):
        return loaded

    with open(env_path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()

            # Ignora linhas em branco e comentários
            if not line or line.startswith("#"):
                continue

            # Remove comentário inline: KEY=value  # comentario
            line = re.sub(r"\s+#.*$", "", line)

            if "=" not in line:
                continue  # linha malformada

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()

            # Remove aspas simples ou duplas envolventes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]

            if not key:
                continue

            loaded[key] = value

            # Só define se não estiver já no ambiente (sistema tem prioridade)
            if key not in os.environ:
                os.environ[key] = value

    return loaded


def require_env(key: str, hint: str = "") -> str:
    """
    Retorna o valor de uma variável de ambiente obrigatória.
    Encerra com mensagem clara se estiver ausente ou com valor padrão.
    """
    value = os.environ.get(key, "").strip()
    placeholder = {"sua-chave-aqui", "sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", ""}

    if not value or value in placeholder:
        msg = f"\n[ERRO] Variável '{key}' não configurada."
        if hint:
            msg += f"\n       {hint}"
        msg += f"\n       Configure em .env: {key}=seu-valor-aqui\n"
        print(msg, file=sys.stderr)
        sys.exit(1)

    return value
