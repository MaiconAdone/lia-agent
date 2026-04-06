"""
main.py — CLI interativa do Book Analytics Agent (Lia).

Uso:
    python main.py

Comandos especiais durante a sessão:
    /overview  — Visão geral do dataset
    /roi       — Impacto financeiro da automação
    /reset     — Nova conversa (limpa histórico)
    /stats     — Estatísticas da sessão
    /health    — Health check do sistema
    /evals     — Roda suíte básica de avaliação
    /help      — Lista comandos e exemplos
    /sair      — Encerra o programa
"""

import os
import sys

# Carrega .env ANTES de qualquer import que precise de variáveis de ambiente
from utils import load_env, require_env
load_env()

import anthropic
import config
from agent import BookAnalyticsAgent
from evals import format_eval_summary, run_eval_suite
from observability import SessionObserver, health_check
from repository import repository
from tools import benchmark_queries

# ─────────────────────────────────────────
# Cores ANSI
# ─────────────────────────────────────────

R    = "\033[0m"       # reset
BOLD = "\033[1m"
CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
GRAY  = "\033[90m"
RED   = "\033[91m"
MAGENTA = "\033[95m"
BLUE  = "\033[94m"


def c(text: str, color: str) -> str:
    return f"{color}{text}{R}"


# ─────────────────────────────────────────
# Banner e textos
# ─────────────────────────────────────────

BANNER = f"""
{CYAN}{BOLD}
  ██╗     ██╗ █████╗
  ██║     ██║██╔══██╗
  ██║     ██║███████║
  ██║     ██║██╔══██║
  ███████╗██║██║  ██║
  ╚══════╝╚═╝╚═╝  ╚═╝   Literary Intelligence Assistant
{R}
  {GRAY}Olá! Eu sou a Lia, sua assistente de análise literária.{R}
  {GRAY}Tenho acesso a mais de 212 mil livros e 5,7 milhões de avaliações.{R}
  {GRAY}Digite {R}{CYAN}/help{R}{GRAY} para ver o que posso fazer por você.{R}
"""

HELP_TEXT = f"""
{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{R}
{CYAN}{BOLD}  Comandos disponíveis{R}
{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{R}

  {CYAN}/overview{R}  Panorama geral do dataset (gêneros, autores, totais)
  {CYAN}/roi{R}       Análise de impacto financeiro da automação
  {CYAN}/reset{R}     Nova conversa — limpa o histórico
  {CYAN}/stats{R}     Estatísticas da sessão atual
  {CYAN}/health{R}    Health check do sistema
  {CYAN}/evals{R}     Executa suíte básica de avaliação
  {CYAN}/bench{R}     Mede latência das queries principais
  {CYAN}/help{R}      Esta mensagem
  {CYAN}/sair{R}      Encerrar o programa

{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{R}
{CYAN}{BOLD}  Exemplos de perguntas{R}
{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{R}

  Sobre autores:
    "Como está o desempenho do Stephen King?"
    "Compare J.K. Rowling com Tolkien"
    "Quais são os livros mais bem avaliados de Agatha Christie?"

  Sobre gêneros:
    "Analise o gênero Fiction"
    "Qual gênero tem melhor avaliação média?"
    "Me mostra os top autores de Romance"

  Sobre usuários:
    "Quem são os reviewers mais ativos?"
    "Encontre usuários com mais de 100 avaliações"
    "Quais leitores têm gosto mais diversificado?"

  Busca e comparação:
    "Busque livros sobre machine learning"
    "Me fale sobre o livro Harry Potter"
    "Qual o impacto de automatizar nossas análises?"

{BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{R}
"""

# ─────────────────────────────────────────
# Formatação de saída
# ─────────────────────────────────────────

def print_response(text: str) -> None:
    print(f"\n{MAGENTA}{BOLD}  Lia:{R}")
    print(f"{MAGENTA}  {'─' * 58}{R}")
    for line in text.splitlines():
        print(f"  {line}")
    print(f"{MAGENTA}  {'─' * 58}{R}\n")


def print_tool_call(tool_name: str, params_str: str) -> None:
    icons = {
        "analyze_author":       "  Analisando autor",
        "analyze_genre":        "  Analisando gênero",
        "find_influential_users": "  Buscando usuários influentes",
        "search_books":         "  Pesquisando livros",
        "get_book_analysis":    "  Analisando livro",
        "compare_authors":      "  Comparando autores",
        "get_dataset_overview": "  Carregando visão geral",
        "calculate_roi_impact": "  Calculando impacto financeiro",
    }
    label = icons.get(tool_name, f"  Executando {tool_name}")
    print(f"{YELLOW}{label}...{R}", flush=True)


def print_error(msg: str) -> None:
    print(f"\n{RED}  Erro: {msg}{R}\n")


def print_separator() -> None:
    print(f"{GRAY}  {'·' * 58}{R}")


# ─────────────────────────────────────────
# Comandos especiais → prompts ao agent
# ─────────────────────────────────────────

COMMAND_PROMPTS = {
    "/overview": (
        "Me dê uma visão geral completa e interessante do nosso dataset: "
        "quantos livros, avaliações, autores e usuários únicos temos. "
        "Mostre os gêneros mais populares, os autores com mais avaliações "
        "e a distribuição geral de notas. Seja curiosa e destaque pontos surpreendentes."
    ),
    "/roi": (
        "Analise e explique de forma clara e convincente o impacto financeiro "
        "de usar um agent de IA para automatizar as análises de avaliações de livros, "
        "comparando com o processo manual atual: 5 analistas, salário de R$5.000 cada, "
        "3 dias por análise. Mostre economia mensal, anual, ganho de velocidade "
        "e os benefícios qualitativos. Apresente como se fosse uma recomendação de negócio."
    ),
}

# ─────────────────────────────────────────
# Loop principal
# ─────────────────────────────────────────

def main() -> None:
    # Ativa cores ANSI no terminal Windows
    if sys.platform == "win32":
        os.system("color")

    # Valida chave de API antes de qualquer coisa
    require_env(
        "ANTHROPIC_API_KEY",
        hint="Obtenha sua chave em https://console.anthropic.com/settings/keys"
    )

    health = health_check(config.DB_PATH)
    if not health["ok"]:
        print_error(f"Health check falhou: {health['checks'].get('db_error', 'erro desconhecido')}")
        sys.exit(1)

    print(BANNER)

    # Inicializa o agent
    try:
        observer = SessionObserver()
        agent = BookAnalyticsAgent(api_key=require_env("ANTHROPIC_API_KEY"), observer=observer)
    except FileNotFoundError as exc:
        print_error(str(exc))
        print(f"\n{YELLOW}  Execute primeiro:{R} python setup_db.py\n")
        sys.exit(1)
    except Exception as exc:
        print_error(str(exc))
        sys.exit(1)

    print(f"{GREEN}  Base de dados conectada com sucesso.{R}")
    print(f"{GREEN}  Health check OK: {health['checks'].get('books_count', 0):,} livros / {health['checks'].get('ratings_count', 0):,} avaliações.{R}")
    print(f"{GRAY}  ({'─' * 56}){R}\n")

    # ─── Mensagem de boas-vindas personalizada da Lia ───
    welcome = agent.greet()
    print_response(welcome)

    while True:
        try:
            user_input = input(f"{CYAN}{BOLD}  Você:{R} ").strip()
        except (EOFError, KeyboardInterrupt):
            farewell = agent.say_goodbye()
            print_response(farewell)
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        # Saída
        if cmd in ("/sair", "/exit", "/quit", "sair", "exit", "tchau", "bye"):
            farewell = agent.say_goodbye()
            print_response(farewell)
            break

        # Reset
        if cmd == "/reset":
            agent.reset()
            print(f"\n{YELLOW}  Conversa reiniciada. Estou pronta para uma nova análise!{R}\n")
            continue

        # Ajuda (sem chamar agent)
        if cmd == "/help":
            print(HELP_TEXT)
            continue

        if cmd == "/stats":
            print(f"\n{BLUE}{agent.get_session_summary()}{R}\n")
            query_metric = repository.last_query_metrics
            if query_metric:
                print(f"{GRAY}  Última query SQL:{R}")
                print(f"  - latency_ms: {query_metric.latency_ms:.1f}")
                print(f"  - rows_returned: {query_metric.rows_returned}")
                print(f"  - cache_hit: {query_metric.cache_hit}")
                print()
            continue

        if cmd == "/health":
            latest_health = health_check(config.DB_PATH)
            status = "OK" if latest_health["ok"] else "FALHOU"
            print(f"\n{BLUE}Health check: {status}{R}")
            for key, value in latest_health["checks"].items():
                print(f"  - {key}: {value}")
            print()
            continue

        if cmd == "/evals":
            print(f"\n{YELLOW}  Executando suíte básica de evals...{R}")
            results = run_eval_suite(agent.ask, output_path=os.path.join("artifacts", "latest_evals.json"))
            print(f"\n{BLUE}{format_eval_summary(results)}{R}\n")
            continue

        if cmd == "/bench":
            print(f"\n{YELLOW}  Executando benchmark rápido de queries...{R}")
            results = benchmark_queries()
            print(f"{BLUE}FTS5 habilitado: {results['fts_enabled']}{R}")
            for item in results["benchmarks"]:
                print(f"  - {item['name']}: {item['latency_ms']} ms | rows={item['rows_returned']} | cache_hit={item['cache_hit']} | ok={item['ok']}")
            print()
            continue

        # Comandos que viram prompts ao agent
        if user_input in COMMAND_PROMPTS:
            user_input = COMMAND_PROMPTS[user_input]

        # ─── Chama o agent ───
        try:
            answer = agent.ask(
                user_input,
                on_tool_call=print_tool_call,
            )
            print_response(answer)
        except anthropic.AuthenticationError:
            print_error("Chave de API inválida. Verifique o valor de ANTHROPIC_API_KEY no arquivo .env")
        except anthropic.RateLimitError:
            print_error("Limite de requisições atingido. Aguarde alguns segundos e tente novamente.")
        except anthropic.APIConnectionError:
            print_error("Sem conexão com a API Anthropic. Verifique sua internet.")
        except anthropic.APIError as exc:
            print_error(f"Erro da API: {exc}")
        except Exception as exc:
            print_error(f"Erro inesperado: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\n{BLUE}{agent.get_session_summary()}{R}\n")
    agent.close()


if __name__ == "__main__":
    main()
