# config.py
from dataclasses import dataclass
from typing import Optional, Tuple
import os

try:
    # se python-dotenv estiver instalado, carrega o .env
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # se não estiver instalado, segue só com o ambiente do SO
    pass


@dataclass(frozen=True)
class Settings:
    github_token: str
    openai_api_key: Optional[str] = None


def get_settings() -> Settings:
    """
    Lê GH_TOKEN e OPENAI_API_KEY do ambiente (.env ou variáveis do SO).
    GH_TOKEN é obrigatório, OPENAI_API_KEY é opcional.
    """
    gh_token = os.getenv("GH_TOKEN")
    if not gh_token:
        raise RuntimeError("GH_TOKEN não configurado nas variáveis de ambiente.")

    openai_key = os.getenv("OPENAI_API_KEY")
    return Settings(github_token=gh_token, openai_api_key=openai_key)


def resolve_credentials(
    cli_token: Optional[str],
    cli_openai: Optional[str],
    settings: Settings,
) -> Tuple[str, Optional[str]]:
    """
    Regra de precedência:
    - se flag de linha de comando foi passada, ela vence o .env
    - senão, usa o valor vindo do Settings (.env)
    """
    gh_token = cli_token or settings.github_token
    openai_key = cli_openai if cli_openai is not None else settings.openai_api_key

    if not gh_token:
        # segurança extra, embora get_settings já garanta isso
        raise RuntimeError("Nenhum token do GitHub foi encontrado.")

    return gh_token, openai_key
