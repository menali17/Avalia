# tests/test_config.py
import os
import sys
import importlib
import types
import pytest

# Adiciona a raiz do projeto no sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config


def _reload_with_env(env: dict) -> types.ModuleType:
    """
    Helper para recarregar o módulo config com um ambiente controlado
    e sem carregar o .env real do projeto.
    """
    # limpa variáveis relevantes do processo atual
    for key in ["GH_TOKEN", "OPENAI_API_KEY"]:
        os.environ.pop(key, None)

    # injeta as que queremos para o teste
    os.environ.update(env)

    # recarrega o módulo config
    mod = importlib.reload(config)

    # DESLIGA o uso de python-dotenv dentro do config,
    # assim ele NÃO chama load_dotenv() e não lê o .env real
    if hasattr(mod, "DOTENV_AVAILABLE"):
        mod.DOTENV_AVAILABLE = False

    return mod


# 1) GH_TOKEN ausente -> erro
def test_get_settings_sem_gh_token_dispara_erro():
    cfg = _reload_with_env({})
    with pytest.raises(RuntimeError):
        cfg.get_settings()


# 2) GH_TOKEN presente, OPENAI_API_KEY ausente -> ok
def test_get_settings_com_gh_token_sem_openai_ok():
    cfg = _reload_with_env({"GH_TOKEN": "fake_gh_token"})
    settings = cfg.get_settings()

    assert settings.github_token == "fake_gh_token"
    assert settings.openai_api_key is None


# 3) Ambos presentes -> ok
def test_get_settings_com_ambos_os_tokens():
    cfg = _reload_with_env(
        {
            "GH_TOKEN": "fake_gh_token",
            "OPENAI_API_KEY": "fake_openai_key",
        }
    )
    settings = cfg.get_settings()

    assert settings.github_token == "fake_gh_token"
    assert settings.openai_api_key == "fake_openai_key"


# 4) resolve_credentials: CLI > ENV para GitHub
def test_resolve_credentials_cli_sobrepoe_env_github():
    settings = config.Settings(
        github_token="token_do_env",
        openai_api_key=None,
    )

    gh_token, openai_key = config.resolve_credentials(
        cli_token="token_da_cli",
        cli_openai=None,
        settings=settings,
    )

    assert gh_token == "token_da_cli"
    assert openai_key is None


# 5) resolve_credentials: CLI > ENV para OpenAI
def test_resolve_credentials_cli_sobrepoe_env_openai():
    settings = config.Settings(
        github_token="token_do_env",
        openai_api_key="openai_do_env",
    )

    gh_token, openai_key = config.resolve_credentials(
        cli_token=None,
        cli_openai="openai_da_cli",
        settings=settings,
    )

    assert gh_token == "token_do_env"      # veio do env
    assert openai_key == "openai_da_cli"   # veio da CLI
