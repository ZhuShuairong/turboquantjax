from __future__ import annotations

import pytest

from turboquant_jax.runtime import cli


def test_parser_includes_required_subcommands() -> None:
    parser = cli.build_parser()
    subcommands = parser._subparsers._group_actions[0].choices.keys()  # type: ignore[attr-defined]
    for name in ["generate", "bench", "validate", "serve", "env"]:
        assert name in subcommands


def test_generate_requires_prompt_source() -> None:
    with pytest.raises(ValueError):
        req = cli._make_request_from_args(  # pylint: disable=protected-access
            cli.argparse.Namespace(
                backend="hf",
                cache="baseline",
                model="tiny-model",
                model_path=None,
                runtime="llamacpp-python",
                prompt=None,
                prompt_file=None,
                max_new_tokens=4,
                temperature=0.0,
                top_p=1.0,
                seed=42,
                bits_k=3,
                bits_v=2,
                device="auto",
                dtype="auto",
                context_length=None,
                export_json=None,
                export_md=None,
                server_url=None,
                llama_cache_type="f16",
                max_eval_layers=None,
            )
        )
        req.validate()


def test_env_command_smoke(capsys) -> None:
    rc = cli.main(["env"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "hf" in captured.out
    assert "gguf" in captured.out
