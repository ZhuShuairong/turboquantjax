from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .bench import BenchmarkConfig, run_benchmark
from .generate import GenerateRequest
from .gguf_backend import GgufBackend
from .hf_backend import HfBackend
from .llamacpp_bridge import LlamaCppBridge
from .quality_eval import default_quality_cases, evaluate_quality_cases
from .serve import ServeConfig, run_server
from .telemetry import export_json, export_markdown_table


def _serialize_result(result) -> dict[str, Any]:
    payload = result.to_dict()
    # Prompt can be large in benchmark mode; keep output artifact concise.
    if len(payload.get("prompt", "")) > 4000:
        payload["prompt"] = payload["prompt"][:4000] + "..."
    return payload


def _write_result_artifacts(result_payload: dict[str, Any], export_json_path: str | None, export_md_path: str | None) -> None:
    if export_json_path:
        export_json(export_json_path, result_payload)

    if export_md_path:
        metrics = result_payload.get("metrics", {})
        row = {
            "model": result_payload.get("model"),
            "backend": result_payload.get("backend"),
            "runtime_mode": result_payload.get("runtime_mode"),
            "cache_mode": result_payload.get("cache_mode"),
            "prompt_length": metrics.get("prompt_tokens"),
            "generation_length": metrics.get("generated_tokens"),
            "ttft_s": metrics.get("ttft_s"),
            "prefill_tps": metrics.get("prefill_tps"),
            "decode_tps": metrics.get("decode_tps"),
            "wall_time_s": metrics.get("wall_time_s"),
            "kv_cache_bytes": metrics.get("kv_cache_bytes"),
            "compression_ratio": metrics.get("compression_ratio"),
            "quality_score": metrics.get("quality_score"),
        }
        export_markdown_table(export_md_path, "TurboQuant Generation Result", [row])


def _common_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["hf", "gguf"], required=True)
    parser.add_argument("--cache", choices=["baseline", "mse", "turboquant"], default="baseline")
    parser.add_argument("--model", help="HF model id or local model directory")
    parser.add_argument("--model-path", help="Local GGUF model file path")
    parser.add_argument("--runtime", default="llamacpp-python", help="GGUF runtime: llamacpp-python|llamacpp-cli|llama-server")
    parser.add_argument("--prompt", help="Inline prompt text")
    parser.add_argument("--prompt-file", help="Prompt file path")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bits-k", type=int, default=3)
    parser.add_argument("--bits-v", type=int, default=2)
    parser.add_argument("--device", default="auto", help="HF device preference: auto|cpu")
    parser.add_argument("--dtype", default="auto", help="HF dtype: auto|fp16|bf16|fp32")
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--export-json", default=None)
    parser.add_argument("--export-md", default=None)


def _make_request_from_args(args: argparse.Namespace) -> GenerateRequest:
    extra = {
        "server_url": getattr(args, "server_url", None),
        "llama_cache_type": getattr(args, "llama_cache_type", "f16"),
        "max_eval_layers": getattr(args, "max_eval_layers", None),
    }
    return GenerateRequest(
        backend=args.backend,
        cache=args.cache,
        model=args.model,
        model_path=args.model_path,
        runtime=args.runtime,
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        bits_k=args.bits_k,
        bits_v=args.bits_v,
        device=args.device,
        dtype=args.dtype,
        context_length=args.context_length,
        export_json=args.export_json,
        export_md=args.export_md,
        extra=extra,
    )


def _print_result(result_payload: dict[str, Any]) -> None:
    print("=" * 72)
    print("TurboQuant Runtime Result")
    print("=" * 72)
    print(json.dumps(result_payload.get("metrics", {}), indent=2))
    print("-" * 72)
    print(result_payload.get("output_text", ""))


def command_generate(args: argparse.Namespace) -> int:
    hf_backend = HfBackend(device=args.device, dtype=args.dtype)
    gguf_backend = GgufBackend(bridge=LlamaCppBridge())

    request = _make_request_from_args(args)
    if request.backend == "hf":
        result = hf_backend.generate(request)
    else:
        result = gguf_backend.generate(request)

    payload = _serialize_result(result)
    _write_result_artifacts(payload, args.export_json, args.export_md)
    _print_result(payload)
    return 0


def command_bench(args: argparse.Namespace) -> int:
    hf_backend = HfBackend(device=args.device, dtype=args.dtype)
    gguf_backend = GgufBackend(bridge=LlamaCppBridge())

    contexts = [int(x) for x in str(args.contexts).split(",") if x.strip()]
    cache_modes = [x.strip() for x in str(args.cache_modes).split(",") if x.strip()]

    config = BenchmarkConfig(
        backend=args.backend,
        model=args.model,
        model_path=args.model_path,
        runtime=args.runtime,
        contexts=contexts,
        cache_modes=cache_modes,
        max_new_tokens=args.max_new_tokens,
        bits_k=args.bits_k,
        bits_v=args.bits_v,
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        export_json_path=args.export_json,
        export_md_path=args.export_md,
        extra={
            "server_url": args.server_url,
            "llama_cache_type": args.llama_cache_type,
            "max_eval_layers": args.max_eval_layers,
        },
    )

    rows = run_benchmark(config=config, hf_backend=hf_backend, gguf_backend=gguf_backend)
    print(json.dumps(rows, indent=2))
    return 0


def command_validate(args: argparse.Namespace) -> int:
    hf_backend = HfBackend(device=args.device, dtype=args.dtype)
    gguf_backend = GgufBackend(bridge=LlamaCppBridge())

    cases = default_quality_cases()
    outputs: dict[str, str] = {}

    for case in cases:
        req = GenerateRequest(
            backend=args.backend,
            cache=args.cache,
            model=args.model,
            model_path=args.model_path,
            runtime=args.runtime,
            prompt=case.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            bits_k=args.bits_k,
            bits_v=args.bits_v,
            device=args.device,
            dtype=args.dtype,
            context_length=args.context_length,
            extra={
                "server_url": args.server_url,
                "llama_cache_type": args.llama_cache_type,
                "max_eval_layers": args.max_eval_layers,
            },
        )

        result = hf_backend.generate(req) if args.backend == "hf" else gguf_backend.generate(req)
        outputs[case.name] = result.output_text

    report = evaluate_quality_cases(outputs=outputs, cases=cases)
    print(json.dumps(report, indent=2))
    if args.export_json:
        export_json(args.export_json, report)
    if args.export_md:
        rows = [{"case": d["case"], "score": d["score"], "hits": ",".join(d["hits"])} for d in report["details"]]
        export_markdown_table(args.export_md, "TurboQuant Validation Report", rows)
    return 0


def command_serve(args: argparse.Namespace) -> int:
    config = ServeConfig(
        backend=args.backend,
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        context_length=args.context_length,
        n_gpu_layers=args.n_gpu_layers,
        binary=args.binary,
    )
    return run_server(config)


def command_env(args: argparse.Namespace) -> int:
    hf_backend = HfBackend(device=args.device, dtype=args.dtype)
    gguf_backend = GgufBackend(bridge=LlamaCppBridge())

    payload = {
        "hf": hf_backend.healthcheck(model=args.model),
        "gguf": gguf_backend.healthcheck(model_path=args.model_path),
    }
    print(json.dumps(payload, indent=2))
    if args.export_json:
        export_json(args.export_json, payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TurboQuant JAX production runtime CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_generate = sub.add_parser("generate", help="Run single prompt generation")
    _common_generation_args(p_generate)
    p_generate.add_argument("--server-url", default=None, help="Required for runtime=llama-server")
    p_generate.add_argument("--llama-cache-type", default="f16")
    p_generate.add_argument("--max-eval-layers", type=int, default=None)
    p_generate.set_defaults(func=command_generate)

    p_bench = sub.add_parser("bench", help="Run benchmark matrix")
    _common_generation_args(p_bench)
    p_bench.add_argument("--contexts", default="2048,4096,8192")
    p_bench.add_argument("--cache-modes", default="baseline,mse,turboquant")
    p_bench.add_argument("--server-url", default=None)
    p_bench.add_argument("--llama-cache-type", default="f16")
    p_bench.add_argument("--max-eval-layers", type=int, default=None)
    p_bench.set_defaults(func=command_bench)

    p_validate = sub.add_parser("validate", help="Run retrieval-oriented quality checks")
    _common_generation_args(p_validate)
    p_validate.add_argument("--server-url", default=None)
    p_validate.add_argument("--llama-cache-type", default="f16")
    p_validate.add_argument("--max-eval-layers", type=int, default=None)
    p_validate.set_defaults(func=command_validate)

    p_serve = sub.add_parser("serve", help="Launch local serving process")
    p_serve.add_argument("--backend", choices=["gguf"], required=True)
    p_serve.add_argument("--model-path", required=True)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--context-length", type=int, default=4096)
    p_serve.add_argument("--n-gpu-layers", type=int, default=-1)
    p_serve.add_argument("--binary", default="llama-server")
    p_serve.set_defaults(func=command_serve)

    p_env = sub.add_parser("env", help="Print backend environment diagnostics")
    p_env.add_argument("--model", default=None)
    p_env.add_argument("--model-path", default=None)
    p_env.add_argument("--device", default="auto")
    p_env.add_argument("--dtype", default="auto")
    p_env.add_argument("--export-json", default=None)
    p_env.set_defaults(func=command_env)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
