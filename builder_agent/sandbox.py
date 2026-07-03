"""Sandboxed code execution environments using subprocesses or containers."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

from builder_agent import config

_NODE_BUILTINS = {
    "assert", "buffer", "child_process", "cluster", "console", "constants",
    "crypto", "dgram", "dns", "domain", "events", "fs", "http", "https",
    "module", "net", "os", "path", "punycode", "querystring", "readline",
    "repl", "stream", "string_decoder", "sys", "timers", "tls", "tty",
    "url", "util", "v8", "vm", "zlib"
}


def _extract_js_dependencies(code: str) -> list[str]:
    deps = []
    esm_patterns = [
        r'''import\s+.*?\s+from\s+['"]([^'"]+)['"]''',
        r'''import\s+['"]([^'"]+)['"]''',
        r'''require\s*\(\s*['"]([^'"]+)['"]\s*\)''',
        r'''import\s*\(\s*['"]([^'"]+)['"]\s*\)'''
    ]
    for pattern in esm_patterns:
        for match in re.finditer(pattern, code):
            dep = match.group(1)
            if not dep.startswith((".", "/", "\\")) and dep not in _NODE_BUILTINS:
                if dep.startswith("@"):
                    parts = dep.split("/")
                    if len(parts) >= 2:
                        deps.append(f"{parts[0]}/{parts[1]}")
                else:
                    deps.append(dep.split("/")[0])
    return sorted(list(set(deps)))


def run_code(
    code: str, timeout: int = 10, language: str = "python"
) -> tuple[bool, str]:
    """Execute code in a sandboxed environment."""
    if config.SANDBOX_BACKEND == "container":
        return _run_in_container(code, timeout, language)
    return _run_in_subprocess(code, timeout, language)


def _run_in_subprocess(code: str, timeout: int, language: str) -> tuple[bool, str]:
    """Execute code locally in a subprocess."""
    npx_cmd = "npx.cmd" if sys.platform == "win32" else "npx"
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"

    if language in ("python", "python_module", "python_package"):
        suffix = ".py"
        cmd = [sys.executable]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False
        ) as f:
            f.write(code)
            f.flush()
            try:
                result = subprocess.run(
                    cmd + [f.name],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                output = result.stdout + result.stderr
                return result.returncode == 0, output.strip()
            except subprocess.TimeoutExpired:
                return False, f"Timeout after {timeout}s"
            finally:
                try:
                    os.unlink(f.name)
                except Exception:
                    pass

    elif language in ("javascript", "typescript"):
        deps = _extract_js_dependencies(code)
        temp_dir = tempfile.mkdtemp()
        try:
            if deps:
                package_json_path = os.path.join(temp_dir, "package.json")
                pkg_data = {
                    "name": "temp-test",
                    "version": "1.0.0",
                    "dependencies": {dep: "latest" for dep in deps}
                }
                if language == "typescript":
                    pkg_data["dependencies"]["@types/node"] = "latest"

                with open(package_json_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(pkg_data))

                # Install dependencies locally with ignore-scripts, no-audit, no-fund
                subprocess.run(
                    [npm_cmd, "install", "--ignore-scripts", "--no-audit", "--no-fund"],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

            if language == "typescript":
                tsconfig_path = os.path.join(temp_dir, "tsconfig.json")
                with open(tsconfig_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "compilerOptions": {
                            "target": "es2020",
                            "module": "commonjs",
                            "esModuleInterop": True,
                            "skipLibCheck": True
                        }
                    }))
                ts_file = os.path.join(temp_dir, "code.ts")
                with open(ts_file, "w", encoding="utf-8") as f:
                    f.write(code)

                # Run tsc compiler
                compile_res = subprocess.run(
                    [npx_cmd, "-y", "tsc"],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                if compile_res.returncode != 0:
                    output = compile_res.stdout + compile_res.stderr
                    return False, f"TypeScript Compilation Failed:\n{output.strip()}"
                js_file = os.path.join(temp_dir, "code.js")
            else:
                js_file = os.path.join(temp_dir, "code.js")
                with open(js_file, "w", encoding="utf-8") as f:
                    f.write(code)

            result = subprocess.run(
                ["node", js_file],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            output = result.stdout + result.stderr
            return result.returncode == 0, output.strip()
        except subprocess.TimeoutExpired:
            return False, f"Timeout after {timeout}s"
        finally:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
    return False, f"Unsupported language: {language}"


def _run_in_container(code: str, timeout: int, language: str) -> tuple[bool, str]:
    """Execute code securely inside a container."""
    engine = config.SANDBOX_ENGINE
    if not shutil.which(engine):
        raise RuntimeError(
            f"Sandbox engine '{engine}' is not available. "
            f"Please verify it is installed and on the system PATH."
        )

    container_name = f"whetstone-sandbox-{uuid.uuid4().hex[:12]}"
    cmd = [engine, "run", "--name", container_name, "--rm", "-i"]

    if not config.SANDBOX_NETWORK_ACCESS:
        cmd.extend(["--network", "none"])
    if config.SANDBOX_MEMORY_LIMIT:
        cmd.extend(["-m", str(config.SANDBOX_MEMORY_LIMIT)])
    if config.SANDBOX_CPU_LIMIT:
        cmd.extend(["--cpus", str(config.SANDBOX_CPU_LIMIT)])

    # Standard security hardening flags
    cmd.extend([
        "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", "32",
        "--user", "65534:65534",
    ])

    if language in ("python", "python_module", "python_package"):
        image = config.SANDBOX_IMAGE
        executable = ["python"]
        payload = code
    elif language in ("javascript", "typescript"):
        image = "node:20-slim"
        deps = _extract_js_dependencies(code)

        # We pass a bootstrap NodeJS script via node -e, and feed JSON payload on stdin
        bootstrap = """
const fs = require('fs');
const { execSync } = require('child_process');
const path = require('path');
let inputData = '';
process.stdin.on('data', chunk => { inputData += chunk; });
process.stdin.on('end', () => {
    try {
        const payload = JSON.parse(inputData);
        const workDir = '/tmp/sandbox';
        fs.mkdirSync(workDir, { recursive: true });
        if (payload.deps && payload.deps.length > 0) {
            const pkg = {
                name: "temp-test",
                version: "1.0.0",
                dependencies: {}
            };
            payload.deps.forEach(d => { pkg.dependencies[d] = "latest"; });
            if (payload.language === "typescript") {
                pkg.dependencies["@types/node"] = "latest";
            }
            fs.writeFileSync(path.join(workDir, 'package.json'), JSON.stringify(pkg));
            execSync(
                'npm install --ignore-scripts --no-audit --no-fund',
                { cwd: workDir, stdio: 'inherit' }
            );
        }
        if (payload.language === "typescript") {
            fs.writeFileSync(path.join(workDir, 'tsconfig.json'), JSON.stringify({
                compilerOptions: {
                    target: "es2020",
                    module: "commonjs",
                    esModuleInterop: true,
                    skipLibCheck: true
                }
            }));
            fs.writeFileSync(path.join(workDir, 'code.ts'), payload.code);
            execSync(
                'npx --yes typescript tsc --project ' +
                path.join(workDir, 'tsconfig.json') +
                ' ' + path.join(workDir, 'code.ts'),
                { cwd: workDir, stdio: 'inherit' }
            );
            require(path.join(workDir, 'code.js'));
        } else {
            fs.writeFileSync(path.join(workDir, 'code.js'), payload.code);
            require(path.join(workDir, 'code.js'));
        }
    } catch (e) {
        console.error(e);
        process.exit(1);
    }
});
"""
        executable = ["node", "-e", bootstrap]
        payload = json.dumps({
            "code": code,
            "deps": deps,
            "language": language
        })
    else:
        return False, f"Unsupported language: {language}"

    cmd.extend([image] + executable)

    try:
        result = subprocess.run(
            cmd,
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Check for container engine/daemon setup issues
        daemon_errors = [
            "cannot connect to the docker daemon",
            "is the docker daemon running",
            "connecting to the service failed",
            "cannot connect to the podman socket",
            "podman service",
            "daemon unreachable",
        ]
        stderr_lower = result.stderr.lower()
        is_daemon_error = any(msg in stderr_lower for msg in daemon_errors)

        if result.returncode == 125 or is_daemon_error:
            raise RuntimeError(
                f"Sandbox engine '{engine}' failed to execute. "
                f"Please verify that the daemon is running. "
                f"Details: {result.stderr.strip()}"
            )

        output = result.stdout + result.stderr
        # Returncode 137 typically indicates container was OOM killed
        if result.returncode == 137:
            return False, "Execution failed: Memory limit exceeded (OOM)"
        return result.returncode == 0, output.strip()
    except subprocess.TimeoutExpired:
        subprocess.run([engine, "rm", "-f", container_name], capture_output=True)
        return False, f"Timeout after {timeout}s"
    except KeyboardInterrupt:
        subprocess.run([engine, "rm", "-f", container_name], capture_output=True)
        raise
    except OSError as e:
        subprocess.run([engine, "rm", "-f", container_name], capture_output=True)
        raise RuntimeError(f"Failed to execute container: {e}") from e
