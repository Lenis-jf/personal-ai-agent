"""Bucle interactivo (REPL) con historial estructurado, logs acumulativos por sesión y árbol en vivo."""
import os
import sys
import threading
import select
import termios
import time
from typing import Any
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import Agent
from llm.local_llm_mlx import LocalLLMMLX

# Estilos ANSI
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
RESET = "\033[0m"


class WorkflowDisplay:
    """Monitor de flujo en árbol con cronómetro en vivo y tiempos finales persistentes en cada rama."""
    PULSE = ["✦", "✧", "✦", "·"]

    def __init__(self) -> None:
        self._message = ""
        self._running = False
        self._step_start = time.perf_counter()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.in_thought = False
        self.in_answer = False
        self._thought_elapsed: float | None = None

    def update_status(self, message: str) -> None:
        self.close_thought_if_open()
        if not message:
            self.stop_active_line()
            return

        with self._lock:
            self._message = message
            self._step_start = time.perf_counter()
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        idx = 0
        while True:
            with self._lock:
                if not self._running:
                    break
                msg = self._message
                elapsed = time.perf_counter() - self._step_start
            icon = self.PULSE[idx % len(self.PULSE)]
            sys.stdout.write(f"\r  {CYAN}{icon}{RESET} {DIM}{msg} ({elapsed:.1f}s){RESET}\033[K")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.1)

    def stop_active_line(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            thread = self._thread
            self._thread = None

        if thread and thread.is_alive():
            thread.join()

        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def log_event(self, kind: str, text: str) -> None:
        self.close_thought_if_open()
        self.stop_active_line()
        icon = "🧭" if kind == "route" else "⚡"
        sys.stdout.write(f"  {DIM}├─{RESET} {icon} {DIM}{text}{RESET}\n")
        sys.stdout.flush()

    def stream_thought(self, token: str) -> None:
        if not self.in_thought:
            self.stop_active_line()
            sys.stdout.write(f"  {DIM}├─{RESET} 💭 {DIM}{ITALIC}")
            self.in_thought = True
            self._thought_elapsed = None
        clean_tok = token.replace("\n", " ")
        sys.stdout.write(clean_tok)
        sys.stdout.flush()

    def mark_thought_done(self, elapsed_sec: float) -> None:
        self._thought_elapsed = elapsed_sec
        self.close_thought_if_open()

    def close_thought_if_open(self) -> None:
        if self.in_thought:
            time_badge = f" ({self._thought_elapsed:.1f}s)" if self._thought_elapsed is not None else ""
            sys.stdout.write(f"{RESET}{DIM}{time_badge}{RESET}\n")
            sys.stdout.flush()
            self.in_thought = False
            self._thought_elapsed = None

    def stream_answer(self, token: str) -> None:
        if not self.in_answer:
            self.close_thought_if_open()
            self.stop_active_line()
            sys.stdout.write(f"\n{GREEN}{BOLD}Asistente{RESET} > ")
            self.in_answer = True
        sys.stdout.write(token)
        sys.stdout.flush()


def render_banner(debug_mode: bool) -> None:
    status_color = GREEN if debug_mode else YELLOW
    status_text = "ACTIVADO" if debug_mode else "DESACTIVADO"
    print(f"""
{CYAN}{BOLD}╔════════════════════════════════════════════════════════════════════════════╗
║                   PERSONAL AI AGENT — LOCAL macOS M4                       ║
╚════════════════════════════════════════════════════════════════════════════╝{RESET}
  {DIM}• Comandos:{RESET} {BOLD}'salir'{RESET} para cerrar | {BOLD}'clear'{RESET} para memoria | {BOLD}'debug'{RESET} para telemetría
  {DIM}• Estado Debug:{RESET} {status_color}{BOLD}[{status_text}]{RESET}
  {DIM}• Logs de sesión:{RESET} {DIM}logs/session_trace.log | logs/session_history.md{RESET}
""")


def print_debug_explanation(active: bool) -> None:
    if active:
        print(f"\n{MAGENTA}{BOLD}>>> [MODO DEBUG: ACTIVADO] <<<{RESET}")
        print(f"{DIM}Verás el contexto inyectado, skill cargada, tiempos de Prefill vs. Generación y salidas crudas.{RESET}\n")
    else:
        print(f"\n{YELLOW}{BOLD}>>> [MODO DEBUG: DESACTIVADO] <<<{RESET}\n")


def print_profiling_card(data: dict) -> None:
    prof = data["profiling"]
    dec = data["decision"]
    trace = data["tools_trace"]
    active_tools = sorted(list(data.get("active_tools", dec.tools)))
    loaded_skill = prof.get("loaded_skill", f"{dec.skill}.md")
    mem_stats = prof.get("memory_stats", {"full_memories": 0, "compacted_memories": 0})

    print(f"\n{BLUE}┌── {BOLD}TELEMETRÍA DETALLADA DEL PIPELINE{RESET}{BLUE} ──────────────────────────────┐{RESET}")

    injected_history = prof.get("injected_history", "").strip()
    if injected_history:
        print(
            f"{BLUE}│{RESET} {BOLD}Contexto de sesión disponible para evaluación autónoma{RESET} "
            f"{DIM}(Memorias completas: {mem_stats['full_memories']} | Compactadas: {mem_stats['compacted_memories']}){RESET}:"
        )
        for line in injected_history.split("\n"):
            print(f"{BLUE}│{RESET}   {DIM}{line}{RESET}")
        print(f"{BLUE}│{RESET} {DIM}{'-'*60}{RESET}")

    print(
        f"{BLUE}│{RESET} {BOLD}Skill activa (Router):{RESET} {CYAN}{BOLD}{dec.skill}{RESET} "
        f"({GREEN}{loaded_skill}{RESET} | complejidad: {dec.complexity}) en {YELLOW}{prof.get('route_time_sec', 0):.2f}s{RESET}"
    )
    print(f"{BLUE}│{RESET} {BOLD}Tools activas en runtime:{RESET} {active_tools}")

    for step in prof.get("steps", []):
        s_num = step["step"]
        ttft = step.get("ttft_sec", 0.0)
        dec_sec = step.get("decode_sec", 0.0)
        tps = step.get("tps", 0.0)
        toks = step.get("tokens", 0)

        if step["type"] == "tool_call":
            print(
                f"{BLUE}│{RESET} {BOLD}Inferencia {s_num} [Tool: {step['action']}]:{RESET} "
                f"Prefill: {ttft:.2f}s | Gen: {dec_sec:.2f}s ({CYAN}{tps} tok/s{RESET}, {toks} tok) | "
                f"Tool: {GREEN}{step['tool_exec_sec']:.3f}s{RESET}"
            )
        elif step["type"] == "circuit_breaker":
            print(f"{BLUE}│{RESET} {BOLD}Inferencia {s_num} [Rompecircuitos]:{RESET} {YELLOW}Bucle repetido bloqueado ({step['action']}){RESET}")
        else:
            print(
                f"{BLUE}│{RESET} {BOLD}Inferencia {s_num} [Respuesta Final]:{RESET} "
                f"Prefill: {ttft:.2f}s | Gen: {dec_sec:.2f}s ({CYAN}{tps} tok/s{RESET}, {toks} tok)"
            )

    if trace:
        tools_called = [t["tool"] for t in trace]
        print(f"{BLUE}│{RESET} {BOLD}Herramientas encadenadas:{RESET} {tools_called}")

    total_sec = prof.get("total_time_sec", 0)
    print(f"{BLUE}│{RESET} {BOLD}Tiempo total transcurrido:{RESET} {MAGENTA}{BOLD}{total_sec:.2f}s{RESET}")
    print(f"{BLUE}│{RESET} {DIM}Logs actualizados (append): logs/session_trace.log | logs/session_history.md{RESET}")
    print(f"{BLUE}└───────────────────────────────────────────────────────────────────┘{RESET}")


def read_clean_user_input(prompt_label: str) -> str:
    """
    1. Descarta del descriptor del kernel (fd) cualquier tecla pulsada mientras el agente estaba ocupado.
    2. Lee la primera línea con input().
    3. Conmuta temporalmente a modo no-canónico (~ICANON) por 80ms para capturar cualquier segunda o
       tercera línea pegada con Cmd+V (incluso si la última línea pegada aún no tiene '\\n' final),
       y si faltaba presionar Enter en la última línea pegada, espera el Enter del usuario.
    """
    fd: int | None = None
    try:
        fd = sys.stdin.fileno()
        termios.tcflush(fd, termios.TCIFLUSH)
    except Exception:
        pass

    first_line = input(prompt_label)
    lines = [first_line]

    if fd is not None:
        extra_bytes = bytearray()
        old_attrs = None
        try:
            old_attrs = termios.tcgetattr(fd)
            new_attrs = termios.tcgetattr(fd)
            # Desactivar ICANON para ver bytes pegados que aún no tienen salto de línea (\n)
            new_attrs[3] = new_attrs[3] & ~termios.ICANON
            new_attrs[6][termios.VMIN] = 0
            new_attrs[6][termios.VTIME] = 1  # 100ms timeout de lectura no bloqueante
            termios.tcsetattr(fd, termios.TCSANOW, new_attrs)

            while True:
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                extra_bytes.extend(chunk)
        except Exception:
            pass
        finally:
            if old_attrs is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSANOW, old_attrs)
                except Exception:
                    pass

        if extra_bytes:
            needs_enter = not extra_bytes.endswith((b"\n", b"\r"))
            extra_text = extra_bytes.decode("utf-8", errors="replace")
            extra_lines = [line.rstrip("\r") for line in extra_text.splitlines()]
            if needs_enter:
                # La última línea pegada no tenía salto de línea final; esperar a que el usuario pulse Enter
                try:
                    tail_input = input("")
                    if extra_lines:
                        extra_lines[-1] += tail_input
                    else:
                        extra_lines.append(tail_input)
                except EOFError:
                    pass
            lines.extend(extra_lines)

    return "\n".join(lines).strip()


def main() -> None:
    print(f"{DIM}Cargando modelos en memoria unificada de macOS...{RESET}")
    llm = LocalLLMMLX(model_path="mlx-community/gemma-4-e4b-it-4bit")
    agent = Agent(llm=llm)

    debug_mode = False
    render_banner(debug_mode)

    history: list[dict[str, Any]] = []

    while True:
        ui = WorkflowDisplay()
        try:
            user_input = read_clean_user_input(f"\n{BOLD}Juan{RESET} > ")

            if not user_input:
                continue

            if user_input.lower() in {"salir", "exit", "q"}:
                print(f"\n{CYAN}¡Listo Juan, sesión cerrada! Nos pillamos.{RESET}")
                break

            if user_input.lower() == "clear":
                history.clear()
                agent.reset_session_logs()
                print(f"\n{YELLOW}[Memoria y logs de sesión reiniciados]{RESET}")
                continue

            if user_input.lower() == "debug":
                debug_mode = not debug_mode
                print_debug_explanation(debug_mode)
                continue

            print()

            resultado = agent.run(
                user_input,
                history=history,
                on_token=ui.stream_answer,
                on_status=ui.update_status,
                on_thought_token=ui.stream_thought,
                on_thought_done=ui.mark_thought_done,
                on_event=ui.log_event,
            )

            ui.close_thought_if_open()
            ui.stop_active_line()

            final_ans = resultado["response"].strip()

            if not ui.in_answer:
                sys.stdout.write(f"\n{GREEN}{BOLD}Asistente{RESET} > {final_ans}\n")
                sys.stdout.flush()
            else:
                sys.stdout.write("\n")
                sys.stdout.flush()

            if debug_mode:
                print_profiling_card(resultado)
                for trace_item in resultado["tools_trace"]:
                    preview = str(trace_item["output"]).strip()
                    if len(preview) > 300:
                        preview = preview[:300] + "..."
                    print(f"{DIM}[Salida {trace_item['tool']}]: {preview}{RESET}")

            stored_memories: list[dict[str, str]] = []
            for trace in resultado["tools_trace"]:
                tool_name = trace["tool"]
                out_str = str(trace["output"]).strip()
                if tool_name == "file_grep":
                    first_lines = "\n".join(out_str.splitlines()[:3])
                    stored_memories.append({"tool": "file_grep", "output": first_lines})
                else:
                    stored_memories.append({"tool": tool_name, "output": out_str})

            history.append({"role": "user", "content": user_input})
            history.append({
                "role": "assistant",
                "content": final_ans,
                "tool_memories": stored_memories,
            })

            prof = resultado.get("profiling", {})
            agent.append_turn_history_log(
                turn_num=prof.get("turn_num", len(history) // 2),
                user_message=user_input,
                final_response=final_ans,
                stored_memories=stored_memories,
                injected_history_snapshot=prof.get("injected_history", ""),
                mem_stats=prof.get("memory_stats", {}),
                total_history_messages=len(history),
            )

        except KeyboardInterrupt:
            ui.stop_active_line()
            print(f"\n\n{YELLOW}Sesión interrumpida. ¡Hasta luego!{RESET}")
            break
        except Exception as err:
            ui.stop_active_line()
            print(f"\n{YELLOW}[Error]: {err}{RESET}")


if __name__ == "__main__":
    main()