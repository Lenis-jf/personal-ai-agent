"""Bucle interactivo (REPL) para chatear en tiempo real con el Asistente Personal."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import Agent
from llm.local_llm_mlx import LocalLLMMLX

BANNER = """
==================================================
           PERSONAL AI AGENT - TERMINAL REPL
   Escribe 'salir', 'exit' o 'q' para terminar.
   Escribe 'clear' para reiniciar la memoria.
   Escribe 'debug' para activar/desactivar la traza.
==================================================
"""


def main() -> None:
    print("Cargando modelos en memoria unificada (Apple Silicon)...")
    llm = LocalLLMMLX(model_path="mlx-community/gemma-2-9b-it-4bit")
    agent = Agent(llm=llm)
    
    print(BANNER)

    history: list[dict[str, str]] = []
    debug_mode = False

    while True:
        try:
            user_input = input("\nJuan > ").strip()

            if not user_input:
                continue

            if user_input.lower() in {"salir", "exit", "q"}:
                print("\n¡Nos pillamos, Juan! Sesión cerrada.")
                break

            if user_input.lower() == "clear":
                history.clear()
                print("\n[Memoria de sesión reiniciada]")
                continue

            if user_input.lower() == "debug":
                debug_mode = not debug_mode
                estado = "ACTIVADO" if debug_mode else "DESACTIVADO"
                print(f"\n[Modo debug {estado}]")
                continue

            # Procesar solicitud a través del agente
            resultado = agent.run(user_input, history=history)

            # Mostrar traza técnica si el modo debug está activo
            if debug_mode:
                dec = resultado["decision"]
                print(f"\n[DEBUG] Skill: {dec.skill} | Complejidad: {dec.complexity} | Tools: {list(dec.tools)}")
                if resultado["tool_executed"]:
                    print(f"[DEBUG] Herramienta llamada: {resultado['tool_executed']}")
                    print(f"[DEBUG] Salida raw:\n{resultado['tool_output']}")

            # Imprimir respuesta al usuario
            print(f"\nAsistente > {resultado['response']}")

            # Guardar en memoria de sesión
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": resultado["response"]})

        except KeyboardInterrupt:
            print("\n\nSesión interrumpida por teclado. ¡Hasta luego!")
            break
        except Exception as err:
            print(f"\n[Error durante la ejecución]: {err}")


if __name__ == "__main__":
    main()