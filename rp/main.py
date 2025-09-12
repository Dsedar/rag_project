import os
import langroid as lr
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from tools import RagSearchTool
# ----------------------------
VLLM_API_BASE = "http://192.168.2.87:8000/v1"  # URL твоего vLLM
MODEL_NAME = "Tlite"                             # Имя твоей модели в vLLM
#DEBUG = True                                    # Режим отладки
PROMPTS_PATH = "src/prompts/"


# ----------------------------
llm_config = OpenAIGPTConfig(
    chat_model=MODEL_NAME,
    api_base=VLLM_API_BASE,
    use_chat_for_completion=True
    #temperature=0.75,
    #max_output_tokens=500,
    #timeout=60,
    #stream=True,  # Важно для потокового вывода
)

# ----------------------------

def load_prompt(file_path: str) -> str:
    """Загружает промпт из текстового файла"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Промпт-файл не найден: {file_path}")
    except Exception as e:
        raise Exception(f"Ошибка при чтении промпт-файла: {e}")

def create_agent(name, prompt):
    return lr.ChatAgent(
        config=lr.ChatAgentConfig(
            name=name,
            llm=llm_config,
            system_message=prompt,
            use_tools = True
        ),
    )

# ----------------------------
# Для демонстрации используем ОДНУ сессию
# В реальном приложении здесь будет менеджер сессий
GM_prompt = load_prompt(PROMPTS_PATH + "GM.txt")
GM_agent = create_agent(name="Game_master", 
                        prompt=GM_prompt)
GM_agent.enable_message(RagSearchTool)
GM_task = lr.Task(
    GM_agent,
    llm_delegate=False,
    single_round=False,
    interactive=True
)
# НОВАЯ ЛОГИКА
def run_game():
    print(">>> Начинаем игру...")
    GM_task.run("Начинаем")

    while True:
        user_input = input("\n[Игрок] ").strip()
        if user_input.lower() in ["выход", "quit", "exit"]:
            break

        # Явно указываем: сначала проверь знания
        result = GM_task.run(
            f"Игрок спрашивает: '{user_input}'. "
            f"Сначала используй rag_search, чтобы найти точную информацию. "
            f"Затем, на основе найденного, дай атмосферный ответ."
        )
        print(f"[ГМ] {result}")

if __name__ == "__main__":
    print("🎮 Запуск игры...")
    GM_task.run("Начинаем")