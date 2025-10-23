from agent import AgarAgent
from pynput import keyboard
import threading


def on_press(key):
    print(f"Detected {key}")
    if key == keyboard.Key.num_lock:
        agar_agent.agent_running = not agar_agent.agent_running
        if agar_agent.agent_running:
            print("Agar Agent running")
        else:
            print("Agar Agent stopping")
    elif key == keyboard.Key.esc:
        print("Agar Agent exiting")
        agar_agent.agent_running = False
        agar_agent.program_running = False
        listener.stop()


if __name__ == "__main__":
    agar_agent = AgarAgent(0.25)
    agent_thread = threading.Thread(target=agar_agent.run)
    agent_thread.start()

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
