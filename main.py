from app.state import AppState
from app.trainer import TrainingController
from app.gui import GUI


def main():
    state   = AppState()
    trainer = TrainingController(state)
    gui     = GUI(state, trainer)
    gui.start()


if __name__ == '__main__':
    main()
