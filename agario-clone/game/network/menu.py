from functools import partial
import pygame_menu
from pygame_menu import themes, widgets, events


class MyMenu:
    MENU_VERTICAL_MARGIN = 25
    HELP = (
        'Control keys',
        '',
        'Press "W" key to shoot',
        'Press "Space" to split into two parts'
    )

    ABOUT = (
        'Python implementation of game agar.io',
        'Authors: @alexandr-gnrk & @vanyabondar',
        '',
        'Github: github.com/alexandr-gnrk/agario'
    )

    def __init__(self, width, height):
        self.width = width
        self.height = height

        # --- Create theme ---
        self.theme = themes.THEME_DARK.copy()
        self.theme.title_bar_style = widgets.MENUBAR_STYLE_ADAPTIVE
        self.theme.widget_selection_effect = widgets.NoneSelection()

        # --- Create menus ---
        self.start_menu = pygame_menu.Menu(
            'Start', width, height,
            theme=self.theme,
            onclose=events.RESET
        )

        self.help_menu = pygame_menu.Menu(
            'Help', width, height,
            theme=self.theme,
            onclose=events.RESET
        )

        self.about_menu = pygame_menu.Menu(
            'About', width, height,
            theme=self.theme,
            onclose=events.RESET
        )

        self.main_menu = pygame_menu.Menu(
            'Main menu', width, height,
            theme=self.theme,
            onclose=events.EXIT
        )

        self.__init_menus_with_widgets()

    def __init_menus_with_widgets(self):
        # Initialize start menu
        self.update_start_menu(lambda *args: None)

        # Help menu
        for line in MyMenu.HELP:
            self.help_menu.add.label(line)
        self.help_menu.add.vertical_margin(MyMenu.MENU_VERTICAL_MARGIN)
        self.help_menu.add.button('Back', events.RESET)

        # About menu
        for line in MyMenu.ABOUT:
            self.about_menu.add.label(line)
        self.about_menu.add.vertical_margin(MyMenu.MENU_VERTICAL_MARGIN)
        self.about_menu.add.button('Back', events.RESET)

        # Main menu
        self.main_menu.add.button(self.start_menu.get_title(), self.start_menu)
        self.main_menu.add.button(self.help_menu.get_title(), self.help_menu)
        self.main_menu.add.button(self.about_menu.get_title(), self.about_menu)
        self.main_menu.add.button('Exit', events.EXIT)

    def update_start_menu(self, connect_callback):
        self.start_menu.clear()

        self.start_menu.add.text_input(
            '        Nickname:   ',
            default='user',
            maxwidth=14,
            textinput_id='nick',
            input_underline='_'
        )

        self.start_menu.add.text_input(
            'Server address:   ',
            default='localhost:9999',
            maxwidth=14,
            textinput_id='addr',
            input_underline='_'
        )

        self.start_menu.add.vertical_margin(MyMenu.MENU_VERTICAL_MARGIN)

        self.start_menu.add.button(
            'Connect',
            partial(connect_callback, self.start_menu.get_input_data)
        )

        self.start_menu.add.button('Back', events.RESET)

    def get_main_menu(self):
        return self.main_menu
