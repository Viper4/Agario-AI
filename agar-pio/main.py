from game import Game



if __name__ == "__main__":    
    g = Game()
    g.start_screen()
    while g.running:
        g.run()
    g.quit()
