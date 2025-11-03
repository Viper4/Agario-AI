from settings import WIDTH, HEIGHT 


class Camera:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.zoom = 0.5

    def centre(self,player):
        self.x = (WIDTH/2-(player.pos.x*self.zoom))
        self.y = (HEIGHT/2-(player.pos.y*self.zoom))
        