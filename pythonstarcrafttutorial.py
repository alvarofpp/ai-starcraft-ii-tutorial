import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer


# General bot class
class SentdeBot(sc2.BotAI):
    async def on_step(self, iteration):
        await self.distribute_workers()


# Run the game
run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, SentdeBot()),
    Computer(Race.Terran, Difficulty.Easy)
], realtime=True)
