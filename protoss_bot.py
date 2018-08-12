import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON


# General bot class
class ProtossBot(sc2.BotAI):

    # Execute at every step
    async def on_step(self, iteration):
        # Initially are 12 workers
        await self.distribute_workers()
        # Build more workers
        await self.build_workers()
        # Build pylons
        await self.build_pylons()

    # Build workers (PROBE sc2.constants)
    async def build_workers(self):
        # Nexus built and without production
        for nexus in self.units(NEXUS).ready.noqueue:
            # If can afford a PROBE
            if self.can_afford(PROBE):
                # Train the probe
                await self.do(nexus.train(PROBE))

    # Build pylons (PYLON sc2.constants)
    async def build_pylons(self):
        if self.supply_left < 5 and not self.already_pending(PYLON):
            # Get nexus
            nexuses = self.units(NEXUS).ready
            # Verify nexus exists
            if nexuses.exists:
                # If can afford a PYLON
                if self.can_afford(PYLON):
                    # Build a PYLON near of the first nexus
                    await self.build(PYLON, near=nexuses.first)


# Run the game
run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, ProtossBot()),
    Computer(Race.Terran, Difficulty.Easy)
], realtime=True)
