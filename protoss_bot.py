import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR


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
        # Build assimilators
        await self.build_assimilators()
        # Expand
        await self.expand()

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
        # Supply capacity left provided by bases and pylon building
        if self.supply_left < 5 and not self.already_pending(PYLON):
            # Get nexus
            nexuses = self.units(NEXUS).ready
            # Check nexus exists
            if nexuses.exists:
                # If can afford a PYLON
                if self.can_afford(PYLON):
                    # Build a PYLON near the first nexus
                    await self.build(PYLON, near=nexuses.first)

    # Build assimilators (ASSIMILATOR sc2.constants)
    async def build_assimilators(self):
        # For each nexus ready
        for nexus in self.units(NEXUS).ready:
            # Geysers near the nexus
            vaspenes = self.state.vespene_geyser.closer_than(10.0, nexus)
            for vaspene in vaspenes:
                # If still can not build the assimilator
                if not self.can_afford(ASSIMILATOR):
                    break
                # Get workers
                worker = self.select_build_worker(vaspene.position)
                # If there are no workers
                if worker is None:
                    break
                # If there are no assimilators near the geysers
                if not self.units(ASSIMILATOR).closer_than(1.0, vaspene).exists:
                    # Build assimilator
                    await self.do(worker.build(ASSIMILATOR, vaspene))

    # Expand the empire
    async def expand(self):
        if self.units(NEXUS).amount < 3 and self.can_afford(NEXUS):
            await self.expand_now()


# Run the game
run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, ProtossBot()),
    Computer(Race.Terran, Difficulty.Easy)
], realtime=False)
