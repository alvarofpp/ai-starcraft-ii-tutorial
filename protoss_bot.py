import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, \
 GATEWAY, CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY
import random
from examples.protoss.cannon_rush import CannonRushBot


# Protoss bot class
class ProtossBot(sc2.BotAI):
    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 165
        self.MAX_WORKERS = 65

    # Execute at every step
    async def on_step(self, iteration):
        self.iteration = iteration

        # Initially are 12 workers
        await self.distribute_workers()
        # Resources
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.expand()
        # Offensive force
        await self.offensive_force_buildings()
        await self.build_offensive_force()
        await self.attack()

    # Build workers (PROBE sc2.constants)
    async def build_workers(self):
        if len(self.units(NEXUS))*16 > len(self.units(PROBE)):
            if len(self.units(PROBE)) < self.MAX_WORKERS:
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
        # Max 3 nexus
        if self.units(NEXUS).amount < 3 and self.can_afford(NEXUS):
            await self.expand_now()

    # Build offensive force buildings (GATEWAY, CYBERNETICSCORE, STARGATE sc2.constants)
    async def offensive_force_buildings(self):
        if self.units(PYLON).ready.exists:
            # Get a random pylon
            pylon = self.units(PYLON).ready.random
            # Build a Cybernetics Core
            if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)
            # Build a Gateway
            elif len(self.units(GATEWAY)) < ((self.iteration / self.ITERATIONS_PER_MINUTE)/2):
                if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                    await self.build(GATEWAY, near=pylon)
            # Build a Stargate
            if self.units(CYBERNETICSCORE).ready.exists:
                if len(self.units(STARGATE)) < ((self.iteration / self.ITERATIONS_PER_MINUTE)/2):
                    if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                        await self.build(STARGATE, near=pylon)

    # Build offensive army (STALKER, VOIDRAY sc2.constants)
    async def build_offensive_force(self):
        # Gateway
        for gateway in self.units(GATEWAY).ready.noqueue:
            # Train stalker
            if not self.units(STALKER).amount > self.units(VOIDRAY).amount:
                if self.can_afford(STALKER) and self.supply_left > 0:
                    await self.do(gateway.train(STALKER))
        # Stargate
        for stargate in self.units(STARGATE).ready.noqueue:
            # Train voidray
            if self.can_afford(VOIDRAY) and self.supply_left > 0:
                await self.do(stargate.train(VOIDRAY))

    # Find location of the enemy (army, structure, etc)
    def find_target(self, state):
        # Army or common unit
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        # Structure
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        # First location known
        else:
            return self.enemy_start_locations[0]

    # Attack the enemy
    async def attack(self):
        # {UNIT: [n to fight, n to defend]}
        aggressive_units = {
            STALKER: [15, 5],
            VOIDRAY: [8, 3]
        }

        for UNIT in aggressive_units:
            # Attack the base
            if self.units(UNIT).amount > aggressive_units[UNIT][0] and self.units(UNIT).amount > aggressive_units[UNIT][1]:
                for unit in self.units(UNIT).idle:
                    await self.do(unit.attack(self.find_target(self.state)))

            # Attack the enemy army or common unit
            elif self.units(UNIT).amount > aggressive_units[UNIT][1]:
                if len(self.known_enemy_units) > 0:
                    for unit in self.units(UNIT).idle:
                        await self.do(unit.attack(random.choice(self.known_enemy_units)))


# Run the game
run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, ProtossBot()),
    Bot(Race.Protoss, CannonRushBot())
], realtime=False)
