import sc2
from sc2 import run_game, maps, Race, Difficulty, position
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, \
 GATEWAY, CYBERNETICSCORE, STARGATE, VOIDRAY, \
 OBSERVER, ROBOTICSFACILITY
import random
import cv2
import numpy as np


# Protoss bot class
class ProtossBot(sc2.BotAI):
    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 165
        self.MAX_WORKERS = 65

    # Execute at every step
    async def on_step(self, iteration):
        self.iteration = iteration
        await self.scout()

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
        # Deep learning
        await self.intel()

    def random_location_variance(self, enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x += ((random.randrange(-20, 20))/100) * enemy_start_location[0]
        y += ((random.randrange(-20, 20))/100) * enemy_start_location[1]

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x,y)))
        return go_to

    async def scout(self):
        if len(self.units(OBSERVER)) > 0:
            scout = self.units(OBSERVER)[0]

            if scout.is_idle:
                enemy_location = self.enemy_start_locations[0]
                move_to = self.random_location_variance(enemy_location)
                await self.do(scout.move(move_to))

            else:
                for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
                    if self.can_afford(OBSERVER) and self.supply_left > 0:
                        await self.do(rf.train(OBSERVER))

    async def intel(self):
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)
        
        # UNIT: [SIZE, (BGR Color)]
        draw_dict = {
            NEXUS: [12, (0, 255, 0)],
            PYLON: [3, (20, 235, 0)],
            PROBE: [1, (55, 200, 0)],
            ASSIMILATOR: [2, (35, 200, 0)],
            GATEWAY: [3, (200, 100, 0)],
            CYBERNETICSCORE: [3, (150, 150, 0)],
            STARGATE: [5, (255, 0, 0)],
            VOIDRAY: [3, (255, 100, 0)],
            ROBOTICSFACILITY: [5, (215, 155, 0)],
        }

        for unit_type in draw_dict:
            for unit in self.units(unit_type).ready:
                pos = unit.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), draw_dict[unit_type][0], draw_dict[unit_type][1], -1)

        main_base_names = ['nexus', 'commandcenter', 'hatchery']
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() not in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212), -1)

        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)

        for enemy_unit in self.known_enemy_units:
            if not enemy_unit.is_structure:
                worker_names = ['probe', 'scv', 'drone']

                pos = enemy_unit.position
                if enemy_unit.name.lower() in worker_names:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
                else:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 155), -1)

        for observer in self.units(OBSERVER).ready:
            pos = observer.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (255, 255, 255), -1)

        # Flip horizontally to make our final fix in visual representation
        flipped = cv2.flip(game_data, 0)
        resize = cv2.resize(flipped, dsize=None, fx=2, fy=2)
        
        cv2.imshow('Intel', resize)
        cv2.waitKey(1)

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
            elif len(self.units(GATEWAY)) < 1:
                if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                    await self.build(GATEWAY, near=pylon)
            # Build a Robotics Facility
            if self.units(CYBERNETICSCORE).ready.exists:
                if len(self.units(ROBOTICSFACILITY)) < 1:
                    if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
                        await self.build(ROBOTICSFACILITY, near=pylon)
            # Build a Stargate
            if self.units(CYBERNETICSCORE).ready.exists:
                if len(self.units(STARGATE)) < (self.iteration / self.ITERATIONS_PER_MINUTE):
                    if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                        await self.build(STARGATE, near=pylon)

    # Build offensive army (VOIDRAY sc2.constants)
    async def build_offensive_force(self):
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
    Computer(Race.Terran, Difficulty.Hard)
], realtime=False)
