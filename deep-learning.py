import sc2
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, \
    GATEWAY, CYBERNETICSCORE, STARGATE, VOIDRAY, \
    OBSERVER, ROBOTICSFACILITY
import random
import cv2
import numpy as np
import time
import keras
import math

HEADLESS = True


# Protoss bot class
class ProtossBot(sc2.BotAI):
    def __init__(self, use_model=False):
        # self.ITERATIONS_PER_MINUTE = 165
        self.MAX_WORKERS = 65
        self.do_something_after = 0
        self.train_data = []
        self.use_model = use_model

        # { UNIT_TAG : LOCATION }
        self.scouts_and_spots = {}

        if self.use_model:
            print("using model")
            self.model = keras.models.load_model("models/BasicCNN-30-epochs-0.0001-LR-4.2")

    def on_end(self, game_result):
        print('--- on_end called ---')
        print(game_result)

        if game_result == Result.Victory:
            np.save("train_data_new/{}.npy".format(str(int(time.time()))), np.array(self.train_data))

    # Execute at every step
    async def on_step(self, iteration):
        # self.iteration = iteration
        self.time = (self.status.game_loop / 22.4) / 60
        await self.build_scout()
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

    async def build_scout(self):
        if len(self.units(OBSERVER)) < math.floor(self.time / 3):
            for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
                print(len(self.units(OBSERVER)), self.time / 3)
                if self.can_afford(OBSERVER) and self.supply_left > 0:
                    await self.do(rf.train(OBSERVER))

    def random_location_variance(self, enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x += random.randrange(-5, 5)
        y += random.randrange(-5, 5)

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x, y)))
        return go_to

    async def scout(self):
        # { DISTANCE_TO_ENEMY_START : EXPANSIONLOC }
        self.expand_dis_dir = {}
        for el in self.expansion_locations:
            distance_to_enemy_start = el.distance_to(self.enemy_start_location[0])
            self.expand_dis_dir[distance_to_enemy_start] = el
        self.ordered_exp_distances = sorted(k for k in self.expand_dis_dir)

        existing_ids = [unit.tag for unit in self.units]
        to_be_removed = []
        for noted_scout in self.scouts_and_spots:
            if noted_scout not in existing_ids:
                to_be_removed.append(noted_scout)

        for scout in to_be_removed:
            del self.scouts_and_spots[scout]

        if len(self.units(ROBOTICSFACILITY).ready) == 0:
            unit_type = PROBE
            unit_limit = 1
        else:
            unit_type = OBSERVER
            unit_limit = 15

        assign_scout = True

        if unit_type == PROBE:
            for unit in self.units(PROBE):
                if unit.tag in self.scouts_and_spots:
                    assign_scout = False

        if assign_scout:
            if len(self.units(unit_type).idle) > 0:
                for obs in self.units(unit_type).idle[:unit_limit]:
                    if obs.tag not in self.scouts_and_spots:
                        for dist in self.ordered_exp_distances:
                            try:
                                location = self.expand_dis_dir[
                                    dist]  # next(value for key, value in self.expand_dis_dir.items if key == dist)
                                active_locations = [self.scouts_and_spots[k] for k in self.scouts_and_spots]

                                if location not in active_locations:
                                    if unit_type == PROBE:
                                        for unit in self.units(PROBE):
                                            if unit.tag in self.scouts_and_spots:
                                                continue
                                    await self.do(obs.move(location))
                                    self.scouts_and_spots[obs.tag] = location
                                    break


                            except Exception as e:
                                # print(str(e))
                                pass

        for obs in self.units(unit_type):
            if obs.tag in self.scouts_and_spots:
                if obs in [probe for probe in self.units(PROBE)]:
                    await self.do(obs.move(self.random_location_variance(self.scouts_and_spots[obs.tag])))

    async def intel(self):
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        # UNIT: [SIZE, (BGR Color)]
        draw_dict = {
            NEXUS: [15, (0, 255, 0)],
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
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 12, (0, 0, 255), -1)

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

        # Draw minerals and vespene
        line_max = 50
        mineral_ratio = self.minerals / 1500
        if mineral_ratio > 1.0:
            mineral_ratio = 1.0

        vespene_ratio = self.vespene / 1500
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0

        population_ratio = self.supply_left / self.supply_cap
        if population_ratio > 1.0:
            population_ratio = 1.0

        plausible_supply = self.supply_cap / 200.0

        military_weight = len(self.units(VOIDRAY)) / (self.supply_cap - self.supply_left)
        if military_weight > 1.0:
            military_weight = 1.0

        cv2.line(game_data, (0, 19), (int(line_max * military_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
        cv2.line(game_data, (0, 15), (int(line_max * plausible_supply), 15), (220, 200, 200),
                 3)  # plausible supply (supply/200.0)
        cv2.line(game_data, (0, 11), (int(line_max * population_ratio), 11), (150, 150, 150),
                 3)  # population ratio (supply_left/supply)
        cv2.line(game_data, (0, 7), (int(line_max * vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
        cv2.line(game_data, (0, 3), (int(line_max * mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500

        # Flip horizontally to make our final fix in visual representation
        self.flipped = cv2.flip(game_data, 0)

        if not HEADLESS:
            resize = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
            cv2.imshow('Intel', resize)
            cv2.waitKey(1)

    # Build workers (PROBE sc2.constants)
    async def build_workers(self):
        if len(self.units(NEXUS)) * 16 > len(self.units(PROBE)):
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
        if self.units(NEXUS).amount < (self.time / 2) and self.can_afford(NEXUS):
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
                if len(self.units(STARGATE)) < self.time:
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
        if len(self.units(VOIDRAY).idle) > 0:
            target = False
            if self.time > self.do_something_after:

                if self.use_model:
                    # Use the model
                    prediction = self.model.predict([self.flipped.reshape([-1, 176, 200, 3])])
                    choice = np.argmax(prediction[0])

                    choice_dict = {
                        0: "No Attack!",
                        1: "Attack close to our nexus!",
                        2: "Attack Enemy Structure!",
                        3: "Attack Enemy Start!"
                    }

                    print("Choice #{}:{}".format(choice, choice_dict[choice]))

                else:
                    choice = random.randrange(0, 4)

                if choice == 0:
                    # no attack
                    wait = random.randrange(7, 100) / 100
                    self.do_something_after = self.time + wait

                elif choice == 1:
                    # attack_unit_closest_nexus
                    if len(self.known_enemy_units) > 0:
                        target = self.known_enemy_units.closest_to(random.choice(self.units(NEXUS)))

                elif choice == 2:
                    # attack enemy structures
                    if len(self.known_enemy_structures) > 0:
                        target = random.choice(self.known_enemy_structures)

                elif choice == 3:
                    # attack_enemy_start
                    target = self.enemy_start_locations[0]

                if target:
                    for vr in self.units(VOIDRAY).idle:
                        await self.do(vr.attack(target))
                y = np.zeros(4)
                y[choice] = 1
                self.train_data.append([y, self.flipped])


# Run the game
run_game(maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, ProtossBot(use_model=True)),
    Computer(Race.Terran, Difficulty.Easy)
], realtime=False)
