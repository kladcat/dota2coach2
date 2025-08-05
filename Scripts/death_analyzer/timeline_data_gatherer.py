### data_gatherer.py
import json
from collections import defaultdict
from math import hypot

NEARBY_RADIUS = 20
WINDOW_SECONDS = 10

class DeathSurvivalDataGatherer:
    def __init__(self, timeline, hero_name, hero_unit):
        self.timeline = timeline
        self.hero_name = hero_name
        self.hero_unit = hero_unit
        self.teammate_ids = []
        self.enemy_ids = []
        self.unit_to_name = {}
        self._identify_units()

    def _identify_units(self):
        hero_ids = set()
        for e in self.timeline:
            if e['type'] == "interval" and e['unit'].startswith("CDOTA_Unit_Hero_"):
                hero_ids.add(e['unit'])
                self.unit_to_name[e['unit']] = e.get("name", e['unit'].replace("CDOTA_Unit_", "npc_dota_"))
        hero_ids = sorted(hero_ids)
        self.enemy_ids = [hid for hid in hero_ids if hid != self.hero_unit][:5]
        self.teammate_ids = [hid for hid in hero_ids if hid != self.hero_unit and hid not in self.enemy_ids][:4]

    @staticmethod
    def detect_hero_name_from_events(timeline, unit_name):
        for e in timeline:
            if e.get("type") == "DOTA_ABILITY_LEVEL" and e.get("unit") == unit_name:
                ability = e.get("ability", "")
                print(f"🦠 Found ability '{ability}' for unit '{unit_name}'")
                if ability.startswith("npc_dota_hero_"):
                    result = ability.split("_ability")[0]
                    print(f"✅ Mapping: {unit_name} → {result}")
                    return result
        fallback = unit_name.replace("CDOTA_Unit_", "npc_dota_").lower()
        print(f"🧪 No ability found for {unit_name}, using fallback: {fallback}")
        return fallback

    def extract_features(self, event_time, verbose=False):

        self.hero_name = self.hero_name.lower()
        self.hero_name_variants = {self.hero_name, self.hero_name.replace("_", "")}

        window_start = event_time - WINDOW_SECONDS
        raw_buckets = defaultdict(lambda: {
            "player_pos": None,
            "player_damage_taken": 0,
            "player_died": False,
            "player_level": None,
            "player_hp_pct": None,
            "teammates": {},
            "enemies": {},
            "damage_events": []
        })

        incapacitation_periods = []
        active_incapacitations = {}

        def detect_incapacitated(e):
            inflictor = e.get("inflictor", "")
            return any(kw in inflictor.lower() for kw in ["stun", "fear", "bash", "silence", "hex", "gaze", "disable"])

        for e in self.timeline:
            t = e["time"]
            if t < window_start or t > event_time:
                continue
            bucket = raw_buckets[t]

#            if e['type'] == "DOTA_COMBATLOG_MODIFIER_ADD" and e.get("targetname", "").lower().replace("_", "") in self.hero_name_variants:
#                if detect_incapacitated(e):
#                    active_incapacitations[e.get("inflictor", "")] = t

#            if e['type'] == "DOTA_COMBATLOG_MODIFIER_REMOVE" and e.get("targetname", "").lower().replace("_", "") in self.hero_name_variants:
#                inflictor = e.get("inflictor", "")
#                if inflictor in active_incapacitations:
#                    start = active_incapacitations.pop(inflictor)
#                    incapacitation_periods.append((start, t))

            # Track incapacitation purely via stun_duration
            if e['type'] == "DOTA_COMBATLOG_MODIFIER_ADD" and e.get("targetname", "").lower().replace("_", "") in self.hero_name_variants:
                stun_duration = e.get("stun_duration")
                if isinstance(stun_duration, (int, float)) and stun_duration > 0:
                    start_time = t
                    end_time = t + int(stun_duration)
                    print(f"🛑 Incapacitation detected via '{e.get('inflictor', 'unknown')}' from t={t} to t={end_time} (duration: {stun_duration}s)")

                    incapacitation_periods.append((start_time, end_time))

            if e['type'] == "interval":
                self._process_interval(e, bucket)
            elif e['type'] == "DOTA_COMBATLOG_DAMAGE":
                self._process_damage(e, bucket, verbose)
            elif e['type'] == "DOTA_COMBATLOG_DEATH":
                self._process_death(e, bucket)

        print(f"🧠 Total incapacitation periods for {self.hero_name}: {len(incapacitation_periods)}")
        incapacitated_ticks = set(tick for start, end in incapacitation_periods for tick in range(start, end + 1))

        output = {}
        for t, v in sorted(raw_buckets.items()):
            pos = v["player_pos"]
            if not pos:
                continue
            output[str(float(t))] = self._construct_output(t, v, pos, incapacitated_ticks, verbose)

        #print(f"🛠️ Built feature window with {len(output)} seconds for {self.hero_name} (event @ {event_time})")

        return output

    def _process_interval(self, e, bucket):
        unit = e['unit']
        
        if unit == self.hero_unit:
            max_hp = e.get("maxHealth")
            hp = e.get("health", 0)
            if isinstance(max_hp, (int, float)) and max_hp > 0:
                hp_pct = round(hp / max_hp, 3)
            else:
                hp_pct = 0

            bucket["player_pos"] = (e['x'], e['y'])
            bucket["player_level"] = e.get("level")
            bucket["player_hp_pct"] = hp_pct

        elif unit in self.teammate_ids + self.enemy_ids:
            max_hp = e.get("maxHealth")
            hp = e.get("health", 0)
            if isinstance(max_hp, (int, float)) and max_hp > 0:
                hp_pct = round(hp / max_hp, 3)
            else:
                hp_pct = 0

            group = "teammates" if unit in self.teammate_ids else "enemies"
            bucket[group].setdefault(unit, {})
            bucket[group][unit].update({
                "pos": (e['x'], e['y']),
                "level": e.get("level"),
                "hp_pct": hp_pct
            })


    def _process_damage(self, e, bucket, verbose):
        tgt = e.get("targetname", "").lower().replace("_", "")
        if tgt in self.hero_name_variants:
            bucket["player_damage_taken"] += e['value']
        norm_name = "CDOTA_Unit_" + e['targetname'].replace("npc_dota_", "")
        if norm_name in self.teammate_ids + self.enemy_ids:
            group = "teammates" if norm_name in self.teammate_ids else "enemies"
            bucket[group].setdefault(norm_name, {}).setdefault("damage_taken", 0)
            bucket[group][norm_name]["damage_taken"] += e['value']
        if verbose and "creep" not in e.get("attackername", "") and "creep" not in e.get("targetname", ""):
            bucket["damage_events"].append({
                "attackername": e.get("attackername"),
                "sourcename": e.get("sourcename"),
                "value": e.get("value"),
                "inflictor": e.get("inflictor"),
                "targetname": e.get("targetname")
            })

    def _process_death(self, e, bucket):
        tgt = e.get("targetname", "").lower().replace("_", "")
        if tgt in self.hero_name_variants:
            bucket["player_died"] = True
        norm_name = "CDOTA_Unit_" + e['targetname'].replace("npc_dota_", "")
        if norm_name in self.teammate_ids + self.enemy_ids:
            group = "teammates" if norm_name in self.teammate_ids else "enemies"
            bucket[group].setdefault(norm_name, {})["died"] = True

    def _construct_output(self, t, v, pos, incapacitated_ticks, verbose):
        def gather_units(units):
            pos_list, dmg_list, died_list, lvl_list, hp_list = [], [], [], [], []
            for uid in units:
                data = v.get("teammates" if uid in self.teammate_ids else "enemies", {}).get(uid)
                if data and "pos" in data and self._distance(pos, data["pos"]) <= NEARBY_RADIUS:
                    pos_list.append(self._normalize(data["pos"], pos))
                    dmg_list.append(data.get("damage_taken", 0))
                    died_list.append(data.get("died", False))
                    lvl_list.append(data.get("level", 1))
                    hp_list.append(data.get("hp_pct", 0))
            return pos_list, dmg_list, died_list, lvl_list, hp_list

        t_pos, t_dmg, t_die, t_lvl, t_hp = gather_units(self.teammate_ids)
        e_pos, e_dmg, e_die, e_lvl, e_hp = gather_units(self.enemy_ids)

        print(f"⏱️ t={t}: player_incapacitated={float(t) in incapacitated_ticks}")


        return {
            "player_pos": [0.0, 0.0],
            "teammates_pos": t_pos,
            "enemies_pos": e_pos,
            "player_damage_taken": v["player_damage_taken"],
            "teammates_damage_taken": t_dmg,
            "enemies_damage_taken": e_dmg,
            "player_died": v["player_died"],
            "teammates_died": t_die,
            "enemies_died": e_die,
            "player_level": v.get("player_level", 1),
            "teammates_levels": t_lvl,
            "enemies_levels": e_lvl,
            "player_hp_pct": v.get("player_hp_pct", 0),
            "teammates_hp_pct": t_hp,
            "enemies_hp_pct": e_hp,
            "player_incapacitated": float(t) in incapacitated_ticks,
            "damage_events": v["damage_events"] if verbose else []
        }

    def _normalize(self, pos, origin):
        return [round(pos[0] - origin[0], 2), round(pos[1] - origin[1], 2)]

    def _distance(self, p1, p2):
        return hypot(p1[0] - p2[0], p1[1] - p2[1])