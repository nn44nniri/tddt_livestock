from __future__ import annotations
from pathlib import Path
import json, sqlite3
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS inner_sensor_log (
 id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, mode TEXT,
 outdoor_temp_c REAL, outdoor_rh_pct REAL, outdoor_wind_m_s REAL,
 indoor_temp_c REAL, indoor_rh_pct REAL, indoor_air_speed_m_s REAL,
 lct_c REAL, uct_c REAL, ventilation_group_pct REAL, heating_group_pct REAL,
 light_on INTEGER, electric_kw REAL, gas_kw REAL, reward REAL,
 climate_context_id TEXT, climate_context_name TEXT, mpc_score REAL,
 comfort_error REAL, energy_cost REAL, gas_penalty REAL, context_penalty REAL,
 guidance_penalty REAL, rl_prior_cost REAL, rl_state_key TEXT, rl_action_key TEXT,
 mpc_components_json TEXT
);
CREATE TABLE IF NOT EXISTS actuator_commands (
 id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, mode TEXT,
 ventilation_group_pct REAL, heating_group_pct REAL, light_on INTEGER,
 safety_status TEXT, command_json TEXT
);
CREATE TABLE IF NOT EXISTS outer_growth_state (
 id INTEGER PRIMARY KEY AUTOINCREMENT, fattening_day INTEGER, growth_date TEXT,
 tbw_kg REAL, feed_intake_kg_dm_day REAL, beef_production_kg REAL,
 heat_production REAL, diet INTEGER, breed INTEGER, status TEXT, raw_json TEXT,
 biological_phase TEXT, adg_kg_day REAL, feed_efficiency REAL, dominant_limitation TEXT,
 daily_climate_context_id TEXT
);
CREATE TABLE IF NOT EXISTS growth_climate_guidance (
 id INTEGER PRIMARY KEY AUTOINCREMENT, fattening_day INTEGER, updated_at TEXT,
 preferred_temp_low_c REAL, preferred_temp_high_c REAL,
 heat_production_feedback REAL, comfort_weight_bias REAL,
 gas_weight_bias REAL, energy_weight_bias REAL, status TEXT,
 best_reference_id TEXT, reference_score REAL, confidence REAL, ventilation_bias REAL, heating_bias REAL, diet_phase_guidance TEXT,
 biological_phase TEXT, climate_context_id TEXT, growth_state_hash TEXT,
 top_k_reference_ids TEXT, reference_distance_delta REAL, preferred_temp_position TEXT,
 growth_priority TEXT, reason_code TEXT
);
CREATE TABLE IF NOT EXISTS sarg_objective_score_ledger (
 id INTEGER PRIMARY KEY AUTOINCREMENT, growth_day INTEGER, biological_phase TEXT,
 climate_context_id TEXT, reference_id TEXT, rank INTEGER,
 growth_score REAL, feed_efficiency_score REAL, feed_cost_penalty REAL,
 heat_production_penalty REAL, thermal_stress_penalty REAL, climate_energy_penalty REAL,
 state_distance_delta REAL, final_score REAL, selected INTEGER
);
CREATE TABLE IF NOT EXISTS rl_policy_memory (
 id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, state_key TEXT, action_key TEXT,
 reward REAL, old_q REAL, new_q REAL, td_error REAL, layer TEXT
);
CREATE TABLE IF NOT EXISTS setd_formal_links (
 id INTEGER PRIMARY KEY AUTOINCREMENT, relation_type TEXT, source_key TEXT, target_key TEXT,
 formula_name TEXT, payload_json TEXT
);
CREATE TABLE IF NOT EXISTS working_runtime (
 key TEXT PRIMARY KEY, value TEXT
);
"""

class SQLiteStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)
        self._migrate_columns()
        self.conn.commit()

    def _migrate_columns(self):
        migrations = {
            "inner_sensor_log": {
                "climate_context_id": "TEXT", "climate_context_name": "TEXT", "mpc_score": "REAL",
                "comfort_error": "REAL", "energy_cost": "REAL", "gas_penalty": "REAL", "context_penalty": "REAL",
                "guidance_penalty": "REAL", "rl_prior_cost": "REAL", "rl_state_key": "TEXT", "rl_action_key": "TEXT", "mpc_components_json": "TEXT",
            },
            "growth_climate_guidance": {
                "best_reference_id": "TEXT", "reference_score": "REAL", "confidence": "REAL", "ventilation_bias": "REAL", "heating_bias": "REAL", "diet_phase_guidance": "TEXT",
                "biological_phase": "TEXT", "climate_context_id": "TEXT", "growth_state_hash": "TEXT", "top_k_reference_ids": "TEXT", "reference_distance_delta": "REAL", "preferred_temp_position": "TEXT", "growth_priority": "TEXT", "reason_code": "TEXT",
            },
            "outer_growth_state": {
                "biological_phase": "TEXT", "adg_kg_day": "REAL", "feed_efficiency": "REAL", "dominant_limitation": "TEXT", "daily_climate_context_id": "TEXT",
            },
        }
        for table, cols in migrations.items():
            existing = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()}
            for col, dtype in cols.items():
                if col not in existing:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {dtype}")

    def close(self):
        self.conn.close()

    def log_inner(self, row: dict):
        comps = row.get("mpc_components") or {}
        cols = ["timestamp","mode","outdoor_temp_c","outdoor_rh_pct","outdoor_wind_m_s","indoor_temp_c","indoor_rh_pct","indoor_air_speed_m_s","lct_c","uct_c","ventilation_group_pct","heating_group_pct","light_on","electric_kw","gas_kw","reward","climate_context_id","climate_context_name","mpc_score","comfort_error","energy_cost","gas_penalty","context_penalty","guidance_penalty","rl_prior_cost","rl_state_key","rl_action_key","mpc_components_json"]
        vals = [
            row.get("timestamp"), row.get("mode"), row.get("outdoor_temp_c"), row.get("outdoor_rh_pct"), row.get("outdoor_wind_m_s"),
            row.get("indoor_temp_c"), row.get("indoor_rh_pct"), row.get("indoor_air_speed_m_s"), row.get("lct_c"), row.get("uct_c"),
            row.get("ventilation_group_pct"), row.get("heating_group_pct"), int(bool(row.get("light_on"))), row.get("electric_kw"), row.get("gas_kw"), row.get("reward"),
            row.get("climate_context_id"), row.get("climate_context_name"), row.get("mpc_score"),
            comps.get("comfort_error", row.get("comfort_error")), comps.get("energy_cost", row.get("energy_cost")), comps.get("gas_penalty", row.get("gas_penalty")), comps.get("context_penalty", row.get("context_penalty")), comps.get("guidance_penalty", row.get("guidance_penalty")),
            row.get("rl_prior_cost"), row.get("rl_state_key"), row.get("rl_action_key"), json.dumps(comps, default=str),
        ]
        self.conn.execute(f"INSERT INTO inner_sensor_log ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})", vals)
        self.conn.commit()

    def log_command(self, timestamp: str, mode: str, command: dict, safety_status: str="OK"):
        self.conn.execute("INSERT INTO actuator_commands (timestamp,mode,ventilation_group_pct,heating_group_pct,light_on,safety_status,command_json) VALUES (?,?,?,?,?,?,?)",
            (timestamp, mode, command.get("ventilation_group_pct"), command.get("heating_group_pct"), int(bool(command.get("light_on"))), safety_status, json.dumps(command)))
        self.conn.commit()

    def log_growth(self, row: dict, status="OK"):
        self.conn.execute("INSERT INTO outer_growth_state (fattening_day,growth_date,tbw_kg,feed_intake_kg_dm_day,beef_production_kg,heat_production,diet,breed,status,raw_json,biological_phase,adg_kg_day,feed_efficiency,dominant_limitation,daily_climate_context_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (row.get("fattening_day"), row.get("growth_date"), row.get("tbw_kg"), row.get("feed_intake_kg_dm_day"), row.get("beef_production_kg"), row.get("heat_production"), row.get("diet"), row.get("breed"), status, json.dumps(row, default=str), row.get("biological_phase"), row.get("adg_kg_day"), row.get("feed_efficiency"), row.get("dominant_limitation"), row.get("daily_climate_context_id")))
        self.conn.commit()

    def log_guidance(self, fattening_day: int, guidance: dict):
        self.conn.execute("INSERT INTO growth_climate_guidance (fattening_day,updated_at,preferred_temp_low_c,preferred_temp_high_c,heat_production_feedback,comfort_weight_bias,gas_weight_bias,energy_weight_bias,status,best_reference_id,reference_score,confidence,ventilation_bias,heating_bias,diet_phase_guidance,biological_phase,climate_context_id,growth_state_hash,top_k_reference_ids,reference_distance_delta,preferred_temp_position,growth_priority,reason_code) VALUES (?,datetime('now'),?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (fattening_day, guidance.get("preferred_temp_low_c"), guidance.get("preferred_temp_high_c"), guidance.get("heat_production_feedback"), guidance.get("comfort_weight_bias"), guidance.get("gas_weight_bias"), guidance.get("energy_weight_bias"), guidance.get("status","OK"), guidance.get("best_reference_id"), guidance.get("reference_score"), guidance.get("confidence"), guidance.get("ventilation_bias"), guidance.get("heating_bias"), guidance.get("diet_phase_guidance"), guidance.get("biological_phase"), guidance.get("climate_context_id"), guidance.get("growth_state_hash"), json.dumps(guidance.get("top_k_reference_ids", []), default=str), guidance.get("reference_distance_delta"), guidance.get("preferred_temp_position"), guidance.get("growth_priority"), guidance.get("reason_code")))
        self.conn.commit()

    def log_sarg_scores(self, rows: list[dict]):
        if not rows: return
        cols = ["growth_day","biological_phase","climate_context_id","reference_id","rank","growth_score","feed_efficiency_score","feed_cost_penalty","heat_production_penalty","thermal_stress_penalty","climate_energy_penalty","state_distance_delta","final_score","selected"]
        vals = [[r.get(c) for c in cols] for r in rows]
        self.conn.executemany(f"INSERT INTO sarg_objective_score_ledger ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})", vals)
        self.conn.commit()

    def log_rl_update(self, update):
        d = update if isinstance(update, dict) else getattr(update, "__dict__", {})
        self.conn.execute("INSERT INTO rl_policy_memory (timestamp,state_key,action_key,reward,old_q,new_q,td_error,layer) VALUES (datetime('now'),?,?,?,?,?,?,?)",
            (d.get("state_key"), d.get("action_key"), d.get("reward"), d.get("old_q"), d.get("new_q"), d.get("td_error"), d.get("layer", "climate_mpc_bias")))
        self.conn.commit()

    def log_formal_link(self, relation_type: str, source_key: str, target_key: str, formula_name: str, payload: dict):
        self.conn.execute("INSERT INTO setd_formal_links (relation_type,source_key,target_key,formula_name,payload_json) VALUES (?,?,?,?,?)", (relation_type, source_key, target_key, formula_name, json.dumps(payload, default=str)))
        self.conn.commit()

    def table(self, name: str) -> pd.DataFrame:
        return pd.read_sql_query(f"SELECT * FROM {name}", self.conn)
