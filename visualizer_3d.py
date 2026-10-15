# filename: visualizer_3d.py
from __future__ import annotations

import json
import math
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import streamlit.components.v1 as components

from data_provider import ElementPropertiesDTO
from dynamics import TransientTrajectoryDTO
from engine import PhaseState, R_GAS, ThermodynamicStateDTO

# Постоянная Авогадро, 1/моль
N_AVOGADRO: float = 6.02214076e23


class TemperatureUnit(Enum):
    """Единицы измерения температуры."""
    KELVIN = ("Кельвин (K)", "K")
    CELSIUS = ("Градус Цельсия (°C)", "°C")
    FAHRENHEIT = ("Градус Фаренгейта (°F)", "°F")

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def symbol(self) -> str:
        return self.value[1]


class PressureUnit(Enum):
    """Единицы измерения чистого (линейного) давления."""
    ATM = ("Атмосфера (атм)", "атм", 101325.0)
    BAR = ("Бар (бар)", "бар", 100000.0)
    MPA = ("Мегапаскаль (МПа)", "МПа", 1e6)
    KPA = ("Килопаскаль (кПа)", "кПа", 1e3)
    PA = ("Паскаль (Па)", "Па", 1.0)

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def symbol(self) -> str:
        return self.value[1]

    @property
    def to_pa_factor(self) -> float:
        return self.value[2]


class UnitConverter:
    """Конвертер физических единиц между ядром СИ (К, Па) и пользовательским UI."""

    @staticmethod
    def temp_from_si(t_kelvin: float, unit: TemperatureUnit) -> float:
        if unit == TemperatureUnit.KELVIN:
            return t_kelvin
        if unit == TemperatureUnit.CELSIUS:
            return t_kelvin - 273.15
        return (t_kelvin - 273.15) * 1.8 + 32.0

    @staticmethod
    def temp_array_from_si(t_arr: NDArray[np.float64], unit: TemperatureUnit) -> NDArray[np.float64]:
        if unit == TemperatureUnit.KELVIN:
            return t_arr
        if unit == TemperatureUnit.CELSIUS:
            return t_arr - 273.15
        return (t_arr - 273.15) * 1.8 + 32.0

    @staticmethod
    def temp_to_si(t_display: float, unit: TemperatureUnit) -> float:
        if unit == TemperatureUnit.KELVIN:
            return max(0.5, t_display)
        if unit == TemperatureUnit.CELSIUS:
            return max(0.5, t_display + 273.15)
        return max(0.5, (t_display - 32.0) / 1.8 + 273.15)

    @staticmethod
    def temp_delta_from_si(dt_kelvin: float, unit: TemperatureUnit) -> float:
        if unit == TemperatureUnit.FAHRENHEIT:
            return dt_kelvin * 1.8
        return dt_kelvin

    @staticmethod
    def pressure_from_si(p_pa: float, unit: PressureUnit) -> float:
        return p_pa / unit.to_pa_factor

    @staticmethod
    def pressure_array_from_si(p_arr: NDArray[np.float64], unit: PressureUnit) -> NDArray[np.float64]:
        return p_arr / unit.to_pa_factor

    @staticmethod
    def pressure_to_si(p_display: float, unit: PressureUnit) -> float:
        return max(1e-4, p_display * unit.to_pa_factor)


class MolecularCubeVisualizer:
    """
    Синхронизированный с RK4 60 FPS 3D-рендерер объема вещества в нанокубе.
    Плавно интерполирует фазовые переходы во времени:
    - Плавление твердого бруска в жидкость с 3D-волнами на мениске;
    - Выкипание жидкого слоя в газовое облако из тысяч мелких молекул;
    - Переход в сверхкритическую опалесцирующую среду без мениска;
    - Синхронное перемещение маркера на соседней 2D фазовой диаграмме Plotly.
    """

    _PHASE_RU_TO_KEY = {
        "Твердое тело": "SOLID",
        "Жидкость": "LIQUID",
        "Газ / Пар": "GAS",
        "Сверхкритический флюид": "SUPERCRITICAL",
    }

    @classmethod
    def render_realtime_molecular_cube(
        cls,
        element: ElementPropertiesDTO,
        state: ThermodynamicStateDTO,
        t_unit: TemperatureUnit,
        p_unit: PressureUnit,
        trajectory: Optional[TransientTrajectoryDTO] = None,
        auto_play_rk4: bool = True,
        playback_speed: float = 1.0,
        auto_rotate_cube: bool = True,
        restart_token: int = 0,
        max_particles: int = 3000,
        height: int = 580,
    ) -> Tuple[float, float]:
        conv = UnitConverter
        v_rms = math.sqrt(3.0 * R_GAS * state.temperature / max(element.molar_mass, 1e-4))
        d_mean_nm = ((state.molar_volume / N_AVOGADRO) ** (1.0 / 3.0)) * 1e9
        box_nm = float(np.clip(d_mean_nm * 14.0, 2.5, 45.0))

        speed_factor = float(np.clip(0.55 + 0.35 * math.log10(max(v_rms, 40.0) / 150.0), 0.4, 1.6))

        # Подготовка временного ряда RK4 для плавной морфинг-анимации фаз в 3D-кубе и на 2D-графике
        timeline_data: List[dict] = []
        if trajectory is not None and len(trajectory.time_s) > 1:
            t_disp_arr = conv.temp_array_from_si(trajectory.temperature_k, t_unit)
            p_disp_arr = conv.pressure_array_from_si(trajectory.pressure_pa, p_unit)
            for idx in range(len(trajectory.time_s)):
                ph_ru = trajectory.phase_labels[idx]
                ph_key = cls._PHASE_RU_TO_KEY.get(ph_ru, "GAS")
                t_si_val = float(trajectory.temperature_k[idx])
                timeline_data.append(
                    {
                        "time_s": round(float(trajectory.time_s[idx]), 2),
                        "t_si": round(t_si_val, 2),
                        "t_disp": round(float(t_disp_arr[idx]), 2),
                        "p_disp": float(f"{p_disp_arr[idx]:.5g}"),
                        "density": round(float(trajectory.density[idx]), 2),
                        "phase": ph_key,
                        "phase_ru": ph_ru,
                        "t_ratio_tm": round(float(np.clip(t_si_val / max(element.tm, 1.0), 0.1, 2.5)), 3),
                        "t_ratio_tc": round(float(np.clip(t_si_val / max(element.tc, 1.0), 0.1, 2.5)), 3),
                    }
                )
        else:
            t_disp_val = conv.temp_from_si(state.temperature, t_unit)
            p_disp_val = conv.pressure_from_si(state.pressure, p_unit)
            ph_ru = state.phase.value.split(" (")[0]
            timeline_data.append(
                {
                    "time_s": 0.0,
                    "t_si": round(state.temperature, 2),
                    "t_disp": round(t_disp_val, 2),
                    "p_disp": float(f"{p_disp_val:.5g}"),
                    "density": round(state.density, 2),
                    "phase": state.phase.name,
                    "phase_ru": ph_ru,
                    "t_ratio_tm": round(float(np.clip(state.temperature / max(element.tm, 1.0), 0.1, 2.5)), 3),
                    "t_ratio_tc": round(float(np.clip(state.temperature / max(element.tc, 1.0), 0.1, 2.5)), 3),
                }
            )

        payload = {
            "formula": element.formula,
            "name": element.name,
            "particles": int(max_particles),
            "v_rms": round(v_rms, 1),
            "box_nm": round(box_nm, 1),
            "t_unit": t_unit.symbol,
            "p_unit": p_unit.symbol,
            "speed_factor": round(speed_factor, 3),
            "auto_play": bool(auto_play_rk4 and len(timeline_data) > 1),
            "playback_speed": float(playback_speed),
            "auto_rotate": bool(auto_rotate_cube),
            "restart_token": int(restart_token),
            "timeline": timeline_data,
        }
        cfg_json = json.dumps(payload, ensure_ascii=False)

        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; user-select: none; }}
            body {{
                background: rgba(15, 23, 42, 0.42);
                border: 1px solid rgba(255, 255, 255, 0.10);
                border-radius: 16px;
                overflow: hidden;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                color: #F8FAFC;
            }}
            # container {{
                position: relative;
                width: 100%;
                height: {height - 4}px;
                display: flex;
                flex-direction: column;
            }}
            .hud-top {{
                padding: 12px 16px 8px 16px;
                background: rgba(15, 23, 42, 0.65);
                border-bottom: 1px solid rgba(255, 255, 255, 0.07);
                display: flex;
                justify-content: space-between;
                align-items: center;
                z-index: 5;
            }}
            .hud-title {{
                font-size: 13.5px;
                font-weight: 600;
                color: #F8FAFC;
            }}
            .hud-sub {{
                font-size: 11.5px;
                color: #94A3B8;
                margin-top: 2px;
            }}
            .phase-badge {{
                padding: 4px 10px;
                border-radius: 8px;
                font-size: 11.5px;
                font-weight: 600;
                background: rgba(56, 189, 248, 0.16);
                border: 1px solid rgba(56, 189, 248, 0.4);
                color: #38BDF8;
                white-space: nowrap;
            }}
            .canvas-wrap {{
                position: relative;
                flex: 1;
                width: 100%;
                overflow: hidden;
            }}
            canvas {{
                width: 100%;
                height: 100%;
                display: block;
                cursor: grab;
            }}
            canvas:active {{ cursor: grabbing; }}
            .timeline-bar {{
                padding: 8px 16px 10px 16px;
                background: rgba(15, 23, 42, 0.75);
                border-top: 1px solid rgba(255, 255, 255, 0.08);
                display: flex;
                align-items: center;
                gap: 12px;
                font-size: 11.5px;
                color: #CBD5E1;
            }}
            .progress-track {{
                flex: 1;
                height: 6px;
                background: rgba(255, 255, 255, 0.12);
                border-radius: 3px;
                overflow: hidden;
                cursor: pointer;
            }}
            .progress-fill {{
                height: 100%;
                width: 0%;
                background: linear-gradient(90deg, #38BDF8, #F472B6);
                border-radius: 3px;
                transition: width 0.05s linear;
            }}
        </style>
        </head>
        <body>
        <div id="container">
            <div class="hud-top">
                <div>
                    <div class="hud-title" id="titleText">3D-объем вещества</div>
                    <div class="hud-sub" id="subText"></div>
                </div>
                <div class="phase-badge" id="phaseBadge"></div>
            </div>
            <div class="canvas-wrap">
                <canvas id="cubeCanvas"></canvas>
            </div>
            <div class="timeline-bar" id="timelineBar">
                <span id="timeLabel">t = 0.0 с</span>
                <div class="progress-track" id="progressTrack">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <span id="tpLabel"></span>
            </div>
        </div>

        <script>
        const CFG = {cfg_json};
        const TL = CFG.timeline;
        const canvas = document.getElementById('cubeCanvas');
        const ctx = canvas.getContext('2d');
        const titleEl = document.getElementById('titleText');
        const subEl = document.getElementById('subText');
        const badgeEl = document.getElementById('phaseBadge');
        const timeLabel = document.getElementById('timeLabel');
        const tpLabel = document.getElementById('tpLabel');
        const progressFill = document.getElementById('progressFill');
        const progressTrack = document.getElementById('progressTrack');
        const timelineBar = document.getElementById('timelineBar');

        if (TL.length <= 1) {{
            timelineBar.style.display = "none";
        }}

        let width = 0, height = 0;
        function resize() {{
            const dpr = window.devicePixelRatio || 1;
            width = canvas.clientWidth;
            height = canvas.clientHeight;
            canvas.width = width * dpr;
            canvas.height = height * dpr;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }}
        window.addEventListener('resize', resize);
        resize();

        let yaw = 0.62;
        let pitch = 0.40;
        let isDragging = false;
        let lastX = 0, lastY = 0;

        canvas.addEventListener('mousedown', e => {{ isDragging = true; lastX = e.clientX; lastY = e.clientY; }});
        window.addEventListener('mouseup', () => {{ isDragging = false; }});
        window.addEventListener('mousemove', e => {{
            if (!isDragging) return;
            yaw += (e.clientX - lastX) * 0.007;
            pitch = Math.max(-1.2, Math.min(1.2, pitch + (e.clientY - lastY) * 0.007));
            lastX = e.clientX;
            lastY = e.clientY;
        }});

        // Перемотка по клику на полосу прогресса RK4
        let simCursor = 0.0;
        progressTrack.addEventListener('click', e => {{
            const rect = progressTrack.getBoundingClientRect();
            const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            simCursor = frac * (TL.length - 1);
            syncPlotly2DMarker(Math.round(simCursor));
        }});

        function project(x, y, z) {{
            const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
            const x1 = x * cosY - y * sinY;
            const y1 = x * sinY + y * cosY;

            const cosP = Math.cos(pitch), sinP = Math.sin(pitch);
            const y2 = y1 * cosP - z * sinP;
            const z2 = y1 * sinP + z * cosP;

            const fov = Math.min(width, height) * 0.82;
            const dist = 3.8;
            const scale = fov / (dist + y2);
            return {{
                sx: width * 0.5 + x1 * scale,
                sy: height * 0.53 - z2 * scale,
                depth: y2,
                scale: scale
            }};
        }}

        // Индексы и массивы частиц
        const N = CFG.particles;
        const px = new Float32Array(N);
        const py = new Float32Array(N);
        const pz = new Float32Array(N);
        const vx = new Float32Array(N);
        const vy = new Float32Array(N);
        const vz = new Float32Array(N);

        // Координаты узлов кристаллической решетки для твердого тела
        const latX = new Float32Array(N);
        const latY = new Float32Array(N);
        const latZ = new Float32Array(N);

        const nSide = Math.max(8, Math.min(15, Math.round(Math.cbrt(N))));
        const totalLattice = nSide * nSide * nSide;

        function buildArrays() {{
            let idx = 0;
            for (let i = 0; i < nSide; i++) {{
                for (let j = 0; j < nSide; j++) {{
                    for (let k = 0; k < nSide; k++) {{
                        if (idx < N) {{
                            latX[idx] = -0.58 + 1.16 * (i / (nSide - 1));
                            latY[idx] = -0.58 + 1.16 * (j / (nSide - 1));
                            latZ[idx] = -0.94 + 1.05 * (k / (nSide - 1));
                            idx++;
                        }}
                    }}
                }}
            }}
            for (; idx < N; idx++) {{
                latX[idx] = (Math.random() - 0.5) * 1.16;
                latY[idx] = (Math.random() - 0.5) * 1.16;
                latZ[idx] = -0.94 + Math.random() * 1.05;
            }}

            const initPhase = TL[0].phase;
            for (let i = 0; i < N; i++) {{
                if (initPhase === "GAS") {{
                    // Компактное ядро внизу куба для красивого газодинамического расширения
                    const r = Math.random() * 0.20;
                    const th = Math.random() * 6.28318;
                    const ph = Math.random() * 3.14159;
                    px[i] = r * Math.sin(ph) * Math.cos(th);
                    py[i] = r * Math.sin(ph) * Math.sin(th);
                    pz[i] = -0.80 + r * Math.cos(ph);
                }} else if (initPhase === "LIQUID") {{
                    px[i] = (Math.random() - 0.5) * 1.88;
                    py[i] = (Math.random() - 0.5) * 1.88;
                    pz[i] = -0.95 + Math.random() * 1.10;
                }} else {{
                    px[i] = latX[i];
                    py[i] = latY[i];
                    pz[i] = latZ[i];
                }}
                const vMag = (0.011 + Math.random() * 0.022) * CFG.speed_factor;
                const vTh = Math.random() * 6.28318;
                const vPh = Math.acos(2 * Math.random() - 1);
                vx[i] = vMag * Math.sin(vPh) * Math.cos(vTh);
                vy[i] = vMag * Math.sin(vPh) * Math.sin(vTh);
                vz[i] = Math.abs(vMag * Math.cos(vPh)) * 1.1 + 0.003;
            }}
        }}
        buildArrays();

        // Плавные веса фаз [0..1] для непрерывного морфинга объема при фазовых переходах
        let wSolid = TL[0].phase === "SOLID" ? 1.0 : 0.0;
        let wLiquid = TL[0].phase === "LIQUID" ? 1.0 : 0.0;
        let wGas = TL[0].phase === "GAS" ? 1.0 : 0.0;
        let wSuper = TL[0].phase === "SUPERCRITICAL" ? 1.0 : 0.0;

        let lastSyncedIdx = -1;
        function syncPlotly2DMarker(frameIdx) {{
            if (frameIdx === lastSyncedIdx || TL.length <= 1) return;
            lastSyncedIdx = frameIdx;
            try {{
                const pt = TL[frameIdx];
                const plots = window.parent.document.querySelectorAll('.js-plotly-plot');
                if (plots && plots.length > 0 && window.parent.Plotly) {{
                    const phasePlot = plots[0];
                    if (phasePlot && phasePlot.data && phasePlot.data.length > 0) {{
                        const markerTraceIdx = phasePlot.data.length - 1;
                        window.parent.Plotly.restyle(
                            phasePlot,
                            {{
                                x: [[pt.t_disp]],
                                y: [[pt.p_disp]],
                                text: [[`  t=${{pt.time_s}}с: ${{pt.phase_ru}} (${{pt.t_disp}} ${{CFG.t_unit}})`]]
                            }},
                            [markerTraceIdx]
                        );
                    }}
                }}
            }} catch (err) {{}}
        }}

        function drawBoxFaces(x0, x1, y0, y1, z0, z1, fillRgba, strokeRgba, lineWidth) {{
            const v = [
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
            ].map(pt => project(pt[0], pt[1], pt[2]));

            const faces = [
                [0, 1, 2, 3], [4, 5, 6, 7],
                [0, 1, 5, 4], [2, 3, 7, 6],
                [0, 3, 7, 4], [1, 2, 6, 5]
            ];

            faces.sort((a, b) => {{
                const da = (v[a[0]].depth + v[a[1]].depth + v[a[2]].depth + v[a[3]].depth) * 0.25;
                const db = (v[b[0]].depth + v[b[1]].depth + v[b[2]].depth + v[b[3]].depth) * 0.25;
                return db - da;
            }});

            ctx.fillStyle = fillRgba;
            ctx.strokeStyle = strokeRgba;
            ctx.lineWidth = lineWidth;

            for (const f of faces) {{
                ctx.beginPath();
                ctx.moveTo(v[f[0]].sx, v[f[0]].sy);
                ctx.lineTo(v[f[1]].sx, v[f[1]].sy);
                ctx.lineTo(v[f[2]].sx, v[f[2]].sy);
                ctx.lineTo(v[f[3]].sx, v[f[3]].sy);
                ctx.closePath();
                if (fillRgba) ctx.fill();
                if (strokeRgba) ctx.stroke();
            }}
        }}

        function drawWavyMeniscus(tSec, zLevel, alpha) {{
            if (alpha < 0.03) return;
            const grid = 9;
            ctx.strokeStyle = `rgba(52, 211, 153, ${{0.58 * alpha}})`;
            ctx.fillStyle = `rgba(16, 185, 129, ${{0.22 * alpha}})`;
            ctx.lineWidth = 1.15;

            for (let ix = 0; ix < grid; ix++) {{
                const xA = -0.96 + (1.92 * ix) / grid;
                const xB = -0.96 + (1.92 * (ix + 1)) / grid;
                for (let iy = 0; iy < grid; iy++) {{
                    const yA = -0.96 + (1.92 * iy) / grid;
                    const yB = -0.96 + (1.92 * (iy + 1)) / grid;

                    const z00 = zLevel + 0.042 * Math.sin(3.2 * xA + tSec * 3.2) * Math.cos(3.2 * yA - tSec * 2.5);
                    const z10 = zLevel + 0.042 * Math.sin(3.2 * xB + tSec * 3.2) * Math.cos(3.2 * yA - tSec * 2.5);
                    const z11 = zLevel + 0.042 * Math.sin(3.2 * xB + tSec * 3.2) * Math.cos(3.2 * yB - tSec * 2.5);
                    const z01 = zLevel + 0.042 * Math.sin(3.2 * xA + tSec * 3.2) * Math.cos(3.2 * yB - tSec * 2.5);

                    const p00 = project(xA, yA, z00);
                    const p10 = project(xB, yA, z10);
                    const p11 = project(xB, yB, z11);
                    const p01 = project(xA, yB, z01);

                    ctx.beginPath();
                    ctx.moveTo(p00.sx, p00.sy);
                    ctx.lineTo(p10.sx, p10.sy);
                    ctx.lineTo(p11.sx, p11.sy);
                    ctx.lineTo(p01.sx, p01.sy);
                    ctx.closePath();
                    ctx.fill();
                    ctx.stroke();
                }}
            }}
        }}

        const badgeColors = {{
            SOLID: ["#60A5FA", "rgba(96, 165, 250, 0.16)", "rgba(96, 165, 250, 0.45)"],
            LIQUID: ["#34D399", "rgba(52, 211, 153, 0.16)", "rgba(52, 211, 153, 0.45)"],
            GAS: ["#FBBF24", "rgba(251, 191, 36, 0.16)", "rgba(251, 191, 36, 0.45)"],
            SUPERCRITICAL: ["#C084FC", "rgba(192, 132, 252, 0.16)", "rgba(192, 132, 252, 0.45)"]
        }};

        let startTime = performance.now();
        let prevNow = startTime;

        function animate(now) {{
            const dtSec = Math.min(0.05, (now - prevNow) * 0.001);
            prevNow = now;
            const tSec = (now - startTime) * 0.001;

            if (CFG.auto_rotate && !isDragging) {{
                yaw += 0.003;
            }}

            // Продвижение курсора времени вдоль RK4-траектории
            if (CFG.auto_play && TL.length > 1) {{
                simCursor += dtSec * 18.0 * CFG.playback_speed;
                if (simCursor >= TL.length) {{
                    simCursor = 0.0;
                }}
            }}

            const curIdx = Math.min(TL.length - 1, Math.max(0, Math.floor(simCursor)));
            const snap = TL[curIdx];

            // Синхронизация маркера на 2D-диаграмме каждые 3 шага
            if (CFG.auto_play && curIdx % 3 === 0) {{
                syncPlotly2DMarker(curIdx);
            }}

            // Плавный морфинг весов фаз (интерполяция переходов Плавление / Кипение / Конденсация)
            const morphRate = 4.2 * dtSec;
            wSolid += ((snap.phase === "SOLID" ? 1.0 : 0.0) - wSolid) * morphRate;
            wLiquid += ((snap.phase === "LIQUID" ? 1.0 : 0.0) - wLiquid) * morphRate;
            wGas += ((snap.phase === "GAS" ? 1.0 : 0.0) - wGas) * morphRate;
            wSuper += ((snap.phase === "SUPERCRITICAL" ? 1.0 : 0.0) - wSuper) * morphRate;

            // Обновление заголовков и таймлайна
            titleEl.textContent = `3D-объем: ${{CFG.name}} (${{CFG.formula}})`;
            subEl.textContent = `Камера ≈ ${{CFG.box_nm}} нм | ρ = ${{snap.density}} кг/м³ | T = ${{snap.t_disp}} ${{CFG.t_unit}}`;
            badgeEl.textContent = snap.phase_ru;
            const bc = badgeColors[snap.phase] || badgeColors.GAS;
            badgeEl.style.color = bc[0];
            badgeEl.style.background = bc[1];
            badgeEl.style.borderColor = bc[2];

            if (TL.length > 1) {{
                const pct = (curIdx / (TL.length - 1)) * 100.0;
                progressFill.style.width = `${{pct.toFixed(1)}}%`;
                timeLabel.textContent = `t = ${{snap.time_s.toFixed(1)}} с`;
                tpLabel.textContent = `${{snap.t_disp}} ${{CFG.t_unit}} | ${{snap.p_disp}} ${{CFG.p_unit}}`;
            }}

            ctx.clearRect(0, 0, width, height);

            // 1. Внешний стеклянный куб
            drawBoxFaces(-1, 1, -1, 1, -1, 1, "rgba(56, 189, 248, 0.025)", "rgba(148, 163, 184, 0.28)", 1.4);

            // 2. Если есть вклад жидкой фазы — рисуем поднимающийся/опускающийся слой жидкости
            const zLiqLevel = -0.96 + 1.15 * wLiquid;
            if (wLiquid > 0.02) {{
                drawBoxFaces(
                    -0.97, 0.97, -0.97, 0.97, -0.98, zLiqLevel,
                    `rgba(5, 150, 105, ${{0.20 * wLiquid}})`,
                    `rgba(16, 185, 129, ${{0.38 * wLiquid}})`,
                    1.2
                );
            }}

            // 3. Движение и морфинг ансамбля микрочастиц
            const nLiqCutoff = Math.floor(N * Math.min(0.92, wLiquid * 0.88));
            const vibAmp = 0.016 * Math.sqrt(snap.t_ratio_tm);

            for (let i = 0; i < N; i++) {{
                if (wSolid > 0.45) {{
                    // Притягивание частиц к узлам кристаллической решетки бруска (кристаллизация / колебания)
                    const tx = latX[i] + vibAmp * Math.sin(tSec * 14.0 + i);
                    const ty = latY[i] + vibAmp * Math.cos(tSec * 14.0 + i * 1.3);
                    const tz = -0.96 + (latZ[i] + 0.96) * wSolid + vibAmp * Math.sin(tSec * 14.0 + i * 0.7);
                    px[i] += (tx - px[i]) * 0.18;
                    py[i] += (ty - py[i]) * 0.18;
                    pz[i] += (tz - pz[i]) * 0.18;
                }} else if (wLiquid > 0.35 && i < nLiqCutoff) {{
                    // Конвективное движение внутри жидкого слоя под мениском
                    const convX = 0.0025 * Math.sin(1.8 * px[i]) * Math.cos(2.2 * pz[i]);
                    const convZ = -0.0025 * Math.cos(1.8 * px[i]) * Math.sin(2.2 * pz[i]);
                    px[i] += vx[i] * 0.45 + convX;
                    py[i] += vy[i] * 0.45;
                    pz[i] += vz[i] * 0.45 + convZ;

                    if (px[i] < -0.95 || px[i] > 0.95) {{ vx[i] *= -1; px[i] = Math.max(-0.95, Math.min(0.95, px[i])); }}
                    if (py[i] < -0.95 || py[i] > 0.95) {{ vy[i] *= -1; py[i] = Math.max(-0.95, Math.min(0.95, py[i])); }}
                    if (pz[i] > zLiqLevel - 0.02) {{
                        pz[i] += (zLiqLevel - 0.15 - pz[i]) * 0.12;
                        vz[i] = -Math.abs(vz[i]);
                    }}
                    if (pz[i] < -0.96) {{ vz[i] = Math.abs(vz[i]); pz[i] = -0.96; }}
                }} else if (wSuper > 0.45) {{
                    // 3D-вихри Тейлора — Грина в сверхкритическом флюиде
                    const wSpd = 0.016 * CFG.speed_factor;
                    px[i] += wSpd * Math.sin(3.14 * px[i] + tSec) * Math.cos(3.14 * py[i]);
                    py[i] += -wSpd * Math.cos(3.14 * px[i]) * Math.sin(3.14 * py[i] + tSec);
                    pz[i] += wSpd * 0.7 * Math.sin(3.14 * pz[i] - tSec * 1.2);
                    if (px[i] < -0.95) px[i] = 0.95; if (px[i] > 0.95) px[i] = -0.95;
                    if (py[i] < -0.95) py[i] = 0.95; if (py[i] > 0.95) py[i] = -0.95;
                    if (pz[i] < -0.95) pz[i] = 0.95; if (pz[i] > 0.95) pz[i] = -0.95;
                }} else {{
                    // Свободный газодинамический разлет по всему кубу
                    px[i] += vx[i];
                    py[i] += vy[i];
                    pz[i] += vz[i];
                    if (px[i] < -0.95 || px[i] > 0.95) {{ vx[i] *= -1; px[i] = Math.max(-0.95, Math.min(0.95, px[i])); }}
                    if (py[i] < -0.95 || py[i] > 0.95) {{ vy[i] *= -1; py[i] = Math.max(-0.95, Math.min(0.95, py[i])); }}
                    const zFloor = wLiquid > 0.1 ? zLiqLevel + 0.03 : -0.95;
                    if (pz[i] < zFloor || pz[i] > 0.95) {{ vz[i] *= -1; pz[i] = Math.max(zFloor, Math.min(0.95, pz[i])); }}
                }}

                const pr = project(px[i], py[i], pz[i]);
                if (wSolid > 0.5) {{
                    ctx.fillStyle = snap.t_ratio_tm > 0.85 ? "rgba(251,146,60,0.85)" : "rgba(125,211,252,0.82)";
                }} else if (wLiquid > 0.35 && i < nLiqCutoff) {{
                    ctx.fillStyle = (i % 2 === 0) ? "rgba(52,211,153,0.78)" : "rgba(16,185,129,0.74)";
                }} else if (wSuper > 0.45) {{
                    ctx.fillStyle = (i % 2 === 0) ? "rgba(192,132,252,0.84)" : "rgba(244,114,182,0.82)";
                }} else {{
                    ctx.fillStyle = (i % 3 === 0) ? "rgba(56,189,248,0.84)" : "rgba(250,204,21,0.82)";
                }}
                const r = Math.max(1.15, pr.scale * 0.011);
                ctx.fillRect(pr.sx - r, pr.sy - r, r * 2, r * 2);
            }}

            // 4. Отрисовка волнового мениска жидкости
            if (wLiquid > 0.04) {{
                drawWavyMeniscus(tSec, zLiqLevel, wLiquid);
            }}

            // 5. Отрисовка монолитного твердого бруска (при плавлении плавно оседает и растворяется в жидкости)
            if (wSolid > 0.03) {{
                const bWidth = 0.62 * (0.65 + 0.35 * wSolid);
                const bTop = -0.98 + 1.08 * wSolid;
                let ingotFill = `rgba(56, 189, 248, ${{0.25 * wSolid}})`;
                let ingotEdge = `rgba(186, 230, 253, ${{0.80 * wSolid}})`;
                if (snap.t_ratio_tm > 0.85) {{
                    ingotFill = `rgba(239, 68, 68, ${{0.28 * wSolid}})`;
                    ingotEdge = `rgba(253, 186, 116, ${{0.85 * wSolid}})`;
                }} else if (snap.t_ratio_tm > 0.45) {{
                    ingotFill = `rgba(148, 163, 184, ${{0.26 * wSolid}})`;
                    ingotEdge = `rgba(241, 245, 249, ${{0.80 * wSolid}})`;
                }}
                drawBoxFaces(-bWidth, bWidth, -bWidth, bWidth, -0.98, bTop, ingotFill, ingotEdge, 2.2);
            }}

            requestAnimationFrame(animate);
        }}

        requestAnimationFrame(animate);
        </script>
        </body>
        </html>
        """
        components.html(html_code, height=height)
        return v_rms, box_nm