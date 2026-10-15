# filename: app.py
from __future__ import annotations

import math
from typing import Dict, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data_provider import ElementDataProvider, ElementPropertiesDTO
from dynamics import (
    ProcessType,
    SimulationConfigDTO,
    TransientProcessSimulator,
    TransientTrajectoryDTO,
)
from engine import PhaseDiagramCurvesDTO, PhaseState, PengRobinsonEOS
from visualizer_3d import (
    MolecularCubeVisualizer,
    PressureUnit,
    TemperatureUnit,
    UnitConverter,
)

PLOTLY_CLEAN_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "scrollZoom": False,
}


def inject_glassmorphism_css() -> None:
    """
    Внедряет кастомный Liquid Glass CSS:
    - Размытие заднего плана (backdrop-filter: blur(20px))
    - Полное скрытие любых плавающих панелей Plotly (.modebar, .updatemenu-container)
    - Устранение обрезки длинных названий фаз
    """
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 12% 18%, rgba(99, 102, 241, 0.16), transparent 38%),
                radial-gradient(circle at 88% 22%, rgba(16, 185, 129, 0.12), transparent 40%),
                radial-gradient(circle at 50% 85%, rgba(236, 72, 153, 0.10), transparent 45%),
                #0B0F19;
        }

        /* Полное скрытие плавающих элементов поверх графиков Plotly */
        .modebar, .modebar-container, .updatemenu-container {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
        }

        section[data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.68) !important;
            backdrop-filter: blur(22px) saturate(180%) !important;
            -webkit-backdrop-filter: blur(22px) saturate(180%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        .glass-card {
            background: rgba(30, 41, 59, 0.48);
            backdrop-filter: blur(18px) saturate(160%);
            -webkit-backdrop-filter: blur(18px) saturate(160%);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 16px;
            padding: 16px 18px;
            margin-bottom: 14px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.32);
            transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
            animation: fadeInGlass 0.45s ease-out;
            min-height: 112px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .glass-card:hover {
            transform: translateY(-3px);
            border-color: rgba(99, 179, 237, 0.42);
            box-shadow: 0 12px 36px 0 rgba(56, 189, 248, 0.18);
        }
        .glass-label {
            font-size: 0.82rem;
            font-weight: 500;
            color: #94A3B8;
            letter-spacing: 0.02em;
            margin-bottom: 6px;
        }
        .glass-value {
            font-size: clamp(1.02rem, 1.32vw, 1.42rem);
            font-weight: 700;
            color: #F8FAFC;
            line-height: 1.25;
            white-space: normal !important;
            overflow-wrap: break-word !important;
            word-break: break-word !important;
        }
        .glass-sub {
            font-size: 0.78rem;
            color: #38BDF8;
            margin-top: 6px;
            font-weight: 500;
        }

        @keyframes fadeInGlass {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_glass_card(label: str, value: str, subtitle: str = "", accent_color: str = "#38BDF8") -> None:
    sub_html = (
        f'<div class="glass-sub" style="color: {accent_color};">{subtitle}</div>'
        if subtitle
        else ""
    )
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="glass-label">{label}</div>
            <div class="glass-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Инициализация базы элементов IUPAC / DIPPR...")
def get_elements_catalog() -> Dict[str, ElementPropertiesDTO]:
    return ElementDataProvider.load_all_elements()


@st.cache_data(show_spinner="Расчет VLE фазового равновесия (φᴸ = φⱽ)...")
def get_cached_phase_curves(
    _engine: PengRobinsonEOS, element: ElementPropertiesDTO
) -> PhaseDiagramCurvesDTO:
    return _engine.generate_phase_diagram(element)


def build_clean_phase_diagram(
    element: ElementPropertiesDTO,
    curves: PhaseDiagramCurvesDTO,
    active_t_si: float,
    active_p_si: float,
    active_time_s: float,
    active_phase_ru: str,
    t_unit: TemperatureUnit,
    p_unit: PressureUnit,
    use_log_pressure_axis: bool,
    trajectory: Optional[TransientTrajectoryDTO] = None,
    show_spinodals: bool = True,
) -> go.Figure:
    """
    Строит абсолютно чистую фазовую P-T диаграмму без плавающих кнопок updatemenus поверх осей.
    Последний трейс — активный маркер состояния, который синхронно двигается вместе с 3D-кубом.
    """
    conv = UnitConverter
    fig = go.Figure()

    t_sub_u = conv.temp_array_from_si(curves.t_sublimation, t_unit)
    p_sub_u = conv.pressure_array_from_si(curves.p_sublimation, p_unit)

    t_melt_u = conv.temp_array_from_si(curves.t_melting, t_unit)
    p_melt_u = conv.pressure_array_from_si(curves.p_melting, p_unit)

    t_vap_u = conv.temp_array_from_si(curves.t_vaporization, t_unit)
    p_vap_u = conv.pressure_array_from_si(curves.p_vaporization, p_unit)

    tt_u = conv.temp_from_si(element.tt, t_unit)
    pt_u = conv.pressure_from_si(curves.pt_eos, p_unit)
    tc_u = conv.temp_from_si(element.tc, t_unit)
    pc_u = conv.pressure_from_si(element.pc, p_unit)

    cur_t_u = conv.temp_from_si(active_t_si, t_unit)
    cur_p_u = conv.pressure_from_si(active_p_si, p_unit)

    # Неоновая подложка размытия (Glow Halos)
    fig.add_trace(
        go.Scatter(
            x=t_vap_u,
            y=p_vap_u,
            mode="lines",
            line=dict(color="rgba(239, 85, 59, 0.22)", width=14),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_melt_u,
            y=p_melt_u,
            mode="lines",
            line=dict(color="rgba(0, 204, 150, 0.20)", width=12),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Основные кривые фазового равновесия
    fig.add_trace(
        go.Scatter(
            x=t_sub_u,
            y=p_sub_u,
            mode="lines",
            name="Сублимация (S ↔ G)",
            line=dict(color="#636EFA", width=3, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_melt_u,
            y=p_melt_u,
            mode="lines",
            name="Плавление (S ↔ L)",
            line=dict(color="#00CC96", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_vap_u,
            y=p_vap_u,
            mode="lines",
            name="Бинодаль VLE (φᴸ = φⱽ)",
            line=dict(color="#EF553B", width=3.5),
        )
    )

    if show_spinodals:
        t_spin_u = conv.temp_array_from_si(curves.t_spinodal, t_unit)
        p_spin_v_u = conv.pressure_array_from_si(curves.p_spinodal_vapor, p_unit)
        p_spin_l_u = conv.pressure_array_from_si(curves.p_spinodal_liquid, p_unit)
        fig.add_trace(
            go.Scatter(
                x=t_spin_u,
                y=p_spin_v_u,
                mode="lines",
                name="Спинодаль пара",
                line=dict(color="rgba(255, 161, 90, 0.55)", width=1.5, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t_spin_u,
                y=p_spin_l_u,
                mode="lines",
                name="Спинодаль жидкости",
                line=dict(color="rgba(171, 99, 250, 0.55)", width=1.5, dash="dash"),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[tt_u],
            y=[pt_u],
            mode="markers+text",
            name="Тройная точка",
            text=["Тройная"],
            textposition="bottom right",
            marker=dict(size=10, color="#FFA15A", symbol="diamond"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[tc_u],
            y=[pc_u],
            mode="markers+text",
            name="Критическая точка",
            text=["Критическая"],
            textposition="top left",
            marker=dict(size=12, color="#AB63FA", symbol="star"),
        )
    )

    if trajectory is not None and len(trajectory.temperature_k) > 1:
        traj_t_u = conv.temp_array_from_si(trajectory.temperature_k, t_unit)
        traj_p_u = conv.pressure_array_from_si(trajectory.pressure_pa, p_unit)

        fig.add_trace(
            go.Scatter(
                x=traj_t_u,
                y=traj_p_u,
                mode="lines",
                line=dict(color="rgba(25, 211, 243, 0.25)", width=10),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=traj_t_u,
                y=traj_p_u,
                mode="lines",
                name="Траектория процесса (RK4)",
                line=dict(color="#19D3F3", width=2.5),
            )
        )

    # Последний трейс — активный маркер состояния (синхронизируется с 3D-кубом)
    fig.add_trace(
        go.Scatter(
            x=[cur_t_u],
            y=[cur_p_u],
            mode="markers+text",
            name="Текущее состояние",
            text=[f"  t={active_time_s:.1f}с: {active_phase_ru} ({cur_t_u:.1f} {t_unit.symbol})"],
            textposition="top right",
            marker=dict(
                size=15,
                color="#FF6692" if trajectory is not None else "#FECB52",
                symbol="circle",
                line=dict(color="#FFFFFF", width=2.5),
            ),
        )
    )

    t_min_u = conv.temp_from_si(0.0, t_unit)
    t_max_si = max(element.tc * 1.55, active_t_si * 1.15)
    if trajectory is not None:
        t_max_si = max(t_max_si, float(trajectory.temperature_k.max()) * 1.1)
    t_max_u = conv.temp_from_si(t_max_si, t_unit)

    p_min_u = max(min(float(p_sub_u[0]), pt_u * 0.02), 1e-6)
    p_max_si = max(element.pc * 15.0, active_p_si * 2.2)
    if trajectory is not None:
        p_max_si = max(p_max_si, float(trajectory.pressure_pa.max()) * 1.5)
    p_max_u = conv.pressure_from_si(p_max_si, p_unit)

    fig.add_vline(x=tc_u, line_width=1, line_dash="dash", line_color="rgba(148, 163, 184, 0.45)")
    fig.add_hline(y=pc_u, line_width=1, line_dash="dash", line_color="rgba(148, 163, 184, 0.45)")

    yaxis_cfg = (
        dict(type="log", range=[math.log10(p_min_u), math.log10(p_max_u * 2.0)])
        if use_log_pressure_axis
        else dict(type="linear", range=[0.0, max(cur_p_u * 1.6, pc_u * 2.2)])
    )

    fig.update_layout(
        title=f"Фазовая диаграмма: {element.name} ({element.formula})",
        xaxis_title=f"Температура T, {t_unit.symbol}",
        yaxis_title=f"Давление P, {p_unit.symbol}",
        xaxis=dict(range=[t_min_u, t_max_u]),
        yaxis=yaxis_cfg,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.42)",
        height=580,
        margin=dict(l=15, r=15, t=48, b=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_transient_dashboard(
    trajectory: TransientTrajectoryDTO,
    t_unit: TemperatureUnit,
    p_unit: PressureUnit,
    selected_time_s: Optional[float] = None,
) -> go.Figure:
    conv = UnitConverter
    t_disp = conv.temp_array_from_si(trajectory.temperature_k, t_unit)
    p_disp = conv.pressure_array_from_si(trajectory.pressure_pa, p_unit)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"Температура T(t), {t_unit.symbol}",
            f"Чистое давление P(t), {p_unit.symbol}",
            "Энтальпия H(t) [кДж/моль] и Энтропия S(t) [Дж/(моль·К)]",
            "Теплоемкости Cp(t) и Cv(t), Дж/(моль·К)",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": True}, {"secondary_y": False}],
        ],
    )

    fig.add_trace(
        go.Scatter(
            x=trajectory.time_s,
            y=t_disp,
            name=f"T(t), {t_unit.symbol}",
            line=dict(color="#FECB52", width=2.8),
            fill="tozeroy",
            fillcolor="rgba(254, 203, 82, 0.08)",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=trajectory.time_s,
            y=p_disp,
            name=f"P(t), {p_unit.symbol}",
            line=dict(color="#19D3F3", width=2.8),
            fill="tozeroy",
            fillcolor="rgba(25, 211, 243, 0.08)",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=trajectory.time_s,
            y=trajectory.enthalpy / 1000.0,
            name="H(t), кДж/моль",
            line=dict(color="#EF553B", width=2.5),
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=trajectory.time_s,
            y=trajectory.entropy,
            name="S(t), Дж/(моль·К)",
            line=dict(color="#AB63FA", width=2.2, dash="dot"),
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=trajectory.time_s,
            y=trajectory.cp,
            name="Cp(t)",
            line=dict(color="#00CC96", width=2.5),
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=trajectory.time_s,
            y=trajectory.cv,
            name="Cv(t)",
            line=dict(color="#FFA15A", width=2, dash="dash"),
        ),
        row=2,
        col=2,
    )

    if selected_time_s is not None:
        fig.add_vline(x=selected_time_s, line_width=1.5, line_dash="dot", line_color="#FF6692")

    fig.update_layout(
        height=600,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.42)",
        title="Временные развертки параметров процесса (RK4)",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Время t, с")
    return fig


def main() -> None:
    st.set_page_config(
        page_title="Термодинамический симулятор PR-VLE + 3D + RK4",
        layout="wide",
    )
    inject_glassmorphism_css()

    if "restart_token" not in st.session_state:
        st.session_state.restart_token = 0

    st.title("Физический симулятор фазовых состояний, 3D-объема вещества и процессов RK4")

    catalog = get_elements_catalog()
    engine = PengRobinsonEOS()
    simulator = TransientProcessSimulator(engine)
    conv = UnitConverter

    with st.sidebar:
        st.header("1. Химический элемент и единицы")
        element_keys = list(catalog.keys())
        default_idx = next(
            (i for i, k in enumerate(element_keys) if "Ar " in k or "N (" in k), 0
        )

        selected_key = st.selectbox(
            "Элемент периодической таблицы:",
            options=element_keys,
            index=default_idx,
        )
        element = catalog[selected_key]
        curves = get_cached_phase_curves(engine, element)

        col_u1, col_u2 = st.columns(2)
        with col_u1:
            t_unit = st.selectbox(
                "Температура:",
                options=list(TemperatureUnit),
                format_func=lambda u: u.symbol,
                index=0,
            )
        with col_u2:
            p_unit = st.selectbox(
                "Давление:",
                options=list(PressureUnit),
                format_func=lambda u: u.symbol,
                index=0,
            )

        st.markdown("---")
        st.header(f"2. Начальная точка ({t_unit.symbol}, {p_unit.symbol})")

        t_min_si = max(1.0, 0.22 * element.tt)
        t_max_si = 1.65 * element.tc
        t_def_si = min(max(element.tt * 0.85, t_min_si), t_max_si)

        t_min_u = float(round(conv.temp_from_si(t_min_si, t_unit), 2))
        t_max_u = float(round(conv.temp_from_si(t_max_si, t_unit), 2))
        t_def_u = float(round(conv.temp_from_si(t_def_si, t_unit), 2))
        t_step_u = max(0.1, round((t_max_u - t_min_u) / 500.0, 2))

        t_val_u = st.slider(
            f"Температура T₀ ({t_unit.label}):",
            min_value=t_min_u,
            max_value=t_max_u,
            value=t_def_u,
            step=t_step_u,
        )
        temperature_si = conv.temp_to_si(t_val_u, t_unit)

        p_range_mode = st.radio(
            "Диапазон ползунка чистого давления:",
            options=[
                "Низкие / Стандартные (до 5 атм)",
                "Критический диапазон (до 2.5·Pc)",
                "Сверхвысокие давления (до 15·Pc)",
            ],
            index=1,
            horizontal=False,
        )

        if p_range_mode == "Низкие / Стандартные (до 5 атм)":
            p_min_si = max(curves.pt_eos * 0.05, 10.0)
            p_max_si = 5.0 * 101325.0
            p_def_si = 101325.0
        elif p_range_mode == "Критический диапазон (до 2.5·Pc)":
            p_min_si = max(curves.pt_eos * 0.1, 100.0)
            p_max_si = 2.5 * element.pc
            p_def_si = min(max(0.35 * element.pc, p_min_si), p_max_si)
        else:
            p_min_si = 0.1 * element.pc
            p_max_si = 15.0 * element.pc
            p_def_si = 1.5 * element.pc

        p_min_u = float(max(1e-5, round(conv.pressure_from_si(p_min_si, p_unit), 5)))
        p_max_u = float(round(conv.pressure_from_si(p_max_si, p_unit), 4))
        p_def_u = float(min(max(round(conv.pressure_from_si(p_def_si, p_unit), 4), p_min_u), p_max_u))
        p_step_u = float(max(1e-5, round((p_max_u - p_min_u) / 400.0, 5)))

        p_val_u = st.slider(
            f"Чистое давление P₀ ({p_unit.symbol}):",
            min_value=p_min_u,
            max_value=p_max_u,
            value=p_def_u,
            step=p_step_u,
        )
        pressure_si = conv.pressure_to_si(p_val_u, p_unit)

        max_particles_3d = st.slider(
            "Количество микрочастиц в 3D-кубе:",
            min_value=1000,
            max_value=5000,
            value=3000,
            step=250,
        )

        use_log_axis = st.checkbox(
            "Логарифмическая ось P на обзорной диаграмме",
            value=True,
        )
        show_spinodals = st.checkbox("Показать спинодали (границы метастабильности)", value=True)

        st.markdown("---")
        st.header("3. Нестационарный процесс (RK4)")
        enable_dynamics = st.checkbox("Включить симуляцию во времени", value=True)

        process_type = st.selectbox(
            "Тип процесса:",
            options=list(ProcessType),
            format_func=lambda p: p.value,
        )

        dt_step = st.slider("Шаг интегрирования dt (с):", 0.05, 2.0, 0.25, 0.05)
        num_steps = st.slider("Число шагов N:", 40, 360, 140, 20)

        heat_power = 0.0
        pressure_rate_si = 0.0
        if process_type in (ProcessType.ISOBARIC, ProcessType.ISOCHORIC):
            max_q = float(round(max(250.0, 12.0 * element.tc), 0))
            heat_power = st.slider(
                "Тепловой поток dQ/dt (Вт/моль):",
                min_value=-max_q,
                max_value=max_q,
                value=float(round(0.35 * max_q, 1)),
                step=max(1.0, round(max_q / 100.0, 1)),
            )
        else:
            max_dp_u = float(round(conv.pressure_from_si(element.pc * 0.15, p_unit), 4))
            dp_rate_u = st.slider(
                f"Скорость изменения давления dP/dt ({p_unit.symbol}/с):",
                min_value=-max_dp_u,
                max_value=max_dp_u,
                value=float(round(0.2 * max_dp_u, 4)),
                step=max(1e-4, round(max_dp_u / 100.0, 4)),
            )
            pressure_rate_si = dp_rate_u * p_unit.to_pa_factor

    # Расчет RK4-траектории
    trajectory: Optional[TransientTrajectoryDTO] = None
    if enable_dynamics:
        sim_cfg = SimulationConfigDTO(
            process_type=process_type,
            t_initial=temperature_si,
            p_initial=pressure_si,
            dt=dt_step,
            num_steps=num_steps,
            heat_power_w_mol=heat_power,
            pressure_rate_pa_s=pressure_rate_si,
        )
        trajectory = simulator.run_simulation(element, sim_cfg)

    # =========================================================================
    # НАТИВНАЯ ПАНЕЛЬ УПРАВЛЕНИЯ АНИМАЦИЕЙ И ВРЕМЕНЕМ ПРОЦЕССА (ВМЕСТО ОКОШКА НА ГРАФИКЕ)
    # =========================================================================
    st.markdown("#### Панель управления процессом и 3D-камерой")
    ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([1.3, 1.0, 1.0, 0.9])
    with ctrl_c1:
        auto_play_rk4 = st.toggle(
            "▶ Авто-цикл процесса RK4 (2D + 3D синхронно)",
            value=True,
            disabled=(trajectory is None),
        )
    with ctrl_c2:
        playback_speed = st.select_slider(
            "Скорость анимации:",
            options=[0.5, 1.0, 2.0, 4.0],
            value=1.0,
            format_func=lambda s: f"{s}×",
        )
    with ctrl_c3:
        auto_rotate_cube = st.toggle("⟳ Автовращение 3D-куба", value=True)
    with ctrl_c4:
        if st.button("↻ Перезапустить с t = 0", use_container_width=True):
            st.session_state.restart_token += 1

    # Если авто-цикл выключен, показываем нативный ползунок ручной перемотки по времени t (с)
    selected_step_idx = 0
    active_time_s = 0.0
    if trajectory is not None and not auto_play_rk4:
        selected_step_idx = st.slider(
            "Ручной выбор момента времени процесса t (с):",
            min_value=0,
            max_value=len(trajectory.time_s) - 1,
            value=0,
            format="%d шаг",
        )
        active_time_s = float(trajectory.time_s[selected_step_idx])
        active_t_si = float(trajectory.temperature_k[selected_step_idx])
        active_p_si = float(trajectory.pressure_pa[selected_step_idx])
        state = engine.calculate_state(element, active_t_si, active_p_si)
    else:
        active_t_si = temperature_si
        active_p_si = pressure_si
        state = engine.calculate_state(element, active_t_si, active_p_si)

    active_t_u = conv.temp_from_si(state.temperature, t_unit)
    active_p_u = conv.pressure_from_si(state.pressure, p_unit)

    phase_colors = {
        PhaseState.SOLID: "#60A5FA",
        PhaseState.LIQUID: "#34D399",
        PhaseState.GAS: "#FBBF24",
        PhaseState.SUPERCRITICAL: "#C084FC",
    }
    phase_ru_title = state.phase.value.split(" (")[0]
    phase_en_sub = state.phase.value.split(" (")[1].rstrip(")")

    # Ряд 1: Метрики состояния
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_glass_card(
            f"Агрегатное состояние (t = {active_time_s:.1f} с)",
            phase_ru_title,
            f"{phase_en_sub} | T = {active_t_u:.2f} {t_unit.symbol}",
            accent_color=phase_colors.get(state.phase, "#38BDF8"),
        )
    with c2:
        render_glass_card(
            "Плотность и Молярный объем",
            f"{state.density:.3f} кг/м³",
            f"Vm = {state.molar_volume * 1e6:.2f} см³/моль",
        )
    with c3:
        render_glass_card(
            "Сжимаемость и Летучесть",
            f"Z = {state.compressibility_z:.4f}",
            f"Коэфф. летучести φ = {state.fugacity_coeff:.4f}",
        )
    with c4:
        render_glass_card(
            f"Чистое давление ({p_unit.symbol})",
            f"{active_p_u:,.4g} {p_unit.symbol}",
            f"{state.pressure:,.1f} Па | c = {state.speed_of_sound:.0f} м/с",
        )

    # Ряд 2: Калорические свойства
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_glass_card(
            "Энтальпия H(T, P)",
            f"{state.enthalpy / 1000.0:.2f} кДж/моль",
            f"Отклонение H_dep: {state.enthalpy_departure / 1000.0:+.2f} кДж/моль",
        )
    with k2:
        render_glass_card(
            "Энтропия S(T, P)",
            f"{state.entropy:.2f} Дж/(моль·К)",
            f"Отклонение S_dep: {state.entropy_departure:+.2f} Дж/(моль·К)",
        )
    with k3:
        render_glass_card(
            "Теплоемкости Cp и Cv",
            f"{state.cp:.2f} / {state.cv:.2f} Дж/(моль·К)",
            f"Показатель адиабаты γ = {state.cp / max(state.cv, 1e-6):.3f}",
        )
    with k4:
        render_glass_card(
            "Энергия Гиббса G(T, P)",
            f"{state.gibbs_energy / 1000.0:.2f} кДж/моль",
            f"Внутренняя U: {state.internal_energy / 1000.0:.2f} кДж/моль",
        )

    # Синхронный блок: 2D фазовая диаграмма (без плавающих кнопок) + 3D-куб с фазовыми переходами
    col_chart, col_cube = st.columns([1.32, 1.0])

    with col_chart:
        fig_phase = build_clean_phase_diagram(
            element=element,
            curves=curves,
            active_t_si=state.temperature,
            active_p_si=state.pressure,
            active_time_s=active_time_s,
            active_phase_ru=phase_ru_title,
            t_unit=t_unit,
            p_unit=p_unit,
            use_log_pressure_axis=use_log_axis,
            trajectory=trajectory,
            show_spinodals=show_spinodals,
        )
        st.plotly_chart(fig_phase, use_container_width=True, config=PLOTLY_CLEAN_CONFIG)

    with col_cube:
        # При ручной перемотке передаем срез в выбранной точке времени, при авто-цикле — всю траекторию RK4
        cube_trajectory = trajectory if auto_play_rk4 else None
        v_rms, box_nm = MolecularCubeVisualizer.render_realtime_molecular_cube(
            element=element,
            state=state,
            t_unit=t_unit,
            p_unit=p_unit,
            trajectory=cube_trajectory,
            auto_play_rk4=auto_play_rk4,
            playback_speed=playback_speed,
            auto_rotate_cube=auto_rotate_cube,
            restart_token=st.session_state.restart_token,
            max_particles=max_particles_3d,
            height=580,
        )

    if trajectory is not None:
        st.subheader("Нестационарная динамика во времени (Runge–Kutta 4)")
        init_s = engine.calculate_state(element, temperature_si, pressure_si)
        end_s = trajectory.final_state
        end_t_u = conv.temp_from_si(end_s.temperature, t_unit)
        dt_u = conv.temp_delta_from_si(end_s.temperature - init_s.temperature, t_unit)
        end_p_u = conv.pressure_from_si(end_s.pressure, p_unit)
        dp_u = conv.pressure_from_si(end_s.pressure - init_s.pressure, p_unit)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_glass_card(
                "Конечная температура T(t_end)",
                f"{end_t_u:.2f} {t_unit.symbol}",
                f"ΔT = {dt_u:+.2f} {t_unit.symbol}",
            )
        with m2:
            render_glass_card(
                "Конечное давление P(t_end)",
                f"{end_p_u:,.4g} {p_unit.symbol}",
                f"ΔP = {dp_u:+.4g} {p_unit.symbol}",
            )
        with m3:
            render_glass_card(
                "Приращение энтальпии ΔH",
                f"{(end_s.enthalpy - init_s.enthalpy) / 1000.0:+.2f} кДж/моль",
                f"ΔS = {end_s.entropy - init_s.entropy:+.2f} Дж/(моль·К)",
            )
        with m4:
            render_glass_card(
                "Фазовый переход процесса",
                f"{init_s.phase.value.split(' (')[0]} → {end_s.phase.value.split(' (')[0]}",
                f"Тепловая скорость ⟨v⟩ = {v_rms:.0f} м/с",
                accent_color=phase_colors.get(end_s.phase, "#38BDF8"),
            )

        fig_dyn = build_transient_dashboard(
            trajectory,
            t_unit=t_unit,
            p_unit=p_unit,
            selected_time_s=active_time_s if not auto_play_rk4 else None,
        )
        st.plotly_chart(fig_dyn, use_container_width=True, config=PLOTLY_CLEAN_CONFIG)

    with st.expander("Фундаментальные константы элемента (в выбранных единицах)", expanded=False):
        tc_disp = conv.temp_from_si(element.tc, t_unit)
        tt_disp = conv.temp_from_si(element.tt, t_unit)
        tb_disp = conv.temp_from_si(element.tb, t_unit)
        pc_disp = conv.pressure_from_si(element.pc, p_unit)
        pt_disp = conv.pressure_from_si(curves.pt_eos, p_unit)

        f1, f2, f3, f4 = st.columns(4)
        f1.write(f"**Формула / CAS:** `{element.formula}` (`{element.cas}`)")
        f1.write(f"**Молярная масса:** `{element.molar_mass * 1000.0:.3f} г/моль`")
        f2.write(f"**Критическая температура ($T_c$):** `{tc_disp:.2f} {t_unit.symbol}`")
        f2.write(f"**Критическое давление ($P_c$):** `{pc_disp:,.4g} {p_unit.symbol}`")
        f3.write(f"**Тройная точка ($T_t$):** `{tt_disp:.2f} {t_unit.symbol}`")
        f3.write(f"**Давление тройной точки ($P_t$):** `{pt_disp:.3e} {p_unit.symbol}`")
        f4.write(f"**Нормальная $T_b$ (1 атм):** `{tb_disp:.2f} {t_unit.symbol}`")
        f4.write(f"**Ацентрический фактор ($\omega$):** `{element.acentric_factor:.4f}`")


if __name__ == "__main__":
    main()