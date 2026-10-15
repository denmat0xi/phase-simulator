# filename: dynamics.py
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from data_provider import ElementPropertiesDTO
from engine import PengRobinsonEOS, R_GAS, ThermodynamicStateDTO


class ProcessType(Enum):
    """Типы нестационарных термодинамических процессов."""
    ISOBARIC = "Изобарный процесс (P = const, нагрев / охлаждение)"
    ISOCHORIC = "Изохорный процесс (V = const, жесткий автоклав)"
    ADIABATIC = "Адиабатический процесс (Q = 0, S = const, сжатие / расширение)"


@dataclass(frozen=True)
class SimulationConfigDTO:
    """Параметры запуска нестационарного численного интегрирования."""
    process_type: ProcessType
    t_initial: float            # Начальная температура T(0), К
    p_initial: float            # Начальное давление P(0), Па
    dt: float                   # Шаг интегрирования по времени dt, с
    num_steps: int              # Число шагов по времени
    heat_power_w_mol: float     # Подводимая/отводимая тепловая мощность dQ/dt, Вт/моль (Дж/(с·моль))
    pressure_rate_pa_s: float   # Скорость изменения давления dP/dt для адиабаты, Па/с


@dataclass(frozen=True)
class TransientTrajectoryDTO:
    """Временные ряды всех макропараметров вдоль рассчитанной RK4-траектории."""
    time_s: NDArray[np.float64]
    temperature_k: NDArray[np.float64]
    pressure_pa: NDArray[np.float64]
    molar_volume: NDArray[np.float64]
    density: NDArray[np.float64]
    compressibility_z: NDArray[np.float64]
    enthalpy: NDArray[np.float64]
    entropy: NDArray[np.float64]
    cp: NDArray[np.float64]
    cv: NDArray[np.float64]
    phase_labels: List[str]
    final_state: ThermodynamicStateDTO


class TransientProcessSimulator:
    """
    Численный интегратор нестационарной термодинамической динамики методом
    Рунге — Кутты 4-го порядка (Runge–Kutta 4th Order, RK4).

    Интегрирует систему ОДУ для вектора состояния y = [T(t), P(t)]^T:
    1. Изобарный (P = const):
         dT/dt = (dQ/dt) / Cp_eff(T, P),   dP/dt = 0
       где Cp_eff включает сглаженный гауссиан скрытой теплоты плавления и парообразования,
       воспроизводящий температурные плато (аресты) при фазовых переходах 1-го рода.
    2. Изохорный (Vm = const):
         dT/dt = (dQ/dt) / Cv(T, P),       dP/dt = (dP/dT)_Vm * (dT/dt)
    3. Адиабатический (Q = 0, dS = 0):
         dP/dt = u_P,                      dT/dt = (dT/dP)_S * (dP/dt)
    """

    def __init__(self, eos: PengRobinsonEOS) -> None:
        self._eos = eos

    def _effective_isobaric_cp(
        self, el: ElementPropertiesDTO, state: ThermodynamicStateDTO, pt_eos: float
    ) -> float:
        """
        Метод эффективной теплоемкости (Effective Heat Capacity Method) для учета скрытой теплоты
        фазовых переходов 1-го рода (плавление и кипение) при интегрировании во времени.
        """
        cp_base = state.cp
        sigma_t = max(0.008 * el.tc, 0.8)  # Полуширина размытия фазового фронта по температуре, К

        # 1. Скрытая теплота плавления на кривой T_melt(P)
        t_melt = self._eos.melting_temperature_at_pressure(el, state.pressure, pt_eos)
        dh_fus = 1.25 * R_GAS * el.tm
        gauss_fus = math.exp(-0.5 * (((state.temperature - t_melt) / sigma_t) ** 2)) / (
            sigma_t * math.sqrt(2.0 * math.pi)
        )

        # 2. Скрытая теплота парообразования (если P_t < P < P_c)
        cp_latent_vap = 0.0
        if pt_eos <= state.pressure < el.pc:
            # Находим температуру кипения при данном давлении и соответствующую dH_vap из PR EOS
            dh_vap = self._eos.latent_heat_vaporization(el, min(state.temperature, 0.995 * el.tc))
            p_sat_local = self._eos.vle_saturation_pressure(el, state.temperature)
            # Оценка близости к линии насыщения через уравнение Клапейрона dP/dT
            dp_dt_sat = max(state.dp_dt_v, 1.0)
            delta_t_sat = (state.pressure - p_sat_local) / dp_dt_sat
            gauss_vap = math.exp(-0.5 * ((delta_t_sat / sigma_t) ** 2)) / (
                sigma_t * math.sqrt(2.0 * math.pi)
            )
            cp_latent_vap = dh_vap * gauss_vap

        return cp_base + dh_fus * gauss_fus + cp_latent_vap

    def _ode_rhs(
        self,
        t_val: float,
        p_val: float,
        el: ElementPropertiesDTO,
        cfg: SimulationConfigDTO,
        pt_eos: float,
    ) -> Tuple[float, float]:
        """
        Правая часть системы дифференциальных уравнений f(t, y) = [dT/dt, dP/dt].
        """
        t_clamped = max(1.0, min(t_val, 5.0 * el.tc))
        p_clamped = max(1e-5, min(p_val, 200.0 * el.pc))
        state = self._eos.calculate_state(el, t_clamped, p_clamped)

        if cfg.process_type == ProcessType.ISOBARIC:
            cp_eff = self._effective_isobaric_cp(el, state, pt_eos)
            dt_dt = cfg.heat_power_w_mol / max(cp_eff, 1.0)
            dp_dt = 0.0
            return dt_dt, dp_dt

        if cfg.process_type == ProcessType.ISOCHORIC:
            dt_dt = cfg.heat_power_w_mol / max(state.cv, 1.0)
            # Для изохоры dVm = 0 => dP = (dP/dT)_Vm * dT
            dp_dt = state.dp_dt_v * dt_dt
            return dt_dt, dp_dt

        # ProcessType.ADIABATIC (Q = 0, изэнтропийное сжатие/расширение)
        dp_dt = cfg.pressure_rate_pa_s
        dt_dt = state.dt_dp_s * dp_dt
        return dt_dt, dp_dt

    def run_simulation(
        self, element: ElementPropertiesDTO, cfg: SimulationConfigDTO
    ) -> TransientTrajectoryDTO:
        """
        Выполняет пошаговое численное интегрирование методом Рунге — Кутты 4-го порядка (RK4):
            k1 = f(y_n)
            k2 = f(y_n + 0.5 * dt * k1)
            k3 = f(y_n + 0.5 * dt * k2)
            k4 = f(y_n + dt * k3)
            y_{n+1} = y_n + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        """
        n = cfg.num_steps + 1
        dt = cfg.dt
        pt_eos = self._eos.vle_saturation_pressure(element, element.tt)

        time_arr = np.zeros(n, dtype=np.float64)
        t_arr = np.zeros(n, dtype=np.float64)
        p_arr = np.zeros(n, dtype=np.float64)
        vm_arr = np.zeros(n, dtype=np.float64)
        rho_arr = np.zeros(n, dtype=np.float64)
        z_arr = np.zeros(n, dtype=np.float64)
        h_arr = np.zeros(n, dtype=np.float64)
        s_arr = np.zeros(n, dtype=np.float64)
        cp_arr = np.zeros(n, dtype=np.float64)
        cv_arr = np.zeros(n, dtype=np.float64)
        phases: List[str] = []

        curr_t = cfg.t_initial
        curr_p = cfg.p_initial
        last_state: ThermodynamicStateDTO | None = None

        for i in range(n):
            state = self._eos.calculate_state(element, curr_t, curr_p)
            last_state = state

            time_arr[i] = i * dt
            t_arr[i] = state.temperature
            p_arr[i] = state.pressure
            vm_arr[i] = state.molar_volume
            rho_arr[i] = state.density
            z_arr[i] = state.compressibility_z
            h_arr[i] = state.enthalpy
            s_arr[i] = state.entropy
            cp_arr[i] = state.cp
            cv_arr[i] = state.cv
            phases.append(state.phase.value.split(" (")[0])

            if i == n - 1:
                break

            # Классический шаг Рунге — Кутты 4-го порядка (RK4)
            k1_t, k1_p = self._ode_rhs(curr_t, curr_p, element, cfg, pt_eos)

            t2 = max(1.0, curr_t + 0.5 * dt * k1_t)
            p2 = max(1e-3, curr_p + 0.5 * dt * k1_p)
            k2_t, k2_p = self._ode_rhs(t2, p2, element, cfg, pt_eos)

            t3 = max(1.0, curr_t + 0.5 * dt * k2_t)
            p3 = max(1e-3, curr_p + 0.5 * dt * k2_p)
            k3_t, k3_p = self._ode_rhs(t3, p3, element, cfg, pt_eos)

            t4 = max(1.0, curr_t + dt * k3_t)
            p4 = max(1e-3, curr_p + dt * k3_p)
            k4_t, k4_p = self._ode_rhs(t4, p4, element, cfg, pt_eos)

            curr_t = max(1.0, min(curr_t + (dt / 6.0) * (k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t), 4.0 * element.tc))
            curr_p = max(1e-2, min(curr_p + (dt / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p), 150.0 * element.pc))

        assert last_state is not None
        return TransientTrajectoryDTO(
            time_s=time_arr,
            temperature_k=t_arr,
            pressure_pa=p_arr,
            molar_volume=vm_arr,
            density=rho_arr,
            compressibility_z=z_arr,
            enthalpy=h_arr,
            entropy=s_arr,
            cp=cp_arr,
            cv=cv_arr,
            phase_labels=phases,
            final_state=last_state,
        )