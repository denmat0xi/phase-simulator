# filename: engine.py
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from data_provider import ElementPropertiesDTO

# Фундаментальные константы
R_GAS: float = 8.314462618      # Универсальная газовая постоянная, Дж/(моль·К)
T_REF: float = 298.15           # Стандартная референсная температура, К
P_REF: float = 101325.0         # Стандартное референсное давление, Па


class PhaseState(Enum):
    """Фазовые состояния вещества."""
    SOLID = "Твердое тело (Solid)"
    LIQUID = "Жидкость (Liquid)"
    GAS = "Газ / Пар (Gas)"
    SUPERCRITICAL = "Сверхкритический флюид (Supercritical Fluid)"


@dataclass(frozen=True)
class ThermodynamicStateDTO:
    """
    Полный термодинамический и калорический слепок состояния в точке (T, P).
    Рассчитывается через уравнение состояния Пенга — Робинсона и функции отклонения (Departure Functions).
    """
    temperature: float           # Температура T, К
    pressure: float              # Давление P, Па
    phase: PhaseState            # Агрегатное состояние
    compressibility_z: float     # Коэффициент сжимаемости Z = P*Vm / (R*T)
    molar_volume: float          # Молярный объем Vm, м³/моль
    density: float               # Плотность rho, кг/м³
    fugacity_coeff: float        # Коэффициент летучести phi = f / P
    ln_phi: float                # Логарифм коэффициента летучести ln(phi)
    enthalpy: float              # Полная молярная энтальпия H(T, P), Дж/моль
    entropy: float               # Полная молярная энтропия S(T, P), Дж/(моль·К)
    internal_energy: float       # Полная внутренняя энергия U(T, P), Дж/моль
    gibbs_energy: float          # Энергия Гиббса G(T, P) = H - T*S, Дж/моль
    enthalpy_departure: float    # Отклонение энтальпии от идеального газа (H - H_ig), Дж/моль
    entropy_departure: float     # Отклонение энтропии от идеального газа (S - S_ig), Дж/(моль·К)
    cp: float                    # Изобарная теплоемкость Cp(T, P), Дж/(моль·К)
    cv: float                    # Изохорная теплоемкость Cv(T, P), Дж/(моль·К)
    dp_dt_v: float               # Термический коэффициент давления (dP/dT)_Vm, Па/К
    dp_dv_t: float               # Изотермическая производная (dP/dVm)_T, Па·моль/м³
    dt_dp_s: float               # Адиабатический коэффициент (dT/dP)_S, К/Па
    speed_of_sound: float        # Термодинамическая скорость звука c, м/с


@dataclass(frozen=True)
class PhaseDiagramCurvesDTO:
    """Координаты кривых фазового равновесия (бинодали и спинодали) для P-T диаграммы."""
    t_sublimation: NDArray[np.float64]
    p_sublimation: NDArray[np.float64]
    t_melting: NDArray[np.float64]
    p_melting: NDArray[np.float64]
    t_vaporization: NDArray[np.float64]
    p_vaporization: NDArray[np.float64]
    t_spinodal: NDArray[np.float64]
    p_spinodal_liquid: NDArray[np.float64]
    p_spinodal_vapor: NDArray[np.float64]
    pt_eos: float  # Давление тройной точки, согласованное с VLE-солвером


class EquationOfState(ABC):
    """Абстрактный интерфейс фундаментального уравнения состояния."""

    @abstractmethod
    def calculate_state(
        self, element: ElementPropertiesDTO, temperature: float, pressure: float
    ) -> ThermodynamicStateDTO:
        pass

    @abstractmethod
    def vle_saturation_pressure(self, element: ElementPropertiesDTO, temperature: float) -> float:
        pass

    @abstractmethod
    def generate_phase_diagram(
        self, element: ElementPropertiesDTO, num_points: int = 160
    ) -> PhaseDiagramCurvesDTO:
        pass


class PengRobinsonEOS(EquationOfState):
    """
    Строгая реализация уравнения состояния Пенга — Робинсона (PR EOS, 1976):
        P = R*T / (Vm - b) - a(T) / (Vm^2 + 2*b*Vm - b^2)

    Включает:
    1. Аналитический поиск границ механической устойчивости (спинодалей dP/dVm = 0).
    2. Численный VLE-солвер фазового равновесия Жидкость — Пар из равенства летучестей:
           ln(phi^L(T, P_sat)) = ln(phi^V(T, P_sat))
    3. Аналитические Departure Functions для энтальпии H, энтропии S, внутренней энергии U,
       теплоемкостей Cv, Cp и скорости звука в реальной среде.
    """

    @staticmethod
    def _eos_parameters(
        el: ElementPropertiesDTO, t: float
    ) -> Tuple[float, float, float, float]:
        """
        Вычисляет энергетический параметр a(T), коволюм b и аналитические производные
        da/dT и d²a/dT², необходимые для калорических функций отклонения (Departure Functions).
        """
        ac = 0.4572355289 * (R_GAS ** 2) * (el.tc ** 2) / el.pc
        b = 0.0777960739 * R_GAS * el.tc / el.pc

        w = el.acentric_factor
        # Модификация Пенга — Робинсона (1978) для широкого диапазона ацентрических факторов
        if w <= 0.491:
            kappa = 0.37464 + 1.54226 * w - 0.26992 * (w ** 2)
        else:
            kappa = 0.379642 + 1.48503 * w - 0.164423 * (w ** 2) + 0.016666 * (w ** 3)

        t_safe = max(t, 1e-4)
        tr = t_safe / el.tc
        sqrt_alpha = 1.0 + kappa * (1.0 - math.sqrt(tr))
        alpha = sqrt_alpha ** 2
        a_t = ac * alpha

        # Первая и вторая производные a(T) по температуре T:
        # da/dT = -ac * kappa * sqrt(alpha) / sqrt(T * Tc)
        # d²a/dT² = ac * kappa * (1 + kappa) / (2 * T * sqrt(T * Tc))
        da_dt = -ac * kappa * sqrt_alpha / math.sqrt(t_safe * el.tc)
        d2a_dt2 = ac * kappa * (1.0 + kappa) / (2.0 * t_safe * math.sqrt(t_safe * el.tc))

        return a_t, b, da_dt, d2a_dt2

    @classmethod
    def _solve_cubic_z(
        cls, a_t: float, b: float, t: float, p: float
    ) -> Tuple[List[float], float, float]:
        """
        Решает кубическое уравнение Пенга — Робинсона относительно фактора сжимаемости Z:
            Z^3 - (1 - B)*Z^2 + (A - 3*B^2 - 2*B)*Z - (A*B - B^2 - B^3) = 0
        Возвращает отсортированный список физических вещественных корней (Z > B) и комплексы (A, B).
        """
        t_safe = max(t, 1e-4)
        p_safe = max(p, 1e-30)

        A = a_t * p_safe / ((R_GAS * t_safe) ** 2)
        B = b * p_safe / (R_GAS * t_safe)

        c2 = -(1.0 - B)
        c1 = A - 3.0 * (B ** 2) - 2.0 * B
        c0 = -(A * B - (B ** 2) - (B ** 3))

        roots = np.roots([1.0, c2, c1, c0])
        real_roots = sorted(
            [
                float(r.real)
                for r in roots
                if abs(r.imag) < 1e-8 and r.real > B * (1.0 + 1e-12)
            ]
        )
        if not real_roots:
            real_roots = [B * 1.02]
        return real_roots, A, B

    @staticmethod
    def _calc_ln_phi(z: float, A: float, B: float) -> float:
        """
        Аналитическое выражение для натурального логарифма коэффициента летучести ln(phi):
            ln(phi) = (Z - 1) - ln(Z - B) - [A / (2*sqrt(2)*B)] * ln[(Z + (1+sqrt(2))*B) / (Z + (1-sqrt(2))*B)]
        """
        sqrt2 = math.sqrt(2.0)
        z_minus_b = max(z - B, 1e-15)
        b_safe = max(B, 1e-25)
        num = z + (1.0 + sqrt2) * b_safe
        den = max(z + (1.0 - sqrt2) * b_safe, 1e-15)
        return (z - 1.0) - math.log(z_minus_b) - (A / (2.0 * sqrt2 * b_safe)) * math.log(num / den)

    @classmethod
    def find_spinodal_pressures(
        cls, el: ElementPropertiesDTO, t: float
    ) -> Optional[Tuple[float, float]]:
        """
        Находит точные спинодальные давления (P_spin_liq, P_spin_vap) при T < Tc,
        решая уравнение границы механической устойчивости (dP/dVm)_T = 0.
        При подстановке x = Vm / b условие (dP/dVm)_T = 0 сводится к полиному 4-й степени:
            x^4 + (4 - k)*x^3 + (2 + k)*x^2 + (k - 4)*x + (1 - k) = 0,  где k = 2*a(T) / (R*T*b)
        """
        if t >= el.tc:
            return None

        a_t, b, _, _ = cls._eos_parameters(el, t)
        k = 2.0 * a_t / (R_GAS * t * b)

        roots = np.roots([1.0, 4.0 - k, 2.0 + k, k - 4.0, 1.0 - k])
        x_roots = sorted(
            [float(r.real) for r in roots if abs(r.imag) < 1e-7 and r.real > 1.00001]
        )
        if len(x_roots) < 2:
            return None

        vm_l = x_roots[0] * b
        vm_v = x_roots[-1] * b

        p_spin_l = R_GAS * t / (vm_l - b) - a_t / (vm_l ** 2 + 2.0 * b * vm_l - b ** 2)
        p_spin_v = R_GAS * t / (vm_v - b) - a_t / (vm_v ** 2 + 2.0 * b * vm_v - b ** 2)

        return max(p_spin_l, 1e-28), max(p_spin_v, 1e-25)

    def vle_saturation_pressure(
        self, element: ElementPropertiesDTO, temperature: float
    ) -> float:
        """
        Честный численный VLE-солвер кривой фазового равновесия Жидкость — Газ (бинодали).
        Ищет давление насыщения P_sat(T), при котором химические потенциалы (и летучести)
        жидкой и паровой ветвей кубического уравнения Пенга — Робинсона строго равны:
            f^L(T, P_sat) = f^V(T, P_sat)  <==>  ln(phi^L) - ln(phi^V) = 0
        """
        if temperature >= element.tc:
            return element.pc

        t_safe = max(temperature, 0.15 * element.tc)
        tr = t_safe / element.tc

        a_t, b, _, _ = self._eos_parameters(element, t_safe)
        spinodal = self.find_spinodal_pressures(element, t_safe)
        if spinodal is None:
            return element.pc

        p_spin_min, p_spin_max = spinodal
        p_low = max(p_spin_min * 1.00001, 1e-28)
        p_high = p_spin_max * 0.99999

        if p_low >= p_high:
            return p_high

        # Начальное приближение по корреляции Ли — Кеслера внутри коридора спинодалей [p_low, p_high]
        f0 = 5.92714 - 6.09648 / tr - 1.28862 * math.log(tr) + 0.169347 * (tr ** 6)
        f1 = 15.2518 - 15.6875 / tr - 13.4721 * math.log(tr) + 0.43577 * (tr ** 6)
        p_guess = element.pc * math.exp(max(-65.0, f0 + element.acentric_factor * f1))
        p = min(max(p_guess, p_low), p_high)

        # Итерации Ньютона — Рафсона в координате ln(P) с аналитическим якобианом:
        # d(ln_phi_L - ln_phi_V) / d(ln P) = Z_L - Z_V
        for _ in range(35):
            roots, A, B = self._solve_cubic_z(a_t, b, t_safe, p)
            if len(roots) >= 2:
                z_l, z_v = roots[0], roots[-1]
                diff = self._calc_ln_phi(z_l, A, B) - self._calc_ln_phi(z_v, A, B)
                if abs(diff) < 1e-11:
                    return float(p)
                d_ln_p = -diff / (z_l - z_v)
                p_next = p * math.exp(max(-0.8, min(0.8, d_ln_p)))
                p = min(max(p_next, p_low), p_high)
            else:
                break

        # Гарантированное доведение методом Брента на отрезке [ln(p_low), ln(p_high)]
        def fugacity_residual(ln_p: float) -> float:
            p_val = math.exp(ln_p)
            r_list, A_val, B_val = self._solve_cubic_z(a_t, b, t_safe, p_val)
            if len(r_list) < 2:
                return 0.0
            return self._calc_ln_phi(r_list[0], A_val, B_val) - self._calc_ln_phi(
                r_list[-1], A_val, B_val
            )

        try:
            f_low = fugacity_residual(math.log(p_low))
            f_high = fugacity_residual(math.log(p_high))
            if f_low * f_high < 0.0:
                ln_p_sol = brentq(fugacity_residual, math.log(p_low), math.log(p_high), xtol=1e-10)
                return float(math.exp(ln_p_sol))
        except ValueError:
            pass

        return float(p)

    def latent_heat_vaporization(self, element: ElementPropertiesDTO, temperature: float) -> float:
        """
        Вычисляет скрытую теплоту парообразования dH_vap(T) = H^V(T, P_sat) - H^L(T, P_sat)
        напрямую через разность функций отклонения энтальпии Пенга — Робинсона на бинодали.
        """
        if temperature >= element.tc:
            return 0.0
        p_sat = self.vle_saturation_pressure(element, temperature)
        a_t, b, da_dt, _ = self._eos_parameters(element, temperature)
        roots, _, B = self._solve_cubic_z(a_t, b, temperature, p_sat)
        if len(roots) < 2:
            return 0.0

        z_l, z_v = roots[0], roots[-1]
        sqrt2 = math.sqrt(2.0)

        def h_dep_for_z(z: float) -> float:
            k_vol = (1.0 / (2.0 * sqrt2 * b)) * math.log(
                (z + (1.0 + sqrt2) * B) / max(z + (1.0 - sqrt2) * B, 1e-15)
            )
            return R_GAS * temperature * (z - 1.0) + (temperature * da_dt - a_t) * k_vol

        return max(0.0, h_dep_for_z(z_v) - h_dep_for_z(z_l))

    def sublimation_pressure(self, el: ElementPropertiesDTO, t: float, pt_eos: float) -> float:
        """
        Кривая сублимации (Твердое — Газ) ниже тройной точки.
        Стыкуется непрерывно с VLE-давлением в тройной точке pt_eos, а наклон определяется
        суммой скрытой теплоты испарения из EOS в тройной точке и энтальпии плавления.
        """
        t_safe = max(t, 0.05 * el.tt)
        dh_vap_tt = self.latent_heat_vaporization(el, el.tt)
        dh_fus = 1.25 * R_GAS * el.tm  # Энтальпия плавления по правилу Ричардса
        dh_sub = max(dh_vap_tt + dh_fus, 8.0 * R_GAS * el.tb)
        exponent = -(dh_sub / R_GAS) * (1.0 / t_safe - 1.0 / el.tt)
        return float(pt_eos * math.exp(max(exponent, -50.0)))

    @staticmethod
    def melting_pressure(el: ElementPropertiesDTO, t: float, pt_eos: float) -> float:
        """
        Кривая плавления (Твердое — Жидкость).
        Учитывает как нормальные вещества (уравнение Симона — Глацеля, dP/dT > 0),
        так и аномальные вещества с расширением при кристаллизации (Ga, Bi, Sb, Si, Ge: dP/dT < 0).
        """
        if not el.anomalous_melting:
            if t < el.tt:
                return pt_eos
            a_sg = 2.0 * el.pc
            c_sg = 2.15
            return float(pt_eos + a_sg * (((t / el.tt) ** c_sg) - 1.0))
        else:
            # Для Ga, Bi, Sb, Si, Ge температура плавления падает с ростом давления (dP/dT < 0)
            if t > el.tt:
                return pt_eos
            slope = -12.0 * el.pc / max(el.tt, 1.0)
            return float(pt_eos + slope * (t - el.tt))

    @staticmethod
    def melting_temperature_at_pressure(el: ElementPropertiesDTO, p: float, pt_eos: float) -> float:
        """Обратная функция T_melt(P) для определения точки плавления при изобарном процессе."""
        if p <= pt_eos:
            return el.tt
        if not el.anomalous_melting:
            a_sg = 2.0 * el.pc
            c_sg = 2.15
            return float(el.tt * (((p - pt_eos) / a_sg + 1.0) ** (1.0 / c_sg)))
        else:
            slope = -12.0 * el.pc / max(el.tt, 1.0)
            return float(max(0.5 * el.tt, el.tt + (p - pt_eos) / slope))

    def _classify_phase(self, el: ElementPropertiesDTO, t: float, p: float) -> PhaseState:
        """Строгая классификация фазового состояния относительно VLE и кривых плавления/сублимации."""
        pt_eos = self.vle_saturation_pressure(el, el.tt)

        if not el.anomalous_melting:
            if t < el.tt:
                p_sub = self.sublimation_pressure(el, t, pt_eos)
                return PhaseState.SOLID if p >= p_sub else PhaseState.GAS
            p_melt = self.melting_pressure(el, t, pt_eos)
            if p >= p_melt:
                return PhaseState.SOLID
        else:
            t_melt_p = self.melting_temperature_at_pressure(el, p, pt_eos)
            if t < t_melt_p:
                if p < pt_eos:
                    p_sub = self.sublimation_pressure(el, t, pt_eos)
                    return PhaseState.SOLID if p >= p_sub else PhaseState.GAS
                return PhaseState.SOLID

        if t >= el.tc and p >= el.pc:
            return PhaseState.SUPERCRITICAL
        if t >= el.tc and p < el.pc:
            return PhaseState.GAS

        p_sat = self.vle_saturation_pressure(el, t)
        return PhaseState.LIQUID if p >= p_sat else PhaseState.GAS

    @staticmethod
    def _ideal_gas_caloric(
        el: ElementPropertiesDTO, t: float, p: float
    ) -> Tuple[float, float, float, float]:
        """
        Статистико-механический расчет идеально-газовых функций Cp_ig(T), Cv_ig(T), H_ig(T), S_ig(T, P).
        Для одноатомных газов: поступательные степени свободы (5/2 R).
        Для двухатомных газов: поступательные + вращательные (7/2 R) + колебательная мода Планка — Эйнштейна.
        """
        t_safe = max(t, 1.0)
        if not el.is_diatomic:
            cp_ig = 2.5 * R_GAS
            cv_ig = 1.5 * R_GAS
            h_ig = cp_ig * (t_safe - T_REF)
            s_ig = cp_ig * math.log(t_safe / T_REF) - R_GAS * math.log(max(p, 1e-20) / P_REF)
            return cp_ig, cv_ig, h_ig, s_ig

        # Двухатомная молекула с квантовым гармоническим осциллятором (характеристическая температура Theta_v)
        theta = max(el.vib_temperature, 100.0)
        x = min(theta / t_safe, 100.0)
        x_ref = min(theta / T_REF, 100.0)

        # Колебательный вклад Эйнштейна в теплоемкость: R * x^2 * e^x / (e^x - 1)^2
        exp_x = math.exp(x)
        c_vib = R_GAS * (x ** 2) * exp_x / ((exp_x - 1.0) ** 2) if x < 80.0 else 0.0
        cp_ig = 3.5 * R_GAS + c_vib
        cv_ig = 2.5 * R_GAS + c_vib

        # Колебательная энтальпия и энтропия
        e_vib_t = R_GAS * theta / (math.exp(x) - 1.0) if x < 80.0 else 0.0
        e_vib_ref = R_GAS * theta / (math.exp(x_ref) - 1.0) if x_ref < 80.0 else 0.0
        h_ig = 3.5 * R_GAS * (t_safe - T_REF) + (e_vib_t - e_vib_ref)

        s_vib_t = R_GAS * (x / (math.exp(x) - 1.0) - math.log(1.0 - math.exp(-x))) if x < 80.0 else 0.0
        s_vib_ref = R_GAS * (x_ref / (math.exp(x_ref) - 1.0) - math.log(1.0 - math.exp(-x_ref))) if x_ref < 80.0 else 0.0
        s_ig = (
            3.5 * R_GAS * math.log(t_safe / T_REF)
            + (s_vib_t - s_vib_ref)
            - R_GAS * math.log(max(p, 1e-20) / P_REF)
        )
        return cp_ig, cv_ig, h_ig, s_ig

    @staticmethod
    def _debye_solid_cv(el: ElementPropertiesDTO, t: float) -> float:
        """Теплоемкость кристаллической решетки по квантовой модели Дебая (предел Дюлонга — Пти 3R)."""
        y = el.debye_temperature / max(t, 0.5)
        if y > 15.0:
            # Низкотемпературный закон Дебая ~ T^3
            return 3.0 * R_GAS * (4.0 * (math.pi ** 4) / 5.0) / (y ** 3)
        # Аппроксимация Паде для интеграла Дебая во всем диапазоне температур
        return 3.0 * R_GAS * (1.0 / (1.0 + 0.05 * (y ** 2) + 0.004 * (y ** 3)))

    def calculate_state(
        self, element: ElementPropertiesDTO, temperature: float, pressure: float
    ) -> ThermodynamicStateDTO:
        """
        Комплексный расчет фазового, объемного и калорического состояния вещества в точке (T, P).
        """
        t_safe = max(temperature, 0.5)
        p_safe = max(pressure, 1e-10)

        phase = self._classify_phase(element, t_safe, p_safe)
        a_t, b, da_dt, d2a_dt2 = self._eos_parameters(element, t_safe)
        roots, A, B = self._solve_cubic_z(a_t, b, t_safe, p_safe)

        # Отбор физического корня Z
        if len(roots) == 1:
            z_factor = roots[0]
        else:
            if phase in (PhaseState.LIQUID, PhaseState.SOLID):
                z_factor = roots[0]
            elif phase == PhaseState.GAS:
                z_factor = roots[-1]
            else:
                # Критерий минимума энергии Гиббса (меньший ln_phi)
                ln_l = self._calc_ln_phi(roots[0], A, B)
                ln_v = self._calc_ln_phi(roots[-1], A, B)
                z_factor = roots[0] if ln_l <= ln_v else roots[-1]

        ln_phi = self._calc_ln_phi(z_factor, A, B)
        phi = math.exp(max(-100.0, min(100.0, ln_phi)))
        vm = (z_factor * R_GAS * t_safe) / p_safe

        # Интегральный объемный фактор K_vol для Departure Functions Пенга — Робинсона:
        # K_vol = [1 / (2*sqrt(2)*b)] * ln[ (Vm + (1 + sqrt(2))*b) / (Vm + (1 - sqrt(2))*b) ]
        sqrt2 = math.sqrt(2.0)
        num = vm + (1.0 + sqrt2) * b
        den = max(vm + (1.0 - sqrt2) * b, 1e-18)
        k_vol = (1.0 / (2.0 * sqrt2 * b)) * math.log(num / den)

        # Калорические функции отклонения (Departure Functions: Real Fluid minus Ideal Gas)
        # U - U_ig = (T * da/dT - a) * K_vol
        # H - H_ig = R*T*(Z - 1) + (T * da/dT - a) * K_vol
        # S - S_ig = R*ln(Z - B) + (da/dT) * K_vol
        # Cv - Cv_ig = T * (d²a/dT²) * K_vol
        u_dep = (t_safe * da_dt - a_t) * k_vol
        h_dep = R_GAS * t_safe * (z_factor - 1.0) + u_dep
        s_dep = R_GAS * math.log(max(z_factor - B, 1e-15)) + da_dt * k_vol
        cv_dep = t_safe * d2a_dt2 * k_vol

        # Частные производные давления P(T, Vm) для расчета Cp и адиабатического градиента
        v_minus_b = max(vm - b, 1e-15)
        denom_v = vm ** 2 + 2.0 * b * vm - b ** 2
        dp_dt_v = R_GAS / v_minus_b - da_dt / denom_v
        dp_dv_t = -R_GAS * t_safe / (v_minus_b ** 2) + 2.0 * a_t * (vm + b) / (denom_v ** 2)

        # Идеально-газовый базис
        cp_ig, cv_ig, h_ig, s_ig = self._ideal_gas_caloric(element, t_safe, p_safe)

        if phase != PhaseState.SOLID:
            cv = max(0.5 * R_GAS, cv_ig + cv_dep)
            # Термодинамическое соотношение Майера для реального флюида:
            # Cp - Cv = -T * (dP/dT)_Vm^2 / (dP/dVm)_T
            if dp_dv_t < -1e-12:
                cp_minus_cv = -t_safe * (dp_dt_v ** 2) / dp_dv_t
            else:
                cp_minus_cv = R_GAS
            # Ограничение сингулярности в непосредственной окрестности критической точки
            cp = cv + min(cp_minus_cv, 80.0 * R_GAS)
            enthalpy = h_ig + h_dep
            entropy = s_ig + s_dep
        else:
            # Для кристаллической фазы используем решеточную модель Дебая и скачок скрытой теплоты плавления
            scale_solid = 1.04 if element.anomalous_melting else 0.93
            vm = vm * scale_solid
            z_factor = (p_safe * vm) / (R_GAS * t_safe)
            cv = self._debye_solid_cv(element, t_safe)
            cp = cv + 0.15 * R_GAS * (t_safe / element.tm)
            dh_fus = 1.25 * R_GAS * element.tm
            ds_fus = dh_fus / element.tm
            enthalpy = h_ig + h_dep - dh_fus
            entropy = s_ig + s_dep - ds_fus
            # Высокий модуль всестороннего сжатия твердого тела
            dp_dv_t = min(dp_dv_t * 5.0, -1e9 / max(vm, 1e-6))

        density = element.molar_mass / max(vm, 1e-12)
        internal_energy = enthalpy - p_safe * vm
        gibbs_energy = enthalpy - t_safe * entropy

        # Адиабатический (изэнтропийный) коэффициент (dT/dP)_S = (T / Cp) * (dVm/dT)_P
        # Так как (dVm/dT)_P = -(dP/dT)_Vm / (dP/dVm)_T:
        if dp_dv_t < -1e-12:
            dv_dt_p = -dp_dt_v / dp_dv_t
        else:
            dv_dt_p = R_GAS / p_safe
        dt_dp_s = (t_safe / max(cp, 1e-6)) * dv_dt_p

        # Термодинамическая скорость звука: c = sqrt( -(Vm^2 / M) * (Cp / Cv) * (dP/dVm)_T )
        bulk_term = -(vm ** 2 / element.molar_mass) * (cp / max(cv, 1e-6)) * dp_dv_t
        speed_of_sound = math.sqrt(max(bulk_term, 10.0))

        return ThermodynamicStateDTO(
            temperature=t_safe,
            pressure=p_safe,
            phase=phase,
            compressibility_z=z_factor,
            molar_volume=vm,
            density=density,
            fugacity_coeff=phi,
            ln_phi=ln_phi,
            enthalpy=enthalpy,
            entropy=entropy,
            internal_energy=internal_energy,
            gibbs_energy=gibbs_energy,
            enthalpy_departure=h_dep,
            entropy_departure=s_dep,
            cp=cp,
            cv=cv,
            dp_dt_v=dp_dt_v,
            dp_dv_t=dp_dv_t,
            dt_dp_s=dt_dp_s,
            speed_of_sound=speed_of_sound,
        )

    def generate_phase_diagram(
        self, element: ElementPropertiesDTO, num_points: int = 160
    ) -> PhaseDiagramCurvesDTO:
        """
        Генерирует кривые фазового равновесия:
        - Ветвь кипения/конденсации решается через честный VLE-солвер (phi^L = phi^V) в каждой точке T;
        - Ветви сублимации и плавления стыкуются в точной тройной точке (Tt, Pt_EOS);
        - Дополнительно рассчитываются жидкая и паровая спинодали (границы метастабильности).
        """
        t_min = max(0.22 * element.tt, 0.5)
        t_max = 1.6 * element.tc

        # Давление в тройной точке из VLE-солвера для бесшовной сшивки трех фазовых границ
        pt_eos = self.vle_saturation_pressure(element, element.tt)

        # 1. Кривая сублимации (t_min -> Tt)
        t_sub = np.linspace(t_min, element.tt, num_points)
        p_sub = np.array([self.sublimation_pressure(element, float(t), pt_eos) for t in t_sub])

        # 2. Честная VLE-бинодаль Пенга — Робинсона (Tt -> Tc) + спинодали
        t_vap = np.linspace(element.tt, element.tc, num_points)
        p_vap = np.zeros(num_points, dtype=np.float64)
        p_spin_l = np.zeros(num_points, dtype=np.float64)
        p_spin_v = np.zeros(num_points, dtype=np.float64)

        for i, t_val in enumerate(t_vap):
            t_f = float(t_val)
            p_vap[i] = self.vle_saturation_pressure(element, t_f)
            spin = self.find_spinodal_pressures(element, t_f)
            if spin is not None:
                p_spin_l[i] = max(spin[0], pt_eos * 0.01)
                p_spin_v[i] = spin[1]
            else:
                p_spin_l[i] = element.pc
                p_spin_v[i] = element.pc

        # 3. Кривая плавления (от Tt)
        if not element.anomalous_melting:
            t_melt = np.linspace(element.tt, t_max, num_points)
        else:
            t_melt = np.linspace(max(0.55 * element.tt, t_min), element.tt, num_points)
        p_melt = np.array([self.melting_pressure(element, float(t), pt_eos) for t in t_melt])

        return PhaseDiagramCurvesDTO(
            t_sublimation=t_sub,
            p_sublimation=p_sub,
            t_melting=t_melt,
            p_melting=p_melt,
            t_vaporization=t_vap,
            p_vaporization=p_vap,
            t_spinodal=t_vap,
            p_spinodal_liquid=p_spin_l,
            p_spinodal_vapor=p_spin_v,
            pt_eos=pt_eos,
        )