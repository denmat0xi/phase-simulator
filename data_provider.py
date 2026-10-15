# filename: data_provider.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Set

import chemicals
from chemicals.elements import periodic_table


@dataclass(frozen=True)
class ElementPropertiesDTO:
    """
    Неизменяемый DTO с фундаментальными термодинамическими параметрами элемента.
    Все величины приведены к системе СИ (К, Па, кг/моль).
    """
    atomic_number: int
    symbol: str
    name: str
    formula: str
    cas: str
    molar_mass: float               # Молярная масса, кг/моль
    tc: float                       # Критическая температура, К
    pc: float                       # Критическое давление, Па
    tb: float                       # Нормальная температура кипения (при 1 атм), К
    tm: float                       # Температура плавления, К
    tt: float                       # Температура тройной точки, К
    pt_exp: float                   # Экспериментальное/оценочное давление тройной точки, Па
    acentric_factor: float          # Ацентрический фактор Питцера (omega)
    is_diatomic: bool               # Флаг двухатомной молекулы (H2, N2, O2, F2, Cl2, Br2, I2)
    vib_temperature: float          # Характеристическая колебательная температура Theta_vib, К
    debye_temperature: float        # Температура Дебая кристаллической решетки Theta_D, К
    anomalous_melting: bool         # Аномалия плавления (dP/dT < 0: Ga, Bi, Sb, Si, Ge)
    is_estimated_critical: bool     # Флаг оценки Tc/Pc по корреляциям Гулдберга — Риделя


class ElementDataProvider:
    """
    Сервис динамической загрузки и нормализации физико-химических свойств
    элементов периодической таблицы из научной библиотеки `chemicals` (`thermo`).
    """

    # Двухатомные простые вещества и их CAS-идентификаторы
    _DIATOMIC_CAS_MAP: Dict[str, str] = {
        "H": "1333-74-0",   # H2
        "N": "7727-37-9",   # N2
        "O": "7782-44-7",   # O2
        "F": "7782-41-4",   # F2
        "Cl": "7782-50-5",  # Cl2
        "Br": "7726-95-6",  # Br2
        "I": "7553-56-2",   # I2
    }

    # Характеристические колебательные температуры двухатомных молекул, К (спектроскопия ИК/КР)
    _VIB_TEMPERATURES: Dict[str, float] = {
        "H": 6332.0,
        "N": 3395.0,
        "O": 2274.0,
        "F": 1320.0,
        "Cl": 805.0,
        "Br": 465.0,
        "I": 308.0,
    }

    # Элементы с уменьшением молярного объема при плавлении (dP_melt / dT < 0)
    _ANOMALOUS_MELTING_SYMBOLS: Set[str] = {"Ga", "Bi", "Sb", "Si", "Ge"}

    _P_ATM: float = 101325.0

    @classmethod
    def load_all_elements(cls) -> Dict[str, ElementPropertiesDTO]:
        """Возвращает упорядоченный по атомному номеру каталог элементов."""
        catalog: Dict[str, ElementPropertiesDTO] = {}
        for el in periodic_table:
            dto = cls._build_element_dto(el)
            if dto is not None:
                key = f"{dto.atomic_number:02d} — {dto.symbol} ({dto.name})"
                catalog[key] = dto
        return dict(sorted(catalog.items(), key=lambda item: item[1].atomic_number))

    @classmethod
    def _build_element_dto(cls, el: object) -> Optional[ElementPropertiesDTO]:
        symbol: str = getattr(el, "symbol")
        atomic_number: int = getattr(el, "number")
        name: str = getattr(el, "name")
        mw_g_mol: Optional[float] = getattr(el, "MW", None)

        if mw_g_mol is None or mw_g_mol <= 0.0:
            return None

        is_diatomic = symbol in cls._DIATOMIC_CAS_MAP
        if is_diatomic:
            cas = cls._DIATOMIC_CAS_MAP[symbol]
            formula = f"{symbol}2"
            molar_mass_kg = (mw_g_mol * 2.0) * 1e-3
        else:
            cas = getattr(el, "CAS")
            formula = symbol
            molar_mass_kg = mw_g_mol * 1e-3

        tb = chemicals.Tb(cas) or getattr(el, "Tb", None)
        tm = chemicals.Tm(cas) or getattr(el, "Tm", None)
        tc = chemicals.Tc(cas)
        pc = chemicals.Pc(cas)
        omega = chemicals.omega(cas)
        tt = chemicals.Tt(cas)
        pt = chemicals.Pt(cas)

        is_estimated = False
        if (tc is None or pc is None) and (tb is not None and tm is not None):
            is_estimated = True
            tc = 1.85 * tb
            pc = cls._P_ATM * math.exp(4.8 * (tc / tb - 1.0))

        if tc is None or pc is None:
            return None

        if tb is None or tb >= tc:
            tb = 0.62 * tc
        if tm is None or tm <= 0.0:
            tm = max(0.95, 0.25 * tb)
        if tm >= tb:
            tm = 0.85 * tb

        if omega is None:
            tr_b = tb / tc
            if 0.1 < tr_b < 0.99:
                omega = (3.0 / 7.0) * (tr_b / (1.0 - tr_b)) * math.log10(pc / cls._P_ATM) - 1.0
                omega = max(-0.38, min(omega, 1.2))
            else:
                omega = 0.0

        if tt is None or not (0.2 * tm <= tt < tc):
            tt = tm
        if pt is None or not (0.0 < pt < pc):
            tr_t = max(tt / tc, 0.18)
            f0 = 5.92714 - 6.09648 / tr_t - 1.28862 * math.log(tr_t) + 0.169347 * (tr_t ** 6)
            f1 = 15.2518 - 15.6875 / tr_t - 13.4721 * math.log(tr_t) + 0.43577 * (tr_t ** 6)
            pt = min(pc * math.exp(max(f0 + omega * f1, -28.0)), 0.4 * pc)

        vib_temp = cls._VIB_TEMPERATURES.get(symbol, 0.0)
        # Оценка температуры Дебая по формуле Линдемана из температуры плавления и молярной массы
        debye_temp = max(40.0, min(1800.0, 135.0 * math.sqrt(tm / max(mw_g_mol, 1.0))))

        return ElementPropertiesDTO(
            atomic_number=atomic_number,
            symbol=symbol,
            name=name,
            formula=formula,
            cas=cas,
            molar_mass=float(molar_mass_kg),
            tc=float(tc),
            pc=float(pc),
            tb=float(tb),
            tm=float(tm),
            tt=float(tt),
            pt_exp=float(pt),
            acentric_factor=float(omega),
            is_diatomic=is_diatomic,
            vib_temperature=float(vib_temp),
            debye_temperature=float(debye_temp),
            anomalous_melting=(symbol in cls._ANOMALOUS_MELTING_SYMBOLS),
            is_estimated_critical=is_estimated,
        )