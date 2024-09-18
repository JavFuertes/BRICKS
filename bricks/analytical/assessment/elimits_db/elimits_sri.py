from dataclasses import dataclass, field
from typing import List, Union, Dict

@dataclass
class LimitEntry:
    source: str
    limits: List[Union[float, int]]
    description: List[str]
    DL: List[int]  # Damage Level indices

@dataclass
class ParameterLimits:
    Beta: List[LimitEntry] = field(default_factory=list)
    ΔSmax: List[LimitEntry] = field(default_factory=list)
    Phi: List[LimitEntry] = field(default_factory=list)
    Omega: List[LimitEntry] = field(default_factory=list)

def empirical_limits() -> ParameterLimits:
    return ParameterLimits(
        Beta=[
            LimitEntry(
                source='Boscardin & Cording (1989)',
                limits=[0, 1e-3, 1.5e-3, 3.25e-3, 6.5e-3, float('inf')],
                description=['No damage', 'Negligible damage', 'Slight', 'Moderate to severe', 'Severe to very severe'],
                DL=[0, 1, 2, 3, 4]
            ),
            LimitEntry(
                source='Skempton & McDonald (1956)',
                limits=[0, 3.33e-3, 6.66e-3, float('inf')],
                description=['No damage', 'Structural damage in beams or columns', 'Cracking in wall panels'],
                DL=[0, 3, 4]
            ),
            LimitEntry(
                source='Bjerrum (1963)',
                limits=[0, 2e-3, 3.33e-3, 6.66e-3, float('inf')],
                description=['No damage', 'Cracking', 'Severe cracking in panel walls', 'Serious cracking in panel walls and brick walls'],
                DL=[0, 1, 3, 4]
            ),
            LimitEntry(
                source='Polshin & Tokar (1957)',
                limits=[0, 5e-3, float('inf')],
                description=['No damage', 'First visible cracking to no infill walls'],
                DL=[0, 1]
            ),
            LimitEntry(
                source='Wood (1958)',
                limits=[0, 2.2e-3, float('inf')],
                description=['No damage', 'First visible cracking to brick panels and walls'],
                DL=[0, 1]
            ),
            LimitEntry(
                source='Bozuzuk (1962)',
                limits=[0, 1e-3, float('inf')],
                description=['No damage', 'Cracking of clay brick units with mortar'],
                DL=[0, 1]
            ),
            LimitEntry(
                source='Meyerhof (1953)',
                limits=[0, 2.5e-3, float('inf')],
                description=['No damage', 'Cracking'],
                DL=[0, 1]
            ),
        ],
        ΔSmax=[
            LimitEntry(
                source="Skempton & McDonald (1956)",
                limits=[0, 0.032, float('inf')],
                description=['No damage', 'Damage in sand (all types of foundation)'],
                DL=[0, 1]
            ),
            LimitEntry(
                source="Skempton & McDonald (1956)",
                limits=[0, 0.045, float('inf')],
                description=['No damage', 'Damage in clay (all types of foundation)'],
                DL=[0, 1]
            ),
        ],
        Phi=[
            LimitEntry(
                source="CUR (1996)",
                limits=[0, 2e-3, 3.3e-3, 10e-3, float('inf')],
                description=['No damage', 'Aesthetic damage', 'Structural damage', 'Risk for residents'],
                DL=[0, 1, 3, 4, 5]
            ),
        ],
        Omega=[
            LimitEntry(
                source="IGWR (2009)",
                limits=[0, 1/66, 1/50, 1/33, float('inf')],
                description=['No damage', 'Acceptable damage', 'Small damage', 'Considerable damage'],
                DL=[0, 1, 2, 3]
            ),
        ]
    )
