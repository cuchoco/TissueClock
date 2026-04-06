from .pan import train as train_pan
from .tissue import train as train_tissue
from .abmil import train as train_abmil
from .abmil_custom import train as train_abmil_custom
from .transmil import train as train_transmil
from .mambamil import train as train_mambamil
from .perceiver import train as train_perceiver

TRAINERS = {
    'pan': train_pan,
    'tissue': train_tissue,
    'abmil': train_abmil,
    'abmil_custom': train_abmil_custom,
    'transmil': train_transmil,
    'mambamil': train_mambamil,
    'perceiver': train_perceiver,
}
