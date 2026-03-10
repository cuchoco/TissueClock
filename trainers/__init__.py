from .pan import train as train_pan
from .tissue import train as train_tissue
from .abmil import train as train_abmil
from .transmil import train as train_transmil
from .mambamil import train as train_mambamil

TRAINERS = {
    'pan': train_pan,
    'tissue': train_tissue,
    'abmil': train_abmil,
    'transmil': train_transmil,
    'mambamil': train_mambamil,
}
